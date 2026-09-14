from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from torch_mlp_stability import load_beam_indices, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "hybrid_ssl_beam_physics_multitask"
SUMMARY_DIR = ROOT / "results" / "summary"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class PhysicsConfig:
    batch_size: int = 128
    max_epochs: int = 700
    patience: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    branch_hidden: int = 64
    hidden: int = 96
    dropout: float = 0.05


class CrossGatedMultiTaskNet(nn.Module):
    def __init__(self, beam_features: int, ssl_features: int, n_traits: int, config: PhysicsConfig) -> None:
        super().__init__()
        h = config.branch_hidden
        self.beam_branch = nn.Sequential(
            nn.Linear(beam_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.ssl_branch = nn.Sequential(
            nn.Linear(ssl_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.beam_to_ssl_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.ssl_to_beam_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.Linear(h * 2, config.hidden),
            nn.LayerNorm(config.hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden, n_traits),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b = self.beam_branch(beam)
        s = self.ssl_branch(ssl)
        s = s * (0.5 + self.beam_to_ssl_gate(b))
        b = b * (0.5 + self.ssl_to_beam_gate(s))
        return self.head(torch.cat([b, s], dim=1))


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).isdigit()]


def load_data(traits: list[str], budget: int, ckpt: dict, device: torch.device, embedding_batch_size: int) -> tuple:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)
    union_idx: set[int] = set()
    selected_by_trait: dict[str, list[float]] = {}
    for trait in traits:
        idx, selected = load_beam_indices(trait, budget, waves)
        union_idx.update(int(i) for i in idx)
        selected_by_trait[trait] = selected
    union_idx_sorted = sorted(union_idx)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    x_test_full = test[waves].to_numpy(dtype=np.float32)
    x_train_beam = x_train_full[:, union_idx_sorted]
    x_test_beam = x_test_full[:, union_idx_sorted]
    emb_train = ssl_embeddings(x_train_full, ckpt, device, embedding_batch_size)
    emb_test = ssl_embeddings(x_test_full, ckpt, device, embedding_batch_size)
    y_train = train[traits].to_numpy(dtype=np.float32)
    y_test = test[traits].to_numpy(dtype=np.float32)
    return x_train_beam, emb_train, y_train, x_test_beam, emb_test, y_test, selected_by_trait


def scale_targets(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.nanmean(y_train, axis=0).astype(np.float32)
    stds = np.nanstd(y_train, axis=0).astype(np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds).astype(np.float32)
    return (y_train - means) / stds, (y_test - means) / stds, means, stds


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - y) * mask
    return torch.sum(diff * diff) / torch.clamp(torch.sum(mask), min=1.0)


def correlation_pairs(y_scaled: np.ndarray, traits: list[str], threshold: float = 0.6) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for i in range(len(traits)):
        for j in range(i + 1, len(traits)):
            mask = np.isfinite(y_scaled[:, i]) & np.isfinite(y_scaled[:, j])
            if int(mask.sum()) < 20:
                continue
            corr = float(np.corrcoef(y_scaled[mask, i], y_scaled[mask, j])[0, 1])
            if np.isfinite(corr) and abs(corr) >= threshold:
                rows.append((i, j, corr))
    return rows


def physics_corr_loss(pred: torch.Tensor, pairs: list[tuple[int, int, float]]) -> torch.Tensor:
    if not pairs:
        return pred.new_tensor(0.0)
    losses = []
    for i, j, target_corr in pairs:
        a = pred[:, i] - pred[:, i].mean()
        b = pred[:, j] - pred[:, j].mean()
        denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b) + 1e-6)
        corr = torch.sum(a * b) / denom
        losses.append((corr - float(target_corr)) ** 2)
    return torch.stack(losses).mean()


def fit_predict(
    x_beam: np.ndarray,
    x_ssl: np.ndarray,
    y: np.ndarray,
    x_beam_test: np.ndarray,
    x_ssl_test: np.ndarray,
    seed: int,
    traits: list[str],
    physics_lambda: float,
    config: PhysicsConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float], list[tuple[int, int, float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    y_scaled, _, means, stds = scale_targets(y, y)
    mask = np.isfinite(y_scaled).astype(np.float32)
    y_filled = np.nan_to_num(y_scaled, nan=0.0).astype(np.float32)
    pairs = correlation_pairs(y_scaled, traits)

    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=seed)
    beam_scaler = StandardScaler()
    ssl_scaler = StandardScaler()
    beam_train = beam_scaler.fit_transform(x_beam[train_idx]).astype(np.float32)
    beam_val = beam_scaler.transform(x_beam[val_idx]).astype(np.float32)
    beam_test = beam_scaler.transform(x_beam_test).astype(np.float32)
    ssl_train = ssl_scaler.fit_transform(x_ssl[train_idx]).astype(np.float32)
    ssl_val = ssl_scaler.transform(x_ssl[val_idx]).astype(np.float32)
    ssl_test = ssl_scaler.transform(x_ssl_test).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(beam_train),
            torch.from_numpy(ssl_train),
            torch.from_numpy(y_filled[train_idx]),
            torch.from_numpy(mask[train_idx]),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )
    beam_val_t = torch.from_numpy(beam_val).to(device)
    ssl_val_t = torch.from_numpy(ssl_val).to(device)
    y_val_t = torch.from_numpy(y_filled[val_idx]).to(device)
    mask_val_t = torch.from_numpy(mask[val_idx]).to(device)
    beam_test_t = torch.from_numpy(beam_test).to(device)
    ssl_test_t = torch.from_numpy(ssl_test).to(device)

    model = CrossGatedMultiTaskNet(x_beam.shape[1], x_ssl.shape[1], len(traits), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for beam_b, ssl_b, yb, mb in train_loader:
            beam_b = beam_b.to(device)
            ssl_b = ssl_b.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(beam_b, ssl_b)
            loss = masked_mse(pred, yb, mb)
            if physics_lambda > 0:
                loss = loss + physics_lambda * physics_corr_loss(pred, pairs)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(beam_val_t, ssl_val_t)
            val_loss = float(masked_mse(val_pred, y_val_t, mask_val_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    train_seconds = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(beam_test_t, ssl_test_t).detach().cpu().numpy()
    pred = pred_scaled * stds.reshape(1, -1) + means.reshape(1, -1)
    return pred, {"train_seconds": train_seconds, "best_epoch": float(best_epoch), "best_val_mse_scaled": best_val}, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked multitask cross-gated model with optional physics/correlation constraints.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=["cab", "cw", "cm", "cbc"], choices=TRAITS)
    parser.add_argument("--physics-lambdas", nargs="+", type=float, default=[0.0, 0.02])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--run-name", default="local_seed0_probe")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    config = PhysicsConfig()
    data = load_data(args.traits, args.budget, ckpt, device, args.embedding_batch_size)
    x_beam, x_ssl, y, x_beam_test, x_ssl_test, y_test, _ = data
    rows: list[dict[str, object]] = []

    print(f"Traits={args.traits} | train={len(y)} test={len(y_test)} union_beam={x_beam.shape[1]} ssl={x_ssl.shape[1]} device={device}")
    for physics_lambda in args.physics_lambdas:
        for seed in args.seeds:
            pred, timing, pairs = fit_predict(
                x_beam, x_ssl, y, x_beam_test, x_ssl_test, seed, args.traits, physics_lambda, config, device
            )
            print("Physics lambda:", physics_lambda, "pairs:", [(args.traits[i], args.traits[j], round(c, 3)) for i, j, c in pairs])
            for t_idx, trait in enumerate(args.traits):
                mask = np.isfinite(y_test[:, t_idx])
                if int(mask.sum()) == 0:
                    continue
                row = {
                    "target": trait,
                    "model": "CrossGatedPhysicsMultitask" if physics_lambda > 0 else "CrossGatedMultitask",
                    "seed": seed,
                    "physics_lambda": physics_lambda,
                    "budget": args.budget,
                    "union_beam_bands": x_beam.shape[1],
                    "ssl_embedding_dim": x_ssl.shape[1],
                    "rmse": rmse(y_test[mask, t_idx], pred[mask, t_idx]),
                    "mae": float(mean_absolute_error(y_test[mask, t_idx], pred[mask, t_idx])),
                    "r2": float(r2_score(y_test[mask, t_idx], pred[mask, t_idx])),
                    "rpd": rpd(y_test[mask, t_idx], pred[mask, t_idx]),
                    "device": str(device),
                    **timing,
                }
                rows.append(row)
                print(f"  {row['model']} {trait}: RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}, epoch={row['best_epoch']:.0f}")

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["target", "model", "physics_lambda", "budget", "union_beam_bands", "ssl_embedding_dim"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd_mean=("rpd", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["model", "target"])
    )
    results_path = RESULTS_DIR / f"hybrid_ssl_beam_physics_multitask_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"hybrid_ssl_beam_physics_multitask_{args.run_name}_summary.csv"
    summary_copy = SUMMARY_DIR / f"hybrid_ssl_beam_physics_multitask_{args.run_name}_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(summary_copy, index=False)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(summary)


if __name__ == "__main__":
    main()
