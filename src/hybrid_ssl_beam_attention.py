from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "hybrid_ssl_beam_attention"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class AttentionConfig:
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    gate_hidden: int = 64
    hidden_1: int = 128
    hidden_2: int = 64
    dropout: float = 0.05
    gate_residual: float = 0.5


class AttentiveFusionRegressor(nn.Module):
    def __init__(self, in_features: int, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Sequential(
            nn.Linear(in_features, config.gate_hidden),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, in_features),
            nn.Sigmoid(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.gate(x)
        fused = x * (self.config.gate_residual + weights)
        return self.regressor(fused).squeeze(-1)


def fit_predict(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    config: AttentionConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.15,
        random_state=seed,
    )
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    x_test_t = torch.from_numpy(x_test_scaled).to(device)

    model = AttentiveFusionRegressor(x_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
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
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        pred_s = model(x_test_t).detach().cpu().numpy().reshape(-1)
    predict_seconds = perf_counter() - start
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }


def run_trait(
    target: str,
    budget: int,
    ckpt: dict,
    seeds: list[int],
    config: AttentionConfig,
    device: torch.device,
    embedding_batch_size: int,
) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    x_test_full = test[waves].to_numpy(dtype=np.float32)
    x_train_beam = x_train_full[:, selected_idx]
    x_test_beam = x_test_full[:, selected_idx]
    emb_train = ssl_embeddings(x_train_full, ckpt, device, embedding_batch_size)
    emb_test = ssl_embeddings(x_test_full, ckpt, device, embedding_batch_size)
    x_train = np.concatenate([x_train_beam, emb_train], axis=1).astype(np.float32)
    x_test = np.concatenate([x_test_beam, emb_test], axis=1).astype(np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    rows: list[dict[str, object]] = []
    print(f"\nTrait={target} | train={len(y_train)} test={len(y_test)} attention_features={x_train.shape[1]} device={device}")
    for seed in seeds:
        pred, timing = fit_predict(x_train, y_train, x_test, seed, config, device)
        row = {
            "target": target,
            "selector": "BeamSearch",
            "model": "SABFNet_SSL_Beam_Attentive_Fusion",
            "seed": seed,
            "budget": budget,
            "beam_bands": budget,
            "ssl_embedding_dim": emb_train.shape[1],
            "features": x_train.shape[1],
            "rmse": rmse(y_test, pred),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
            "rpd": rpd(y_test, pred),
            "device": str(device),
            "selected_wavelengths": selected_wavelengths,
            **timing,
        }
        rows.append(row)
        print(f"  seed={seed}: RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}, epoch={row['best_epoch']:.0f}")
    return rows


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ["r2", "rmse"]:
        data = summary.sort_values("target")
        plt.figure(figsize=(8.5, 4.8))
        plt.errorbar(
            data["target"],
            data[f"{metric}_mean"],
            yerr=data[f"{metric}_std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=4,
            label="SABF-Net",
        )
        plt.xlabel("Trait")
        plt.ylabel(metric.upper())
        plt.title(f"SABF-Net hybrid SSL + BeamSearch: {metric.upper()}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hybrid_ssl_beam_attention_{run_name}_{metric}.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SABF-Net: SSL-guided attentive fusion over BeamSearch bands and SSL embeddings.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=["cab", "cw", "cm", "cbc"], choices=TRAITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--run-name", default="contiguous_b64_pilot")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    config = AttentionConfig()

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, ckpt, args.seeds, config, device, args.embedding_batch_size))
    results = pd.DataFrame(rows).sort_values(["target", "seed"])
    summary = (
        results.groupby(["target", "selector", "model", "budget", "beam_bands", "ssl_embedding_dim", "features"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            predict_seconds_mean=("predict_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values("target")
    )
    results_path = RESULTS_DIR / f"hybrid_ssl_beam_attention_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"hybrid_ssl_beam_attention_{args.run_name}_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(SUMMARY_DIR / f"hybrid_ssl_beam_attention_{args.run_name}_summary.csv", index=False)
    save_figures(summary, args.run_name)

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(SUMMARY_DIR / f"hybrid_ssl_beam_attention_{args.run_name}_summary.csv")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
