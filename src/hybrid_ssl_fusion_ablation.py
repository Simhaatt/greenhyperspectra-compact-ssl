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

from beam_supervised_modern_heads import HeadConfig, make_model as make_labelled_model
from hybrid_ssl_beam_gated_heads import GatedHeadConfig, make_model as make_cross_gated_model
from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "hybrid_ssl_fusion_ablation"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class FusionConfig:
    batch_size: int = 128
    max_epochs: int = 800
    patience: int = 60
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    hidden_1: int = 128
    hidden_2: int = 64
    fusion_hidden: int = 128
    dropout: float = 0.05


class PlainMLP(nn.Module):
    def __init__(self, in_features: int, config: FusionConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AdditiveFusion(nn.Module):
    def __init__(self, beam_features: int, ssl_features: int, config: FusionConfig) -> None:
        super().__init__()
        h = config.fusion_hidden
        self.beam_proj = nn.Sequential(nn.Linear(beam_features, h), nn.LayerNorm(h), nn.SiLU())
        self.ssl_proj = nn.Sequential(nn.Linear(ssl_features, h), nn.LayerNorm(h), nn.SiLU())
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(h, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        return self.head(self.beam_proj(beam) + self.ssl_proj(ssl)).squeeze(-1)


class FiLMFusion(nn.Module):
    def __init__(self, beam_features: int, ssl_features: int, config: FusionConfig) -> None:
        super().__init__()
        h = config.fusion_hidden
        self.beam_proj = nn.Sequential(nn.Linear(beam_features, h), nn.LayerNorm(h), nn.SiLU())
        self.film = nn.Sequential(
            nn.Linear(ssl_features, h),
            nn.SiLU(),
            nn.Linear(h, h * 2),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Dropout(config.dropout),
            nn.Linear(h, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b = self.beam_proj(beam)
        gamma, beta = self.film(ssl).chunk(2, dim=1)
        conditioned = b * (1.0 + 0.25 * torch.tanh(gamma)) + 0.25 * torch.tanh(beta)
        return self.head(conditioned).squeeze(-1)


def single_model(model_key: str, in_features: int, config: FusionConfig) -> nn.Module:
    if model_key == "beam_only_residual":
        head_config = HeadConfig(
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
        )
        return make_labelled_model("residual", in_features, head_config)
    if model_key in {"ssl_only_mlp", "concat_mlp"}:
        return PlainMLP(in_features, config)
    raise ValueError(f"Unknown single-input model: {model_key}")


def dual_model(model_key: str, beam_features: int, ssl_features: int, config: FusionConfig) -> nn.Module:
    if model_key == "additive_fusion":
        return AdditiveFusion(beam_features, ssl_features, config)
    if model_key == "film_fusion":
        return FiLMFusion(beam_features, ssl_features, config)
    if model_key == "cross_gated_nam":
        gated_config = GatedHeadConfig(
            batch_size=config.batch_size,
            max_epochs=config.max_epochs,
            patience=config.patience,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            dropout=config.dropout,
        )
        return make_cross_gated_model("cross_gated_nam_head", beam_features, ssl_features, gated_config)
    raise ValueError(f"Unknown dual-input model: {model_key}")


def fit_single(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    model_key: str,
    seed: int,
    config: FusionConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.15, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = single_model(model_key, x_train.shape[1], config).to(device)
    return train_single_loop(model, loader, torch.from_numpy(x_val).to(device), torch.from_numpy(y_val_s).to(device), torch.from_numpy(x_test_s).to(device), y_scaler, config)


def fit_dual(
    beam_train_all: np.ndarray,
    ssl_train_all: np.ndarray,
    y_train_all: np.ndarray,
    beam_test: np.ndarray,
    ssl_test: np.ndarray,
    model_key: str,
    seed: int,
    config: FusionConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(len(y_train_all))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=seed)
    beam_scaler = StandardScaler()
    ssl_scaler = StandardScaler()
    y_scaler = StandardScaler()
    beam_train = beam_scaler.fit_transform(beam_train_all[train_idx]).astype(np.float32)
    beam_val = beam_scaler.transform(beam_train_all[val_idx]).astype(np.float32)
    beam_test_s = beam_scaler.transform(beam_test).astype(np.float32)
    ssl_train = ssl_scaler.fit_transform(ssl_train_all[train_idx]).astype(np.float32)
    ssl_val = ssl_scaler.transform(ssl_train_all[val_idx]).astype(np.float32)
    ssl_test_s = ssl_scaler.transform(ssl_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train_all[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_train_all[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(beam_train), torch.from_numpy(ssl_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = dual_model(model_key, beam_train.shape[1], ssl_train.shape[1], config).to(device)
    return train_dual_loop(
        model,
        loader,
        torch.from_numpy(beam_val).to(device),
        torch.from_numpy(ssl_val).to(device),
        torch.from_numpy(y_val_s).to(device),
        torch.from_numpy(beam_test_s).to(device),
        torch.from_numpy(ssl_test_s).to(device),
        y_scaler,
        config,
    )


def train_single_loop(
    model: nn.Module,
    loader: DataLoader,
    x_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    x_test_t: torch.Tensor,
    y_scaler: StandardScaler,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(x_val_t.device)
            yb = yb.to(x_val_t.device)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        pred_s = model(x_test_t).detach().cpu().numpy().reshape(-1)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {"train_seconds": train_seconds, "best_epoch": float(best_epoch), "best_val_mse_scaled": best_val}


def train_dual_loop(
    model: nn.Module,
    loader: DataLoader,
    beam_val_t: torch.Tensor,
    ssl_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    beam_test_t: torch.Tensor,
    ssl_test_t: torch.Tensor,
    y_scaler: StandardScaler,
    config: FusionConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for beam_b, ssl_b, yb in loader:
            beam_b = beam_b.to(y_val_t.device)
            ssl_b = ssl_b.to(y_val_t.device)
            yb = yb.to(y_val_t.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(beam_b, ssl_b), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(beam_val_t, ssl_val_t), y_val_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        pred_s = model(beam_test_t, ssl_test_t).detach().cpu().numpy().reshape(-1)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {"train_seconds": train_seconds, "best_epoch": float(best_epoch), "best_val_mse_scaled": best_val}


MODEL_META = {
    "beam_only_residual": {
        "model": "BeamK30_ResidualMLP",
        "fusion_method": "Beam only",
        "family": "labelled_only",
        "comment": "Strong compact labelled baseline.",
    },
    "ssl_only_mlp": {
        "model": "SSL128_Only_MLP",
        "fusion_method": "SSL only",
        "family": "ssl_ablation",
        "comment": "Tests whether the SSL embedding alone carries trait signal.",
    },
    "concat_mlp": {
        "model": "BeamK30_SSL128_ConcatMLP",
        "fusion_method": "Beam + SSL concat",
        "family": "fusion_ablation",
        "comment": "Simple concatenation baseline.",
    },
    "additive_fusion": {
        "model": "BeamK30_SSL128_AdditiveFusion",
        "fusion_method": "Beam + SSL additive",
        "family": "fusion_ablation",
        "comment": "Projects both streams to a shared latent space and adds them.",
    },
    "film_fusion": {
        "model": "BeamK30_SSL128_FiLMFusion",
        "fusion_method": "FiLM conditioning",
        "family": "fusion_ablation",
        "comment": "Uses SSL context to scale and shift the Beam representation.",
    },
    "cross_gated_nam": {
        "model": "Hybrid_BeamK30_SSL_CrossGatedNAMHead",
        "fusion_method": "Cross-gated NAM",
        "family": "proposed_hybrid",
        "comment": "Proposed hybrid fusion/head combination.",
    },
}


def run_trait(
    target: str,
    budget: int,
    ckpt: dict,
    models: list[str],
    seeds: list[int],
    config: FusionConfig,
    device: torch.device,
    embedding_batch_size: int,
) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    x_test_full = test[waves].to_numpy(dtype=np.float32)
    x_train_beam = x_train_full[:, selected_idx]
    x_test_beam = x_test_full[:, selected_idx]
    y_train = train[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    print(f"\nTrait={target} train={len(y_train)} test={len(y_test)} | embedding full spectra...")
    emb_train = ssl_embeddings(x_train_full, ckpt, device, embedding_batch_size)
    emb_test = ssl_embeddings(x_test_full, ckpt, device, embedding_batch_size)
    concat_train = np.concatenate([x_train_beam, emb_train], axis=1).astype(np.float32)
    concat_test = np.concatenate([x_test_beam, emb_test], axis=1).astype(np.float32)

    rows: list[dict[str, object]] = []
    for model_key in models:
        meta = MODEL_META[model_key]
        for seed in seeds:
            if model_key == "beam_only_residual":
                pred, timing = fit_single(x_train_beam, y_train, x_test_beam, model_key, seed, config, device)
                features = x_train_beam.shape[1]
                ssl_dim = 0
            elif model_key == "ssl_only_mlp":
                pred, timing = fit_single(emb_train, y_train, emb_test, model_key, seed, config, device)
                features = emb_train.shape[1]
                ssl_dim = emb_train.shape[1]
            elif model_key == "concat_mlp":
                pred, timing = fit_single(concat_train, y_train, concat_test, model_key, seed, config, device)
                features = concat_train.shape[1]
                ssl_dim = emb_train.shape[1]
            else:
                pred, timing = fit_dual(x_train_beam, emb_train, y_train, x_test_beam, emb_test, model_key, seed, config, device)
                features = x_train_beam.shape[1] + emb_train.shape[1]
                ssl_dim = emb_train.shape[1]
            row = {
                "target": target,
                "family": meta["family"],
                "fusion_method": meta["fusion_method"],
                "model": meta["model"],
                "model_key": model_key,
                "seed": seed,
                "budget": budget,
                "beam_bands": budget if model_key != "ssl_only_mlp" else 0,
                "ssl_embedding_dim": ssl_dim,
                "features": features,
                "rmse": rmse(y_test, pred),
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
                "rpd": rpd(y_test, pred),
                "device": str(device),
                "selected_wavelengths": selected_wavelengths if model_key != "ssl_only_mlp" else "",
                "comment": meta["comment"],
                **timing,
            }
            rows.append(row)
            print(
                f"  {target} {meta['fusion_method']} seed={seed}: "
                f"R2={row['r2']:.4f} RMSE={row['rmse']:.6f} epoch={row['best_epoch']:.0f}"
            )
    return rows


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        results.groupby(
            ["target", "family", "fusion_method", "model", "model_key", "budget", "beam_bands", "ssl_embedding_dim", "features", "comment"],
            as_index=False,
        )
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
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )
    rows = []
    for (fusion_method, model), sub in summary.groupby(["fusion_method", "model"]):
        top_traits = sub.sort_values("r2_mean", ascending=False).head(4)
        comment = str(sub["comment"].iloc[0])
        rows.append(
            {
                "fusion_method": fusion_method,
                "model": model,
                "mean_r2": float(sub["r2_mean"].mean()),
                "mean_rmse": float(sub["rmse_mean"].mean()),
                "best_traits": ", ".join(f"{r.target} ({r.r2_mean:.3f})" for r in top_traits.itertuples()),
                "comment": comment,
            }
        )
    fusion_table = pd.DataFrame(rows).sort_values("mean_r2", ascending=False)
    return summary, fusion_table


def save_figures(summary: pd.DataFrame, fusion_table: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = fusion_table.sort_values("mean_r2")
    plt.figure(figsize=(9.8, 5.0))
    plt.barh(data["fusion_method"], data["mean_r2"], color="#4C78A8")
    plt.xlabel("Mean R2 across traits")
    plt.title("Fusion ablation leaderboard")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hybrid_ssl_fusion_ablation_{run_name}_leaderboard.png", dpi=220)
    plt.close()

    pivot = summary.pivot_table(index="target", columns="fusion_method", values="r2_mean", aggfunc="mean")
    plt.figure(figsize=(11.5, 5.6))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=col)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    plt.ylabel("R2")
    plt.title("Fusion ablation by trait")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"hybrid_ssl_fusion_ablation_{run_name}_by_trait.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results" / "ssl" / "masked_autoencoder_contiguous_b64.pt")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--models", nargs="+", default=list(MODEL_META), choices=list(MODEL_META))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=512)
    parser.add_argument("--run-name", default="fusion_ablation_all8_3seeds")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    ckpt = load_checkpoint(args.checkpoint, device)
    config = FusionConfig(max_epochs=args.max_epochs, patience=args.patience, batch_size=args.batch_size)

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, ckpt, args.models, args.seeds, config, device, args.embedding_batch_size))
    results = pd.DataFrame(rows)
    summary, fusion_table = summarize(results)

    results_path = RESULTS_DIR / f"hybrid_ssl_fusion_ablation_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"hybrid_ssl_fusion_ablation_{args.run_name}_summary.csv"
    fusion_path = RESULTS_DIR / f"hybrid_ssl_fusion_ablation_{args.run_name}_fusion_table.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    fusion_table.to_csv(fusion_path, index=False)
    summary.to_csv(SUMMARY_DIR / summary_path.name, index=False)
    fusion_table.to_csv(SUMMARY_DIR / fusion_path.name, index=False)
    save_figures(summary, fusion_table, args.run_name)
    print("\nFusion ablation table:")
    print(fusion_table.round(4).to_string(index=False))
    print("Saved:", results_path)
    print("Saved:", summary_path)
    print("Saved:", fusion_path)


if __name__ == "__main__":
    main()
