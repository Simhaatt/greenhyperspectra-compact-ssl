from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ssl_pretrain_mae import SpectralEncoder
from torch_mlp_stability import load_split, rpd, rmse


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
SSL_DIR = ROOT / "results" / "ssl"
OUT_DIR = ROOT / "results" / "ssl_label_efficiency"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


class SSLRegressor(nn.Module):
    def __init__(self, encoder: SpectralEncoder, latent_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z).squeeze(-1)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_xy(target: str, wave_cols: list[str], mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train, test, _ = load_split(target)
    x_train = train[wave_cols].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test = test[wave_cols].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)
    return x_train, y_train, x_test, y_test


def fit_predict(
    ckpt: dict,
    x_all: np.ndarray,
    y_all: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    fraction: float,
    device: torch.device,
    freeze_encoder: bool,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + int(fraction * 1000))
    n_total = len(y_all)
    n_subset = max(24, int(round(n_total * fraction)))
    n_subset = min(n_subset, n_total)
    subset_idx = rng.choice(n_total, size=n_subset, replace=False)
    x_subset = x_all[subset_idx]
    y_subset = y_all[subset_idx]

    x_train, x_val, y_train, y_val = train_test_split(x_subset, y_subset, test_size=0.2, random_state=seed)
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train).unsqueeze(1), torch.from_numpy(y_train_s)),
        batch_size=min(batch_size, len(y_train_s)),
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).unsqueeze(1).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    x_test_t = torch.from_numpy(x_test).unsqueeze(1).to(device)

    latent_channels = int(ckpt.get("latent_channels", 128))
    encoder = SpectralEncoder(latent_channels=latent_channels)
    encoder.load_state_dict(ckpt["encoder_state"])
    model = SSLRegressor(encoder, latent_channels=latent_channels).to(device)
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.001, weight_decay=0.0005)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()

    for epoch in range(1, epochs + 1):
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
        if stale >= patience:
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
        "train_subset_rows": n_subset,
        "train_total_rows": n_total,
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }


def save_figures(summary: pd.DataFrame, suffix: str) -> None:
    for metric in ["r2", "rmse"]:
        plt.figure(figsize=(10, 5.6))
        for trait in TRAITS:
            data = summary[summary["target"] == trait].sort_values("label_percent")
            if data.empty:
                continue
            plt.errorbar(
                data["label_percent"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"].fillna(0),
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=trait,
            )
        plt.xlabel("Training labels used (%)")
        plt.ylabel(metric.upper())
        plt.title(f"SSL encoder label efficiency: {metric.upper()}")
        plt.xticks([10, 25, 50, 100])
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        if suffix == "finetuned":
            figure_name = f"ssl_label_efficiency_{metric}.png"
        else:
            figure_name = f"ssl_label_efficiency_{suffix}_{metric}.png"
        plt.savefig(FIG_DIR / figure_name, dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SSL-pretrained spectral encoder for label-efficiency regression.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder.pt"))
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--run-name", default=None, help="Optional output suffix, e.g. contiguous_b64.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.run_name if args.run_name else ("frozen" if args.freeze_encoder else "finetuned")
    results_path = OUT_DIR / f"ssl_label_efficiency_{suffix}_results.csv"
    summary_path = OUT_DIR / f"ssl_label_efficiency_{suffix}_summary.csv"
    summary_copy_path = SUMMARY_DIR / f"ssl_label_efficiency_{suffix}_summary.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    wave_cols = ckpt["wave_cols"]
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    completed: set[tuple[str, float, int]] = set()
    if results_path.exists():
        existing = pd.read_csv(results_path)
        if not existing.empty:
            rows = existing.to_dict("records")
            completed = {
                (str(row["target"]), round(float(row["label_fraction"]), 4), int(row["seed"]))
                for row in rows
            }
            print(f"Resuming from {results_path} with {len(completed)} completed runs.")

    for trait in args.traits:
        x_train, y_train, x_test, y_test = load_xy(trait, wave_cols, mean, std)
        for fraction in args.fractions:
            for seed in args.seeds:
                key = (trait, round(float(fraction), 4), int(seed))
                if key in completed:
                    print(f"skip trait={trait} fraction={fraction:.2f} seed={seed} already complete")
                    continue
                print(f"trait={trait} fraction={fraction:.2f} seed={seed} device={device}")
                pred, extra = fit_predict(
                    ckpt,
                    x_train,
                    y_train,
                    x_test,
                    seed,
                    fraction,
                    device,
                    args.freeze_encoder,
                    args.epochs,
                    args.patience,
                    args.batch_size,
                )
                row = {
                    "target": trait,
                    "model": (
                        f"SSL_Encoder_Regressor_{suffix}"
                        if args.run_name
                        else ("SSL_Encoder_Regressor_Frozen" if args.freeze_encoder else "SSL_Encoder_Regressor")
                    ),
                    "label_fraction": fraction,
                    "label_percent": int(round(fraction * 100)),
                    "seed": seed,
                    "test_rows": len(y_test),
                    "rmse": rmse(y_test, pred),
                    "mae": float(mean_absolute_error(y_test, pred)),
                    "r2": float(r2_score(y_test, pred)),
                    "rpd": rpd(y_test, pred),
                    "device": str(device),
                    **extra,
                }
                rows.append(row)
                pd.DataFrame(rows).sort_values(["target", "label_fraction", "seed"]).to_csv(results_path, index=False)
                print(f"  rows={row['train_subset_rows']} RMSE={row['rmse']:.6g} R2={row['r2']:.4f}")

    results = pd.DataFrame(rows).sort_values(["target", "label_fraction", "seed"])
    summary = (
        results.groupby(["target", "model", "label_fraction", "label_percent"], as_index=False)
        .agg(
            train_subset_rows_mean=("train_subset_rows", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["target", "label_fraction"])
    )

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(summary_copy_path, index=False)
    save_figures(summary, suffix)
    print("\nSaved:")
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
