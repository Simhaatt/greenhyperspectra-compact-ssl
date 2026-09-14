from __future__ import annotations
import os

import argparse
from dataclasses import dataclass
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

from torch_mlp_stability import load_beam_indices, load_split, rpd, rmse


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results"
OUT_DIR = RESULTS_DIR / "label_efficiency"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIG_DIR = RESULTS_DIR / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


@dataclass
class MLPConfig:
    batch_size: int = 64
    max_epochs: int = 800
    patience: int = 70
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.05


class TorchMLP(nn.Module):
    def __init__(self, in_features: int, config: MLPConfig) -> None:
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


def fit_predict(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    config: MLPConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if len(y_train_all) < 20:
        raise ValueError("Too few training samples after sub-sampling.")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=seed,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled)),
        batch_size=min(config.batch_size, len(y_train_scaled)),
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_scaled).to(device)
    x_test_t = torch.from_numpy(x_test_scaled).to(device)

    model = TorchMLP(x_train.shape[1], config).to(device)
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
        pred_scaled = model(x_test_t).detach().cpu().numpy().reshape(-1)
    predict_seconds = perf_counter() - start
    pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    return pred, {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }


def run_trait_fraction(
    trait: str,
    fraction: float,
    seed: int,
    config: MLPConfig,
    device: torch.device,
    budget: int,
) -> dict[str, object]:
    train, test, waves = load_split(trait)
    selected_idx, selected_wavelengths = load_beam_indices(trait, budget, waves)
    x_all = train[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_all = train[trait].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_test = test[trait].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed + int(fraction * 1000))
    n_total = len(y_all)
    n_subset = max(24, int(round(n_total * fraction)))
    n_subset = min(n_subset, n_total)
    subset_idx = rng.choice(n_total, size=n_subset, replace=False)

    pred, extra = fit_predict(x_all[subset_idx], y_all[subset_idx], x_test, seed, config, device)
    return {
        "target": trait,
        "selector": "BeamSearch",
        "model": "TorchMLP",
        "budget": budget,
        "bands": budget,
        "label_fraction": fraction,
        "label_percent": int(round(fraction * 100)),
        "seed": seed,
        "train_subset_rows": n_subset,
        "train_total_rows": n_total,
        "test_rows": len(y_test),
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rpd": rpd(y_test, pred),
        "device": str(device),
        "selected_wavelengths": selected_wavelengths,
        **extra,
    }


def save_figures(summary: pd.DataFrame) -> None:
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
        plt.title(f"Label efficiency: BeamSearch K=30 + TorchMLP ({metric.upper()})")
        plt.xticks([10, 25, 50, 100])
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"label_efficiency_{metric}.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-efficiency experiment for BeamSearch K=30 + TorchMLP.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config = MLPConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, object]] = []

    for trait in args.traits:
        for fraction in args.fractions:
            for seed in args.seeds:
                print(f"trait={trait} fraction={fraction:.2f} seed={seed}")
                row = run_trait_fraction(trait, fraction, seed, config, device, args.budget)
                rows.append(row)
                print(f"  rows={row['train_subset_rows']} RMSE={row['rmse']:.6g} R2={row['r2']:.4f}")

    results = pd.DataFrame(rows).sort_values(["target", "label_fraction", "seed"])
    summary = (
        results.groupby(["target", "selector", "model", "budget", "bands", "label_fraction", "label_percent"], as_index=False)
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

    results.to_csv(OUT_DIR / "label_efficiency_results.csv", index=False)
    summary.to_csv(OUT_DIR / "label_efficiency_summary.csv", index=False)
    summary.to_csv(SUMMARY_DIR / "label_efficiency_summary.csv", index=False)
    save_figures(summary)

    print("\nSaved:")
    print(OUT_DIR / "label_efficiency_results.csv")
    print(OUT_DIR / "label_efficiency_summary.csv")
    print(SUMMARY_DIR / "label_efficiency_summary.csv")
    print(FIG_DIR / "label_efficiency_r2.png")
    print(FIG_DIR / "label_efficiency_rmse.png")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
