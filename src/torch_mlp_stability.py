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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
BEAM_DIR = ROOT / "results" / "beam"
RESULTS_DIR = ROOT / "results" / "compact_regressors"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class MLPConfig:
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 80
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


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_split(target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)

    train = train.dropna(subset=[target]).reset_index(drop=True)
    test = test.dropna(subset=[target]).reset_index(drop=True)
    train = train[np.isfinite(train[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    test = test[np.isfinite(test[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    return train, test, waves


def load_beam_indices(target: str, budget: int, waves: list[str]) -> tuple[np.ndarray, str]:
    beam = pd.read_csv(BEAM_DIR / f"{target}_beam_band_selection_results.csv")
    row = beam[beam["budget"] == budget].sort_values("rmse").iloc[0]
    selected_waves = str(row["selected_wavelengths"]).split(",")
    wave_to_idx = {wave: idx for idx, wave in enumerate(waves)}
    selected_idx = np.asarray([wave_to_idx[wave] for wave in selected_waves], dtype=int)
    return selected_idx, ",".join(selected_waves)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = rmse(y_true, y_pred)
    if value == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / value)


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

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
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


def run_trait(target: str, budget: int, seeds: list[int], config: MLPConfig, device: torch.device) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)

    x_train = train[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_test = test[target].to_numpy(dtype=np.float32)

    rows: list[dict[str, object]] = []
    print(f"\nTrait={target} | train={len(y_train)} test={len(y_test)} budget={budget} device={device}")
    for seed in seeds:
        pred, timing = fit_predict(x_train, y_train, x_test, seed, config, device)
        row = {
            "target": target,
            "selector": "BeamSearch",
            "model": "TorchMLP",
            "seed": seed,
            "budget": budget,
            "bands": budget,
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


def save_figures(summary: pd.DataFrame) -> None:
    for metric in ["r2", "rmse"]:
        plt.figure(figsize=(8.3, 4.7))
        data = summary.sort_values("target")
        plt.errorbar(
            data["target"],
            data[f"{metric}_mean"],
            yerr=data[f"{metric}_std"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=4,
            label="TorchMLP",
        )
        plt.xlabel("Trait")
        plt.ylabel(metric.upper())
        plt.title(f"BeamSearch K=30 TorchMLP stability: {metric.upper()}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"beam_k30_torch_mlp_stability_{metric}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable PyTorch MLP on BeamSearch-selected compact wavelengths.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=30)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config = MLPConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, args.seeds, config, device))

    results = pd.DataFrame(rows).sort_values(["target", "seed"])
    summary = (
        results.groupby(["target", "selector", "model", "budget", "bands"], as_index=False)
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
        .sort_values(["target"])
    )

    results_path = RESULTS_DIR / f"beam_k{args.budget}_torch_mlp_stability_results.csv"
    summary_path = RESULTS_DIR / f"beam_k{args.budget}_torch_mlp_stability_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(SUMMARY_DIR / f"beam_k{args.budget}_torch_mlp_stability_summary.csv", index=False)
    save_figures(summary)

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(SUMMARY_DIR / f"beam_k{args.budget}_torch_mlp_stability_summary.csv")
    print(FIG_DIR / "beam_k30_torch_mlp_stability_r2.png")
    print(FIG_DIR / "beam_k30_torch_mlp_stability_rmse.png")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
