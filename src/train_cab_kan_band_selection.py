from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

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
BAND_DIR = ROOT / "results" / "band_selection"
RESULTS_DIR = ROOT / "results" / "kan"
TARGET = "cab"
RANDOM_STATE = 42
BUDGETS = [5, 10, 20, 30]
RANDOM_REPEATS = 5


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_epochs: int = 500
    patience: int = 50
    learning_rate: float = 0.003
    weight_decay: float = 1e-4
    hidden_1: int = 32
    hidden_2: int = 16
    grid_size: int = 12


class KANLayer(nn.Module):
    """Efficient KAN-style layer using learnable radial spline bases per edge."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 12) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        centers = torch.linspace(-2.5, 2.5, grid_size)
        spacing = float(centers[1] - centers[0]) if grid_size > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (spacing * spacing)

        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.spline_weight = nn.Parameter(torch.empty(in_features, out_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,io->bo", x, self.base_weight)
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        spline = torch.einsum("big,iog->bo", basis, self.spline_weight)
        return base + spline + self.bias


class KANRegressor(nn.Module):
    def __init__(self, in_features: int, config: TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            KANLayer(in_features, config.hidden_1, config.grid_size),
            nn.SiLU(),
            KANLayer(config.hidden_1, config.hidden_2, config.grid_size),
            nn.SiLU(),
            KANLayer(config.hidden_2, 1, config.grid_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_split() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)

    train = train.dropna(subset=[TARGET]).reset_index(drop=True)
    test = test.dropna(subset=[TARGET]).reset_index(drop=True)

    train = train[np.isfinite(train[waves + [TARGET]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    test = test[np.isfinite(test[waves + [TARGET]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    return train, test, waves


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    score_rmse = rmse(y_true, y_pred)
    if score_rmse == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / score_rmse)


def selected_indices_from_csv(waves: list[str]) -> dict[tuple[str, int], np.ndarray]:
    selected = pd.read_csv(BAND_DIR / "cab_selected_wavelengths.csv")
    wave_to_idx = {wave: idx for idx, wave in enumerate(waves)}
    lookup: dict[tuple[str, int], np.ndarray] = {}
    for _, row in selected.iterrows():
        selected_waves = str(row["selected_wavelengths"]).split(",")
        idx = [wave_to_idx[wave] for wave in selected_waves if wave in wave_to_idx]
        lookup[(str(row["selector"]), int(row["budget"]))] = np.asarray(idx, dtype=int)
    return lookup


def fit_predict_kan(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_idx: np.ndarray,
    device: torch.device,
    config: TrainConfig,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all[:, selected_idx],
        y_train_all,
        test_size=0.15,
        random_state=seed,
    )
    x_test_selected = x_test[:, selected_idx]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test_selected).astype(np.float32)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_scaled).to(device)
    x_test_t = torch.from_numpy(x_test_scaled).to(device)

    model = KANRegressor(len(selected_idx), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

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
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
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

    metrics = {
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rpd": rpd(y_test, pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }
    return metrics, pred


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, test, waves = load_split()
    x_train = train[waves].to_numpy(dtype=np.float32)
    y_train = train[TARGET].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[TARGET].to_numpy(dtype=np.float32)

    selected_lookup = selected_indices_from_csv(waves)
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"Target: {TARGET}")
    print(f"Device: {device}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)} | Full bands: {len(waves)}")

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for budget in BUDGETS:
        print(f"\nBudget {budget}")
        selector_sets: list[tuple[str, int, np.ndarray]] = []
        for selector in ["AbsCorr", "MutualInfo"]:
            selector_sets.append((selector, 0, selected_lookup[(selector, budget)]))

        for repeat in range(RANDOM_REPEATS):
            selector_sets.append(("Random", repeat, rng.choice(len(waves), size=budget, replace=False)))

        for selector, repeat, selected_idx in selector_sets:
            seed = RANDOM_STATE + repeat + budget * 100 + (0 if selector != "MutualInfo" else 10000)
            metrics, pred = fit_predict_kan(x_train, y_train, x_test, y_test, selected_idx, device, config, seed)
            row: dict[str, object] = {
                "selector": selector,
                "model": "KAN",
                "target": TARGET,
                "budget": budget,
                "bands": len(selected_idx),
                "repeat": repeat,
                "device": str(device),
                "selected_wavelengths": ",".join(waves[i] for i in selected_idx),
            }
            row.update(metrics)
            rows.append(row)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "selector": selector,
                        "budget": budget,
                        "repeat": repeat,
                        "y_true": y_test,
                        "y_pred": pred,
                    }
                )
            )
            print(f"{selector}+KAN repeat={repeat}: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")

    detail = pd.DataFrame(rows)
    summary_rows: list[pd.DataFrame] = []
    for selector in ["AbsCorr", "MutualInfo"]:
        summary_rows.append(detail[detail["selector"] == selector].copy())

    random_summary = (
        detail[detail["selector"] == "Random"]
        .groupby(["selector", "model", "target", "budget", "bands", "device"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae=("mae", "mean"),
            mae_std=("mae", "std"),
            r2=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            train_seconds=("train_seconds", "mean"),
            predict_seconds=("predict_seconds", "mean"),
            best_epoch=("best_epoch", "mean"),
        )
    )
    random_summary["repeat"] = "mean"
    random_summary["selected_wavelengths"] = "random_repeat_mean"
    summary_rows.append(random_summary)

    summary = pd.concat(summary_rows, ignore_index=True, sort=False).sort_values(["budget", "rmse"])
    predictions = pd.concat(prediction_frames, ignore_index=True)

    detail.to_csv(RESULTS_DIR / "cab_kan_band_selection_detail.csv", index=False)
    summary.to_csv(RESULTS_DIR / "cab_kan_band_selection_summary.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "cab_kan_band_selection_predictions.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / "cab_kan_band_selection_detail.csv")
    print(RESULTS_DIR / "cab_kan_band_selection_summary.csv")
    print(RESULTS_DIR / "cab_kan_band_selection_predictions.csv")
    print("\nTop KAN results:")
    print(summary.sort_values("rmse")[["selector", "model", "budget", "rmse", "mae", "r2", "rpd", "device"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
