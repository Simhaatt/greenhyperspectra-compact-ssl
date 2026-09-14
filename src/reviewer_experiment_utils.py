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
RESULTS_DIR = ROOT / "results"
BEAM_DIR = RESULTS_DIR / "beam"
SUMMARY_DIR = RESULTS_DIR / "summary"
PAPER_TABLE_DIR = RESULTS_DIR / "paper_tables"
FIG_DIR = RESULTS_DIR / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
DEFAULT_TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car"]


@dataclass
class TrainConfig:
    batch_size: int = 128
    max_epochs: int = 600
    patience: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.05


class ResidualBlock(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, features),
        )
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualMLP(nn.Module):
    def __init__(self, in_features: int, config: TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            ResidualBlock(config.hidden_1, config.hidden_1, config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [str(col) for col in df.columns if str(col).isdigit()]


def finite_frame(df: pd.DataFrame, waves: list[str], target: str) -> pd.DataFrame:
    out = df.dropna(subset=[target]).reset_index(drop=True)
    finite = np.isfinite(out[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)
    return out[finite].reset_index(drop=True)


def load_split(target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)
    return finite_frame(train, waves, target), finite_frame(test, waves, target), waves


def load_union(target: str) -> tuple[pd.DataFrame, list[str]]:
    train, test, waves = load_split(target)
    return pd.concat([train.assign(original_split="train"), test.assign(original_split="test")], ignore_index=True), waves


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = rmse(y_true, y_pred)
    if value == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / value)


def abs_corr_ranking(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    y = y_train.astype(np.float64) - float(np.mean(y_train))
    x = x_train.astype(np.float64) - np.mean(x_train, axis=0, keepdims=True)
    numerator = np.abs(x.T @ y)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.argsort(corr)[::-1]


def selected_wavelengths_for_budget(
    target: str,
    budget: int,
    waves: list[str],
    x_train_full: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    beam_path = BEAM_DIR / f"{target}_beam_band_selection_results.csv"
    beam = pd.read_csv(beam_path)
    wave_to_idx = {wave: idx for idx, wave in enumerate(waves)}

    exact = beam[beam["budget"] == budget]
    if not exact.empty:
        row = exact.sort_values("rmse").iloc[0]
        selected_waves = [w.strip() for w in str(row["selected_wavelengths"]).split(",") if w.strip()]
        selected_idx = np.asarray([wave_to_idx[w] for w in selected_waves], dtype=int)
        return selected_idx, ",".join(selected_waves), "BeamSearch"

    lower = beam[beam["budget"] < budget]
    if lower.empty:
        raise ValueError(f"No BeamSearch seed selection is available for {target} budget {budget}")
    row = lower.sort_values(["budget", "rmse"], ascending=[False, True]).iloc[0]
    selected_waves = [w.strip() for w in str(row["selected_wavelengths"]).split(",") if w.strip()]
    selected_idx = [wave_to_idx[w] for w in selected_waves]
    selected_set = set(selected_idx)

    for idx in abs_corr_ranking(x_train_full, y_train):
        if int(idx) not in selected_set:
            selected_idx.append(int(idx))
            selected_set.add(int(idx))
        if len(selected_idx) >= budget:
            break

    out_idx = np.asarray(selected_idx[:budget], dtype=int)
    out_waves = [waves[i] for i in out_idx]
    return out_idx, ",".join(out_waves), "BeamSearch+CorrelationExtension"


def fit_residual_mlp(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    seed: int,
    config: TrainConfig,
    device: torch.device,
) -> tuple[ResidualMLP, StandardScaler, StandardScaler, dict[str, float]]:
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
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = ResidualMLP(x_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_scaled).to(device)

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

    if best_state is not None:
        model.load_state_dict(best_state)
    timing = {
        "train_seconds": perf_counter() - start,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": float(best_val),
    }
    return model, x_scaler, y_scaler, timing


def predict_residual_mlp(
    model: ResidualMLP,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    x: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_scaled = x_scaler.transform(x).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_scaled).to(device)).detach().cpu().numpy().reshape(-1)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)


def metric_row(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, pred),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "rpd": rpd(y_true, pred),
    }


def ensure_output_dirs() -> None:
    for path in [RESULTS_DIR, SUMMARY_DIR, PAPER_TABLE_DIR, FIG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
