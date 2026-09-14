from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from hybrid_ssl_label_efficiency import train_hybrid, train_labelled
from torch_mlp_stability import load_beam_indices, rmse, rpd, wavelength_columns


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
BEAM_DIR = RESULTS_DIR / "beam"
SSL_DIR = RESULTS_DIR / "ssl"
SUMMARY_DIR = RESULTS_DIR / "summary"
FIG_DIR = RESULTS_DIR / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
OOD_TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car"]
MODEL_KEYS = ["labelled_residual", "hybrid_cross_gated_nam"]
MODEL_DISPLAY = {
    "labelled_residual": ("BeamK30 ResidualMLP", "labelled_only"),
    "hybrid_cross_gated_nam": ("Hybrid Cross-Gated NAM", "hybrid_ssl"),
}


def load_labelled_union() -> tuple[pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    df = pd.concat([train.assign(original_split="train"), test.assign(original_split="test")], ignore_index=True)
    return df, wavelength_columns(df)


def finite_trait_frame(df: pd.DataFrame, waves: list[str], target: str) -> pd.DataFrame:
    cols = waves + [target, "dataset"]
    out = df.dropna(subset=[target, "dataset"]).reset_index(drop=True)
    finite = np.isfinite(out[cols].to_numpy(dtype=np.float64)).all(axis=1)
    return out[finite].reset_index(drop=True)


def choose_sources(df: pd.DataFrame, max_sources: int, min_test_labels: int, source_mode: str) -> list[int]:
    counts = df.groupby("dataset").size().sort_values(ascending=False)
    counts = counts[counts >= min_test_labels]
    if source_mode == "largest":
        selected = counts.head(max_sources)
    elif source_mode == "smallest":
        selected = counts.sort_values(ascending=True).head(max_sources)
    else:
        if len(counts) <= max_sources:
            selected = counts
        else:
            idx = np.linspace(0, len(counts) - 1, max_sources).round().astype(int)
            selected = counts.iloc[idx]
    return [int(x) for x in selected.index.tolist()]


def snv(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def source_zscore(x: np.ndarray, datasets: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for source in np.unique(datasets):
        mask = datasets == source
        mean = x[mask].mean(axis=0, keepdims=True)
        std = x[mask].std(axis=0, keepdims=True)
        std = np.where(std < eps, 1.0, std)
        out[mask] = ((x[mask] - mean) / std).astype(np.float32)
    return out


def apply_preprocessing(x: np.ndarray, datasets: np.ndarray, variant: str) -> np.ndarray:
    if variant == "baseline":
        return x.astype(np.float32)
    if variant == "snv":
        return snv(x)
    if variant == "per_source_zscore":
        return source_zscore(x, datasets)
    if variant == "snv_per_source_zscore":
        return source_zscore(snv(x), datasets)
    raise ValueError(f"Unknown preprocessing variant: {variant}")


def split_calibration_indices(n: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    if fraction <= 0:
        return np.asarray([], dtype=int), idx
    rng = np.random.default_rng(seed)
    n_cal = max(2, int(round(n * fraction)))
    n_cal = min(n - 2, n_cal)
    if n_cal < 2:
        return np.asarray([], dtype=int), idx
    cal = np.sort(rng.choice(idx, size=n_cal, replace=False))
    test = np.setdiff1d(idx, cal, assume_unique=True)
    return cal, test


def linear_calibrate(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(pred) < 2 or float(np.std(pred)) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(float), y.astype(float), deg=1)
    return float(a), float(b)


def metric_row(
    target: str,
    model_key: str,
    heldout_source: int,
    seed: int,
    y_true: np.ndarray,
    pred: np.ndarray,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    display, family = MODEL_DISPLAY[model_key]
    row = {
        "target": target,
        "family": family,
        "model": display,
        "model_key": model_key,
        "heldout_dataset": int(heldout_source),
        "seed": int(seed),
        "test_rows": int(len(y_true)),
        "rmse": rmse(y_true, pred),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "rpd": rpd(y_true, pred),
    }
    if extra:
        row.update(extra)
    return row


def train_predict_model(
    model_key: str,
    x_beam_train: np.ndarray,
    x_ssl_train: np.ndarray | None,
    y_train: np.ndarray,
    x_beam_test: np.ndarray,
    x_ssl_test: np.ndarray | None,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    subset = np.arange(len(y_train))
    if model_key == "labelled_residual":
        return train_labelled(
            x_beam_train,
            y_train,
            x_beam_test,
            subset,
            model_key,
            seed,
            max_epochs,
            patience,
            device,
        )
    if model_key == "hybrid_cross_gated_nam":
        if x_ssl_train is None or x_ssl_test is None:
            raise ValueError("Hybrid model requires SSL embeddings.")
        return train_hybrid(
            x_beam_train,
            x_ssl_train,
            y_train,
            x_beam_test,
            x_ssl_test,
            subset,
            model_key,
            seed,
            max_epochs,
            patience,
            device,
        )
    raise ValueError(f"Unsupported model: {model_key}")


def maybe_ssl_embeddings(
    x_full: np.ndarray,
    checkpoint_path: Path,
    needs_hybrid: bool,
    device: torch.device,
    batch_size: int,
) -> np.ndarray | None:
    if not needs_hybrid:
        return None
    ckpt = load_checkpoint(checkpoint_path, device)
    return ssl_embeddings(x_full, ckpt, device, batch_size)


def summarize_model_table(results: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        results.groupby(group_cols, as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            mean_rpd=("rpd", "mean"),
            n_evaluations=("r2", "size"),
        )
        .sort_values(group_cols)
    )
