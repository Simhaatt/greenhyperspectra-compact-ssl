from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_cab_kan_band_selection import TrainConfig, fit_predict_kan


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
BEAM_DIR = ROOT / "results" / "beam"
RESULTS_DIR = ROOT / "results" / "compact_regressors"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm"]


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


def metric_row(
    target: str,
    model: str,
    seed: int,
    budget: int,
    selected_wavelengths: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    predict_seconds: float,
    device: str = "cpu",
) -> dict[str, object]:
    return {
        "target": target,
        "selector": "BeamSearch",
        "model": model,
        "seed": seed,
        "budget": budget,
        "bands": budget,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rpd": rpd(y_true, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "device": device,
        "selected_wavelengths": selected_wavelengths,
    }


def train_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=0.001,
                    batch_size=128,
                    learning_rate_init=0.001,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=50,
                    random_state=seed,
                ),
            ),
        ]
    )
    start = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = np.asarray(model.predict(x_test)).reshape(-1)
    predict_seconds = perf_counter() - start
    return pred, train_seconds, predict_seconds


def run_trait(target: str, budget: int, seeds: list[int], skip_kan: bool) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)

    x_train_all = train[waves].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test_all = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)
    x_train = x_train_all[:, selected_idx]
    x_test = x_test_all[:, selected_idx]

    rows: list[dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kan_config = TrainConfig(max_epochs=500, patience=60)

    print(f"\nTrait={target} | train={len(y_train)} test={len(y_test)} budget={budget}")
    for seed in seeds:
        print(f"  MLP seed={seed}...")
        pred, train_seconds, predict_seconds = train_mlp(x_train, y_train, x_test, seed)
        row = metric_row(target, "MLP", seed, budget, selected_wavelengths, y_test, pred, train_seconds, predict_seconds)
        rows.append(row)
        print(f"    RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}")

        if not skip_kan:
            print(f"  KAN seed={seed} on {device}...")
            metrics, pred = fit_predict_kan(
                x_train_all,
                y_train,
                x_test_all,
                y_test,
                selected_idx,
                device,
                kan_config,
                seed,
            )
            row = {
                "target": target,
                "selector": "BeamSearch",
                "model": "KAN",
                "seed": seed,
                "budget": budget,
                "bands": budget,
                "device": str(device),
                "selected_wavelengths": selected_wavelengths,
                **metrics,
            }
            rows.append(row)
            print(f"    RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}")

    return rows


def save_figures(summary: pd.DataFrame) -> None:
    for metric in ["r2", "rmse"]:
        plt.figure(figsize=(8.5, 4.8))
        for model, marker in [("MLP", "o"), ("KAN", "s")]:
            data = summary[summary["model"] == model].sort_values("target")
            if data.empty:
                continue
            plt.errorbar(
                data["target"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"].fillna(0),
                marker=marker,
                linewidth=2,
                capsize=4,
                label=model,
            )
        plt.xlabel("Trait")
        plt.ylabel(metric.upper())
        plt.title(f"BeamSearch K=30 compact neural regressor stability: {metric.upper()}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"beam_k30_neural_stability_{metric}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MLP/KAN stability on BeamSearch-selected compact wavelengths.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--skip-kan", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, args.seeds, args.skip_kan))

    results = pd.DataFrame(rows).sort_values(["target", "model", "seed"])
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
        )
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )

    results_path = RESULTS_DIR / f"beam_k{args.budget}_neural_stability_results.csv"
    summary_path = RESULTS_DIR / f"beam_k{args.budget}_neural_stability_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(SUMMARY_DIR / f"beam_k{args.budget}_neural_stability_summary.csv", index=False)
    save_figures(summary)

    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(SUMMARY_DIR / f"beam_k{args.budget}_neural_stability_summary.csv")
    print(FIG_DIR / "beam_k30_neural_stability_r2.png")
    print(FIG_DIR / "beam_k30_neural_stability_rmse.png")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
