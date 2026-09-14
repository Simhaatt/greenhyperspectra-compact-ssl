from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from beam_supervised_modern_heads import HeadConfig, fit_predict
from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "finite_bandpass"
PAPER_TABLE_DIR = ROOT / "results" / "paper_tables"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


def gaussian_matrix(wavelengths: np.ndarray, centers: np.ndarray, fwhm: float) -> np.ndarray:
    sigma = fwhm / 2.354820045
    weights = np.exp(-0.5 * ((wavelengths[None, :] - centers[:, None]) / sigma) ** 2)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def merge_centers(centers: np.ndarray, max_gap: float) -> list[list[int]]:
    order = np.argsort(centers)
    clusters: list[list[int]] = []
    current = [int(order[0])]
    for idx in order[1:]:
        idx = int(idx)
        if centers[idx] - centers[current[-1]] <= max_gap:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    return clusters


def transform_features(
    x_full: np.ndarray,
    wavelengths: np.ndarray,
    selected_idx: np.ndarray,
    selected_centers: np.ndarray,
    variant: str,
    fwhm: float | None,
) -> tuple[np.ndarray, np.ndarray, int, str]:
    if variant == "raw_nm":
        return x_full[:, selected_idx].astype(np.float32), selected_centers, len(selected_centers), "raw selected 1 nm samples"

    if fwhm is None:
        raise ValueError("Finite bandpass variants require fwhm.")

    if variant == "gaussian":
        weights = gaussian_matrix(wavelengths, selected_centers, fwhm)
        return (x_full @ weights.T).astype(np.float32), selected_centers, len(selected_centers), f"{fwhm:g} nm FWHM, 30 overlapping filters"

    if variant == "merged_gaussian":
        clusters = merge_centers(selected_centers, max_gap=fwhm)
        merged_centers = np.asarray([float(np.mean(selected_centers[c])) for c in clusters], dtype=np.float32)
        weights = gaussian_matrix(wavelengths, merged_centers, fwhm)
        cluster_text = ";".join(
            ",".join(str(int(selected_centers[i])) for i in cluster) for cluster in clusters
        )
        return (
            (x_full @ weights.T).astype(np.float32),
            merged_centers,
            len(merged_centers),
            f"{fwhm:g} nm FWHM, centers within {fwhm:g} nm merged | {cluster_text}",
        )

    raise ValueError(f"Unknown variant: {variant}")


def run_trait(target: str, budget: int, seeds: list[int], device: torch.device, config: HeadConfig) -> tuple[list[dict], list[dict]]:
    train, test, waves = load_split(target)
    wavelengths = np.asarray([float(w) for w in waves], dtype=np.float32)
    selected_idx, selected_text = load_beam_indices(target, budget, waves)
    selected_centers = wavelengths[selected_idx]

    x_train_full = train[waves].to_numpy(dtype=np.float32)
    x_test_full = test[waves].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    variants = [
        ("raw_nm", None),
        ("gaussian", 10.0),
        ("gaussian", 20.0),
        ("merged_gaussian", 10.0),
        ("merged_gaussian", 20.0),
    ]
    rows: list[dict] = []
    audit_rows: list[dict] = []

    for variant, fwhm in variants:
        x_train, centers_used, effective_channels, note = transform_features(
            x_train_full, wavelengths, selected_idx, selected_centers, variant, fwhm
        )
        x_test, _, _, _ = transform_features(x_test_full, wavelengths, selected_idx, selected_centers, variant, fwhm)
        variant_name = variant if fwhm is None else f"{variant}_fwhm{int(fwhm)}"
        print(
            f"\nTrait={target} variant={variant_name} train={len(y_train)} test={len(y_test)} "
            f"features={x_train.shape[1]} device={device}"
        )
        audit_rows.append(
            {
                "target": target,
                "budget": budget,
                "variant": variant_name,
                "selected_wavelengths": selected_text,
                "effective_channels": effective_channels,
                "used_channel_centers_nm": ",".join(str(int(round(c))) for c in centers_used),
                "note": note,
            }
        )

        for seed in seeds:
            start = perf_counter()
            pred, timing = fit_predict(x_train, y_train, x_test, seed, "residual", config, device)
            elapsed = perf_counter() - start
            row = {
                "target": target,
                "model": f"BeamK{budget}_ResidualMLP",
                "seed": seed,
                "variant": variant_name,
                "budget": budget,
                "fwhm_nm": fwhm,
                "input_features": x_train.shape[1],
                "effective_channels": effective_channels,
                "rmse": rmse(y_test, pred),
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
                "rpd": rpd(y_test, pred),
                "device": str(device),
                "elapsed_seconds": elapsed,
                **timing,
            }
            rows.append(row)
            print(f"  seed={seed}: R2={row['r2']:.4f} RMSE={row['rmse']:.6f} channels={effective_channels}")
    return rows, audit_rows


def save_outputs(results: pd.DataFrame, audit: pd.DataFrame, run_name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = (
        results.groupby(["target", "model", "budget", "variant", "fwhm_nm", "input_features", "effective_channels"], dropna=False, as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            train_seconds_mean=("train_seconds", "mean"),
        )
    )

    raw_means = summary[summary["variant"] == "raw_nm"][["target", "r2_mean"]].rename(columns={"r2_mean": "raw_r2_mean"})
    summary = summary.merge(raw_means, on="target", how="left")
    summary["delta_r2_vs_raw"] = summary["r2_mean"] - summary["raw_r2_mean"]

    results.to_csv(RESULTS_DIR / f"finite_bandpass_{run_name}_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / f"finite_bandpass_{run_name}_summary.csv", index=False)
    audit.to_csv(RESULTS_DIR / f"finite_bandpass_{run_name}_channel_audit.csv", index=False)
    summary.to_csv(PAPER_TABLE_DIR / "table56_finite_bandpass_simulation.csv", index=False)
    audit.to_csv(PAPER_TABLE_DIR / "table57_finite_bandpass_channel_audit.csv", index=False)

    overall = (
        summary.groupby(["variant", "fwhm_nm"], dropna=False, as_index=False)
        .agg(
            mean_r2=("r2_mean", "mean"),
            mean_delta_r2_vs_raw=("delta_r2_vs_raw", "mean"),
            mean_effective_channels=("effective_channels", "mean"),
            min_effective_channels=("effective_channels", "min"),
            max_effective_channels=("effective_channels", "max"),
        )
        .sort_values(["variant", "fwhm_nm"])
    )
    overall.to_csv(RESULTS_DIR / f"finite_bandpass_{run_name}_overall.csv", index=False)
    overall.to_csv(PAPER_TABLE_DIR / "table56b_finite_bandpass_overall.csv", index=False)

    plt.figure(figsize=(9, 5))
    order = ["raw_nm", "gaussian_fwhm10", "gaussian_fwhm20", "merged_gaussian_fwhm10", "merged_gaussian_fwhm20"]
    plot_data = summary.copy()
    for variant in order:
        sub = plot_data[plot_data["variant"] == variant].sort_values("target")
        if len(sub) == 0:
            continue
        plt.plot(sub["target"], sub["r2_mean"], marker="o", linewidth=2, label=variant)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Mean test R2")
    plt.xlabel("Trait")
    plt.title("Finite-bandpass simulation for selected wavelength channels")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"finite_bandpass_{run_name}_r2_by_trait.png", dpi=240)
    plt.close()

    print("\nSaved outputs:")
    print(RESULTS_DIR / f"finite_bandpass_{run_name}_summary.csv")
    print(PAPER_TABLE_DIR / "table56_finite_bandpass_simulation.csv")
    print(PAPER_TABLE_DIR / "table56b_finite_bandpass_overall.csv")
    print(FIG_DIR / f"finite_bandpass_{run_name}_r2_by_trait.png")
    print("\nOverall:")
    print(overall)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-bandpass simulation for BeamSearch-selected wavelengths.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--run-name", default="all8_3seeds")
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HeadConfig(max_epochs=args.max_epochs, patience=args.patience)
    print("Device:", device)
    print("Config:", asdict(config))

    rows: list[dict] = []
    audit_rows: list[dict] = []
    for trait in args.traits:
        trait_rows, trait_audit = run_trait(trait, args.budget, args.seeds, device, config)
        rows.extend(trait_rows)
        audit_rows.extend(trait_audit)
    save_outputs(pd.DataFrame(rows), pd.DataFrame(audit_rows), args.run_name)


if __name__ == "__main__":
    main()
