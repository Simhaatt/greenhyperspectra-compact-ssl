from __future__ import annotations
import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from torch_mlp_stability import MLPConfig, fit_predict, load_beam_indices, load_split, rpd, rmse


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "paper_tables"
FIG_DIR = RESULTS_DIR / "paper_figures"
TRAITS = ["cab", "cw", "cm"]
TRAIT_LABELS = {
    "cab": "Cab",
    "cw": "Cw",
    "cm": "Cm",
}


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def numeric_wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def table1_dataset_summary() -> pd.DataFrame:
    raw = RESULTS_DIR.parent / "data" / "raw"
    train = pd.read_parquet(raw / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(raw / "greenhyperspectra_labeled_test.parquet")
    unlabeled = None
    unlabeled_dir = raw / "greenhyperspectra_unlabeled_hf"
    if unlabeled_dir.exists():
        from datasets import load_from_disk

        unlabeled = load_from_disk(str(unlabeled_dir))

    waves = numeric_wavelength_columns(train)
    rows = []
    for trait in TRAITS:
        rows.append(
            {
                "target": trait,
                "description": {"cab": "chlorophyll content", "cw": "leaf water content", "cm": "dry matter / leaf mass trait"}[trait],
                "train_labeled_samples": int(train[trait].notna().sum()),
                "test_labeled_samples": int(test[trait].notna().sum()),
                "total_labeled_samples": int(train[trait].notna().sum() + test[trait].notna().sum()),
                "spectral_bands": len(waves),
                "wavelength_range_nm": f"{waves[0]}-{waves[-1]}",
                "unlabeled_spectra_available": int(len(unlabeled["train"])) if unlabeled is not None else 0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "table1_dataset_summary.csv", index=False)
    return df


def table2_full_spectrum_baselines() -> pd.DataFrame:
    frames = []
    for trait in TRAITS:
        df = pd.read_csv(RESULTS_DIR / "baselines" / f"{trait}_cpu_baselines.csv")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values(["target", "rmse"])
    out.to_csv(TABLE_DIR / "table2_full_spectrum_baselines.csv", index=False)
    return out


def table3_band_selection_comparison() -> pd.DataFrame:
    rows = []
    for trait in TRAITS:
        conventional = pd.read_csv(RESULTS_DIR / "band_selection" / f"{trait}_band_selection_results.csv")
        conventional = conventional[conventional["selector"] != "FullSpectrum"].copy()
        conventional["method"] = conventional["selector"] + "+" + conventional["model"]

        rl = pd.read_csv(RESULTS_DIR / "rl" / f"{trait}_rl_band_selection_results.csv")
        rl["method"] = "EpsilonGreedyRL+" + rl["model"]

        beam = pd.read_csv(RESULTS_DIR / "beam" / f"{trait}_beam_band_selection_results.csv")
        beam["method"] = "BeamSearch+" + beam["model"]

        combined = pd.concat(
            [
                conventional[["target", "method", "budget", "rmse", "mae", "r2", "rpd", "selected_wavelengths"]],
                rl[["target", "method", "budget", "rmse", "mae", "r2", "rpd", "selected_wavelengths"]],
                beam[["target", "method", "budget", "rmse", "mae", "r2", "rpd", "selected_wavelengths"]],
            ],
            ignore_index=True,
        )
        best_per_method = combined.sort_values("rmse").groupby(["target", "method"], as_index=False).first()
        rows.append(best_per_method)
    out = pd.concat(rows, ignore_index=True).sort_values(["target", "rmse"])
    out.to_csv(TABLE_DIR / "table3_band_selection_comparison.csv", index=False)
    return out


def table4_compact_model_comparison() -> pd.DataFrame:
    final_summary = pd.read_csv(RESULTS_DIR / "summary" / "final_compact_model_summary.csv")
    final_summary.to_csv(TABLE_DIR / "table4_compact_model_comparison.csv", index=False)
    return final_summary


def table5_stability_results() -> pd.DataFrame:
    beam = pd.read_csv(RESULTS_DIR / "beam_stability" / "beam_stability_summary.csv")
    beam = beam[beam["budget"] == 30].copy()
    beam = beam.rename(columns={"trait": "target"})
    beam["experiment"] = "BeamSearch+PLSR selector stability"
    beam = beam[
        [
            "experiment",
            "target",
            "model",
            "budget",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "r2_mean",
            "r2_std",
            "rpd_mean",
            "rpd_std",
        ]
    ]

    neural = pd.read_csv(RESULTS_DIR / "summary" / "beam_k30_neural_stability_summary.csv")
    neural["experiment"] = "BeamSearch+KAN stability" 
    neural = neural[neural["model"] == "KAN"].copy()
    neural = neural[
        [
            "experiment",
            "target",
            "model",
            "budget",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "r2_mean",
            "r2_std",
            "rpd_mean",
            "rpd_std",
        ]
    ]

    torch_mlp = pd.read_csv(RESULTS_DIR / "summary" / "beam_k30_torch_mlp_stability_summary.csv")
    torch_mlp["experiment"] = "BeamSearch+TorchMLP stability"
    torch_mlp = torch_mlp[
        [
            "experiment",
            "target",
            "model",
            "budget",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "r2_mean",
            "r2_std",
            "rpd_mean",
            "rpd_std",
        ]
    ]

    out = pd.concat([beam, neural, torch_mlp], ignore_index=True).sort_values(["target", "experiment"])
    out.to_csv(TABLE_DIR / "table5_stability_results.csv", index=False)
    return out


def fig1_selected_wavelengths() -> None:
    plt.figure(figsize=(10, 4.8))
    y_positions = np.arange(len(TRAITS))
    for y, trait in zip(y_positions, TRAITS):
        selected = pd.read_csv(RESULTS_DIR / "beam" / f"{trait}_beam_selected_wavelengths.csv")
        row = selected[selected["budget"] == 30].iloc[0]
        wavelengths = [int(value) for value in str(row["selected_wavelengths"]).split(",")]
        plt.scatter(wavelengths, [y] * len(wavelengths), s=42, label=TRAIT_LABELS[trait])
    plt.yticks(y_positions, [TRAIT_LABELS[trait] for trait in TRAITS])
    plt.xlabel("Wavelength (nm)")
    plt.title("BeamSearch-selected wavelengths (K=30)")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_selected_wavelengths.png", dpi=240)
    plt.close()


def fig2_fig3_metric_comparison(final_summary: pd.DataFrame) -> None:
    for metric, ylabel, filename in [
        ("r2", "R2", "fig2_r2_comparison.png"),
        ("rmse", "RMSE", "fig3_rmse_comparison.png"),
    ]:
        plt.figure(figsize=(8.5, 4.9))
        for group, marker in [
            ("Best full-spectrum", "o"),
            ("Beam K=30 + KAN", "s"),
            ("Beam K=30 + TorchMLP", "^"),
        ]:
            data = final_summary[final_summary["group"] == group].sort_values("trait")
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in data.columns:
                y = data[mean_col]
                yerr = data[std_col].fillna(0) if std_col in data.columns else None
            else:
                y = data[metric]
                yerr = None
            plt.errorbar(data["trait"], y, yerr=yerr, marker=marker, linewidth=2, capsize=4, label=group)
        plt.xlabel("Target trait")
        plt.ylabel(ylabel)
        plt.title(f"Final compact model {ylabel} comparison")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=240)
        plt.close()


def representative_seed_for_trait(trait: str) -> int:
    results = pd.read_csv(RESULTS_DIR / "compact_regressors" / "beam_k30_torch_mlp_stability_results.csv")
    summary = pd.read_csv(RESULTS_DIR / "compact_regressors" / "beam_k30_torch_mlp_stability_summary.csv")
    trait_results = results[results["target"] == trait].copy()
    mean_r2 = float(summary[summary["target"] == trait]["r2_mean"].iloc[0])
    trait_results["distance"] = (trait_results["r2"] - mean_r2).abs()
    return int(trait_results.sort_values("distance").iloc[0]["seed"])


def predicted_vs_actual_plots() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MLPConfig()
    predictions_dir = RESULTS_DIR / "paper_predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for trait in TRAITS:
        train, test, waves = load_split(trait)
        selected_idx, selected_wavelengths = load_beam_indices(trait, 30, waves)
        seed = representative_seed_for_trait(trait)

        x_train = train[waves].to_numpy(dtype=np.float32)[:, selected_idx]
        y_train = train[trait].to_numpy(dtype=np.float32)
        x_test = test[waves].to_numpy(dtype=np.float32)[:, selected_idx]
        y_test = test[trait].to_numpy(dtype=np.float32)

        pred, _ = fit_predict(x_train, y_train, x_test, seed, config, device)
        pred_df = pd.DataFrame(
            {
                "target": trait,
                "seed": seed,
                "selected_wavelengths": selected_wavelengths,
                "y_true": y_test,
                "y_pred": pred,
            }
        )
        pred_df.to_csv(predictions_dir / f"torch_mlp_representative_predictions_{trait}.csv", index=False)

        plot_rmse = rmse(y_test, pred)
        plot_r2 = float(r2_score(y_test, pred))
        plot_rpd = rpd(y_test, pred)

        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(y_test, pred, s=18, alpha=0.65)
        low = min(float(np.min(y_test)), float(np.min(pred)))
        high = max(float(np.max(y_test)), float(np.max(pred)))
        plt.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.2)
        plt.xlabel(f"Actual {TRAIT_LABELS[trait]}")
        plt.ylabel(f"Predicted {TRAIT_LABELS[trait]}")
        plt.title(f"{TRAIT_LABELS[trait]} predicted vs actual\nRMSE={plot_rmse:.4g}, R2={plot_r2:.3f}, RPD={plot_rpd:.3f}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"fig{4 + TRAITS.index(trait)}_predicted_vs_actual_{trait}.png", dpi=240)
        plt.close()


def main() -> None:
    ensure_dirs()
    table1_dataset_summary()
    table2_full_spectrum_baselines()
    table3_band_selection_comparison()
    final_summary = table4_compact_model_comparison()
    table5_stability_results()

    fig1_selected_wavelengths()
    fig2_fig3_metric_comparison(final_summary)
    predicted_vs_actual_plots()

    print("Paper tables written to:", TABLE_DIR)
    print("Paper figures written to:", FIG_DIR)


if __name__ == "__main__":
    main()
