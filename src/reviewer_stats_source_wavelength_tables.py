from __future__ import annotations
import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import ttest_rel, wilcoxon
except Exception:  # pragma: no cover - fallback for minimal local runtimes
    ttest_rel = None
    wilcoxon = None

from export_selected_wavelengths_interpretation import main as export_wavelength_main


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results"
PAPER_TABLE_DIR = RESULTS_DIR / "paper_tables"
SUMMARY_DIR = RESULTS_DIR / "summary"


def ensure_output_dirs() -> None:
    for path in [RESULTS_DIR, PAPER_TABLE_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def compact_vs_full_from_existing() -> pd.DataFrame:
    compact_path = RESULTS_DIR / "compact_regressors" / "beam_k30_torch_mlp_stability_results.csv"
    baseline_files = sorted((RESULTS_DIR / "baselines").glob("*_cpu_baselines.csv"))
    if not compact_path.exists() or not baseline_files:
        return pd.DataFrame()

    compact = pd.read_csv(compact_path)
    compact = compact[["target", "seed", "r2", "rmse", "rpd"]].rename(
        columns={"r2": "compact_r2", "rmse": "compact_rmse", "rpd": "compact_rpd"}
    )
    baseline_rows = []
    for path in baseline_files:
        df = pd.read_csv(path)
        if "model" not in df.columns:
            continue
        target = path.name.replace("_cpu_baselines.csv", "")
        candidates = df[df["model"].astype(str).str.contains("Full|PLSR|Ridge|RandomForest|SVR", case=False, na=False)].copy()
        if candidates.empty:
            candidates = df.copy()
        best = candidates.sort_values("r2", ascending=False).iloc[0]
        baseline_rows.append(
            {
                "target": target,
                "baseline_model": best.get("model", "full_spectrum_best"),
                "full_r2": float(best["r2"]),
                "full_rmse": float(best["rmse"]),
                "full_rpd": float(best["rpd"]),
            }
        )
    baseline = pd.DataFrame(baseline_rows)
    paired = compact.merge(baseline, on="target", how="inner")
    if paired.empty:
        return pd.DataFrame()
    paired["delta_r2_compact_minus_full"] = paired["compact_r2"] - paired["full_r2"]
    paired["delta_rmse_compact_minus_full"] = paired["compact_rmse"] - paired["full_rmse"]
    paired["comparison"] = "BeamK30 ResidualMLP - best full-spectrum baseline"
    return paired


def significance_table(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    rows = []
    values = paired["delta_r2_compact_minus_full"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(values)
    try:
        if wilcoxon is None:
            raise RuntimeError("scipy unavailable")
        w_p = float(wilcoxon(values).pvalue)
    except Exception:
        w_p = 1.0
    try:
        if ttest_rel is None:
            raise RuntimeError("scipy unavailable")
        t_p = float(ttest_rel(paired["compact_r2"], paired["full_r2"]).pvalue)
    except Exception:
        t_p = float("nan")
    trait_delta = paired.groupby("target")["delta_r2_compact_minus_full"].mean()
    rows.append(
        {
            "comparison": "BeamK30 ResidualMLP - best full-spectrum baseline",
            "n_pairs": int(len(values)),
            "mean_delta_r2": float(np.mean(values)),
            "std_delta_r2": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "median_delta_r2": float(np.median(values)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "paired_ttest_p_value": t_p,
            "wilcoxon_p_value": w_p,
            "traits_improved": int((trait_delta > 0).sum()),
            "traits_compared": int(trait_delta.shape[0]),
        }
    )
    return pd.DataFrame(rows)


def source_shift_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_path = RESULTS_DIR / "ood_source_validation" / "ood_source_validation_top5_sources_3seeds_results.csv"
    if not source_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    df = pd.read_csv(source_path)
    by_source = (
        df.groupby(["heldout_dataset", "model", "model_key"], as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            std_r2=("r2", "std"),
            mean_rmse=("rmse", "mean"),
            mean_rpd=("rpd", "mean"),
            n_evaluations=("r2", "size"),
            mean_test_rows=("test_rows", "mean"),
        )
        .sort_values(["mean_r2", "heldout_dataset"], ascending=[True, True])
    )
    by_trait_source = (
        df.groupby(["target", "heldout_dataset", "model", "model_key"], as_index=False)
        .agg(mean_r2=("r2", "mean"), mean_rmse=("rmse", "mean"), mean_rpd=("rpd", "mean"), n_runs=("r2", "size"))
        .sort_values(["target", "mean_r2"])
    )
    return by_source, by_trait_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick reviewer tables: wavelength interpretation, statistics, source-wise shift.")
    parser.add_argument("--run-name", default="reviewer_stats_source_wavelength")
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = RESULTS_DIR / "reviewer_stats_source_wavelength"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_wavelength_main()

    paired = compact_vs_full_from_existing()
    sig = significance_table(paired)
    by_source, by_trait_source = source_shift_tables()

    if not paired.empty:
        paired.to_csv(out_dir / f"compact_vs_full_paired_{args.run_name}.csv", index=False)
        paired.to_csv(PAPER_TABLE_DIR / "table52_compact_vs_full_paired_results.csv", index=False)
    if not sig.empty:
        sig.to_csv(out_dir / f"statistical_significance_{args.run_name}.csv", index=False)
        sig.to_csv(PAPER_TABLE_DIR / "table53_statistical_significance.csv", index=False)
        sig.to_csv(SUMMARY_DIR / "table53_statistical_significance.csv", index=False)
    if not by_source.empty:
        by_source.to_csv(out_dir / f"source_shift_by_source_{args.run_name}.csv", index=False)
        by_trait_source.to_csv(out_dir / f"source_shift_by_trait_source_{args.run_name}.csv", index=False)
        by_source.to_csv(PAPER_TABLE_DIR / "table54_source_shift_by_source.csv", index=False)
        by_trait_source.to_csv(PAPER_TABLE_DIR / "table55_source_shift_by_trait_source.csv", index=False)
        by_source.to_csv(SUMMARY_DIR / "table54_source_shift_by_source.csv", index=False)

    readme = out_dir / "reviewer_tables_summary.md"
    readme.write_text(
        "\n".join(
            [
                "# Reviewer Tables Summary",
                "",
                "Generated reviewer-support tables for wavelength interpretation, compact-vs-full statistical testing, and source-wise domain-shift analysis.",
                "",
                "Key paper tables:",
                "- table44_selected_wavelengths_by_trait.csv",
                "- table45_selected_wavelengths_long_interpretation.csv",
                "- table52_compact_vs_full_paired_results.csv",
                "- table53_statistical_significance.csv",
                "- table54_source_shift_by_source.csv",
                "- table55_source_shift_by_trait_source.csv",
                "",
                "Use the source-wise tables to name the hardest held-out datasets rather than presenting source shift as a single aggregate failure.",
            ]
        ),
        encoding="utf-8",
    )

    print("Saved reviewer stats/source/wavelength outputs under:", out_dir)
    if not sig.empty:
        print(sig.to_string(index=False))
    if not by_source.empty:
        print(by_source.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
