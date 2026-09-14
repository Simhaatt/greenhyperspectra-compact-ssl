from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from reviewer_experiment_utils import (
    DEFAULT_TRAITS,
    FIG_DIR,
    PAPER_TABLE_DIR,
    RESULTS_DIR,
    SUMMARY_DIR,
    TrainConfig,
    ensure_output_dirs,
    fit_residual_mlp,
    load_split,
    metric_row,
    predict_residual_mlp,
    selected_wavelengths_for_budget,
)


def plot_budget_curve(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.5, 5.2))
    for target, sub in summary.groupby("target"):
        sub = sub.sort_values("budget")
        plt.plot(sub["budget"], sub["r2_mean"], marker="o", linewidth=1.8, label=target)
    plt.axvline(30, color="black", linestyle="--", linewidth=1, alpha=0.55)
    plt.xlabel("Number of selected wavelengths")
    plt.ylabel("Mean test R2")
    plt.title("Band-budget study for compact ResidualMLP")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"band_budget_{run_name}_r2_curve.png", dpi=240)
    plt.close()

    overall = summary.groupby("budget", as_index=False).agg(r2_mean=("r2_mean", "mean"), r2_std=("r2_mean", "std"))
    plt.figure(figsize=(7.8, 4.6))
    plt.errorbar(overall["budget"], overall["r2_mean"], yerr=overall["r2_std"].fillna(0), marker="o", linewidth=2.2, capsize=4)
    plt.axvline(30, color="black", linestyle="--", linewidth=1, alpha=0.55)
    plt.xlabel("Number of selected wavelengths")
    plt.ylabel("Mean R2 across traits")
    plt.title("Overall accuracy vs band budget")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"band_budget_{run_name}_overall_r2.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer band-budget study for compact ResidualMLP.")
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--budgets", nargs="+", type=int, default=[5, 10, 20, 30, 50, 100])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--run-name", default="reviewer_band_budget")
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = RESULTS_DIR / "reviewer_band_budget"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)
    rows: list[dict[str, object]] = []

    print("Device:", device)
    print("Traits:", args.traits)
    print("Budgets:", args.budgets)
    for target in args.traits:
        train, test, waves = load_split(target)
        x_train_full = train[waves].to_numpy(dtype=np.float32)
        y_train = train[target].to_numpy(dtype=np.float32)
        x_test_full = test[waves].to_numpy(dtype=np.float32)
        y_test = test[target].to_numpy(dtype=np.float32)
        print(f"\nTrait={target} train={len(train)} test={len(test)}")

        for budget in args.budgets:
            selected_idx, selected_waves, selector = selected_wavelengths_for_budget(
                target, budget, waves, x_train_full, y_train
            )
            x_train = x_train_full[:, selected_idx]
            x_test = x_test_full[:, selected_idx]
            print(f"  budget={budget} selector={selector} bands={len(selected_idx)}")
            for seed in args.seeds:
                model, x_scaler, y_scaler, timing = fit_residual_mlp(x_train, y_train, seed, config, device)
                pred = predict_residual_mlp(model, x_scaler, y_scaler, x_test, device)
                row: dict[str, object] = {
                    "target": target,
                    "model": "BeamBudget ResidualMLP",
                    "selector": selector,
                    "budget": int(budget),
                    "bands": int(len(selected_idx)),
                    "seed": int(seed),
                    "selected_wavelengths": selected_waves,
                    "device": str(device),
                    **metric_row(y_test, pred),
                    **timing,
                }
                rows.append(row)
                print(f"    seed={seed}: R2={row['r2']:.4f}, RMSE={row['rmse']:.4f}, epoch={row['best_epoch']:.0f}")

    results = pd.DataFrame(rows).sort_values(["target", "budget", "seed"])
    summary = (
        results.groupby(["target", "model", "selector", "budget", "bands"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            n_runs=("r2", "size"),
        )
        .sort_values(["target", "budget"])
    )
    overall = (
        summary.groupby("budget", as_index=False)
        .agg(mean_r2_across_traits=("r2_mean", "mean"), mean_rpd_across_traits=("rpd_mean", "mean"))
        .sort_values("budget")
    )

    results_path = out_dir / f"band_budget_{args.run_name}_results.csv"
    summary_path = out_dir / f"band_budget_{args.run_name}_summary.csv"
    overall_path = out_dir / f"band_budget_{args.run_name}_overall.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    overall.to_csv(overall_path, index=False)
    summary.to_csv(SUMMARY_DIR / "table49_band_budget_study.csv", index=False)
    summary.to_csv(PAPER_TABLE_DIR / "table49_band_budget_study.csv", index=False)
    overall.to_csv(PAPER_TABLE_DIR / "table49b_band_budget_overall.csv", index=False)
    plot_budget_curve(summary, args.run_name)

    print("\nSaved:")
    for path in [
        results_path,
        summary_path,
        overall_path,
        SUMMARY_DIR / "table49_band_budget_study.csv",
        PAPER_TABLE_DIR / "table49_band_budget_study.csv",
        PAPER_TABLE_DIR / "table49b_band_budget_overall.csv",
        FIG_DIR / f"band_budget_{args.run_name}_r2_curve.png",
        FIG_DIR / f"band_budget_{args.run_name}_overall_r2.png",
    ]:
        print(path)
    print("\nOverall:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
