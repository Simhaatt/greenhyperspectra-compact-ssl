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


def gradient_importance(model, x_scaler, x_raw: np.ndarray, device: torch.device) -> np.ndarray:
    x_scaled = x_scaler.transform(x_raw).astype(np.float32)
    x = torch.from_numpy(x_scaled).to(device)
    x.requires_grad_(True)
    model.eval()
    pred = model(x).sum()
    pred.backward()
    grad = x.grad.detach().cpu().numpy()
    return np.mean(np.abs(grad * x.detach().cpu().numpy()), axis=0)


def shap_or_gradient_importance(model, x_scaler, background_raw: np.ndarray, explain_raw: np.ndarray, device: torch.device) -> tuple[np.ndarray, str]:
    try:
        import shap  # type: ignore

        background = torch.from_numpy(x_scaler.transform(background_raw).astype(np.float32)).to(device)
        explain = torch.from_numpy(x_scaler.transform(explain_raw).astype(np.float32)).to(device)
        explainer = shap.GradientExplainer(model, background)
        values = explainer.shap_values(explain)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., 0]
        return np.mean(np.abs(values), axis=0), "GradientSHAP"
    except Exception as exc:
        print("SHAP failed; using gradient x input fallback:", repr(exc))
        return gradient_importance(model, x_scaler, explain_raw, device), "GradientInputFallback"


def plot_noise(noise_summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for perturbation, sub in noise_summary.groupby("perturbation"):
        plt.figure(figsize=(8.5, 4.8))
        for target, g in sub.groupby("target"):
            g = g.sort_values("level")
            plt.plot(g["level"], g["r2_mean"], marker="o", linewidth=1.7, label=target)
        plt.xlabel("Perturbation level")
        plt.ylabel("Mean R2")
        plt.title(f"Robustness to {perturbation}")
        plt.grid(True, alpha=0.25)
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        safe = perturbation.replace(" ", "_").lower()
        plt.savefig(FIG_DIR / f"robustness_{safe}_{run_name}.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviewer SHAP/importance and noise robustness for BeamK30 ResidualMLP.")
    parser.add_argument("--traits", nargs="+", default=DEFAULT_TRAITS)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--shap-seed", type=int, default=0)
    parser.add_argument("--background-size", type=int, default=128)
    parser.add_argument("--explain-size", type=int, default=128)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--dropout-levels", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.30])
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--run-name", default="reviewer_shap_noise")
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = RESULTS_DIR / "reviewer_shap_noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainConfig(max_epochs=args.max_epochs, patience=args.patience)
    rng = np.random.default_rng(42)
    robustness_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    print("Device:", device)
    for target in args.traits:
        train, test, waves = load_split(target)
        x_train_full = train[waves].to_numpy(dtype=np.float32)
        y_train = train[target].to_numpy(dtype=np.float32)
        x_test_full = test[waves].to_numpy(dtype=np.float32)
        y_test = test[target].to_numpy(dtype=np.float32)
        selected_idx, selected_waves, selector = selected_wavelengths_for_budget(
            target, args.budget, waves, x_train_full, y_train
        )
        selected_wave_list = selected_waves.split(",")
        x_train = x_train_full[:, selected_idx]
        x_test = x_test_full[:, selected_idx]
        train_mean = x_train.mean(axis=0, keepdims=True)
        train_std = x_train.std(axis=0, keepdims=True)
        train_std = np.where(train_std < 1e-6, 1.0, train_std)
        print(f"\nTrait={target} budget={args.budget} selector={selector}")

        for seed in args.seeds:
            model, x_scaler, y_scaler, timing = fit_residual_mlp(x_train, y_train, seed, config, device)
            clean_pred = predict_residual_mlp(model, x_scaler, y_scaler, x_test, device)
            clean = metric_row(y_test, clean_pred)

            for level in args.noise_levels:
                if level == 0:
                    x_noisy = x_test.copy()
                else:
                    x_noisy = x_test + rng.normal(0, level, size=x_test.shape).astype(np.float32) * train_std
                pred = predict_residual_mlp(model, x_scaler, y_scaler, x_noisy, device)
                robustness_rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "budget": args.budget,
                        "perturbation": "gaussian_noise",
                        "level": float(level),
                        "clean_r2": clean["r2"],
                        "selector": selector,
                        **metric_row(y_test, pred),
                        **timing,
                    }
                )

            for level in args.dropout_levels:
                if level == 0:
                    x_drop = x_test.copy()
                else:
                    mask = rng.random(x_test.shape) < level
                    x_drop = x_test.copy()
                    x_drop[mask] = np.broadcast_to(train_mean, x_test.shape)[mask]
                pred = predict_residual_mlp(model, x_scaler, y_scaler, x_drop, device)
                robustness_rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "budget": args.budget,
                        "perturbation": "band_dropout",
                        "level": float(level),
                        "clean_r2": clean["r2"],
                        "selector": selector,
                        **metric_row(y_test, pred),
                        **timing,
                    }
                )

            if seed == args.shap_seed:
                bg_n = min(args.background_size, len(x_train))
                ex_n = min(args.explain_size, len(x_test))
                bg_idx = rng.choice(np.arange(len(x_train)), size=bg_n, replace=False)
                ex_idx = rng.choice(np.arange(len(x_test)), size=ex_n, replace=False)
                importance, method = shap_or_gradient_importance(model, x_scaler, x_train[bg_idx], x_test[ex_idx], device)
                order = np.argsort(importance)[::-1]
                total = float(np.sum(importance)) if float(np.sum(importance)) > 0 else 1.0
                for rank, feat_idx in enumerate(order, start=1):
                    importance_rows.append(
                        {
                            "target": target,
                            "budget": args.budget,
                            "seed": seed,
                            "method": method,
                            "rank": rank,
                            "wavelength_nm": selected_wave_list[int(feat_idx)],
                            "importance": float(importance[int(feat_idx)]),
                            "relative_importance": float(importance[int(feat_idx)] / total),
                            "selected_wavelengths": selected_waves,
                        }
                    )

    robustness = pd.DataFrame(robustness_rows).sort_values(["target", "perturbation", "level", "seed"])
    robustness_summary = (
        robustness.groupby(["target", "perturbation", "level", "budget"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rmse_mean=("rmse", "mean"),
            rpd_mean=("rpd", "mean"),
            n_runs=("r2", "size"),
        )
        .sort_values(["perturbation", "target", "level"])
    )
    importance_df = pd.DataFrame(importance_rows).sort_values(["target", "rank"])
    top_importance = importance_df[importance_df["rank"] <= 15].copy()

    robustness.to_csv(out_dir / f"robustness_{args.run_name}_results.csv", index=False)
    robustness_summary.to_csv(out_dir / f"robustness_{args.run_name}_summary.csv", index=False)
    importance_df.to_csv(out_dir / f"shap_importance_{args.run_name}_all.csv", index=False)
    top_importance.to_csv(out_dir / f"shap_importance_{args.run_name}_top15.csv", index=False)
    robustness_summary.to_csv(PAPER_TABLE_DIR / "table50_noise_band_dropout_robustness.csv", index=False)
    top_importance.to_csv(PAPER_TABLE_DIR / "table51_shap_top_wavelengths.csv", index=False)
    robustness_summary.to_csv(SUMMARY_DIR / "table50_noise_band_dropout_robustness.csv", index=False)
    top_importance.to_csv(SUMMARY_DIR / "table51_shap_top_wavelengths.csv", index=False)
    plot_noise(robustness_summary, args.run_name)

    print("\nSaved reviewer SHAP/noise outputs under:", out_dir)
    print("Top importance:")
    print(top_importance.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
