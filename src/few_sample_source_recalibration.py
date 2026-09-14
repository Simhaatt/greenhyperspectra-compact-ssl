from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from hybrid_ssl_label_efficiency import train_hybrid, train_labelled
from ood_source_validation import choose_sources, finite_trait_frame, load_labelled_union
from torch_mlp_stability import load_beam_indices, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "source_recalibration"
SUMMARY_DIR = ROOT / "results" / "summary"
TABLE_DIR = ROOT / "results" / "paper_tables"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
MODELS = {
    "labelled_residual": ("BeamK30 ResidualMLP", "labelled_only"),
    "hybrid_cross_gated_nam": ("Hybrid Cross-Gated NAM", "hybrid_ssl"),
}


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
    eval_idx = np.setdiff1d(idx, cal, assume_unique=True)
    return cal, eval_idx


def linear_calibrate(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(pred) < 2 or float(np.std(pred)) < 1e-8:
        return 1.0, 0.0
    a, b = np.polyfit(pred.astype(float), y.astype(float), deg=1)
    return float(a), float(b)


def metric_row(
    target: str,
    model_key: str,
    source: int,
    seed: int,
    fraction: float,
    y_eval: np.ndarray,
    pred_eval: np.ndarray,
    cal_rows: int,
    eval_rows: int,
    a: float,
    b: float,
) -> dict[str, object]:
    display, family = MODELS[model_key]
    return {
        "target": target,
        "family": family,
        "model": display,
        "model_key": model_key,
        "heldout_dataset": int(source),
        "seed": int(seed),
        "target_calibration_fraction": fraction,
        "target_calibration_percent": int(round(fraction * 100)),
        "calibration_rows": int(cal_rows),
        "test_rows": int(eval_rows),
        "calibration_a": a,
        "calibration_b": b,
        "rmse": rmse(y_eval, pred_eval),
        "mae": float(mean_absolute_error(y_eval, pred_eval)),
        "r2": float(r2_score(y_eval, pred_eval)),
        "rpd": rpd(y_eval, pred_eval),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-sample source recalibration using y_calibrated = a*y_pred + b.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--min-test-labels", type=int, default=30)
    parser.add_argument("--source-mode", choices=["largest", "smallest", "spread"], default="largest")
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--run-name", default="top5_sources_3seeds")
    parser.add_argument("--skip-paper-table", action="store_true", help="Use for smoke runs so partial results do not overwrite the paper table.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all, waves = load_labelled_union()
    needs_hybrid = any(model in {"hybrid_cross_gated_nam"} for model in args.models)
    ckpt = load_checkpoint(Path(args.checkpoint), device) if needs_hybrid else None
    rows: list[dict[str, object]] = []

    print("Device:", device)
    print("Traits:", args.traits)
    print("Models:", args.models)
    print("Calibration fractions:", args.fractions)

    for target in args.traits:
        df = finite_trait_frame(df_all, waves, target)
        sources = choose_sources(df, args.max_sources, args.min_test_labels, args.source_mode)
        if not sources:
            print(f"Skipping {target}: no source with enough labels.")
            continue
        selected_idx, _ = load_beam_indices(target, args.budget, waves)
        x_full = df[waves].to_numpy(dtype=np.float32)
        x_beam = x_full[:, selected_idx]
        y = df[target].to_numpy(dtype=np.float32)
        datasets = df["dataset"].to_numpy()
        x_ssl = None
        if needs_hybrid:
            assert ckpt is not None
            print(f"Embedding full spectra for {target}...")
            x_ssl = ssl_embeddings(x_full, ckpt, device, args.embedding_batch_size)

        for source in sources:
            heldout_mask = datasets == source
            train_mask = ~heldout_mask
            if int(heldout_mask.sum()) < args.min_test_labels or int(train_mask.sum()) < 30:
                continue

            for seed in args.seeds:
                for model_key in args.models:
                    if model_key == "labelled_residual":
                        pred_heldout, _ = train_labelled(
                            x_beam[train_mask],
                            y[train_mask],
                            x_beam[heldout_mask],
                            np.arange(int(train_mask.sum())),
                            model_key,
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )
                    else:
                        assert x_ssl is not None
                        pred_heldout, _ = train_hybrid(
                            x_beam[train_mask],
                            x_ssl[train_mask],
                            y[train_mask],
                            x_beam[heldout_mask],
                            x_ssl[heldout_mask],
                            np.arange(int(train_mask.sum())),
                            model_key,
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )

                    y_heldout = y[heldout_mask]
                    for fraction in args.fractions:
                        cal_idx, eval_idx = split_calibration_indices(len(y_heldout), fraction, seed + int(source) * 1000)
                        if len(eval_idx) < 2:
                            continue
                        if len(cal_idx) > 0:
                            a, b = linear_calibrate(pred_heldout[cal_idx], y_heldout[cal_idx])
                        else:
                            a, b = 1.0, 0.0
                        pred_eval = a * pred_heldout[eval_idx] + b
                        rows.append(
                            metric_row(
                                target,
                                model_key,
                                int(source),
                                seed,
                                float(fraction),
                                y_heldout[eval_idx],
                                pred_eval,
                                len(cal_idx),
                                len(eval_idx),
                                a,
                                b,
                            )
                        )
                    print(f"  {target} source={source} seed={seed} model={MODELS[model_key][0]} done")

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No recalibration rows were produced.")

    group_cols = ["model", "model_key", "target_calibration_percent"]
    summary = (
        detail.groupby(group_cols, as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            rmse=("rmse", "mean"),
            rpd=("rpd", "mean"),
            n_evaluations=("r2", "size"),
        )
        .sort_values(["model", "target_calibration_percent"])
    )
    base = (
        detail[detail["target_calibration_percent"] == 0]
        .groupby(["model_key", "target"], as_index=False)["r2"]
        .mean()
        .rename(columns={"r2": "base_target_r2"})
    )
    by_trait = detail.groupby(["model_key", "target_calibration_percent", "target"], as_index=False)["r2"].mean()
    by_trait = by_trait.merge(base, on=["model_key", "target"], how="left")
    improved = (
        by_trait.assign(improved=by_trait["r2"] > by_trait["base_target_r2"])
        .groupby(["model_key", "target_calibration_percent"], as_index=False)["improved"]
        .sum()
        .rename(columns={"improved": "traits_improved"})
    )
    summary = summary.merge(improved, on=["model_key", "target_calibration_percent"], how="left")
    summary = summary[
        ["model", "target_calibration_percent", "mean_r2", "median_r2", "rmse", "rpd", "traits_improved", "n_evaluations"]
    ]

    detail_path = RESULTS_DIR / f"few_sample_source_recalibration_{args.run_name}_detail.csv"
    summary_path = RESULTS_DIR / f"few_sample_source_recalibration_{args.run_name}_summary.csv"
    table_path = TABLE_DIR / "table42_few_sample_source_recalibration.csv"
    summary_copy = SUMMARY_DIR / "table42_few_sample_source_recalibration.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    if not args.skip_paper_table:
        summary.to_csv(table_path, index=False)
        summary.to_csv(summary_copy, index=False)
    print("\nSaved:")
    print(detail_path)
    print(summary_path)
    if not args.skip_paper_table:
        print(table_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
