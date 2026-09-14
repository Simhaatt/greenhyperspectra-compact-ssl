from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from domain_shift_experiment_utils import (
    FIG_DIR,
    MODEL_KEYS,
    OOD_TRAITS,
    RESULTS_DIR,
    SSL_DIR,
    apply_preprocessing,
    choose_sources,
    finite_trait_frame,
    linear_calibrate,
    load_beam_indices,
    load_labelled_union,
    maybe_ssl_embeddings,
    metric_row,
    split_calibration_indices,
    summarize_model_table,
    train_predict_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot held-out-source adaptation sweep.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=OOD_TRAITS)
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--preprocessing", default="baseline", choices=["baseline", "snv", "per_source_zscore", "snv_per_source_zscore"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-sources", type=int, default=5)
    parser.add_argument("--min-test-labels", type=int, default=30)
    parser.add_argument("--source-mode", choices=["largest", "smallest", "spread"], default="largest")
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--run-name", default="top5_sources_3seeds")
    args = parser.parse_args()

    out_dir = RESULTS_DIR / "few_shot_finetuning"
    summary_dir = RESULTS_DIR / "summary"
    table_dir = RESULTS_DIR / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_hybrid = "hybrid_cross_gated_nam" in args.models
    df_all, waves = load_labelled_union()
    rows: list[dict[str, object]] = []

    print("Device:", device)
    print("Few-shot preprocessing:", args.preprocessing)
    for target in args.traits:
        df = finite_trait_frame(df_all, waves, target)
        sources = choose_sources(df, args.max_sources, args.min_test_labels, args.source_mode)
        if not sources:
            print(f"Skipping {target}: no valid sources")
            continue
        selected_idx, selected_wavelengths = load_beam_indices(target, args.budget, waves)
        raw_full = df[waves].to_numpy(dtype=np.float32)
        x_full = apply_preprocessing(raw_full, df["dataset"].to_numpy(), args.preprocessing)
        x_beam = x_full[:, selected_idx]
        x_ssl = maybe_ssl_embeddings(x_full, Path(args.checkpoint), needs_hybrid, device, args.embedding_batch_size)
        y = df[target].to_numpy(dtype=np.float32)
        datasets = df["dataset"].to_numpy()
        print(f"\nTrait={target} sources={sources}")
        for source in sources:
            heldout_mask = datasets == source
            train_mask = ~heldout_mask
            heldout_idx = np.where(heldout_mask)[0]
            if len(heldout_idx) < args.min_test_labels or int(train_mask.sum()) < 30:
                continue
            for seed in args.seeds:
                for model_key in args.models:
                    # One zero-shot prediction per source/model/seed; head-only uses it for linear head calibration.
                    zero_pred, zero_timing = train_predict_model(
                        model_key,
                        x_beam[train_mask],
                        None if x_ssl is None else x_ssl[train_mask],
                        y[train_mask],
                        x_beam[heldout_mask],
                        None if x_ssl is None else x_ssl[heldout_mask],
                        seed,
                        args.max_epochs,
                        args.patience,
                        device,
                    )
                    for fraction in args.fractions:
                        cal_local, eval_local = split_calibration_indices(len(heldout_idx), fraction, seed + int(source) * 1000)
                        if len(eval_local) < 2:
                            continue
                        y_heldout = y[heldout_mask]
                        if len(cal_local) > 0:
                            a, b = linear_calibrate(zero_pred[cal_local], y_heldout[cal_local])
                        else:
                            a, b = 1.0, 0.0
                        head_pred = a * zero_pred[eval_local] + b
                        rows.append(
                            metric_row(
                                target,
                                model_key,
                                int(source),
                                seed,
                                y_heldout[eval_local],
                                head_pred,
                                {
                                    "adaptation_mode": "head_only_linear_calibration",
                                    "target_fraction": fraction,
                                    "target_percent": int(round(fraction * 100)),
                                    "calibration_rows": int(len(cal_local)),
                                    "preprocessing": args.preprocessing,
                                    "budget": args.budget,
                                    "selected_wavelengths": selected_wavelengths,
                                    "device": str(device),
                                    **zero_timing,
                                },
                            )
                        )
                        # Practical full-update proxy: refit the same architecture with target calibration samples included.
                        if len(cal_local) > 0:
                            cal_global = heldout_idx[cal_local]
                            eval_global = heldout_idx[eval_local]
                            full_train_mask = train_mask.copy()
                            full_train_mask[cal_global] = True
                            full_pred, timing = train_predict_model(
                                model_key,
                                x_beam[full_train_mask],
                                None if x_ssl is None else x_ssl[full_train_mask],
                                y[full_train_mask],
                                x_beam[eval_global],
                                None if x_ssl is None else x_ssl[eval_global],
                                seed,
                                args.max_epochs,
                                args.patience,
                                device,
                            )
                            rows.append(
                                metric_row(
                                    target,
                                    model_key,
                                    int(source),
                                    seed,
                                    y[eval_global],
                                    full_pred,
                                    {
                                        "adaptation_mode": "full_target_augmented_refit",
                                        "target_fraction": fraction,
                                        "target_percent": int(round(fraction * 100)),
                                        "calibration_rows": int(len(cal_local)),
                                        "preprocessing": args.preprocessing,
                                        "budget": args.budget,
                                        "selected_wavelengths": selected_wavelengths,
                                        "device": str(device),
                                        **timing,
                                    },
                                )
                            )
                    print(f"  {target} source={source} seed={seed} {model_key} done")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No few-shot rows were produced.")
    summary = summarize_model_table(results, ["adaptation_mode", "model", "model_key", "target_percent"])
    results_path = out_dir / f"few_shot_finetuning_{args.run_name}_results.csv"
    summary_path = out_dir / f"few_shot_finetuning_{args.run_name}_summary.csv"
    paper_path = table_dir / "table47_few_shot_finetuning_sweep.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paper_path, index=False)
    summary.to_csv(summary_dir / paper_path.name, index=False)

    fig_path = FIG_DIR / f"few_shot_finetuning_{args.run_name}_mean_r2.png"
    plt.figure(figsize=(8.6, 5.0))
    plot_df = summary.groupby(["adaptation_mode", "target_percent"], as_index=False)["mean_r2"].mean()
    for mode, sub in plot_df.groupby("adaptation_mode"):
        sub = sub.sort_values("target_percent")
        plt.plot(sub["target_percent"], sub["mean_r2"], marker="o", linewidth=2, label=mode)
    plt.axhline(0.62, color="black", linestyle="--", linewidth=1, alpha=0.6, label="in-distribution reference R2=0.62")
    plt.axhline(0, color="gray", linewidth=1, alpha=0.5)
    plt.xlabel("Held-out-source labels used for adaptation (%)")
    plt.ylabel("Mean OOD R2")
    plt.title("Few-shot held-out-source adaptation")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=240)
    plt.close()
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(paper_path)
    print(fig_path)


if __name__ == "__main__":
    main()
