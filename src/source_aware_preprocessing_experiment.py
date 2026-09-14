from __future__ import annotations

import argparse
from pathlib import Path

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
    load_beam_indices,
    load_labelled_union,
    maybe_ssl_embeddings,
    metric_row,
    summarize_model_table,
    train_predict_model,
)


VARIANTS = ["baseline", "snv", "per_source_zscore", "snv_per_source_zscore"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Source-aware preprocessing for leave-source-out validation.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=OOD_TRAITS)
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
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

    out_dir = RESULTS_DIR / "source_aware_preprocessing"
    summary_dir = RESULTS_DIR / "summary"
    table_dir = RESULTS_DIR / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    needs_hybrid = "hybrid_cross_gated_nam" in args.models
    df_all, waves = load_labelled_union()
    rows: list[dict[str, object]] = []

    print("Device:", device)
    print("Variants:", args.variants)
    for target in args.traits:
        df = finite_trait_frame(df_all, waves, target)
        sources = choose_sources(df, args.max_sources, args.min_test_labels, args.source_mode)
        if not sources:
            print(f"Skipping {target}: no valid sources")
            continue
        selected_idx, selected_wavelengths = load_beam_indices(target, args.budget, waves)
        raw_full = df[waves].to_numpy(dtype=np.float32)
        y = df[target].to_numpy(dtype=np.float32)
        datasets = df["dataset"].to_numpy()
        print(f"\nTrait={target} sources={sources}")
        for variant in args.variants:
            x_full = apply_preprocessing(raw_full, datasets, variant)
            x_beam = x_full[:, selected_idx]
            x_ssl = maybe_ssl_embeddings(x_full, Path(args.checkpoint), needs_hybrid, device, args.embedding_batch_size)
            for source in sources:
                test_mask = datasets == source
                train_mask = ~test_mask
                if int(test_mask.sum()) < args.min_test_labels or int(train_mask.sum()) < 30:
                    continue
                for seed in args.seeds:
                    for model_key in args.models:
                        pred, timing = train_predict_model(
                            model_key,
                            x_beam[train_mask],
                            None if x_ssl is None else x_ssl[train_mask],
                            y[train_mask],
                            x_beam[test_mask],
                            None if x_ssl is None else x_ssl[test_mask],
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )
                        row = metric_row(
                            target,
                            model_key,
                            int(source),
                            seed,
                            y[test_mask],
                            pred,
                            {
                                "preprocessing": variant,
                                "budget": args.budget,
                                "selected_wavelengths": selected_wavelengths,
                                "device": str(device),
                                **timing,
                            },
                        )
                        rows.append(row)
                        print(f"  {target} {variant} source={source} seed={seed} {model_key}: R2={row['r2']:.4f}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No source-aware preprocessing rows were produced.")
    summary = summarize_model_table(results, ["preprocessing", "model", "model_key"])
    by_trait = summarize_model_table(results, ["preprocessing", "target", "model", "model_key"])
    results_path = out_dir / f"source_aware_preprocessing_{args.run_name}_results.csv"
    summary_path = out_dir / f"source_aware_preprocessing_{args.run_name}_summary.csv"
    trait_path = out_dir / f"source_aware_preprocessing_{args.run_name}_trait_summary.csv"
    paper_path = table_dir / "table46_source_aware_preprocessing.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_trait.to_csv(trait_path, index=False)
    summary.to_csv(paper_path, index=False)
    summary.to_csv(summary_dir / paper_path.name, index=False)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(trait_path)
    print(paper_path)


if __name__ == "__main__":
    main()
