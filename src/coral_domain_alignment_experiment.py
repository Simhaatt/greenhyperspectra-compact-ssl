from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from domain_shift_experiment_utils import (
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


def coral_transform(source: np.ndarray, target: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    # Align source feature covariance to target covariance with a standard CORAL transform.
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    xs = source - source_mean
    xt = target - target_mean
    cs = np.cov(xs, rowvar=False) + np.eye(source.shape[1]) * eps
    ct = np.cov(xt, rowvar=False) + np.eye(target.shape[1]) * eps
    es, vs = np.linalg.eigh(cs)
    et, vt = np.linalg.eigh(ct)
    cs_inv_sqrt = vs @ np.diag(1.0 / np.sqrt(np.maximum(es, eps))) @ vs.T
    ct_sqrt = vt @ np.diag(np.sqrt(np.maximum(et, eps))) @ vt.T
    return ((xs @ cs_inv_sqrt @ ct_sqrt) + target_mean).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot CORAL feature alignment for leave-source-out validation.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=OOD_TRAITS)
    parser.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    parser.add_argument("--preprocessing", default="baseline", choices=["baseline", "snv", "per_source_zscore", "snv_per_source_zscore"])
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

    out_dir = RESULTS_DIR / "coral_domain_alignment"
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
    print("CORAL preprocessing:", args.preprocessing)
    for target in args.traits:
        df = finite_trait_frame(df_all, waves, target)
        sources = choose_sources(df, args.max_sources, args.min_test_labels, args.source_mode)
        if not sources:
            continue
        selected_idx, selected_wavelengths = load_beam_indices(target, args.budget, waves)
        raw_full = df[waves].to_numpy(dtype=np.float32)
        x_full = apply_preprocessing(raw_full, df["dataset"].to_numpy(), args.preprocessing)
        x_beam_base = x_full[:, selected_idx]
        x_ssl_base = maybe_ssl_embeddings(x_full, Path(args.checkpoint), needs_hybrid, device, args.embedding_batch_size)
        y = df[target].to_numpy(dtype=np.float32)
        datasets = df["dataset"].to_numpy()
        print(f"\nTrait={target} sources={sources}")
        for source in sources:
            test_mask = datasets == source
            train_mask = ~test_mask
            if int(test_mask.sum()) < args.min_test_labels or int(train_mask.sum()) < 30:
                continue
            x_beam_coral = x_beam_base.copy()
            x_beam_coral[train_mask] = coral_transform(x_beam_base[train_mask], x_beam_base[test_mask])
            if x_ssl_base is not None:
                x_ssl_coral = x_ssl_base.copy()
                x_ssl_coral[train_mask] = coral_transform(x_ssl_base[train_mask], x_ssl_base[test_mask])
            else:
                x_ssl_coral = None
            for seed in args.seeds:
                for model_key in args.models:
                    for alignment, xb, xs in [
                        ("none", x_beam_base, x_ssl_base),
                        ("coral_train_to_heldout_unlabelled", x_beam_coral, x_ssl_coral),
                    ]:
                        pred, timing = train_predict_model(
                            model_key,
                            xb[train_mask],
                            None if xs is None else xs[train_mask],
                            y[train_mask],
                            xb[test_mask],
                            None if xs is None else xs[test_mask],
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
                                "alignment": alignment,
                                "preprocessing": args.preprocessing,
                                "budget": args.budget,
                                "selected_wavelengths": selected_wavelengths,
                                "device": str(device),
                                **timing,
                            },
                        )
                        rows.append(row)
                        print(f"  {target} source={source} seed={seed} {model_key} {alignment}: R2={row['r2']:.4f}")

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No CORAL rows were produced.")
    summary = summarize_model_table(results, ["alignment", "model", "model_key"])
    results_path = out_dir / f"coral_domain_alignment_{args.run_name}_results.csv"
    summary_path = out_dir / f"coral_domain_alignment_{args.run_name}_summary.csv"
    paper_path = table_dir / "table48_coral_domain_alignment.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(paper_path, index=False)
    summary.to_csv(summary_dir / paper_path.name, index=False)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(paper_path)


if __name__ == "__main__":
    main()
