from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from hybrid_ssl_label_efficiency import HYBRID_MODELS, LABELLED_MODELS, train_hybrid, train_labelled
from torch_mlp_stability import load_beam_indices, rmse, rpd, wavelength_columns
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "ood_source_validation"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


def load_labelled_union() -> tuple[pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    df = pd.concat([train.assign(original_split="train"), test.assign(original_split="test")], ignore_index=True)
    waves = wavelength_columns(df)
    return df, waves


def finite_trait_frame(df: pd.DataFrame, waves: list[str], target: str) -> pd.DataFrame:
    cols = waves + [target, "dataset"]
    out = df.dropna(subset=[target, "dataset"]).reset_index(drop=True)
    finite = np.isfinite(out[cols].to_numpy(dtype=np.float64)).all(axis=1)
    return out[finite].reset_index(drop=True)


def choose_sources(
    df: pd.DataFrame,
    max_sources: int,
    min_test_labels: int,
    source_mode: str,
) -> list[int]:
    counts = df.groupby("dataset").size().sort_values(ascending=False)
    counts = counts[counts >= min_test_labels]
    if source_mode == "largest":
        selected = counts.head(max_sources)
    elif source_mode == "smallest":
        selected = counts.sort_values(ascending=True).head(max_sources)
    else:
        if len(counts) <= max_sources:
            selected = counts
        else:
            positions = np.linspace(0, len(counts) - 1, max_sources).round().astype(int)
            selected = counts.iloc[positions]
    return [int(x) for x in selected.index.tolist()]


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mean = (
        summary.groupby(["target", "family", "model"], as_index=False)
        .agg(r2_mean=("r2_mean", "mean"), r2_std=("r2_mean", "std"))
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )
    plt.figure(figsize=(11, 5.6))
    for model, sub in mean.groupby("model"):
        sub = sub.sort_values("target")
        plt.plot(sub["target"], sub["r2_mean"], marker="o", linewidth=2, label=model.replace("Hybrid_BeamK30_SSL_", "Hybrid ").replace("BeamK30_", "Beam "))
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    plt.xlabel("Trait")
    plt.ylabel("Mean leave-source-out R2")
    plt.title("OOD/source validation by model")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ood_source_validation_{run_name}_r2_by_trait.png", dpi=240)
    plt.close()

    model_mean = summary.groupby(["family", "model"], as_index=False)["r2_mean"].mean().sort_values("r2_mean", ascending=False)
    plt.figure(figsize=(9.5, 4.8))
    labels = [m.replace("Hybrid_BeamK30_SSL_", "Hybrid ").replace("BeamK30_", "Beam ") for m in model_mean["model"]]
    colors = ["#21A67A" if f == "hybrid_ssl" else "#246BFE" for f in model_mean["family"]]
    plt.barh(range(len(model_mean)), model_mean["r2_mean"], color=colors)
    plt.yticks(range(len(model_mean)), labels, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean leave-source-out R2")
    plt.title("OOD/source validation leaderboard")
    plt.grid(axis="x", alpha=0.25)
    for i, value in enumerate(model_mean["r2_mean"]):
        plt.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"ood_source_validation_{run_name}_leaderboard.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-source-out/OOD validation using the GreenHyperSpectra dataset column.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=["cab", "cw", "cm", "LAI", "cp", "cbc", "car"], choices=TRAITS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["labelled_residual", "hybrid_cross_gated_nam", "hybrid_cross_gated_dcn", "hybrid_cross_gated_mlp"],
        choices=list(LABELLED_MODELS) + list(HYBRID_MODELS),
    )
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all, waves = load_labelled_union()
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []

    print("Device:", device)
    print("Traits:", args.traits)
    print("Models:", args.models)
    print("Source mode:", args.source_mode)

    for target in args.traits:
        df = finite_trait_frame(df_all, waves, target)
        sources = choose_sources(df, args.max_sources, args.min_test_labels, args.source_mode)
        source_counts = df.groupby("dataset").size().to_dict()
        print(f"\nTrait={target} labelled_rows={len(df)} candidate_sources={sources}")
        if len(sources) < 1:
            print(f"Skipping {target}: no source has at least {args.min_test_labels} labels")
            continue
        selected_idx, selected_wavelengths = load_beam_indices(target, args.budget, waves)
        x_full = df[waves].to_numpy(dtype=np.float32)
        x_beam = x_full[:, selected_idx]
        y = df[target].to_numpy(dtype=np.float32)
        datasets = df["dataset"].to_numpy()
        needs_hybrid = any(m in HYBRID_MODELS for m in args.models)
        x_ssl = None
        if needs_hybrid:
            print(f"Embedding full spectra for {target}...")
            x_ssl = ssl_embeddings(x_full, ckpt, device, args.embedding_batch_size)

        for source in sources:
            test_mask = datasets == source
            train_mask = ~test_mask
            test_rows = int(test_mask.sum())
            train_rows = int(train_mask.sum())
            source_rows.append(
                {
                    "target": target,
                    "heldout_dataset": int(source),
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "source_total_labels": int(source_counts[source]),
                }
            )
            if train_rows < 30 or test_rows < args.min_test_labels:
                continue
            for seed in args.seeds:
                train_idx = np.where(train_mask)[0]
                for model_key in args.models:
                    if model_key in LABELLED_MODELS:
                        display_name, _ = LABELLED_MODELS[model_key]
                        pred, timing = train_labelled(
                            x_beam[train_mask],
                            y[train_mask],
                            x_beam[test_mask],
                            np.arange(train_rows),
                            model_key,
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )
                        family = "labelled_only"
                    else:
                        assert x_ssl is not None
                        display_name, _ = HYBRID_MODELS[model_key]
                        pred, timing = train_hybrid(
                            x_beam[train_mask],
                            x_ssl[train_mask],
                            y[train_mask],
                            x_beam[test_mask],
                            x_ssl[test_mask],
                            np.arange(train_rows),
                            model_key,
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )
                        family = "hybrid_ssl"
                    y_test = y[test_mask]
                    row = {
                        "target": target,
                        "family": family,
                        "model": display_name,
                        "model_key": model_key,
                        "heldout_dataset": int(source),
                        "seed": seed,
                        "train_rows": train_rows,
                        "test_rows": test_rows,
                        "budget": args.budget,
                        "rmse": rmse(y_test, pred),
                        "mae": float(mean_absolute_error(y_test, pred)),
                        "r2": float(r2_score(y_test, pred)),
                        "rpd": rpd(y_test, pred),
                        "device": str(device),
                        "selected_wavelengths": selected_wavelengths,
                        **timing,
                    }
                    rows.append(row)
                    print(
                        f"  {target} source={source} {display_name} seed={seed}: "
                        f"R2={row['r2']:.4f} RMSE={row['rmse']:.6f} test={test_rows}"
                    )

    results = pd.DataFrame(rows)
    source_audit = pd.DataFrame(source_rows)
    if results.empty:
        raise RuntimeError("No OOD results were produced. Lower --min-test-labels or check source labels.")
    summary = (
        results.groupby(["target", "family", "model", "model_key", "heldout_dataset", "budget"], as_index=False)
        .agg(
            train_rows=("train_rows", "mean"),
            test_rows=("test_rows", "mean"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            train_seconds_mean=("train_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .sort_values(["target", "heldout_dataset", "model"])
    )
    model_summary = (
        summary.groupby(["target", "family", "model", "model_key", "budget"], as_index=False)
        .agg(
            heldout_sources=("heldout_dataset", "nunique"),
            test_rows_mean=("test_rows", "mean"),
            rmse_mean=("rmse_mean", "mean"),
            r2_mean=("r2_mean", "mean"),
            r2_std_across_sources=("r2_mean", "std"),
            rpd_mean=("rpd_mean", "mean"),
        )
        .sort_values(["target", "r2_mean"], ascending=[True, False])
    )

    results_path = RESULTS_DIR / f"ood_source_validation_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"ood_source_validation_{args.run_name}_summary_by_source.csv"
    model_summary_path = RESULTS_DIR / f"ood_source_validation_{args.run_name}_model_summary.csv"
    source_path = RESULTS_DIR / f"ood_source_validation_{args.run_name}_source_audit.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    model_summary.to_csv(model_summary_path, index=False)
    source_audit.to_csv(source_path, index=False)
    model_summary.to_csv(SUMMARY_DIR / f"ood_source_validation_{args.run_name}_model_summary.csv", index=False)
    save_figures(summary, args.run_name)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(model_summary_path)
    print(source_path)


if __name__ == "__main__":
    main()
