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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from beam_supervised_modern_heads import HeadConfig, make_model as make_labelled_model
from hybrid_ssl_beam_gated_heads import GatedHeadConfig, make_model as make_hybrid_model
from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "hybrid_ssl_label_efficiency"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


LABELLED_MODELS = {
    "labelled_residual": ("BeamK30_ResidualMLP", "residual"),
    "labelled_mlp": ("BeamK30_MLP", "mlp"),
    "labelled_polynomial": ("BeamK30_Polynomial", "polynomial"),
}

HYBRID_MODELS = {
    "hybrid_cross_gated_nam": ("Hybrid_BeamK30_SSL_CrossGatedNAMHead", "cross_gated_nam_head"),
    "hybrid_cross_gated_dcn": ("Hybrid_BeamK30_SSL_CrossGatedDCNHead", "cross_gated_dcn_head"),
    "hybrid_cross_gated_mlp": ("Hybrid_BeamK30_SSL_CrossGatedFusion", "cross_gated_fusion"),
}


def subset_indices(n_rows: int, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if fraction >= 0.999:
        return np.arange(n_rows)
    n_subset = int(round(n_rows * fraction))
    n_subset = max(8, min(n_rows, n_subset))
    return np.sort(rng.choice(n_rows, size=n_subset, replace=False))


def train_labelled(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    subset_idx: np.ndarray,
    model_key: str,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = HeadConfig(max_epochs=max_epochs, patience=patience)
    x_subset = x_train_all[subset_idx]
    y_subset = y_train_all[subset_idx]
    x_train, x_val, y_train, y_val = train_test_split(x_subset, y_subset, test_size=0.2, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_s = x_scaler.transform(x_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    x_test_t = torch.from_numpy(x_test_s).to(device)
    _, internal = LABELLED_MODELS[model_key]
    model = make_labelled_model(internal, x_train.shape[1], config).to(device)
    return _fit_single_input(model, loader, x_val_t, y_val_t, x_test_t, y_scaler, config, device)


def train_hybrid(
    beam_train_all: np.ndarray,
    ssl_train_all: np.ndarray,
    y_train_all: np.ndarray,
    beam_test: np.ndarray,
    ssl_test: np.ndarray,
    subset_idx: np.ndarray,
    model_key: str,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    config = GatedHeadConfig(max_epochs=max_epochs, patience=patience)
    beam_subset = beam_train_all[subset_idx]
    ssl_subset = ssl_train_all[subset_idx]
    y_subset = y_train_all[subset_idx]
    idx = np.arange(len(y_subset))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    beam_scaler = StandardScaler()
    ssl_scaler = StandardScaler()
    y_scaler = StandardScaler()
    beam_train = beam_scaler.fit_transform(beam_subset[train_idx]).astype(np.float32)
    beam_val = beam_scaler.transform(beam_subset[val_idx]).astype(np.float32)
    beam_test_s = beam_scaler.transform(beam_test).astype(np.float32)
    ssl_train = ssl_scaler.fit_transform(ssl_subset[train_idx]).astype(np.float32)
    ssl_val = ssl_scaler.transform(ssl_subset[val_idx]).astype(np.float32)
    ssl_test_s = ssl_scaler.transform(ssl_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_subset[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_subset[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(beam_train), torch.from_numpy(ssl_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    beam_val_t = torch.from_numpy(beam_val).to(device)
    ssl_val_t = torch.from_numpy(ssl_val).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    beam_test_t = torch.from_numpy(beam_test_s).to(device)
    ssl_test_t = torch.from_numpy(ssl_test_s).to(device)
    _, internal = HYBRID_MODELS[model_key]
    model = make_hybrid_model(internal, beam_train.shape[1], ssl_train.shape[1], config).to(device)
    return _fit_two_input(model, loader, beam_val_t, ssl_val_t, y_val_t, beam_test_t, ssl_test_t, y_scaler, config, device)


def _fit_single_input(
    model: nn.Module,
    loader: DataLoader,
    x_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    x_test_t: torch.Tensor,
    y_scaler: StandardScaler,
    config: HeadConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_t), y_val_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    train_seconds = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(x_test_t).detach().cpu().numpy().reshape(-1)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {"train_seconds": train_seconds, "best_epoch": float(best_epoch), "best_val_mse_scaled": best_val}


def _fit_two_input(
    model: nn.Module,
    loader: DataLoader,
    beam_val_t: torch.Tensor,
    ssl_val_t: torch.Tensor,
    y_val_t: torch.Tensor,
    beam_test_t: torch.Tensor,
    ssl_test_t: torch.Tensor,
    y_scaler: StandardScaler,
    config: GatedHeadConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for beam_b, ssl_b, yb in loader:
            beam_b = beam_b.to(device)
            ssl_b = ssl_b.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(beam_b, ssl_b), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(beam_val_t, ssl_val_t), y_val_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    train_seconds = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(beam_test_t, ssl_test_t).detach().cpu().numpy().reshape(-1)
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {"train_seconds": train_seconds, "best_epoch": float(best_epoch), "best_val_mse_scaled": best_val}


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ["r2", "rmse"]:
        for target, sub in summary.groupby("target"):
            plt.figure(figsize=(8.4, 5.0))
            for model, data in sub.sort_values("label_percent").groupby("model"):
                plt.errorbar(
                    data["label_percent"],
                    data[f"{metric}_mean"],
                    yerr=data[f"{metric}_std"].fillna(0),
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    label=model.replace("Hybrid_BeamK30_SSL_", "Hybrid ").replace("BeamK30_", "Beam "),
                )
            plt.xlabel("Training labels used (%)")
            plt.ylabel(metric.upper())
            plt.title(f"Hybrid SSL label efficiency: {target} {metric.upper()}")
            plt.grid(True, alpha=0.25)
            plt.legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"hybrid_ssl_label_efficiency_{run_name}_{target}_{metric}.png", dpi=220)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-efficiency comparison for labelled-only and SSL-hybrid BeamSearch models.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["labelled_residual", "hybrid_cross_gated_nam", "hybrid_cross_gated_dcn", "hybrid_cross_gated_mlp"],
        choices=list(LABELLED_MODELS) + list(HYBRID_MODELS),
    )
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=700)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--run-name", default="all8_3seeds")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    rows: list[dict[str, object]] = []
    print("Device:", device)
    print("Models:", args.models)
    print("Fractions:", args.fractions)

    for target in args.traits:
        train, test, waves = load_split(target)
        selected_idx, selected_wavelengths = load_beam_indices(target, args.budget, waves)
        x_train_full = train[waves].to_numpy(dtype=np.float32)
        x_test_full = test[waves].to_numpy(dtype=np.float32)
        beam_train = x_train_full[:, selected_idx]
        beam_test = x_test_full[:, selected_idx]
        y_train = train[target].to_numpy(dtype=np.float32)
        y_test = test[target].to_numpy(dtype=np.float32)
        needs_hybrid = any(m in HYBRID_MODELS for m in args.models)
        ssl_train = ssl_test = None
        if needs_hybrid:
            print(f"Embedding full spectra for {target}...")
            ssl_train = ssl_embeddings(x_train_full, ckpt, device, args.embedding_batch_size)
            ssl_test = ssl_embeddings(x_test_full, ckpt, device, args.embedding_batch_size)
        print(f"\\nTrait={target} train={len(y_train)} test={len(y_test)}")
        for fraction in args.fractions:
            for seed in args.seeds:
                idx = subset_indices(len(y_train), fraction, seed)
                for model_key in args.models:
                    if model_key in LABELLED_MODELS:
                        display_name, _ = LABELLED_MODELS[model_key]
                        pred, timing = train_labelled(
                            beam_train, y_train, beam_test, idx, model_key, seed, args.max_epochs, args.patience, device
                        )
                        family = "labelled_only"
                    else:
                        display_name, _ = HYBRID_MODELS[model_key]
                        assert ssl_train is not None and ssl_test is not None
                        pred, timing = train_hybrid(
                            beam_train,
                            ssl_train,
                            y_train,
                            beam_test,
                            ssl_test,
                            idx,
                            model_key,
                            seed,
                            args.max_epochs,
                            args.patience,
                            device,
                        )
                        family = "hybrid_ssl"
                    row = {
                        "target": target,
                        "family": family,
                        "model": display_name,
                        "model_key": model_key,
                        "label_fraction": fraction,
                        "label_percent": int(round(fraction * 100)),
                        "seed": seed,
                        "train_subset_rows": len(idx),
                        "train_total_rows": len(y_train),
                        "test_rows": len(y_test),
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
                        f"  {target} {display_name} labels={row['label_percent']}% seed={seed}: "
                        f"R2={row['r2']:.4f} RMSE={row['rmse']:.6f} rows={len(idx)}"
                    )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["target", "family", "model", "model_key", "label_fraction", "label_percent", "budget"], as_index=False)
        .agg(
            train_subset_rows_mean=("train_subset_rows", "mean"),
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
        .sort_values(["target", "model", "label_fraction"])
    )
    results_path = RESULTS_DIR / f"hybrid_ssl_label_efficiency_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"hybrid_ssl_label_efficiency_{args.run_name}_summary.csv"
    summary_copy = SUMMARY_DIR / f"hybrid_ssl_label_efficiency_{args.run_name}_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(summary_copy, index=False)
    save_figures(summary, args.run_name)
    print("\\nSaved:")
    print(results_path)
    print(summary_path)
    print(summary_copy)


if __name__ == "__main__":
    main()
