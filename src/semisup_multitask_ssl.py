from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from itertools import cycle
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ssl_pretrain_mae import SpectralEncoder, make_contiguous_mask, make_random_mask
from torch_mlp_stability import rpd, rmse


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "semisup_ssl"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class SemiSupConfig:
    traits: list[str]
    epochs: int = 60
    warmup_epochs: int = 8
    patience: int = 12
    unlabeled_batch_size: int = 256
    labeled_batch_size: int = 128
    steps_per_epoch: int = 220
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    mask_ratio: float = 0.30
    mask_mode: str = "contiguous"
    block_size: int = 64
    trait_loss_weight: float = 1.0
    labeled_recon_weight: float = 0.5
    val_fraction: float = 0.15
    latent_channels: int = 128
    seed: int = 42
    max_unlabeled_samples: int | None = 50000


class LabeledMultiTraitDataset(Dataset):
    def __init__(self, spectra: np.ndarray, y: np.ndarray, y_mask: np.ndarray) -> None:
        self.spectra = spectra.astype(np.float32)
        self.y = y.astype(np.float32)
        self.y_mask = y_mask.astype(np.float32)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.spectra[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.y_mask[idx]),
        )


class SemiSupervisedSpectralRegressor(nn.Module):
    def __init__(self, n_bands: int, n_traits: int, latent_channels: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.encoder = SpectralEncoder(latent_channels=latent_channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(latent_channels, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, n_traits),
        )

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)[..., : self.n_bands]

    def predict_traits(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.regression_head(z)


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_unlabeled_spectra(wave_cols: list[str], max_samples: int | None, seed: int) -> np.ndarray:
    ds_loaded = load_from_disk(str(RAW_DIR / "greenhyperspectra_unlabeled_hf"))
    ds = ds_loaded["train"] if hasattr(ds_loaded, "keys") and "train" in ds_loaded.keys() else ds_loaded
    available = [col for col in wave_cols if col in ds.column_names]
    if len(available) != len(wave_cols):
        raise ValueError("Unlabeled wavelength columns do not match labelled split.")
    if max_samples is not None and max_samples < len(ds):
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(ds), size=max_samples, replace=False))
        ds = ds.select(idx.tolist())
    arr = ds.select_columns(wave_cols).to_pandas().to_numpy(dtype=np.float32)
    finite = np.isfinite(arr).all(axis=1)
    return arr[finite]


def standardize_with_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def make_spectral_mask(x: torch.Tensor, config: SemiSupConfig) -> torch.Tensor:
    if config.mask_mode == "random":
        return make_random_mask(x, config.mask_ratio)
    if config.mask_mode == "contiguous":
        return make_contiguous_mask(x, config.mask_ratio, config.block_size)
    raise ValueError(f"Unsupported mask mode: {config.mask_mode}")


def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((recon - target).pow(2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def masked_trait_loss(pred: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor) -> torch.Tensor:
    return ((pred - y).pow(2) * y_mask).sum() / torch.clamp(y_mask.sum(), min=1.0)


def build_labeled_arrays(
    df: pd.DataFrame,
    wave_cols: list[str],
    traits: list[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_spectra = np.isfinite(df[wave_cols].to_numpy(dtype=np.float64)).all(axis=1)
    has_trait = df[traits].notna().any(axis=1).to_numpy()
    df = df[valid_spectra & has_trait].reset_index(drop=True)
    x = standardize_with_stats(df[wave_cols].to_numpy(dtype=np.float32), x_mean, x_std)
    y_raw = df[traits].to_numpy(dtype=np.float32)
    y_mask = np.isfinite(y_raw).astype(np.float32)
    y_filled = np.where(np.isfinite(y_raw), y_raw, target_mean)
    target_std = np.where(target_std < 1e-6, 1.0, target_std)
    y_scaled = ((y_filled - target_mean) / target_std).astype(np.float32)
    return x, y_scaled, y_mask


def compute_target_stats(train_df: pd.DataFrame, traits: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for trait in traits:
        values = train_df[trait].dropna().to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        means.append(float(values.mean()))
        std = float(values.std())
        stds.append(std if std > 1e-6 else 1.0)
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def run_epoch(
    model: SemiSupervisedSpectralRegressor,
    unlabeled_iter,
    labeled_iter,
    steps: int,
    device: torch.device,
    config: SemiSupConfig,
    optimizer: torch.optim.Optimizer,
    effective_trait_loss_weight: float,
) -> dict[str, float]:
    model.train()
    total_recon_unlabeled = 0.0
    total_recon_labeled = 0.0
    total_trait = 0.0

    for _ in range(steps):
        xu = next(unlabeled_iter)[0].to(device).unsqueeze(1)
        xl, y, y_mask = next(labeled_iter)
        xl = xl.to(device).unsqueeze(1)
        y = y.to(device)
        y_mask = y_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        mu = make_spectral_mask(xu, config)
        recon_u = model.reconstruct(xu * (1.0 - mu))
        loss_u = masked_recon_loss(recon_u, xu, mu)

        ml = make_spectral_mask(xl, config)
        recon_l = model.reconstruct(xl * (1.0 - ml))
        loss_l = masked_recon_loss(recon_l, xl, ml)

        if effective_trait_loss_weight > 0:
            pred = model.predict_traits(xl)
            loss_trait = masked_trait_loss(pred, y, y_mask)
        else:
            loss_trait = torch.zeros((), device=device)

        loss = loss_u + config.labeled_recon_weight * loss_l + effective_trait_loss_weight * loss_trait
        loss.backward()
        optimizer.step()

        total_recon_unlabeled += float(loss_u.item())
        total_recon_labeled += float(loss_l.item())
        total_trait += float(loss_trait.item())

    return {
        "train_unlabeled_recon": total_recon_unlabeled / steps,
        "train_labeled_recon": total_recon_labeled / steps,
        "train_trait_mse_scaled": total_trait / steps,
        "effective_trait_loss_weight": effective_trait_loss_weight,
    }


def evaluate_validation(
    model: SemiSupervisedSpectralRegressor,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_mask = 0.0
    with torch.no_grad():
        for x, y, y_mask in loader:
            x = x.to(device).unsqueeze(1)
            y = y.to(device)
            y_mask = y_mask.to(device)
            pred = model.predict_traits(x)
            total_loss += float(((pred - y).pow(2) * y_mask).sum().item())
            total_mask += float(y_mask.sum().item())
    return total_loss / max(total_mask, 1.0)


def evaluate_test(
    model: SemiSupervisedSpectralRegressor,
    test_df: pd.DataFrame,
    wave_cols: list[str],
    traits: list[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_spectra = np.isfinite(test_df[wave_cols].to_numpy(dtype=np.float64)).all(axis=1)
    test_df = test_df[valid_spectra].reset_index(drop=True)
    x = standardize_with_stats(test_df[wave_cols].to_numpy(dtype=np.float32), x_mean, x_std)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)

    preds_scaled = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            preds_scaled.append(model.predict_traits(xb.to(device).unsqueeze(1)).detach().cpu().numpy())
    pred_scaled = np.vstack(preds_scaled)
    pred = pred_scaled * target_std.reshape(1, -1) + target_mean.reshape(1, -1)

    rows = []
    pred_rows = []
    for trait_idx, trait in enumerate(traits):
        y_true = test_df[trait].to_numpy(dtype=np.float32)
        mask = np.isfinite(y_true)
        if not mask.any():
            continue
        y_t = y_true[mask]
        y_p = pred[mask, trait_idx]
        rows.append(
            {
                "target": trait,
                "test_rows": int(mask.sum()),
                "rmse": rmse(y_t, y_p),
                "mae": float(mean_absolute_error(y_t, y_p)),
                "r2": float(r2_score(y_t, y_p)),
                "rpd": rpd(y_t, y_p),
            }
        )
        for idx, true_value, pred_value in zip(np.where(mask)[0], y_t, y_p):
            pred_rows.append(
                {
                    "target": trait,
                    "row_index": int(idx),
                    "actual": float(true_value),
                    "predicted": float(pred_value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    ordered = summary.sort_values("target")
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(ordered["target"], ordered["r2"], color="#2E74B5")
    plt.axhline(0, color="#333333", linewidth=0.8)
    plt.ylabel("R2")
    plt.title(f"Semi-supervised multitask SSL test R2 ({run_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"semisup_multitask_{run_name}_r2.png", dpi=240)
    plt.close()

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(ordered["target"], ordered["rmse"], color="#7A5195")
    plt.ylabel("RMSE")
    plt.title(f"Semi-supervised multitask SSL test RMSE ({run_name})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"semisup_multitask_{run_name}_rmse.png", dpi=240)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semi-supervised multitask SSL for GreenHyperSpectra trait regression.")
    parser.add_argument("--traits", nargs="+", default=["cab", "cw", "cm", "cbc"], choices=TRAITS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--unlabeled-batch-size", type=int, default=256)
    parser.add_argument("--labeled-batch-size", type=int, default=128)
    parser.add_argument("--steps-per-epoch", type=int, default=220)
    parser.add_argument("--max-unlabeled-samples", type=int, default=50000)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-mode", choices=["random", "contiguous"], default="contiguous")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--trait-loss-weight", type=float, default=1.0)
    parser.add_argument("--labeled-recon-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="pilot_contiguous_b64")
    args = parser.parse_args()

    config = SemiSupConfig(
        traits=args.traits,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        unlabeled_batch_size=args.unlabeled_batch_size,
        labeled_batch_size=args.labeled_batch_size,
        steps_per_epoch=args.steps_per_epoch,
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        block_size=args.block_size,
        trait_loss_weight=args.trait_loss_weight,
        labeled_recon_weight=args.labeled_recon_weight,
        learning_rate=args.learning_rate,
        latent_channels=args.latent_channels,
        seed=args.seed,
        max_unlabeled_samples=args.max_unlabeled_samples,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_all = perf_counter()

    train_df = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test_df = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    wave_cols = wavelength_columns(train_df)

    print("Loading unlabeled spectra...")
    x_unlabeled_raw = load_unlabeled_spectra(wave_cols, config.max_unlabeled_samples, config.seed)
    x_mean = x_unlabeled_raw.mean(axis=0).astype(np.float32)
    x_std = x_unlabeled_raw.std(axis=0).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    x_unlabeled = standardize_with_stats(x_unlabeled_raw, x_mean, x_std)

    target_mean, target_std = compute_target_stats(train_df, config.traits)
    x_labeled, y_labeled, y_mask = build_labeled_arrays(train_df, wave_cols, config.traits, x_mean, x_std, target_mean, target_std)
    train_idx, val_idx = train_test_split(
        np.arange(len(x_labeled)),
        test_size=config.val_fraction,
        random_state=config.seed,
        shuffle=True,
    )

    unlabeled_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_unlabeled)),
        batch_size=config.unlabeled_batch_size,
        shuffle=True,
        drop_last=True,
    )
    labeled_train = LabeledMultiTraitDataset(x_labeled[train_idx], y_labeled[train_idx], y_mask[train_idx])
    labeled_val = LabeledMultiTraitDataset(x_labeled[val_idx], y_labeled[val_idx], y_mask[val_idx])
    labeled_loader = DataLoader(labeled_train, batch_size=config.labeled_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(labeled_val, batch_size=config.labeled_batch_size, shuffle=False)

    model = SemiSupervisedSpectralRegressor(
        n_bands=len(wave_cols),
        n_traits=len(config.traits),
        latent_channels=config.latent_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    print(
        f"Device={device} | unlabeled={len(x_unlabeled)} | labeled={len(x_labeled)} | val={len(val_idx)} "
        f"| traits={config.traits} | mask={config.mask_mode}:{config.mask_ratio} block={config.block_size} "
        f"| warmup_epochs={config.warmup_epochs} | lr={config.learning_rate} | trait_weight={config.trait_loss_weight}"
    )

    for epoch in range(1, config.epochs + 1):
        start = perf_counter()
        effective_trait_loss_weight = 0.0 if epoch <= config.warmup_epochs else config.trait_loss_weight
        metrics = run_epoch(
            model,
            cycle(unlabeled_loader),
            cycle(labeled_loader),
            config.steps_per_epoch,
            device,
            config,
            optimizer,
            effective_trait_loss_weight,
        )
        val_trait = evaluate_validation(model, val_loader, device)
        seconds = perf_counter() - start
        row = {"epoch": epoch, "val_trait_mse_scaled": val_trait, "seconds": seconds, **metrics}
        history.append(row)
        print(
            f"epoch={epoch:03d} trait_w={effective_trait_loss_weight:.3f} trait_train={metrics['train_trait_mse_scaled']:.5f} "
            f"val_trait={val_trait:.5f} recon_u={metrics['train_unlabeled_recon']:.5f} seconds={seconds:.1f}"
        )
        if epoch <= config.warmup_epochs:
            continue
        if val_trait < best_val - 1e-5:
            best_val = val_trait
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_epoch = config.epochs
        best_val = evaluate_validation(model, val_loader, device)

    summary, predictions = evaluate_test(
        model,
        test_df,
        wave_cols,
        config.traits,
        x_mean,
        x_std,
        target_mean,
        target_std,
        device,
        config.labeled_batch_size,
    )
    summary.insert(0, "run_name", args.run_name)
    summary.insert(1, "model", "SemiSup_Multitask_SSL")
    summary["best_epoch"] = best_epoch
    summary["best_val_trait_mse_scaled"] = best_val
    summary["device"] = str(device)

    checkpoint_path = OUT_DIR / f"semisup_multitask_{args.run_name}.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "wave_cols": wave_cols,
            "traits": config.traits,
            "x_mean": x_mean,
            "x_std": x_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "config": asdict(config),
            "best_epoch": best_epoch,
            "best_val_trait_mse_scaled": best_val,
        },
        checkpoint_path,
    )

    history_path = OUT_DIR / f"semisup_multitask_{args.run_name}_history.csv"
    summary_path = OUT_DIR / f"semisup_multitask_{args.run_name}_summary.csv"
    pred_path = OUT_DIR / f"semisup_multitask_{args.run_name}_predictions.csv"
    meta_path = OUT_DIR / f"semisup_multitask_{args.run_name}_meta.json"
    pd.DataFrame(history).to_csv(history_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(SUMMARY_DIR / f"semisup_multitask_{args.run_name}_summary.csv", index=False)
    predictions.to_csv(pred_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary": str(summary_path),
                "history": str(history_path),
                "predictions": str(pred_path),
                "total_seconds": perf_counter() - start_all,
                "config": asdict(config),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_figures(summary, args.run_name)

    print("\nTest summary:")
    print(summary[["target", "test_rows", "rmse", "mae", "r2", "rpd"]].to_string(index=False))
    print("\nSaved:")
    for path in [checkpoint_path, history_path, summary_path, pred_path, meta_path]:
        print(path)


if __name__ == "__main__":
    main()
