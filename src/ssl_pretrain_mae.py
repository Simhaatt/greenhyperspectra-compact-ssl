from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from time import perf_counter

import json
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results" / "ssl"


@dataclass
class SSLConfig:
    mask_ratio: float = 0.30
    mask_mode: str = "random"
    block_size: int = 64
    batch_size: int = 256
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    val_fraction: float = 0.05
    seed: int = 42


class SpectraDataset(Dataset):
    def __init__(self, spectra: np.ndarray) -> None:
        self.spectra = spectra.astype(np.float32)

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.spectra[idx])


class SpectralEncoder(nn.Module):
    def __init__(self, latent_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, latent_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(latent_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedSpectralAutoencoder(nn.Module):
    def __init__(self, n_bands: int, latent_channels: int = 128) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out[..., : self.n_bands]


def load_unlabeled_spectra(max_samples: int | None = None) -> tuple[np.ndarray, list[str]]:
    ds_loaded = load_from_disk(str(RAW_DIR / "greenhyperspectra_unlabeled_hf"))
    ds = ds_loaded["train"] if hasattr(ds_loaded, "keys") and "train" in ds_loaded.keys() else ds_loaded
    wave_cols = [col for col in ds.column_names if str(col).isdigit()]
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    arr = ds.select_columns(wave_cols).to_pandas().to_numpy(dtype=np.float32)
    finite = np.isfinite(arr).all(axis=1)
    arr = arr[finite]
    return arr, wave_cols


def standardize_spectra(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def make_random_mask(x: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    return (torch.rand_like(x) < mask_ratio).float()


def make_contiguous_mask(x: torch.Tensor, mask_ratio: float, block_size: int) -> torch.Tensor:
    batch_size, _, n_bands = x.shape
    target_masked = max(1, int(round(n_bands * mask_ratio)))
    block_size = max(1, min(block_size, n_bands))
    n_blocks = max(1, int(np.ceil(target_masked / block_size)))
    mask = torch.zeros_like(x)

    for row in range(batch_size):
        for _ in range(n_blocks):
            start = int(torch.randint(0, n_bands - block_size + 1, (1,), device=x.device).item())
            mask[row, :, start : start + block_size] = 1.0

        masked_count = int(mask[row].sum().item())
        if masked_count > target_masked:
            masked_positions = torch.nonzero(mask[row, 0] > 0, as_tuple=False).flatten()
            remove_count = masked_count - target_masked
            remove_idx = masked_positions[torch.randperm(len(masked_positions), device=x.device)[:remove_count]]
            mask[row, :, remove_idx] = 0.0
        elif masked_count < target_masked:
            unmasked_positions = torch.nonzero(mask[row, 0] == 0, as_tuple=False).flatten()
            add_count = min(target_masked - masked_count, len(unmasked_positions))
            add_idx = unmasked_positions[torch.randperm(len(unmasked_positions), device=x.device)[:add_count]]
            mask[row, :, add_idx] = 1.0

    return mask


def make_mask(x: torch.Tensor, config: SSLConfig) -> torch.Tensor:
    if config.mask_mode == "random":
        return make_random_mask(x, config.mask_ratio)
    if config.mask_mode == "contiguous":
        return make_contiguous_mask(x, config.mask_ratio, config.block_size)
    raise ValueError(f"Unsupported mask mode: {config.mask_mode}")


def run_epoch(
    model: MaskedSpectralAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: SSLConfig,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_count = 0
    loss_fn = nn.MSELoss(reduction="sum")

    for spectra in loader:
        spectra = spectra.to(device).unsqueeze(1)
        mask = make_mask(spectra, config)
        corrupted = spectra * (1.0 - mask)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            recon = model(corrupted)
            loss = loss_fn(recon * mask, spectra * mask) / torch.clamp(mask.sum(), min=1.0)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = spectra.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain a masked spectral autoencoder on GreenHyperSpectra unlabeled spectra.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-mode", choices=["random", "contiguous"], default="random")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="masked_autoencoder.pt")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = SSLConfig(
        mask_ratio=args.mask_ratio,
        mask_mode=args.mask_mode,
        block_size=args.block_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading unlabeled spectra...")
    x_raw, wave_cols = load_unlabeled_spectra(args.max_samples)
    x, mean, std = standardize_spectra(x_raw)
    dataset = SpectraDataset(x)
    val_size = max(1, int(round(len(dataset) * config.val_fraction)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed))
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = MaskedSpectralAutoencoder(n_bands=x.shape[1], latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []

    print(
        f"Device: {device} | samples={len(dataset)} | bands={x.shape[1]} | train={train_size} val={val_size} "
        f"| mask_mode={config.mask_mode} | mask_ratio={config.mask_ratio} | block_size={config.block_size}"
    )
    start_all = perf_counter()
    for epoch in range(1, config.epochs + 1):
        start = perf_counter()
        train_loss = run_epoch(model, train_loader, device, config, optimizer)
        val_loss = run_epoch(model, val_loader, device, config, None)
        seconds = perf_counter() - start
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": seconds})
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} seconds={seconds:.1f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
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

    ckpt_path = OUT_DIR / args.out
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder_state": model.encoder.state_dict(),
            "wave_cols": wave_cols,
            "mean": mean,
            "std": std,
            "config": asdict(config),
            "latent_channels": args.latent_channels,
            "n_bands": x.shape[1],
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )
    pd.DataFrame(history).to_csv(OUT_DIR / "masked_autoencoder_history.csv", index=False)
    (OUT_DIR / "masked_autoencoder_meta.json").write_text(
        json.dumps({"checkpoint": str(ckpt_path), "best_epoch": best_epoch, "best_val_loss": best_val, "total_seconds": perf_counter() - start_all}, indent=2),
        encoding="utf-8",
    )
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
