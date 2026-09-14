from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "beam_supervised_modern_heads"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class HeadConfig:
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    hidden_1: int = 64
    hidden_2: int = 32
    dropout: float = 0.05
    polynomial_rank: int = 16
    nam_hidden: int = 12
    grid_size: int = 10


class MLPHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, features),
        )
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            ResidualBlock(config.hidden_1, config.hidden_1, config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CrossLayer(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features))
        self.bias = nn.Parameter(torch.zeros(features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class DCNHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.cross1 = CrossLayer(in_features)
        self.cross2 = CrossLayer(in_features)
        self.deep = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, config.hidden_2),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(in_features + config.hidden_2, config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = self.cross2(x, self.cross1(x, x))
        xd = self.deep(x)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


class PolynomialInteractionLayer(nn.Module):
    def __init__(self, features: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.empty(features, rank))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summed = x @ self.proj
        return 0.5 * ((summed * summed) - ((x * x) @ (self.proj * self.proj)))


class PolynomialHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.poly = PolynomialInteractionLayer(in_features, config.polynomial_rank)
        self.net = nn.Sequential(
            nn.Linear(in_features + config.polynomial_rank, config.hidden_1),
            nn.LayerNorm(config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.poly(x)], dim=1)).squeeze(-1)


class KANLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int) -> None:
        super().__init__()
        centers = torch.linspace(-2.5, 2.5, grid_size)
        spacing = float(centers[1] - centers[0]) if grid_size > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (spacing * spacing)
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.spline_weight = nn.Parameter(torch.empty(in_features, out_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.012)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,io->bo", x, self.base_weight)
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        spline = torch.einsum("big,iog->bo", basis, self.spline_weight)
        return base + spline + self.bias


class KANHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, config.hidden_1),
            nn.LayerNorm(config.hidden_1),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            KANLayer(config.hidden_1, config.hidden_2, config.grid_size),
            nn.LayerNorm(config.hidden_2),
            nn.SiLU(),
            nn.Linear(config.hidden_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NAMHead(nn.Module):
    def __init__(self, in_features: int, config: HeadConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(config.dropout)
        self.weight_1 = nn.Parameter(torch.empty(in_features, config.nam_hidden))
        self.bias_1 = nn.Parameter(torch.zeros(in_features, config.nam_hidden))
        self.weight_2 = nn.Parameter(torch.empty(in_features, config.nam_hidden))
        self.linear_skip = nn.Parameter(torch.zeros(in_features))
        self.bias_out = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.weight_1, mean=0.0, std=0.08)
        nn.init.normal_(self.weight_2, mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.norm(x))
        hidden = F.silu(x.unsqueeze(-1) * self.weight_1 + self.bias_1)
        return torch.sum(hidden * self.weight_2, dim=(1, 2)) + torch.sum(x * self.linear_skip, dim=1) + self.bias_out


def make_model(model_name: str, in_features: int, config: HeadConfig) -> nn.Module:
    if model_name == "mlp":
        return MLPHead(in_features, config)
    if model_name == "residual":
        return ResidualHead(in_features, config)
    if model_name == "dcn":
        return DCNHead(in_features, config)
    if model_name == "polynomial":
        return PolynomialHead(in_features, config)
    if model_name == "kan":
        return KANHead(in_features, config)
    if model_name == "nam":
        return NAMHead(in_features, config)
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict(
    x_train_all: np.ndarray,
    y_train_all: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    model_name: str,
    config: HeadConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size=0.15, random_state=seed)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train).astype(np.float32)
    x_val = x_scaler.transform(x_val).astype(np.float32)
    x_test_scaled = x_scaler.transform(x_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    x_test_t = torch.from_numpy(x_test_scaled).to(device)

    model = make_model(model_name, x_train.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
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


def run_trait(target: str, budget: int, seeds: list[int], models: list[str], config: HeadConfig, device: torch.device) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)
    x_train = train[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)[:, selected_idx]
    y_test = test[target].to_numpy(dtype=np.float32)
    rows: list[dict[str, object]] = []
    names = {
        "mlp": "BeamK30_MLP",
        "residual": "BeamK30_ResidualMLP",
        "dcn": "BeamK30_DCN",
        "polynomial": "BeamK30_Polynomial",
        "kan": "BeamK30_KAN",
        "nam": "BeamK30_NAM",
    }
    print(f"\nTrait={target} | train={len(y_train)} test={len(y_test)} features={x_train.shape[1]} device={device}")
    for model_name in models:
        for seed in seeds:
            pred, timing = fit_predict(x_train, y_train, x_test, seed, model_name, config, device)
            row = {
                "target": target,
                "selector": "BeamSearch",
                "model": names[model_name],
                "seed": seed,
                "budget": budget,
                "features": x_train.shape[1],
                "rmse": rmse(y_test, pred),
                "mae": float(mean_absolute_error(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
                "rpd": rpd(y_test, pred),
                "device": str(device),
                "selected_wavelengths": selected_wavelengths,
                **timing,
            }
            rows.append(row)
            print(f"  {names[model_name]} seed={seed}: RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}, epoch={row['best_epoch']:.0f}")
    return rows


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ["r2", "rmse"]:
        plt.figure(figsize=(10, 5.2))
        for model, data in summary.sort_values("target").groupby("model"):
            plt.errorbar(
                data["target"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=model,
            )
        plt.xlabel("Trait")
        plt.ylabel(metric.upper())
        plt.title(f"Labelled-only BeamSearch K=30 modern heads: {metric.upper()}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"beam_supervised_modern_heads_{run_name}_{metric}.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Labelled-only BeamSearch K=30 modern regression heads.")
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--models", nargs="+", default=["mlp", "residual", "dcn", "polynomial", "kan", "nam"], choices=["mlp", "residual", "dcn", "polynomial", "kan", "nam"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--run-name", default="all8_3seeds")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = HeadConfig()
    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, args.seeds, args.models, config, device))

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["target", "selector", "model", "budget", "features"], as_index=False)
        .agg(
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
        .sort_values(["model", "target"])
    )
    results_path = RESULTS_DIR / f"beam_supervised_modern_heads_{args.run_name}_results.csv"
    summary_path = RESULTS_DIR / f"beam_supervised_modern_heads_{args.run_name}_summary.csv"
    summary_copy = SUMMARY_DIR / f"beam_supervised_modern_heads_{args.run_name}_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(summary_copy, index=False)
    save_figures(summary, args.run_name)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(summary)


if __name__ == "__main__":
    main()
