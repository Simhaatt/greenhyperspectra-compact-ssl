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

from hybrid_ssl_beam_mlp import load_checkpoint, ssl_embeddings
from torch_mlp_stability import load_beam_indices, load_split, rmse, rpd


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS_DIR = ROOT / "results" / "hybrid_ssl_beam_gated_heads"
SUMMARY_DIR = ROOT / "results" / "summary"
FIG_DIR = ROOT / "results" / "figures"
SSL_DIR = ROOT / "results" / "ssl"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]


@dataclass
class GatedHeadConfig:
    batch_size: int = 128
    max_epochs: int = 1000
    patience: int = 80
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    dropout: float = 0.05
    gate_hidden: int = 64
    residual_hidden: int = 128
    tail_hidden: int = 64
    branch_hidden: int = 64
    kan_hidden: int = 32
    grid_size: int = 10
    moe_experts: int = 3
    cross_layers: int = 2
    bilinear_rank: int = 32
    polynomial_rank: int = 32
    nam_hidden: int = 12
    sparse_moe_top_k: int = 2
    hyper_hidden: int = 64


class ResidualGatedMLP(nn.Module):
    def __init__(self, in_features: int, config: GatedHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Sequential(
            nn.Linear(in_features, config.gate_hidden),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, in_features),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(in_features, config.residual_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.residual_hidden, in_features),
        )
        self.norm = nn.LayerNorm(in_features)
        self.tail = nn.Sequential(
            nn.Linear(in_features, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        x = torch.cat([beam, ssl], dim=1)
        gate = self.gate(x)
        x = x * (0.5 + gate)
        x = self.norm(x + self.residual(x))
        return self.tail(x).squeeze(-1)


class CrossGatedBeamSSLFusion(nn.Module):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__()
        h = config.branch_hidden
        self.beam_branch = nn.Sequential(
            nn.Linear(beam_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.ssl_branch = nn.Sequential(
            nn.Linear(ssl_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.beam_to_ssl_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.ssl_to_beam_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(h * 2, config.tail_hidden),
            nn.LayerNorm(config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b = self.beam_branch(beam)
        s = self.ssl_branch(ssl)
        s = s * (0.5 + self.beam_to_ssl_gate(b))
        b = b * (0.5 + self.ssl_to_beam_gate(s))
        return self.fusion(torch.cat([b, s], dim=1)).squeeze(-1)


class KANLayer(nn.Module):
    """Small KAN-style radial basis layer for the fused representation head."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 10) -> None:
        super().__init__()
        centers = torch.linspace(-2.5, 2.5, grid_size)
        spacing = float(centers[1] - centers[0]) if grid_size > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (spacing * spacing)
        self.base_weight = nn.Parameter(torch.empty(in_features, out_features))
        self.spline_weight = nn.Parameter(torch.empty(in_features, out_features, grid_size))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.012)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = torch.einsum("bi,io->bo", x, self.base_weight)
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        spline = torch.einsum("big,iog->bo", basis, self.spline_weight)
        return base + spline + self.bias


class CrossGatedBase(nn.Module):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__()
        h = config.branch_hidden
        self.beam_branch = nn.Sequential(
            nn.Linear(beam_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.ssl_branch = nn.Sequential(
            nn.Linear(ssl_features, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )
        self.beam_to_ssl_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())
        self.ssl_to_beam_gate = nn.Sequential(nn.Linear(h, h), nn.Sigmoid())

    def encode_gated(self, beam: torch.Tensor, ssl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.beam_branch(beam)
        s = self.ssl_branch(ssl)
        s = s * (0.5 + self.beam_to_ssl_gate(b))
        b = b * (0.5 + self.ssl_to_beam_gate(s))
        return b, s

    def fused(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b, s = self.encode_gated(beam, ssl)
        return torch.cat([b, s], dim=1)


class CrossGatedKANHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.head = nn.Sequential(
            KANLayer(fused_dim, config.kan_hidden, config.grid_size),
            nn.LayerNorm(config.kan_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.kan_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused(beam, ssl)).squeeze(-1)


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


class CrossGatedResidualHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.head = nn.Sequential(
            ResidualBlock(fused_dim, config.residual_hidden, config.dropout),
            ResidualBlock(fused_dim, config.residual_hidden, config.dropout),
            nn.Linear(fused_dim, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused(beam, ssl)).squeeze(-1)


class CrossGatedMoEHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, config.tail_hidden),
            nn.SiLU(),
            nn.Linear(config.tail_hidden, config.moe_experts),
            nn.Softmax(dim=1),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, config.tail_hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.tail_hidden, 1),
                )
                for _ in range(config.moe_experts)
            ]
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        z = self.fused(beam, ssl)
        weights = self.gate(z)
        preds = torch.cat([expert(z) for expert in self.experts], dim=1)
        return torch.sum(weights * preds, dim=1)


class CrossLayer(nn.Module):
    """DCN-v1 cross layer: x_{l+1} = x0 * (w^T xl) + b + xl."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(features))
        self.bias = nn.Parameter(torch.zeros(features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        scale = torch.sum(x * self.weight, dim=1, keepdim=True)
        return x0 * scale + self.bias + x


class CrossGatedDCNHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.cross_layers = nn.ModuleList([CrossLayer(fused_dim) for _ in range(config.cross_layers)])
        self.deep = nn.Sequential(
            nn.Linear(fused_dim, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, config.tail_hidden),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim + config.tail_hidden, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        x0 = self.fused(beam, ssl)
        x_cross = x0
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)
        x_deep = self.deep(x0)
        return self.head(torch.cat([x_cross, x_deep], dim=1)).squeeze(-1)


class CrossGatedBilinearFusion(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        h = config.branch_hidden
        r = config.bilinear_rank
        self.beam_proj = nn.Linear(h, r, bias=False)
        self.ssl_proj = nn.Linear(h, r, bias=False)
        self.head = nn.Sequential(
            nn.Linear(h * 2 + r, config.tail_hidden),
            nn.LayerNorm(config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b, s = self.encode_gated(beam, ssl)
        interaction = self.beam_proj(b) * self.ssl_proj(s)
        return self.head(torch.cat([b, s, interaction], dim=1)).squeeze(-1)


class HighwayBlock(nn.Module):
    def __init__(self, features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(features, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, features),
        )
        self.gate = nn.Sequential(nn.Linear(features, features), nn.Sigmoid())
        self.norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        transformed = self.transform(x)
        return self.norm(gate * transformed + (1.0 - gate) * x)


class CrossGatedHighwayHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.head = nn.Sequential(
            HighwayBlock(fused_dim, config.residual_hidden, config.dropout),
            HighwayBlock(fused_dim, config.residual_hidden, config.dropout),
            nn.Linear(fused_dim, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        return self.head(self.fused(beam, ssl)).squeeze(-1)


class PolynomialInteractionLayer(nn.Module):
    """Low-rank second-order interaction layer, similar to a factorization machine."""

    def __init__(self, features: int, rank: int) -> None:
        super().__init__()
        self.proj = nn.Parameter(torch.empty(features, rank))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        summed = x @ self.proj
        squared_sum = summed * summed
        sum_squared = (x * x) @ (self.proj * self.proj)
        return 0.5 * (squared_sum - sum_squared)


class CrossGatedPolynomialHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.poly = PolynomialInteractionLayer(fused_dim, config.polynomial_rank)
        self.head = nn.Sequential(
            nn.Linear(fused_dim + config.polynomial_rank, config.tail_hidden),
            nn.LayerNorm(config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        z = self.fused(beam, ssl)
        interactions = self.poly(z)
        return self.head(torch.cat([z, interactions], dim=1)).squeeze(-1)


class NAMLayer(nn.Module):
    """Vectorized neural additive model over fused features."""

    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.weight_1 = nn.Parameter(torch.empty(features, hidden))
        self.bias_1 = nn.Parameter(torch.zeros(features, hidden))
        self.weight_2 = nn.Parameter(torch.empty(features, hidden))
        self.bias_out = nn.Parameter(torch.zeros(1))
        self.linear_skip = nn.Parameter(torch.zeros(features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight_1, mean=0.0, std=0.08)
        nn.init.normal_(self.weight_2, mean=0.0, std=0.03)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(x.unsqueeze(-1) * self.weight_1 + self.bias_1)
        additive = torch.sum(hidden * self.weight_2, dim=(1, 2))
        skip = torch.sum(x * self.linear_skip, dim=1)
        return additive + skip + self.bias_out


class CrossGatedNAMHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.nam = NAMLayer(fused_dim, config.nam_hidden)

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.norm(self.fused(beam, ssl)))
        return self.nam(z)


class CrossGatedSparseMoEHead(CrossGatedBase):
    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.top_k = min(config.sparse_moe_top_k, config.moe_experts)
        self.gate_logits = nn.Sequential(
            nn.Linear(fused_dim, config.tail_hidden),
            nn.SiLU(),
            nn.Linear(config.tail_hidden, config.moe_experts),
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, config.tail_hidden),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.tail_hidden, 1),
                )
                for _ in range(config.moe_experts)
            ]
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        z = self.fused(beam, ssl)
        logits = self.gate_logits(z)
        top_values, top_indices = torch.topk(logits, k=self.top_k, dim=1)
        sparse_weights = torch.zeros_like(logits)
        sparse_weights.scatter_(1, top_indices, torch.softmax(top_values, dim=1))
        preds = torch.cat([expert(z) for expert in self.experts], dim=1)
        return torch.sum(sparse_weights * preds, dim=1)


class CrossGatedHypernetworkHead(CrossGatedBase):
    """SSL context generates a dynamic linear regressor over gated Beam features."""

    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        h = config.branch_hidden
        self.hyper = nn.Sequential(
            nn.Linear(h, config.hyper_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hyper_hidden, h + 1),
        )
        self.context_head = nn.Sequential(
            nn.Linear(h * 2, config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )
        self.scale = h ** -0.5

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        b, s = self.encode_gated(beam, ssl)
        params = self.hyper(s)
        weights = params[:, :-1]
        bias = params[:, -1]
        dynamic = torch.sum(b * weights, dim=1) * self.scale + bias
        context = self.context_head(torch.cat([b, s], dim=1)).squeeze(-1)
        return 0.5 * dynamic + 0.5 * context


class CrossGatedMultiScaleHead(CrossGatedBase):
    """Parallel shallow, medium, and residual branches over the fused representation."""

    def __init__(self, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> None:
        super().__init__(beam_features, ssl_features, config)
        fused_dim = config.branch_hidden * 2
        self.small = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.SiLU(),
        )
        self.medium = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        self.global_branch = nn.Sequential(
            ResidualBlock(fused_dim, config.residual_hidden, config.dropout),
            nn.Linear(fused_dim, 32),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(96, config.tail_hidden),
            nn.LayerNorm(config.tail_hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.tail_hidden, 1),
        )

    def forward(self, beam: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        z = self.fused(beam, ssl)
        return self.head(torch.cat([self.small(z), self.medium(z), self.global_branch(z)], dim=1)).squeeze(-1)


def make_model(model_name: str, beam_features: int, ssl_features: int, config: GatedHeadConfig) -> nn.Module:
    if model_name == "residual_gated_mlp":
        return ResidualGatedMLP(beam_features + ssl_features, config)
    if model_name == "cross_gated_fusion":
        return CrossGatedBeamSSLFusion(beam_features, ssl_features, config)
    if model_name == "cross_gated_kan_head":
        return CrossGatedKANHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_residual_head":
        return CrossGatedResidualHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_moe_head":
        return CrossGatedMoEHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_dcn_head":
        return CrossGatedDCNHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_bilinear_fusion":
        return CrossGatedBilinearFusion(beam_features, ssl_features, config)
    if model_name == "cross_gated_highway_head":
        return CrossGatedHighwayHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_polynomial_head":
        return CrossGatedPolynomialHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_nam_head":
        return CrossGatedNAMHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_sparse_moe_head":
        return CrossGatedSparseMoEHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_hypernetwork_head":
        return CrossGatedHypernetworkHead(beam_features, ssl_features, config)
    if model_name == "cross_gated_multiscale_head":
        return CrossGatedMultiScaleHead(beam_features, ssl_features, config)
    raise ValueError(f"Unknown model: {model_name}")


def fit_predict(
    x_beam_all: np.ndarray,
    x_ssl_all: np.ndarray,
    y_train_all: np.ndarray,
    x_beam_test: np.ndarray,
    x_ssl_test: np.ndarray,
    seed: int,
    model_name: str,
    config: GatedHeadConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    idx = np.arange(len(y_train_all))
    train_idx, val_idx = train_test_split(idx, test_size=0.15, random_state=seed)

    beam_scaler = StandardScaler()
    ssl_scaler = StandardScaler()
    y_scaler = StandardScaler()

    beam_train = beam_scaler.fit_transform(x_beam_all[train_idx]).astype(np.float32)
    beam_val = beam_scaler.transform(x_beam_all[val_idx]).astype(np.float32)
    beam_test = beam_scaler.transform(x_beam_test).astype(np.float32)
    ssl_train = ssl_scaler.fit_transform(x_ssl_all[train_idx]).astype(np.float32)
    ssl_val = ssl_scaler.transform(x_ssl_all[val_idx]).astype(np.float32)
    ssl_test = ssl_scaler.transform(x_ssl_test).astype(np.float32)
    y_train_s = y_scaler.fit_transform(y_train_all[train_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)
    y_val_s = y_scaler.transform(y_train_all[val_idx].reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(beam_train), torch.from_numpy(ssl_train), torch.from_numpy(y_train_s)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    beam_val_t = torch.from_numpy(beam_val).to(device)
    ssl_val_t = torch.from_numpy(ssl_val).to(device)
    y_val_t = torch.from_numpy(y_val_s).to(device)
    beam_test_t = torch.from_numpy(beam_test).to(device)
    ssl_test_t = torch.from_numpy(ssl_test).to(device)

    model = make_model(model_name, x_beam_all.shape[1], x_ssl_all.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start = perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for beam_b, ssl_b, yb in train_loader:
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
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    train_seconds = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    start = perf_counter()
    model.eval()
    with torch.no_grad():
        pred_s = model(beam_test_t, ssl_test_t).detach().cpu().numpy().reshape(-1)
    predict_seconds = perf_counter() - start
    pred = y_scaler.inverse_transform(pred_s.reshape(-1, 1)).reshape(-1)
    return pred, {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
        "best_epoch": float(best_epoch),
        "best_val_mse_scaled": best_val,
    }


def run_trait(
    target: str,
    budget: int,
    ckpt: dict,
    seeds: list[int],
    model_names: list[str],
    config: GatedHeadConfig,
    device: torch.device,
    embedding_batch_size: int,
) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    selected_idx, selected_wavelengths = load_beam_indices(target, budget, waves)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    x_test_full = test[waves].to_numpy(dtype=np.float32)
    x_train_beam = x_train_full[:, selected_idx]
    x_test_beam = x_test_full[:, selected_idx]
    emb_train = ssl_embeddings(x_train_full, ckpt, device, embedding_batch_size)
    emb_test = ssl_embeddings(x_test_full, ckpt, device, embedding_batch_size)
    y_train = train[target].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    display_names = {
        "residual_gated_mlp": "Hybrid_BeamK30_SSL_ResidualGatedMLP",
        "cross_gated_fusion": "Hybrid_BeamK30_SSL_CrossGatedFusion",
        "cross_gated_kan_head": "Hybrid_BeamK30_SSL_CrossGatedKANHead",
        "cross_gated_residual_head": "Hybrid_BeamK30_SSL_CrossGatedResidualHead",
        "cross_gated_moe_head": "Hybrid_BeamK30_SSL_CrossGatedMoEHead",
        "cross_gated_dcn_head": "Hybrid_BeamK30_SSL_CrossGatedDCNHead",
        "cross_gated_bilinear_fusion": "Hybrid_BeamK30_SSL_CrossGatedBilinearFusion",
        "cross_gated_highway_head": "Hybrid_BeamK30_SSL_CrossGatedHighwayHead",
        "cross_gated_polynomial_head": "Hybrid_BeamK30_SSL_CrossGatedPolynomialHead",
        "cross_gated_nam_head": "Hybrid_BeamK30_SSL_CrossGatedNAMHead",
        "cross_gated_sparse_moe_head": "Hybrid_BeamK30_SSL_CrossGatedSparseMoEHead",
        "cross_gated_hypernetwork_head": "Hybrid_BeamK30_SSL_CrossGatedHypernetworkHead",
        "cross_gated_multiscale_head": "Hybrid_BeamK30_SSL_CrossGatedMultiScaleHead",
    }
    rows: list[dict[str, object]] = []
    print(
        f"\nTrait={target} | train={len(y_train)} test={len(y_test)} "
        f"beam={x_train_beam.shape[1]} ssl={emb_train.shape[1]} device={device}"
    )
    for model_name in model_names:
        for seed in seeds:
            pred, timing = fit_predict(
                x_train_beam,
                emb_train,
                y_train,
                x_test_beam,
                emb_test,
                seed,
                model_name,
                config,
                device,
            )
            row = {
                "target": target,
                "selector": "BeamSearch",
                "model": display_names[model_name],
                "seed": seed,
                "budget": budget,
                "beam_bands": budget,
                "ssl_embedding_dim": emb_train.shape[1],
                "features": x_train_beam.shape[1] + emb_train.shape[1],
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
                f"  {display_names[model_name]} seed={seed}: "
                f"RMSE={row['rmse']:.6f}, R2={row['r2']:.4f}, epoch={row['best_epoch']:.0f}"
            )
    return rows


def save_figures(summary: pd.DataFrame, run_name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for metric in ["r2", "rmse"]:
        plt.figure(figsize=(9.5, 5.0))
        for model, data in summary.sort_values("target").groupby("model"):
            plt.errorbar(
                data["target"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=model.replace("Hybrid_BeamK30_SSL_", ""),
            )
        plt.xlabel("Trait")
        plt.ylabel(metric.upper())
        plt.title(f"Hybrid BeamSearch K=30 + SSL gated heads: {metric.upper()}")
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"hybrid_ssl_beam_gated_heads_{run_name}_{metric}.png", dpi=240)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual gated MLP and cross-gated Beam/SSL fusion heads.")
    parser.add_argument("--checkpoint", default=str(SSL_DIR / "masked_autoencoder_contiguous_b64.pt"))
    parser.add_argument("--traits", nargs="+", default=["cab", "cw", "cm", "cbc"], choices=TRAITS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["residual_gated_mlp", "cross_gated_fusion"],
        choices=[
            "residual_gated_mlp",
            "cross_gated_fusion",
            "cross_gated_kan_head",
            "cross_gated_residual_head",
            "cross_gated_moe_head",
            "cross_gated_dcn_head",
            "cross_gated_bilinear_fusion",
            "cross_gated_highway_head",
            "cross_gated_polynomial_head",
            "cross_gated_nam_head",
            "cross_gated_sparse_moe_head",
            "cross_gated_hypernetwork_head",
            "cross_gated_multiscale_head",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--run-name", default="contiguous_b64_pilot")
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(Path(args.checkpoint), device)
    config = GatedHeadConfig()

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        rows.extend(run_trait(trait, args.budget, ckpt, args.seeds, args.models, config, device, args.embedding_batch_size))

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["target", "selector", "model", "budget", "beam_bands", "ssl_embedding_dim", "features"], as_index=False)
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

    run_name = args.run_name
    results_path = RESULTS_DIR / f"hybrid_ssl_beam_gated_heads_{run_name}_results.csv"
    summary_path = RESULTS_DIR / f"hybrid_ssl_beam_gated_heads_{run_name}_summary.csv"
    summary_copy_path = SUMMARY_DIR / f"hybrid_ssl_beam_gated_heads_{run_name}_summary.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(summary_copy_path, index=False)
    save_figures(summary, run_name)
    print("\nSaved:")
    print(results_path)
    print(summary_path)
    print(summary_copy_path)
    print(summary)


if __name__ == "__main__":
    main()
