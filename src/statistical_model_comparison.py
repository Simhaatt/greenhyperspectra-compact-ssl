from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "results"
INPUT = RESULTS / "hybrid_ssl_fusion_ablation" / "hybrid_ssl_fusion_ablation_fusion_ablation_all8_3seeds_results.csv"
OUT = RESULTS / "paper_tables" / "table40_statistical_model_comparison.csv"
SUMMARY_OUT = RESULTS / "summary" / "table40_statistical_model_comparison.csv"

BASELINE_KEY = "beam_only_residual"
COMPARISONS = [
    ("Hybrid NAM - Beam ResidualMLP", "cross_gated_nam"),
    ("Additive - Beam ResidualMLP", "additive_fusion"),
    ("FiLM - Beam ResidualMLP", "film_fusion"),
    ("SSL-only - Beam ResidualMLP", "ssl_only_mlp"),
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def interpretation(mean_delta: float, p_value: float) -> str:
    if mean_delta <= -0.15 and p_value < 0.05:
        return "clearly weaker"
    if abs(mean_delta) <= 0.05:
        return "competitive"
    if mean_delta < 0:
        return "slightly lower"
    return "higher"


def main() -> None:
    df = pd.read_csv(INPUT)
    baseline = df[df["model_key"] == BASELINE_KEY][["target", "seed", "r2"]].rename(columns={"r2": "baseline_r2"})
    rows: list[dict[str, object]] = []

    for label, model_key in COMPARISONS:
        model = df[df["model_key"] == model_key][["target", "seed", "r2"]].rename(columns={"r2": "model_r2"})
        paired = baseline.merge(model, on=["target", "seed"], how="inner")
        if paired.empty:
            raise RuntimeError(f"No paired rows found for {model_key}")
        deltas = (paired["model_r2"] - paired["baseline_r2"]).to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(deltas)
        trait_delta = paired.groupby("target").apply(
            lambda g: float((g["model_r2"] - g["baseline_r2"]).mean()),
            include_groups=False,
        )
        try:
            p_value = float(wilcoxon(deltas).pvalue)
        except ValueError:
            p_value = 1.0
        effect = float(deltas.mean() / deltas.std(ddof=1)) if len(deltas) > 1 and deltas.std(ddof=1) > 0 else 0.0
        rows.append(
            {
                "comparison": label,
                "n_pairs": int(len(deltas)),
                "mean_delta_r2": float(deltas.mean()),
                "median_delta_r2": float(np.median(deltas)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "wilcoxon_p_value": p_value,
                "effect_size_cohen_dz": effect,
                "traits_improved": int((trait_delta > 0).sum()),
                "traits_compared": int(trait_delta.shape[0]),
                "interpretation": interpretation(float(deltas.mean()), p_value),
            }
        )

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    out.to_csv(SUMMARY_OUT, index=False)
    print(OUT)
    print(SUMMARY_OUT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
