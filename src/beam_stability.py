from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "beam_stability"
FIG_DIR = ROOT / "results" / "figures"
TRAITS = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
BUDGETS = [5, 10, 20, 30]


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_split(target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")
    waves = wavelength_columns(train)

    train = train.dropna(subset=[target]).reset_index(drop=True)
    test = test.dropna(subset=[target]).reset_index(drop=True)
    train = train[np.isfinite(train[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    test = test[np.isfinite(test[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    return train, test, waves


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    value = rmse(y_true, y_pred)
    if value == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / value)


def plsr_for_budget(n_bands: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", PLSRegression(n_components=min(20, n_bands))),
        ]
    )


def score_validation(
    selected_idx: tuple[int, ...],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    selected = np.asarray(selected_idx, dtype=int)
    model = plsr_for_budget(len(selected))
    model.fit(x_train[:, selected], y_train)
    pred = np.asarray(model.predict(x_val[:, selected])).reshape(-1)
    return rmse(y_val, pred)


def score_test(
    selected_idx: tuple[int, ...],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    selected = np.asarray(selected_idx, dtype=int)
    model = plsr_for_budget(len(selected))

    start = perf_counter()
    model.fit(x_train[:, selected], y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = np.asarray(model.predict(x_test[:, selected])).reshape(-1)
    predict_seconds = perf_counter() - start

    return {
        "rmse": rmse(y_test, pred),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "rpd": rpd(y_test, pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }


def abs_corr_ranking(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    y_centered = y_train - y_train.mean()
    x_centered = x_train - x_train.mean(axis=0, keepdims=True)
    numerator = np.abs(x_centered.T @ y_centered)
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.argsort(corr)[::-1]


def build_candidate_pool(
    corr_rank: np.ndarray,
    mi_rank: np.ndarray,
    n_bands: int,
    top_pool: int,
    extra_random: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ranked_pool = np.unique(np.concatenate([corr_rank[:top_pool], mi_rank[:top_pool]])).astype(int)
    remaining = np.setdiff1d(np.arange(n_bands), ranked_pool, assume_unique=False)
    if extra_random > 0 and len(remaining) > 0:
        random_extra = rng.choice(remaining, size=min(extra_random, len(remaining)), replace=False)
        ranked_pool = np.unique(np.concatenate([ranked_pool, random_extra])).astype(int)
    return ranked_pool


def beam_search(
    max_budget: int,
    candidate_pool: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    beam_width: int,
    expansion_limit: int,
) -> dict[int, tuple[int, ...]]:
    beam: list[tuple[float, tuple[int, ...]]] = [(float("inf"), tuple())]
    best_by_budget: dict[int, tuple[int, ...]] = {}

    for step in range(1, max_budget + 1):
        expanded: list[tuple[float, tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()

        for _, selected in beam:
            selected_set = set(selected)
            available = [int(idx) for idx in candidate_pool if int(idx) not in selected_set]
            available = available[:expansion_limit]

            for idx in available:
                candidate = selected + (idx,)
                key = tuple(sorted(candidate))
                if key in seen:
                    continue
                seen.add(key)
                value = score_validation(candidate, x_train, y_train, x_val, y_val)
                expanded.append((value, candidate))

        expanded.sort(key=lambda item: item[0])
        beam = expanded[:beam_width]
        best_by_budget[step] = beam[0][1]

    return best_by_budget


def run_one(
    target: str,
    seed: int,
    beam_width: int,
    top_pool: int,
    extra_random: int,
    expansion_limit: int,
) -> list[dict[str, object]]:
    train, test, waves = load_split(target)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    y_train_full = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    x_selector_train, x_val, y_selector_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed,
    )

    corr_rank = abs_corr_ranking(x_selector_train, y_selector_train)
    mi_rank = np.argsort(mutual_info_regression(x_selector_train, y_selector_train, random_state=seed, n_neighbors=5))[::-1]
    candidate_pool = build_candidate_pool(corr_rank, mi_rank, len(waves), top_pool, extra_random, seed)

    best_by_budget = beam_search(
        max_budget=max(BUDGETS),
        candidate_pool=candidate_pool,
        x_train=x_selector_train,
        y_train=y_selector_train,
        x_val=x_val,
        y_val=y_val,
        beam_width=beam_width,
        expansion_limit=expansion_limit,
    )

    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        selected = best_by_budget[budget]
        validation_rmse = score_validation(selected, x_selector_train, y_selector_train, x_val, y_val)
        metrics = score_test(selected, x_train_full, y_train_full, x_test, y_test)
        rows.append(
            {
                "trait": target,
                "selector": "BeamSearch",
                "model": "PLSR",
                "seed": seed,
                "budget": budget,
                "bands": len(selected),
                "validation_rmse": validation_rmse,
                "selected_band_indices": ",".join(str(idx) for idx in selected),
                "selected_wavelengths": ",".join(waves[idx] for idx in selected),
                "beam_width": beam_width,
                "top_pool": top_pool,
                "extra_random": extra_random,
                "expansion_limit": expansion_limit,
                **metrics,
            }
        )
    return rows


def save_frequency_figures(results: pd.DataFrame) -> None:
    for trait in sorted(results["trait"].unique()):
        trait_df = results[(results["trait"] == trait) & (results["budget"] == 30)]
        counts: dict[int, int] = {}
        for value in trait_df["selected_wavelengths"]:
            for wavelength in str(value).split(","):
                wave = int(wavelength)
                counts[wave] = counts.get(wave, 0) + 1

        freq = pd.DataFrame({"wavelength": list(counts.keys()), "count": list(counts.values())}).sort_values("wavelength")
        freq.to_csv(RESULTS_DIR / f"{trait}_beam_k30_wavelength_frequency.csv", index=False)

        top = freq.sort_values("count", ascending=False).head(40).sort_values("wavelength")
        plt.figure(figsize=(8.5, 4.3))
        plt.bar(top["wavelength"], top["count"], width=8)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Selection count across seeds")
        plt.title(f"{trait.upper()} BeamSearch K=30 wavelength frequency")
        plt.grid(True, axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"beam_selection_frequency_{trait}.png", dpi=220)
        plt.close()


def save_summary_figures(summary: pd.DataFrame) -> None:
    for metric in ["rmse", "r2"]:
        plt.figure(figsize=(8.5, 4.8))
        for trait in TRAITS:
            data = summary[summary["trait"] == trait].sort_values("budget")
            plt.errorbar(
                data["budget"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"].fillna(0),
                marker="o",
                linewidth=2,
                capsize=4,
                label=trait,
            )
        plt.xlabel("Selected wavelengths")
        plt.ylabel(metric.upper() if metric == "r2" else metric.upper())
        plt.title(f"BeamSearch stability: {metric.upper()} mean +/- std")
        plt.xticks(BUDGETS)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"beam_stability_{metric}.png", dpi=220)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed BeamSearch stability experiments.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--traits", nargs="+", default=TRAITS, choices=TRAITS)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--top-pool", type=int, default=160)
    parser.add_argument("--extra-random", type=int, default=30)
    parser.add_argument("--expansion-limit", type=int, default=180)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for trait in args.traits:
        for seed in args.seeds:
            print(f"Running trait={trait}, seed={seed}...")
            start = perf_counter()
            seed_rows = run_one(
                target=trait,
                seed=seed,
                beam_width=args.beam_width,
                top_pool=args.top_pool,
                extra_random=args.extra_random,
                expansion_limit=args.expansion_limit,
            )
            rows.extend(seed_rows)
            elapsed = perf_counter() - start
            best30 = [row for row in seed_rows if row["budget"] == 30][0]
            print(
                f"  done in {elapsed:.1f}s | K=30 RMSE={best30['rmse']:.6f}, "
                f"R2={best30['r2']:.4f}"
            )

    results = pd.DataFrame(rows).sort_values(["trait", "budget", "seed"])
    summary = (
        results.groupby(["trait", "selector", "model", "budget", "bands"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            rpd_mean=("rpd", "mean"),
            rpd_std=("rpd", "std"),
            validation_rmse_mean=("validation_rmse", "mean"),
            validation_rmse_std=("validation_rmse", "std"),
        )
        .sort_values(["trait", "budget"])
    )

    results.to_csv(RESULTS_DIR / "beam_stability_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "beam_stability_summary.csv", index=False)
    save_frequency_figures(results)
    save_summary_figures(summary)

    print("\nSaved:")
    print(RESULTS_DIR / "beam_stability_results.csv")
    print(RESULTS_DIR / "beam_stability_summary.csv")
    print(FIG_DIR / "beam_stability_rmse.png")
    print(FIG_DIR / "beam_stability_r2.png")
    for trait in args.traits:
        print(FIG_DIR / f"beam_selection_frequency_{trait}.png")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
