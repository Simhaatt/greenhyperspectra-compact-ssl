from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

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
RESULTS_DIR = ROOT / "results" / "beam"
TARGET_CHOICES = ["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"]
RANDOM_STATE = 42
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
    model = plsr_for_budget(len(selected_idx))
    selected = np.asarray(selected_idx, dtype=int)
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
    model = plsr_for_budget(len(selected_idx))
    selected = np.asarray(selected_idx, dtype=int)

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
) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
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
) -> tuple[dict[int, tuple[int, ...]], list[dict[str, object]]]:
    beam: list[tuple[float, tuple[int, ...]]] = [(float("inf"), tuple())]
    best_by_budget: dict[int, tuple[int, ...]] = {}
    rows: list[dict[str, object]] = []

    for step in range(1, max_budget + 1):
        expanded: list[tuple[float, tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()

        for _, selected in beam:
            selected_set = set(selected)
            available = [int(idx) for idx in candidate_pool if int(idx) not in selected_set]
            if expansion_limit > 0:
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
        best_score, best_selected = beam[0]
        best_by_budget[step] = best_selected

        rows.append(
            {
                "step": step,
                "beam_width": beam_width,
                "expanded_candidates": len(expanded),
                "best_validation_rmse": best_score,
                "best_band_indices": ",".join(str(idx) for idx in best_selected),
            }
        )
        print(f"step={step:02d}: val_RMSE={best_score:.6f}, expanded={len(expanded)}")

    return best_by_budget, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Beam-search compact wavelength selector using validation PLSR reward.")
    parser.add_argument("--target", default="cab", choices=TARGET_CHOICES)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--top-pool", type=int, default=220)
    parser.add_argument("--extra-random", type=int, default=40)
    parser.add_argument("--expansion-limit", type=int, default=260)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = args.target

    train, test, waves = load_split(target)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    y_train_full = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    x_selector_train, x_val, y_selector_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Target: {target}")
    print(f"Selector train rows: {len(y_selector_train)} | Validation rows: {len(y_val)}")
    print(f"Final train rows: {len(y_train_full)} | Test rows: {len(y_test)} | Bands: {len(waves)}")
    print("Computing correlation and mutual-information rankings...")
    corr_rank = abs_corr_ranking(x_selector_train, y_selector_train)
    mi_rank = np.argsort(mutual_info_regression(x_selector_train, y_selector_train, random_state=RANDOM_STATE, n_neighbors=5))[::-1]
    candidate_pool = build_candidate_pool(corr_rank, mi_rank, len(waves), args.top_pool, args.extra_random)
    print(f"Candidate pool size: {len(candidate_pool)}")

    best_by_budget, trajectory_rows = beam_search(
        max_budget=max(BUDGETS),
        candidate_pool=candidate_pool,
        x_train=x_selector_train,
        y_train=y_selector_train,
        x_val=x_val,
        y_val=y_val,
        beam_width=args.beam_width,
        expansion_limit=args.expansion_limit,
    )

    result_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        selected = best_by_budget[budget]
        validation_rmse = score_validation(selected, x_selector_train, y_selector_train, x_val, y_val)
        metrics = score_test(selected, x_train_full, y_train_full, x_test, y_test)
        selected_wavelengths = [waves[idx] for idx in selected]
        row: dict[str, object] = {
            "selector": "BeamSearch",
            "model": "PLSR",
            "target": target,
            "budget": budget,
            "bands": len(selected),
            "validation_rmse": validation_rmse,
            "selected_wavelengths": ",".join(selected_wavelengths),
            "selected_band_indices": ",".join(str(idx) for idx in selected),
            "beam_width": args.beam_width,
            "top_pool": args.top_pool,
            "extra_random": args.extra_random,
            "expansion_limit": args.expansion_limit,
        }
        row.update(metrics)
        result_rows.append(row)
        selected_rows.append(
            {
                "selector": "BeamSearch",
                "target": target,
                "budget": budget,
                "validation_rmse": validation_rmse,
                "selected_wavelengths": ",".join(selected_wavelengths),
                "selected_band_indices": ",".join(str(idx) for idx in selected),
            }
        )
        print(f"Beam+PLSR K={budget}: test_RMSE={metrics['rmse']:.6f}, R2={metrics['r2']:.4f}")

    results = pd.DataFrame(result_rows).sort_values(["budget", "rmse"])
    selected_df = pd.DataFrame(selected_rows)
    trajectory = pd.DataFrame(trajectory_rows)

    results.to_csv(RESULTS_DIR / f"{target}_beam_band_selection_results.csv", index=False)
    selected_df.to_csv(RESULTS_DIR / f"{target}_beam_selected_wavelengths.csv", index=False)
    trajectory.to_csv(RESULTS_DIR / f"{target}_beam_trajectory.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / f"{target}_beam_band_selection_results.csv")
    print(RESULTS_DIR / f"{target}_beam_selected_wavelengths.csv")
    print(RESULTS_DIR / f"{target}_beam_trajectory.csv")
    print("\nResults:")
    print(results[["selector", "model", "budget", "rmse", "mae", "r2", "rpd", "selected_wavelengths"]].to_string(index=False))


if __name__ == "__main__":
    main()
