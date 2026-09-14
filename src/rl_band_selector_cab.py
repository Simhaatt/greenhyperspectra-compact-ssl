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

from train_cab_kan_band_selection import TrainConfig, fit_predict_kan

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "rl"
RANDOM_STATE = 42
BUDGETS = [5, 10, 20, 30]
EPISODES = 4
TOP_POOL = 250
TOP_CANDIDATES = 35
RANDOM_CANDIDATES = 35
EPSILON_START = 0.25
EPSILON_END = 0.05


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


def score_plsr_validation(
    selected_idx: list[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    model = plsr_for_budget(len(selected_idx))
    model.fit(x_train[:, selected_idx], y_train)
    pred = np.asarray(model.predict(x_val[:, selected_idx])).reshape(-1)
    return rmse(y_val, pred)


def score_plsr_test(
    selected_idx: list[int],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    start = perf_counter()
    model = plsr_for_budget(len(selected_idx))
    model.fit(x_train[:, selected_idx], y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    pred = np.asarray(model.predict(x_test[:, selected_idx])).reshape(-1)
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


def make_candidate_pool(corr_rank: np.ndarray, mi_rank: np.ndarray, n_bands: int) -> np.ndarray:
    pool = np.unique(np.concatenate([corr_rank[:TOP_POOL], mi_rank[:TOP_POOL]]))
    if len(pool) < min(n_bands, TOP_POOL):
        missing = np.setdiff1d(np.arange(n_bands), pool, assume_unique=False)
        pool = np.concatenate([pool, missing[: min(len(missing), TOP_POOL - len(pool))]])
    return pool.astype(int)


def candidate_actions(
    selected: list[int],
    corr_rank: np.ndarray,
    mi_rank: np.ndarray,
    pool: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    selected_set = set(selected)

    ranked_candidates: list[int] = []
    for ranking in [corr_rank, mi_rank]:
        for idx in ranking:
            idx_int = int(idx)
            if idx_int not in selected_set:
                ranked_candidates.append(idx_int)
            if len(ranked_candidates) >= TOP_CANDIDATES * 2:
                break

    available_pool = np.asarray([idx for idx in pool if int(idx) not in selected_set], dtype=int)
    random_count = min(RANDOM_CANDIDATES, len(available_pool))
    random_candidates = rng.choice(available_pool, size=random_count, replace=False) if random_count else np.asarray([], dtype=int)

    candidates = np.unique(np.concatenate([np.asarray(ranked_candidates, dtype=int), random_candidates]))
    return candidates.astype(int)


def run_episode(
    budget: int,
    episode: int,
    x_sel_train: np.ndarray,
    y_sel_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    corr_rank: np.ndarray,
    mi_rank: np.ndarray,
    pool: np.ndarray,
) -> tuple[list[int], list[dict[str, object]]]:
    rng = np.random.default_rng(RANDOM_STATE + budget * 1000 + episode)
    selected: list[int] = []
    trajectory: list[dict[str, object]] = []

    for step in range(1, budget + 1):
        epsilon = EPSILON_START + (EPSILON_END - EPSILON_START) * ((step - 1) / max(1, budget - 1))
        candidates = candidate_actions(selected, corr_rank, mi_rank, pool, rng)

        if rng.random() < epsilon:
            action = int(rng.choice(candidates))
            validation_rmse = score_plsr_validation(selected + [action], x_sel_train, y_sel_train, x_val, y_val)
            policy = "explore"
        else:
            best_action = None
            best_rmse = float("inf")
            for candidate in candidates:
                candidate_idx = int(candidate)
                value = score_plsr_validation(selected + [candidate_idx], x_sel_train, y_sel_train, x_val, y_val)
                if value < best_rmse:
                    best_rmse = value
                    best_action = candidate_idx
            action = int(best_action)
            validation_rmse = best_rmse
            policy = "exploit"

        selected.append(action)
        trajectory.append(
            {
                "budget": budget,
                "episode": episode,
                "step": step,
                "action_band_index": action,
                "validation_rmse": validation_rmse,
                "reward": -validation_rmse,
                "epsilon": epsilon,
                "policy": policy,
                "candidate_count": len(candidates),
            }
        )

    return selected, trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run epsilon-greedy RL-style band selection for one trait.")
    parser.add_argument("--target", default="cab", choices=["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"])
    parser.add_argument("--skip-kan", action="store_true", help="Evaluate only RL+PLSR.")
    args = parser.parse_args()
    target = args.target

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train, test, waves = load_split(target)
    x_train_full = train[waves].to_numpy(dtype=np.float32)
    y_train_full = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    x_sel_train, x_val, y_sel_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Target: {target}")
    print(f"Selector train rows: {len(y_sel_train)} | Validation rows: {len(y_val)}")
    print(f"Final train rows: {len(y_train_full)} | Test rows: {len(y_test)} | Bands: {len(waves)}")
    print("Computing correlation and mutual-information rankings...")
    corr_rank = abs_corr_ranking(x_sel_train, y_sel_train)
    mi_rank = np.argsort(mutual_info_regression(x_sel_train, y_sel_train, random_state=RANDOM_STATE, n_neighbors=5))[::-1]
    pool = make_candidate_pool(corr_rank, mi_rank, len(waves))
    print(f"Candidate pool size: {len(pool)}")

    selected_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []

    for budget in BUDGETS:
        print(f"\nBudget {budget}")
        for episode in range(EPISODES):
            selected, trajectory = run_episode(
                budget,
                episode,
                x_sel_train,
                y_sel_train,
                x_val,
                y_val,
                corr_rank,
                mi_rank,
                pool,
            )
            final_val_rmse = trajectory[-1]["validation_rmse"]
            selected_wavelengths = [waves[i] for i in selected]
            selected_rows.append(
                {
                    "selector": "EpsilonGreedyRL",
                    "budget": budget,
                    "episode": episode,
                    "validation_rmse": final_val_rmse,
                    "selected_band_indices": ",".join(str(i) for i in selected),
                    "selected_wavelengths": ",".join(selected_wavelengths),
                }
            )
            for row in trajectory:
                row = dict(row)
                row["action_wavelength"] = waves[int(row["action_band_index"])]
                trajectory_rows.append(row)
            print(f"episode={episode}: val_RMSE={final_val_rmse:.4f}, bands={','.join(selected_wavelengths)}")

    selected_df = pd.DataFrame(selected_rows)
    trajectory_df = pd.DataFrame(trajectory_rows)

    best_selected = selected_df.loc[selected_df.groupby("budget")["validation_rmse"].idxmin()].copy()

    result_rows: list[dict[str, object]] = []
    kan_config = TrainConfig(max_epochs=400, patience=40)
    if torch is not None and not args.skip_kan:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None

    print("\nEvaluating best RL selections on held-out test set...")
    for _, row in best_selected.iterrows():
        budget = int(row["budget"])
        selected_idx = [int(value) for value in str(row["selected_band_indices"]).split(",")]

        plsr_metrics = score_plsr_test(selected_idx, x_train_full, y_train_full, x_test, y_test)
        result_row: dict[str, object] = {
            "selector": "EpsilonGreedyRL",
            "model": "PLSR",
            "target": target,
            "budget": budget,
            "bands": len(selected_idx),
            "episode": int(row["episode"]),
            "validation_rmse": float(row["validation_rmse"]),
            "selected_wavelengths": row["selected_wavelengths"],
        }
        result_row.update(plsr_metrics)
        result_rows.append(result_row)
        print(f"RL+PLSR K={budget}: RMSE={plsr_metrics['rmse']:.4f}, R2={plsr_metrics['r2']:.4f}")

        if device is not None:
            kan_metrics, _ = fit_predict_kan(
                x_train_full,
                y_train_full,
                x_test,
                y_test,
                np.asarray(selected_idx, dtype=int),
                device,
                kan_config,
                RANDOM_STATE + budget,
            )
            kan_row: dict[str, object] = {
                "selector": "EpsilonGreedyRL",
                "model": "KAN",
                "target": target,
                "budget": budget,
                "bands": len(selected_idx),
                "episode": int(row["episode"]),
                "validation_rmse": float(row["validation_rmse"]),
                "selected_wavelengths": row["selected_wavelengths"],
                "device": str(device),
            }
            kan_row.update(kan_metrics)
            result_rows.append(kan_row)
            print(f"RL+KAN  K={budget}: RMSE={kan_metrics['rmse']:.4f}, R2={kan_metrics['r2']:.4f}, device={device}")

    results_df = pd.DataFrame(result_rows).sort_values(["budget", "rmse"])
    selected_df.to_csv(RESULTS_DIR / f"{target}_rl_selected_wavelengths_all_episodes.csv", index=False)
    trajectory_df.to_csv(RESULTS_DIR / f"{target}_rl_trajectories.csv", index=False)
    best_selected.to_csv(RESULTS_DIR / f"{target}_rl_selected_wavelengths.csv", index=False)
    results_df.to_csv(RESULTS_DIR / f"{target}_rl_band_selection_results.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / f"{target}_rl_selected_wavelengths_all_episodes.csv")
    print(RESULTS_DIR / f"{target}_rl_trajectories.csv")
    print(RESULTS_DIR / f"{target}_rl_selected_wavelengths.csv")
    print(RESULTS_DIR / f"{target}_rl_band_selection_results.csv")
    print("\nBest RL test results:")
    print(results_df[["selector", "model", "budget", "rmse", "mae", "r2", "rpd", "selected_wavelengths"]].to_string(index=False))


if __name__ == "__main__":
    main()
