from __future__ import annotations
import os

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


ROOT = Path(os.environ.get("HSI_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results" / "baselines"
RANDOM_STATE = 42


def wavelength_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).isdigit()]


def load_split(target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_train.parquet")
    test = pd.read_parquet(RAW_DIR / "greenhyperspectra_labeled_test.parquet")

    waves = wavelength_columns(train)
    train = train.dropna(subset=[target]).reset_index(drop=True)
    test = test.dropna(subset=[target]).reset_index(drop=True)

    # Keep only spectra with finite reflectance values and target labels.
    train = train[np.isfinite(train[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
    test = test[np.isfinite(test[waves + [target]].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)

    return train, test, waves


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    prediction_rmse = rmse(y_true, y_pred)
    if prediction_rmse == 0:
        return float("inf")
    return float(np.std(y_true, ddof=1) / prediction_rmse)


def evaluate(target: str, name: str, model: object, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict[str, float | str], np.ndarray]:
    start = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - start

    start = perf_counter()
    y_pred = model.predict(x_test)
    predict_seconds = perf_counter() - start
    y_pred = np.asarray(y_pred).reshape(-1)

    row: dict[str, float | str] = {
        "model": name,
        "target": target,
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "bands": int(x_train.shape[1]),
        "rmse": rmse(y_test, y_pred),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "rpd": rpd(y_test, y_pred),
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }
    return row, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full-spectrum CPU baselines for one GreenHyperSpectra trait.")
    parser.add_argument("--target", default="cab", choices=["cab", "cw", "cm", "LAI", "cp", "cbc", "car", "anth"])
    args = parser.parse_args()
    target = args.target

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    train, test, waves = load_split(target)

    x_train = train[waves].to_numpy(dtype=np.float32)
    y_train = train[target].to_numpy(dtype=np.float32)
    x_test = test[waves].to_numpy(dtype=np.float32)
    y_test = test[target].to_numpy(dtype=np.float32)

    models: list[tuple[str, object]] = [
        (
            "PLSR_20",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(n_components=20)),
                ]
            ),
        ),
        (
            "Ridge",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "ElasticNet",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "SVR_RBF",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(C=10.0, epsilon=0.1, gamma="scale")),
                ]
            ),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "MLP",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(256, 128),
                            activation="relu",
                            alpha=0.001,
                            batch_size=128,
                            learning_rate_init=0.001,
                            max_iter=500,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=25,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    ]

    if XGBRegressor is not None:
        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            )
        )

    rows: list[dict[str, float | str]] = []
    predictions = pd.DataFrame({"y_true": y_test})

    print(f"Target: {target}")
    print(f"Train rows: {len(y_train)} | Test rows: {len(y_test)} | Bands: {len(waves)}")

    for name, model in models:
        print(f"Training {name}...")
        row, y_pred = evaluate(target, name, model, x_train, y_train, x_test, y_test)
        rows.append(row)
        predictions[name] = y_pred
        print(
            f"{name}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, "
            f"R2={row['r2']:.4f}, RPD={row['rpd']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values("rmse")
    results.to_csv(RESULTS_DIR / f"{target}_cpu_baselines.csv", index=False)
    predictions.to_csv(RESULTS_DIR / f"{target}_cpu_predictions.csv", index=False)

    print("\nSaved:")
    print(RESULTS_DIR / f"{target}_cpu_baselines.csv")
    print(RESULTS_DIR / f"{target}_cpu_predictions.csv")
    print("\nResults:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
