"""Small gradient-boosting forecaster and transparent baselines."""

from datetime import datetime, timezone

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from threadpoolctl import threadpool_limits

from .features import Example, FEATURE_NAMES, history_features


def arrays(examples: list[Example]):
    return np.vstack([item.features for item in examples]), np.array([item.target_soh for item in examples])


def fit_model(examples: list[Example], horizon=10, window=12, seed=42, source="synthetic_demo"):
    x, y = arrays(examples)
    # Predict future change, rather than ask a tree to extrapolate absolute capacity.
    estimator = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, learning_rate=0.06,
                                             l2_regularization=1.0, early_stopping=False, random_state=seed)
    with threadpool_limits(limits=1):
        estimator.fit(x, y - x[:, 1])
    return {"estimator": estimator, "metadata": {
        "format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source, "horizon_cycles": horizon, "window_cycles": window,
        "features": FEATURE_NAMES, "training_examples": len(examples), "seed": seed,
        "sklearn_version": sklearn.__version__, "target": "future_capacity_ah / nominal_capacity_ah",
    }}


def predict_array(bundle, x):
    with threadpool_limits(limits=1):
        return x[:, 1] + bundle["estimator"].predict(x)


def predict_history(bundle, history):
    metadata = bundle["metadata"]
    x = history_features(history, metadata["window_cycles"])[None, :]
    prediction = float(predict_array(bundle, x)[0])
    latest = max(history, key=lambda row: row.cycle)
    return {
        "battery_id": latest.battery_id, "origin_cycle": latest.cycle,
        "target_cycle": latest.cycle + metadata["horizon_cycles"],
        "predicted_soh": prediction,
        "predicted_capacity_ah": prediction * latest.nominal_capacity_ah,
        "persistence_soh": float(x[0, 1]),
        "linear_trend_soh": float(x[0, 1] + metadata["horizon_cycles"] * x[0, 2]),
        "model_source": metadata["source"],
        "note": "Experimental forecast; synthetic demo models are not validated on real batteries.",
    }


def evaluate(bundle, examples):
    x, y = arrays(examples)
    predictions = {
        "persistence": x[:, 1],
        "linear_trend": x[:, 1] + bundle["metadata"]["horizon_cycles"] * x[:, 2],
        "gradient_boosting": predict_array(bundle, x),
    }
    metrics = {}
    for name, predicted in predictions.items():
        per_battery = {}
        for battery_id in sorted({item.battery_id for item in examples}):
            mask = np.array([item.battery_id == battery_id for item in examples])
            per_battery[battery_id] = float(mean_absolute_error(y[mask], predicted[mask]))
        metrics[name] = {
            "mae_soh": float(mean_absolute_error(y, predicted)),
            "rmse_soh": float(np.sqrt(mean_squared_error(y, predicted))),
            "mean_error_soh": float(np.mean(predicted - y)),
            "macro_battery_mae_soh": float(np.mean(list(per_battery.values()))),
            "per_battery_mae_soh": per_battery,
        }
    return metrics, predictions
