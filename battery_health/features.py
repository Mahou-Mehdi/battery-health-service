"""Every feature is calculated at or before the forecast origin."""

from dataclasses import dataclass

import numpy as np

from .data import Reading, validate_readings

FEATURE_NAMES = ["cycle", "current_soh", "soh_slope", "soh_std", "mean_temperature_c",
                 "resistance_mohm", "resistance_slope", "mean_charge_rate_c"]


@dataclass
class Example:
    battery_id: str
    origin_cycle: int
    target_cycle: int
    features: np.ndarray
    target_soh: float


def history_features(history: list[Reading], window: int = 12) -> np.ndarray:
    groups = validate_readings(history)
    if len(groups) != 1 or len(history) < window:
        raise ValueError(f"Provide at least {window} readings from one battery")
    rows = next(iter(groups.values()))[-window:]
    cycles = np.array([row.cycle for row in rows], dtype=float)
    if not np.all(np.diff(cycles) == 1):
        raise ValueError("The last history window must contain consecutive cycle numbers")
    soh = np.array([row.capacity_ah / row.nominal_capacity_ah for row in rows])
    resistance = np.array([row.resistance_mohm for row in rows])
    centered = cycles - cycles.mean()
    slope = float(centered @ (soh - soh.mean()) / (centered @ centered))
    resistance_slope = float(centered @ (resistance - resistance.mean()) / (centered @ centered))
    return np.array([cycles[-1], soh[-1], slope, soh.std(),
                     np.mean([row.temperature_c for row in rows]), resistance[-1],
                     resistance_slope, np.mean([row.charge_rate_c for row in rows])])


def make_examples(readings: list[Reading], horizon: int = 10, window: int = 12) -> list[Example]:
    if horizon < 1 or window < 2:
        raise ValueError("horizon must be positive and window at least 2")
    examples = []
    for battery_id, rows in validate_readings(readings).items():
        by_cycle = {row.cycle: row for row in rows}
        for index in range(window - 1, len(rows)):
            origin = rows[index].cycle
            future = by_cycle.get(origin + horizon)
            if future is None:
                continue
            history = rows[index - window + 1:index + 1]
            # Gaps are excluded rather than silently changing the forecasting horizon.
            if any(b.cycle - a.cycle != 1 for a, b in zip(history, history[1:])):
                continue
            features = history_features(history, window)
            examples.append(Example(battery_id, origin, future.cycle, features,
                                    future.capacity_ah / future.nominal_capacity_ah))
    return examples


def split_examples(examples: list[Example], seed: int = 42,
                   temporal_fraction: float = 0.7) -> tuple[dict[str, list[Example]], dict]:
    """Battery holdout plus a horizon-aware chronological split for seen batteries.

    No training target is later than its battery cutoff. Temporal test origins
    are strictly after that cutoff. Windows crossing the boundary are dropped.
    Past context is allowed for a new forecast; future targets never are.
    """
    battery_ids = sorted({example.battery_id for example in examples})
    if len(battery_ids) < 5:
        raise ValueError("At least 5 batteries with valid forecasting windows are required")
    if not 0.3 <= temporal_fraction <= 0.85:
        raise ValueError("temporal_fraction must be between 0.3 and 0.85")
    rng = np.random.default_rng(seed)
    unseen = set(rng.choice(battery_ids, size=max(1, round(len(battery_ids) * 0.2)), replace=False))
    splits = {"train": [], "temporal_test": [], "unseen_battery_test": []}
    cutoffs = {}
    for battery_id in battery_ids:
        rows = [item for item in examples if item.battery_id == battery_id]
        if battery_id in unseen:
            splits["unseen_battery_test"].extend(rows)
            continue
        first = min(item.origin_cycle for item in rows)
        last = max(item.target_cycle for item in rows)
        cutoff = int(first + (last - first) * temporal_fraction)
        cutoffs[battery_id] = cutoff
        splits["train"].extend(item for item in rows if item.target_cycle <= cutoff)
        splits["temporal_test"].extend(item for item in rows if item.origin_cycle > cutoff)
    if any(not values for values in splits.values()):
        raise ValueError("Not enough cycles for nonempty train and evaluation sets")
    metadata = {"seed": seed, "unseen_battery_ids": sorted(unseen), "train_battery_cutoffs": cutoffs,
                "counts": {key: len(value) for key, value in splits.items()},
                "boundary_windows_discarded": len(examples) - sum(map(len, splits.values()))}
    return splits, metadata
