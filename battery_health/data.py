"""Validated CSV input and an explicitly synthetic capacity-degradation demo."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Reading:
    battery_id: str
    cycle: int
    capacity_ah: float
    nominal_capacity_ah: float
    temperature_c: float
    resistance_mohm: float
    charge_rate_c: float


def validate_readings(readings: list[Reading]) -> dict[str, list[Reading]]:
    groups: dict[str, list[Reading]] = {}
    seen = set()
    for row in readings:
        if not row.battery_id.strip() or row.cycle < 0 or int(row.cycle) != row.cycle:
            raise ValueError("battery_id must be nonempty and cycle a nonnegative integer")
        values = [row.capacity_ah, row.nominal_capacity_ah, row.temperature_c,
                  row.resistance_mohm, row.charge_rate_c]
        if not np.isfinite(values).all():
            raise ValueError("All measurements must be finite")
        if min(row.capacity_ah, row.nominal_capacity_ah, row.resistance_mohm, row.charge_rate_c) <= 0:
            raise ValueError("Capacities, resistance, and charge rate must be positive")
        if row.capacity_ah > row.nominal_capacity_ah * 1.2:
            raise ValueError("capacity_ah exceeds 120% of nominal capacity; check units")
        if (row.battery_id, row.cycle) in seen:
            raise ValueError("Duplicate battery_id/cycle pair")
        seen.add((row.battery_id, row.cycle))
        groups.setdefault(row.battery_id, []).append(row)
    for battery_id, rows in groups.items():
        rows.sort(key=lambda item: item.cycle)
        if len({row.nominal_capacity_ah for row in rows}) != 1:
            raise ValueError(f"Nominal capacity changes within battery {battery_id}")
    return groups


def load_csv(path: Path) -> list[Reading]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(Reading.__dataclass_fields__)
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV needs columns: " + ", ".join(sorted(required)))
        for line, row in enumerate(reader, start=2):
            try:
                rows.append(Reading(battery_id=row["battery_id"], cycle=int(row["cycle"]),
                                    **{key: float(row[key]) for key in required - {"battery_id", "cycle"}}))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid measurement at CSV line {line}: {exc}") from exc
    validate_readings(rows)
    return rows


def write_csv(readings: list[Reading], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Reading.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in readings)


def synthetic_readings(seed: int = 42, batteries: int = 30, cycles: int = 180) -> list[Reading]:
    """Toy sensor traces, NOT a validated electrochemical or real-fleet model."""
    if batteries < 5 or cycles < 60:
        raise ValueError("Synthetic demo needs at least 5 batteries and 60 cycles")
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(batteries):
        nominal = float(rng.uniform(2.5, 3.5))
        base_fade = rng.uniform(0.00035, 0.0008)
        knee = rng.uniform(65, 120)
        acceleration = rng.uniform(0.000010, 0.000020)
        ambient, charge = rng.uniform(20, 34), rng.uniform(0.6, 1.6)
        base_resistance = rng.uniform(35, 55)
        true_soh = rng.uniform(0.99, 1.01)
        for cycle in range(cycles):
            temp = ambient + 2.5 * np.sin(cycle / 17) + rng.normal(0, 0.6)
            rate = charge + 0.08 * np.sin(cycle / 11) + rng.normal(0, 0.015)
            degradation = base_fade * (1 + 0.035 * (temp - 25)) * (0.7 + 0.3 * rate)
            degradation += max(cycle - knee, 0) * acceleration
            true_soh = max(0.30, true_soh - degradation)
            capacity = nominal * (true_soh + rng.normal(0, 0.0015))
            resistance = base_resistance + 90 * (1 - true_soh) + rng.normal(0, 0.2)
            rows.append(Reading(f"SYN-{idx:03d}", cycle, float(capacity), nominal,
                                float(temp), float(resistance), float(rate)))
    return rows
