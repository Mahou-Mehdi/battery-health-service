from dataclasses import asdict, replace
from pathlib import Path
import unittest
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np

from battery_health.api import create_app
from battery_health.data import load_csv, synthetic_readings, validate_readings, write_csv
from battery_health.features import make_examples, split_examples
from battery_health.model import fit_model, predict_history


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_readings(seed=11, batteries=6, cycles=75)
        cls.examples = make_examples(cls.rows)
        cls.splits, cls.metadata = split_examples(cls.examples, seed=11)
        cls.bundle = fit_model(cls.splits["train"], seed=11)

    def test_temporal_split_never_trains_on_future_test_targets(self):
        for battery_id, cutoff in self.metadata["train_battery_cutoffs"].items():
            train = [item for item in self.splits["train"] if item.battery_id == battery_id]
            test = [item for item in self.splits["temporal_test"] if item.battery_id == battery_id]
            self.assertLessEqual(max(item.target_cycle for item in train), cutoff)
            self.assertGreater(min(item.origin_cycle for item in test), cutoff)

    def test_unseen_batteries_are_entirely_excluded_from_training(self):
        train_ids = {item.battery_id for item in self.splits["train"]}
        test_ids = {item.battery_id for item in self.splits["unseen_battery_test"]}
        self.assertTrue(train_ids.isdisjoint(test_ids))

    def test_future_values_cannot_change_features_at_earlier_origins(self):
        modified = [replace(row, capacity_ah=row.capacity_ah * 0.8) if row.cycle > 30 else row
                    for row in self.rows]
        before = [item for item in self.examples if item.origin_cycle <= 30]
        after = [item for item in make_examples(modified) if item.origin_cycle <= 30]
        for first, second in zip(before, after):
            np.testing.assert_array_equal(first.features, second.features)

    def test_duplicate_cycles_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_readings(self.rows + self.rows[:1])

    def test_csv_roundtrip(self):
        # Ordinary file creation preserves inherited ACLs on Windows sandboxes.
        path = Path(__file__).resolve().parent / f"fixture-{uuid4().hex}.csv"
        try:
            write_csv(self.rows, path)
            self.assertEqual(load_csv(path), self.rows)
        finally:
            path.unlink(missing_ok=True)

    def test_api_forecast_and_history_validation(self):
        history = self.rows[:12]
        request = {"readings": [asdict(row) for row in history]}
        with TestClient(create_app(self.bundle)) as client:
            self.assertTrue(client.get("/health").json()["model_ready"])
            result = client.post("/predict", json=request)
            self.assertEqual(result.status_code, 200)
            expected = predict_history(self.bundle, history)
            self.assertAlmostEqual(result.json()["predicted_soh"], expected["predicted_soh"])
            self.assertEqual(result.json()["target_cycle"], 21)
            request["readings"][-1]["cycle"] = 13
            self.assertEqual(client.post("/predict", json=request).status_code, 422)

    def test_synthetic_data_reproducible(self):
        self.assertEqual(self.rows, synthetic_readings(seed=11, batteries=6, cycles=75))


if __name__ == "__main__":
    unittest.main()
