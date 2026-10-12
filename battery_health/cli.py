"""Train, evaluate, and forecast from the terminal."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import sklearn

from .data import Reading, load_csv, synthetic_readings, validate_readings, write_csv
from .features import make_examples, split_examples
from .model import evaluate, fit_model, predict_history


def train(csv_path, output, report, source="user_csv", seed=42, horizon=10, window=12):
    readings = load_csv(csv_path)
    examples = make_examples(readings, horizon, window)
    splits, split_metadata = split_examples(examples, seed)
    start = perf_counter()
    bundle = fit_model(splits["train"], horizon, window, seed, source)
    elapsed = perf_counter() - start
    bundle["metadata"]["input_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    bundle["metadata"]["split"] = split_metadata
    result = {"model": bundle["metadata"], "fit_seconds": elapsed,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "scikit_learn": sklearn.__version__},
              "evaluation": {}}
    for name in ("temporal_test", "unseen_battery_test"):
        metrics, _ = evaluate(bundle, splits[name])
        result["evaluation"][name] = metrics
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "model.joblib")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(result, report.with_suffix(".md"))
    print(report.with_suffix(".md").read_text(encoding="utf-8"))
    print(f"Model saved to {output / 'model.joblib'}")
    return bundle, readings


def write_report(result, path):
    lines = ["# Evaluation results", "", f"Data source: **{result['model']['source']}**.", "",
             "SOH = measured capacity / nominal capacity. MAE and RMSE below are SOH percentage points.",
             "For example, 0.3 percentage points means an absolute SOH error of 0.003.", "",
             "| Evaluation | Method | MAE (pp) | RMSE (pp) | Macro battery MAE (pp) |",
             "|---|---|---:|---:|---:|"]
    for split, methods in result["evaluation"].items():
        for name, metrics in methods.items():
            lines.append(f"| {split} | {name} | {100 * metrics['mae_soh']:.3f} | "
                         f"{100 * metrics['rmse_soh']:.3f} | {100 * metrics['macro_battery_mae_soh']:.3f} |")
    lines += ["", "Temporal test: later forecast origins on batteries represented in training.",
              "Unseen-battery test: all windows from batteries excluded entirely from training.",
              "The held-out metrics are evaluation only; hyperparameters are fixed in code.",
              "Synthetic demo results establish software behavior, not real-world battery accuracy."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "train"):
        child = sub.add_parser(command)
        child.add_argument("--output", type=Path, default=Path("artifacts"))
        child.add_argument("--report", type=Path, default=Path("results/demo-metrics.json" if command == "demo" else "results/csv-metrics.json"))
        child.add_argument("--seed", type=int, default=42)
        child.add_argument("--horizon", type=int, default=10)
        child.add_argument("--window", type=int, default=12)
        if command == "train":
            child.add_argument("--csv", type=Path, required=True)
        else:
            child.add_argument("--batteries", type=int, default=30)
            child.add_argument("--cycles", type=int, default=180)
    generate = sub.add_parser("generate")
    generate.add_argument("--csv", type=Path, default=Path("artifacts/synthetic-batteries.csv"))
    generate.add_argument("--seed", type=int, default=42)
    predict = sub.add_parser("predict")
    predict.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    predict.add_argument("--history", type=Path, default=Path("examples/predict-request.json"))
    args = parser.parse_args()
    if args.command == "generate":
        write_csv(synthetic_readings(args.seed), args.csv)
        print(f"Synthetic data written to {args.csv}")
    elif args.command in ("demo", "train"):
        if args.command == "demo":
            csv_path = args.output / "synthetic-batteries.csv"
            write_csv(synthetic_readings(args.seed, args.batteries, args.cycles), csv_path)
        else:
            csv_path = args.csv
        bundle, readings = train(csv_path, args.output, args.report,
                                 "synthetic_demo" if args.command == "demo" else "user_csv",
                                 args.seed, args.horizon, args.window)
        if args.command == "demo":
            example_id = bundle["metadata"]["split"]["unseen_battery_ids"][0]
            history = validate_readings(readings)[example_id][:max(args.window, 25)]
            example_dir = Path("examples")
            example_dir.mkdir(exist_ok=True)
            (example_dir / "predict-request.json").write_text(
                json.dumps({"readings": [asdict(row) for row in history]}, indent=2) + "\n", encoding="utf-8")
            (example_dir / "predict-response.json").write_text(
                json.dumps(predict_history(bundle, history), indent=2) + "\n", encoding="utf-8")
    elif args.command == "predict":
        # joblib uses pickle: only load locally created or otherwise trusted artifacts.
        bundle = joblib.load(args.model)
        data = json.loads(args.history.read_text(encoding="utf-8"))
        history = [Reading(**row) for row in data["readings"]]
        print(json.dumps(predict_history(bundle, history), indent=2))


if __name__ == "__main__":
    main()
