"""Check saved experiment metrics, predictions, and file hashes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    assert completion["status"] == "completed" and len(completion["states"]) == 29
    assert all(value == "completed" for value in completion["states"].values())
    train_split, val_split = read_csv(root / "training_split.csv"), read_csv(root / "validation_split.csv")
    train_indices = {int(row["source_index"]) for row in train_split}
    val_indices = [int(row["source_index"]) for row in val_split]
    assert len(train_indices) == 5894 and len(val_indices) == 1474
    assert not train_indices.intersection(val_indices)
    assert train_indices.union(val_indices) == set(range(7368))
    mlp_runs, epoch_rows, prediction_rows = 0, 0, 0
    for run_id in completion["states"]:
        directory = root / "runs" / run_id
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        checks = json.loads((directory / "file_checksums.json").read_text(encoding="utf-8"))
        for name, digest in checks["public"].items():
            assert Path(name).name == name
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, (run_id, name)
        prediction = read_csv(directory / "validation_predictions.csv")
        assert len(prediction) == 1474
        assert [int(row["source_index"]) for row in prediction] == val_indices
        correct = [int(row["true_label"]) == int(row["predicted_label"]) for row in prediction]
        assert all((row["is_correct"].lower() == "true") == value for row, value in zip(prediction, correct))
        assert math.isclose(sum(correct) / len(correct), summary["val_accuracy"], rel_tol=0, abs_tol=1e-12)
        assert len(correct) - sum(correct) == summary["wrong_count"]
        prediction_rows += len(prediction)
        if config["model"] == "MLP":
            history = read_csv(directory / "history.csv")
            assert [int(row["epoch"]) for row in history] == list(range(1, config["epochs"] + 1))
            for field in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
                assert all(math.isfinite(float(row[field])) for row in history)
                assert math.isclose(float(history[-1][field]), summary[field], rel_tol=0, abs_tol=1e-12)
            assert (directory / "curves.png").is_file()
            mlp_runs += 1
            epoch_rows += len(history)
    assert (mlp_runs, epoch_rows, prediction_rows) == (17, 1430, 42746)
    print(json.dumps({"configurations": 29, "MLP_runs": mlp_runs, "epoch_rows": epoch_rows,
                      "validation_prediction_rows": prediction_rows, "numeric_checks": "passed"}))


if __name__ == "__main__":
    main()
