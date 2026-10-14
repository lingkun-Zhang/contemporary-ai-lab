"""Fit the selected configuration on all labeled data; preserve test row order."""
import argparse
import csv
import hashlib
import json
import platform
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits


CONFIG = {
    "model": "LogisticRegression", "selection_run_id": "lr_vocab10000",
    "max_features": 10000, "ngram_range": [1, 1], "sublinear_tf": False,
    "C": 100.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 1000,
    "tol": 1e-4, "random_state": 42, "class_weight": None,
    "training_threads": 2,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_submission_csv(path, labels):
    # The teacher's example uses one label per row, with no header or index.
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows([[int(label)] for label in labels])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "final_outputs")
    parser.add_argument("--model-output", type=Path, help="optional private local model path; do not upload it")
    parser.add_argument("--json-only", action="store_true", help="retain labels for an external exact-format CSV export")
    args = parser.parse_args()
    train_file = args.data_dir / "train_data.csv"
    test_file = args.data_dir / "test_data_unlabeled.csv"
    input_hashes = {"train_data.csv": sha256(train_file), "test_data_unlabeled.csv": sha256(test_file)}
    train, test = pd.read_csv(train_file), pd.read_csv(test_file)
    if not {"text", "target"}.issubset(train.columns) or "text" not in test.columns:
        raise ValueError("Expected text,target columns in train data and text in test data.")
    if len(train) != 7368 or len(test) != 2457:
        raise ValueError("This lab expects 7368 labeled and 2457 unlabeled rows.")
    if train[["text", "target"]].isna().any().any() or test["text"].isna().any():
        raise ValueError("Missing text or target data; do not silently change sample count.")
    labels = train["target"].to_numpy()
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("Expected integer class labels 0 through 9.")
    outputs = ["final_training.json", "test_predictions.json", "prediction.csv"]
    if any((args.output / name).exists() for name in outputs):
        raise FileExistsError("Choose a new output directory to preserve existing predictions.")
    if args.model_output and args.model_output.exists():
        raise FileExistsError("Model output already exists; choose a new path.")
    train_texts = train["text"].astype(str).tolist()
    test_texts = test["text"].astype(str).tolist()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=False)),
        ("classifier", LogisticRegression(C=100, penalty="l2", solver="lbfgs", max_iter=1000,
                                           tol=1e-4, random_state=42, class_weight=None)),
    ])
    print("Final fit: all 7368 labeled texts; TF-IDF cap=10000; LR C=100; no class weighting.", flush=True)
    started_at = datetime.now(timezone.utc).isoformat()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter("always")
        start = perf_counter()
        pipeline.fit(train_texts, labels)
        training_seconds = perf_counter() - start
        predictions = pipeline.predict(test_texts).astype(int)
        probabilities = pipeline.predict_proba(test_texts)
        fit_predictions = pipeline.predict(train_texts)
        fit_probabilities = pipeline.predict_proba(train_texts)
    warning_records = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("Final fit did not converge; predictions were not exported.")
    if len(predictions) != len(test) or not np.isin(predictions, np.arange(10)).all():
        raise ValueError("Invalid prediction row count or labels.")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid predicted probabilities.")
    if input_hashes != {"train_data.csv": sha256(train_file), "test_data_unlabeled.csv": sha256(test_file)}:
        raise RuntimeError("Input files changed during training.")
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_records = {
        "row_order": "Original test_data_unlabeled.csv order, zero-based, no sorting or filtering.",
        "predictions": predictions.tolist(),
        "classes": pipeline.classes_.astype(int).tolist(),
        "probabilities": probabilities.tolist(),
    }
    save_json(args.output / "test_predictions.json", prediction_records)
    if not args.json_only:
        write_submission_csv(args.output / "prediction.csv", predictions)
    if args.model_output:
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "configuration": CONFIG, "input_hashes": input_hashes},
                    args.model_output, compress=3)
    training_record = {
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "configuration": CONFIG, "input_sha256": input_hashes,
        "script_sha256": sha256(__file__), "training_rows": len(train), "test_rows": len(test),
        "all_labeled_rows_used": True, "test_used_for_fit": False,
        "actual_vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "optimizer_iterations": pipeline.named_steps["classifier"].n_iter_.astype(int).tolist(),
        "training_seconds": training_seconds,
        "full_fit_training_accuracy": float(accuracy_score(labels, fit_predictions)),
        "full_fit_training_cross_entropy": float(log_loss(labels, fit_probabilities, labels=pipeline.classes_)),
        "prediction_label_counts": {str(k): v for k, v in sorted(Counter(predictions.tolist()).items())},
        "prediction_order_sha256": hashlib.sha256(predictions.astype("<i8").tobytes()).hexdigest(),
        "test_text_order_sha256": hashlib.sha256(json.dumps(test_texts, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "warnings": warning_records, "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "test_accuracy": None, "test_accuracy_note": "The official test set has no labels; no test accuracy is claimed.",
        "model_saved_locally": bool(args.model_output),
    }
    save_json(args.output / "final_training.json", training_record)
    print(json.dumps({"test_rows": len(test), "vocabulary": training_record["actual_vocabulary_size"],
                      "iterations": training_record["optimizer_iterations"], "warnings": warning_records,
                      "label_counts": training_record["prediction_label_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
