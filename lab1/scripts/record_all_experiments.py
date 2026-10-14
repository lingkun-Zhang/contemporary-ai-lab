"""Run the 29 text-classification configurations and save their results."""

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import traceback
import warnings
from contextlib import contextmanager
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_info, threadpool_limits


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def package_versions():
    return {name: version(name) for name in
            ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl")}


@contextmanager
def recorded_warnings(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            yield caught
        finally:
            save_json(path, [{"category": item.category.__name__, "message": str(item.message)} for item in caught])


def plan():
    runs = []

    def add(run_id, model, **options):
        runs.append({"run_id": run_id, "model": model, "max_features": 5000,
                     "ngram_range": [1, 1], "sublinear_tf": False, "seed": 42,
                     "weight_class": None, "relative_weight": 1.0, **options})

    for c in (.1, 1, 10, 100):
        add(f"lr_c{c:g}", "LR", C=c)
    for factor in (1.5, 2.0):
        add(f"lr_class7_w{factor:g}", "LR", C=100, weight_class=7, relative_weight=factor)
    add("lr_vocab10000", "LR", C=100, max_features=10000)

    def mlp(run_id, **changes):
        add(run_id, "MLP", **{"hidden_units": 50, "learning_rate_init": .0001,
                            "alpha": .005, "batch_size": 64, "epochs": 100, **changes})

    mlp("mlp_lr001_e30", learning_rate_init=.001, alpha=.001, epochs=30)
    mlp("mlp_lr0001_e30", alpha=.001, epochs=30)
    mlp("mlp_alpha001_e60", alpha=.001, epochs=60)
    mlp("mlp_alpha005_e50", epochs=50)
    mlp("mlp_alpha005_e60", epochs=60)
    mlp("mlp_width50_s42")
    mlp("mlp_alpha05_e100", alpha=.05)
    mlp("mlp_width25_s42", hidden_units=25)
    mlp("mlp_width100_s42", hidden_units=100)
    for seed in (43, 44):
        for width in (50, 100):
            mlp(f"mlp_width{width}_s{seed}", hidden_units=width, seed=seed)
    mlp("mlp_class2_w1.5", weight_class=2, relative_weight=1.5)
    mlp("mlp_bigram", hidden_units=100, ngram_range=[1, 2])
    mlp("mlp_logtf", hidden_units=100, sublinear_tf=True)
    mlp("mlp_vocab10000", hidden_units=100, max_features=10000)

    for c in (.1, 1, 10, 100):
        add(f"svm_c{c:g}", "SVM", C=c)
    add("svm_vocab10000", "SVM", C=100, max_features=10000)
    return runs


COMPARISONS = {
    "LR_C": ["lr_c0.1", "lr_c1", "lr_c10", "lr_c100"],
    "LR_weight": ["lr_c100", "lr_class7_w1.5", "lr_class7_w2"],
    "MLP_learning_rate": ["mlp_lr001_e30", "mlp_lr0001_e30"],
    "MLP_epochs_alpha001": ["mlp_lr0001_e30", "mlp_alpha001_e60"],
    "MLP_epochs_alpha005": ["mlp_alpha005_e50", "mlp_alpha005_e60", "mlp_width50_s42"],
    "MLP_regularization": ["mlp_width50_s42", "mlp_alpha05_e100"],
    "MLP_width": ["mlp_width25_s42", "mlp_width50_s42", "mlp_width100_s42"],
    "MLP_seed42": ["mlp_width50_s42", "mlp_width100_s42"],
    "MLP_seed43": ["mlp_width50_s43", "mlp_width100_s43"],
    "MLP_seed44": ["mlp_width50_s44", "mlp_width100_s44"],
    "MLP_weight": ["mlp_width50_s42", "mlp_class2_w1.5"],
    "MLP_ngram": ["mlp_width100_s42", "mlp_bigram"],
    "MLP_tf_scaling": ["mlp_width100_s42", "mlp_logtf"],
    "SVM_C": ["svm_c0.1", "svm_c1", "svm_c10", "svm_c100"],
    "LR_vocabulary": ["lr_c100", "lr_vocab10000"],
    "MLP_vocabulary": ["mlp_width100_s42", "mlp_vocab10000"],
    "SVM_vocabulary": ["svm_c100", "svm_vocab10000"],
}


def sample_weights(labels, config):
    if config["weight_class"] is None or config["relative_weight"] == 1:
        return None
    weights = np.where(labels == config["weight_class"], config["relative_weight"], 1.0)
    return weights / weights.mean()


def make_model(config):
    if config["model"] == "LR":
        return LogisticRegression(C=config["C"], penalty="l2", solver="lbfgs", max_iter=1000,
                                  tol=1e-4, random_state=config["seed"], class_weight=None)
    if config["model"] == "SVM":
        return SVC(C=config["C"], kernel="linear", probability=False, tol=.001,
                   max_iter=-1, cache_size=512, class_weight=None,
                   decision_function_shape="ovr", break_ties=False, random_state=config["seed"])
    return MLPClassifier(hidden_layer_sizes=(config["hidden_units"],), activation="relu",
                         solver="adam", learning_rate_init=config["learning_rate_init"],
                         alpha=config["alpha"], batch_size=config["batch_size"],
                         early_stopping=False, random_state=np.random.RandomState(config["seed"]))


def json_parameters(model):
    result = model.get_params()
    if isinstance(result.get("random_state"), np.random.RandomState):
        result["random_state"] = "persistent numpy.random.RandomState(seed in run_config)"
    if isinstance(result.get("hidden_layer_sizes"), tuple):
        result["hidden_layer_sizes"] = list(result["hidden_layer_sizes"])
    return result


def evaluate(model, X_fit, X_val, y_fit, y_val):
    train_pred, val_pred = model.predict(X_fit), model.predict(X_val)
    row = {"train_accuracy": float(accuracy_score(y_fit, train_pred)),
           "val_accuracy": float(accuracy_score(y_val, val_pred)),
           "wrong_count": int(np.sum(val_pred != y_val))}
    val_proba = None
    if not isinstance(model, SVC):
        train_proba, val_proba = model.predict_proba(X_fit), model.predict_proba(X_val)
        for probabilities in (train_proba, val_proba):
            if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
                raise ValueError("Invalid predicted probabilities")
        row.update(train_loss=float(log_loss(y_fit, train_proba, labels=model.classes_)),
                   val_loss=float(log_loss(y_val, val_proba, labels=model.classes_)))
    return row, train_pred, val_pred, val_proba


def save_predictions(directory, indices, truth, prediction, probabilities, classes):
    frame = pd.DataFrame({"source_index": indices, "true_label": truth,
                          "predicted_label": prediction, "is_correct": truth == prediction})
    if probabilities is not None:
        for column, label in enumerate(classes):
            frame[f"probability_class_{label}"] = probabilities[:, column]
    frame.to_csv(directory / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    frame.loc[~frame.is_correct].to_csv(directory / "wrong_predictions.csv", index=False, encoding="utf-8-sig")
    save_json(directory / "classification_report.json",
              classification_report(truth, prediction, output_dict=True, zero_division=0))
    pd.DataFrame(confusion_matrix(truth, prediction, labels=classes), index=classes, columns=classes).to_csv(
        directory / "confusion_matrix.csv", encoding="utf-8-sig")


def plot_history(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    history = pd.read_csv(directory / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for phase, color in (("train", "#2166AC"), ("val", "#D6604D")):
        for ax, field in zip(axes, ("loss", "accuracy")):
            ax.plot(history.epoch, history[f"{phase}_{field}"], color=color, marker=".",
                    markersize=3, linewidth=1, label=phase)
    axes[0].set(ylabel="Mean cross-entropy (no L2 term)")
    axes[1].set(ylabel="Accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set(xlabel="Epoch", xlim=(1, max(2, history.epoch.max())))
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle(directory.name)
    fig.savefig(directory / "curves.png", dpi=200)
    plt.close(fig)


def feature_set(data, config, cache, public_root):
    key = (config["max_features"], tuple(config["ngram_range"]), config["sublinear_tf"])
    if key not in cache:
        vectorizer = TfidfVectorizer(max_features=key[0], ngram_range=key[1], sublinear_tf=key[2])
        X_fit = vectorizer.fit_transform(data["fit_texts"])
        X_val = vectorizer.transform(data["val_texts"])
        names = vectorizer.get_feature_names_out()
        metadata = {"actual_feature_count": int(X_fit.shape[1]),
                    "word_features": int(sum(" " not in word for word in names)),
                    "bigram_features": int(sum(" " in word for word in names)),
                    "vocabulary_sha256": hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest(),
                    "idf_sha256": array_sha(vectorizer.idf_, "<f8"),
                    "fit_nnz": int(X_fit.nnz), "val_nnz": int(X_val.nnz)}
        parameters = vectorizer.get_params()
        parameters["dtype"] = np.dtype(parameters["dtype"]).name
        metadata["tfidf_parameters"] = parameters
        cache[key] = (vectorizer, X_fit, X_val, metadata)
        save_json(public_root / f"features_{key[0]}_{key[1][1]}_{int(key[2])}.json", metadata)
    return cache[key]


def run_one(config, data, cached_features, public_root, private_root):
    directory = public_root / "runs" / config["run_id"]
    directory.mkdir(parents=True, exist_ok=False)
    save_json(directory / "status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    save_json(directory / "config.json", config)
    vectorizer, X_fit, X_val, feature_metadata = feature_set(data, config, cached_features, public_root)
    model = make_model(config)
    save_json(directory / "model_parameters.json", json_parameters(model))
    save_json(directory / "feature_metadata.json", feature_metadata)
    y_fit, y_val = data["y_fit"], data["y_val"]
    weights = sample_weights(y_fit, config)
    if weights is not None:
        pd.DataFrame({"source_index": data["fit_indices"], "weight": weights}).to_csv(
            directory / "training_weights.csv", index=False, encoding="utf-8-sig")
    training_seconds = 0.0
    warning_records = []
    with (directory / "console.log").open("w", encoding="utf-8") as console:
        def report(message):
            print(message, flush=True)
            console.write(message + "\n")
            console.flush()
        report("START " + config["run_id"] + " " + json.dumps(config, ensure_ascii=False))
        with threadpool_limits(limits=2), recorded_warnings(directory / "warnings.json") as caught:
            if config["model"] == "MLP":
                with (directory / "history.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                    fields = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy",
                              "wrong_count", "training_seconds", "optimizer_reported_loss", "samples_seen"]
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    for epoch in range(1, config["epochs"] + 1):
                        started = perf_counter()
                        model.partial_fit(X_fit, y_fit, classes=np.unique(y_fit), sample_weight=weights)
                        training_seconds += perf_counter() - started
                        row, train_pred, val_pred, val_proba = evaluate(model, X_fit, X_val, y_fit, y_val)
                        row.update(epoch=epoch, training_seconds=training_seconds,
                                   optimizer_reported_loss=float(model.loss_), samples_seen=int(model.t_))
                        writer.writerow(row)
                        stream.flush()
                        report(f"{config['run_id']} epoch={epoch} trainCE={row['train_loss']:.8f} "
                               f"valCE={row['val_loss']:.8f} trainAcc={row['train_accuracy']:.8f} "
                               f"valAcc={row['val_accuracy']:.8f}")
            else:
                started = perf_counter()
                if weights is None:
                    model.fit(X_fit, y_fit)
                else:
                    model.fit(X_fit, y_fit, sample_weight=weights)
                training_seconds = perf_counter() - started
                row, train_pred, val_pred, val_proba = evaluate(model, X_fit, X_val, y_fit, y_val)
            warning_records = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
        summary = {**config, **row, **{key: value for key, value in feature_metadata.items() if key != "tfidf_parameters"},
                   "training_seconds": training_seconds, "accuracy_gap": row["train_accuracy"] - row["val_accuracy"],
                   "result_origin": "fresh_reproduction", "evaluation": "final configured epoch or completed fit"}
        if config["model"] == "MLP":
            summary["parameter_count"] = int(sum(a.size for a in model.coefs_ + model.intercepts_))
        elif config["model"] == "LR":
            summary.update(parameter_count=int(model.coef_.size + model.intercept_.size),
                           optimizer_iterations=int(model.n_iter_.max()),
                           converged=not any(issubclass(item.category, ConvergenceWarning) for item in caught))
        else:
            summary.update(support_vectors=int(model.support_.size), fit_status=int(model.fit_status_),
                           solver_iterations_per_pair=model.n_iter_.tolist())
        save_json(directory / "warnings.json", warning_records)
        save_json(directory / "summary.json", summary)
        save_predictions(directory, data["val_indices"], y_val, val_pred, val_proba, model.classes_)
        pd.DataFrame({"source_index": data["fit_indices"], "true_label": y_fit,
                      "predicted_label": train_pred}).to_csv(directory / "training_predictions.csv", index=False,
                                                            encoding="utf-8-sig")
        local_dir = private_root / config["run_id"]
        local_dir.mkdir(parents=True, exist_ok=False)
        local = pd.read_csv(directory / "validation_predictions.csv")
        local["text"] = data["val_texts"]
        local.to_csv(local_dir / "validation_predictions_with_text.csv", index=False, encoding="utf-8-sig")
        joblib.dump({"model": model, "vectorizer": vectorizer, "config": config}, local_dir / "model.joblib", compress=3)
        if config["model"] == "MLP":
            try:
                plot_history(directory)
            except Exception:
                (directory / "plot_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                report("PLOT FAILED; raw training data remain saved")
        save_json(directory / "status.json", {"status": "completed", "ended_at": datetime.now(timezone.utc).isoformat()})
        report("DONE " + config["run_id"] + " " + json.dumps(summary, ensure_ascii=False))
    save_json(directory / "file_checksums.json", {
        "public": {path.name: sha256(path) for path in directory.iterdir() if path.is_file()},
        "private": {path.name: sha256(path) for path in local_dir.iterdir() if path.is_file()},
    })
    return summary


def verify_completed(directory, local_dir, config, data):
    checksums = json.loads((directory / "file_checksums.json").read_text(encoding="utf-8"))
    for category, root in (("public", directory), ("private", local_dir)):
        for name, digest in checksums[category].items():
            if Path(name).name != name or sha256(root / name) != digest:
                raise ValueError("Resume rejected: saved result missing or changed")
    saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if saved != config:
        raise ValueError("Resume rejected: run configuration mismatch")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    records = pd.read_csv(directory / "validation_predictions.csv")
    if (not np.array_equal(records.source_index, data["val_indices"])
            or not np.array_equal(records.true_label, data["y_val"])
            or not np.array_equal(records.is_correct, records.true_label == records.predicted_label)
            or not np.isclose(records.is_correct.mean(), summary["val_accuracy"], rtol=0, atol=1e-12)):
        raise ValueError("Resume rejected: prediction metrics or sample order mismatch")
    if config["model"] == "MLP":
        history = pd.read_csv(directory / "history.csv")
        if not np.array_equal(history.epoch, np.arange(1, config["epochs"] + 1)):
            raise ValueError("Resume rejected: incomplete epoch log")
        for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            if not np.isfinite(history[key]).all() or not np.isclose(history.iloc[-1][key], summary[key], rtol=0, atol=1e-12):
                raise ValueError("Resume rejected: invalid epoch metrics")


def finish_comparisons(public_root, states, configurations):
    rows = []
    for config in configurations:
        run_id = config["run_id"]
        summary_file = public_root / "runs" / run_id / "summary.json"
        row = {**config, "status": states[run_id]}
        if summary_file.exists():
            row.update(json.loads(summary_file.read_text(encoding="utf-8")))
        rows.append(row)
    pd.DataFrame(rows).to_csv(public_root / "all_results.csv", index=False, encoding="utf-8-sig")
    comparisons = public_root / "comparisons"
    comparisons.mkdir(exist_ok=True)
    for name, ids in COMPARISONS.items():
        selected = [row for row in rows if row["run_id"] in ids]
        pd.DataFrame(selected).to_csv(comparisons / f"{name}.csv", index=False, encoding="utf-8-sig")
        baseline = public_root / "runs" / ids[0] / "validation_predictions.csv"
        if not baseline.exists():
            continue
        first = pd.read_csv(baseline)
        for run_id in ids[1:]:
            prediction_file = public_root / "runs" / run_id / "validation_predictions.csv"
            if not prediction_file.exists():
                continue
            other = pd.read_csv(prediction_file)
            if not np.array_equal(first.source_index, other.source_index) or not np.array_equal(first.true_label, other.true_label):
                raise ValueError("Comparison prediction order mismatch")
            difference = first[["source_index", "true_label"]].copy()
            difference["baseline_prediction"] = first.predicted_label
            difference["comparison_prediction"] = other.predicted_label
            difference["corrected"] = ~first.is_correct & other.is_correct
            difference["introduced"] = first.is_correct & ~other.is_correct
            difference.to_csv(comparisons / f"{name}__{run_id}__all_cases.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "local_runs")
    parser.add_argument("--check-only", action="store_true", help="validate plan and data; do not fit")
    parser.add_argument("--resume", type=Path, help="existing session root; skip only completed, never discard failures")
    args = parser.parse_args()
    configurations = plan()
    assert len(configurations) == 29 and len({c["run_id"] for c in configurations}) == 29
    file = args.data_dir / "train_data.csv"
    frame = pd.read_csv(file)
    if not {"text", "target"}.issubset(frame):
        raise ValueError("train_data.csv requires text and target columns")
    if frame[["text", "target"]].isna().any().any():
        raise ValueError("train_data.csv contains missing text or target values")
    fit_texts, val_texts, y_fit, y_val, fit_indices, val_indices = train_test_split(
        frame.text.astype(str).tolist(), frame.target.to_numpy(), np.arange(len(frame)),
        test_size=.2, stratify=frame.target.to_numpy(), random_state=42)
    data = dict(fit_texts=fit_texts, val_texts=val_texts, y_fit=y_fit, y_val=y_val,
                fit_indices=fit_indices, val_indices=val_indices)
    hashes = {"train_data_sha256": sha256(file), "train_indices_sha256": array_sha(fit_indices, "<i8"),
              "val_indices_sha256": array_sha(val_indices, "<i8"), "runner_sha256": sha256(__file__)}
    print(f"PLAN: 29 configurations, fit={len(y_fit)}, validation={len(y_val)}.", flush=True)
    if args.check_only:
        print(json.dumps({"runs": configurations, "comparisons": COMPARISONS, "hashes": hashes}, ensure_ascii=False, indent=2))
        return
    if args.resume:
        session = args.resume.resolve()
        metadata = json.loads((session / "public" / "session.json").read_text(encoding="utf-8"))
        if metadata["hashes"] != hashes or metadata["configurations"] != configurations or metadata["packages"] != package_versions():
            raise ValueError("Resume rejected: source, data, split, plan, or package versions changed")
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session = args.output.resolve() / ("reproduction_" + stamp)
        session.mkdir(parents=True, exist_ok=False)
    public_root, private_root = session / "public", session / "private"
    public_root.mkdir(exist_ok=True)
    private_root.mkdir(exist_ok=True)
    if not args.resume:
        shutil.copy2(__file__, public_root / "record_all_experiments.py")
        save_json(public_root / "session.json", {
            "started_at": datetime.now(timezone.utc).isoformat(), "hashes": hashes,
            "configurations": configurations, "comparison_groups": COMPARISONS,
            "python": sys.version, "platform": platform.platform(),
            "packages": package_versions(),
            "threadpools": threadpool_info(), "training_threads": 2,
            "train_count": len(y_fit), "validation_count": len(y_val),
            "loss_definition": "epoch-end full-data unweighted mean cross-entropy, excludes L2",
            "optimizer_reported_loss_definition": "sklearn loss_: internal minibatch-averaged objective; may contain weights/L2, not directly comparable to val_loss",
            "all_configurations_recorded": True, "official_test_used": False,
            "old_results_overwritten": False, "historical_results_reused": False,
            "adapter_note": "Known hyperparameters are reproduced. This new adapter is not the exact historical script; early source snapshots are unavailable.",
            "shared_baseline_note": "One fresh fit for each unique configuration; comparison groups refer to that run, not independent repeated fits.",
            "private_note": "private contains raw texts and model/vocabulary artifacts. Do not publish private or teacher datasets by default.",
            "rng_caveat": "Same seed across shapes does not mean identical initialization or batch shuffle stream.",
            "weight_caveat": "Global mean-one weights do not equalize all minibatch weight sums or MLP L2 scales.",
        })
        pd.DataFrame({"source_index": fit_indices, "label": y_fit}).to_csv(public_root / "training_split.csv", index=False)
        pd.DataFrame({"source_index": val_indices, "label": y_val}).to_csv(public_root / "validation_split.csv", index=False)
    print("SESSION " + str(session), flush=True)
    states = {}
    for config in configurations:
        status_file = public_root / "runs" / config["run_id"] / "status.json"
        states[config["run_id"]] = json.loads(status_file.read_text(encoding="utf-8"))["status"] if status_file.exists() else "pending"
    save_json(public_root / "progress.json", states)
    cache = {}
    for position, config in enumerate(configurations, 1):
        run_id = config["run_id"]
        if states[run_id] == "completed":
            verify_completed(public_root / "runs" / run_id, private_root / run_id, config, data)
            continue
        directory = public_root / "runs" / run_id
        if directory.exists():
            # Preserve any failed/interrupted attempt before retrying it.
            resolved = directory.resolve()
            if not resolved.is_relative_to((public_root / "runs").resolve()):
                raise ValueError("Unsafe retry source path")
            attempt_id = f"{run_id}_{datetime.now().strftime('%H%M%S_%f')}"
            retry = public_root / "prior_attempts" / attempt_id
            retry.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(directory), str(retry))
            if (private_root / run_id).exists():
                private_retry = private_root / "prior_attempts" / attempt_id
                private_retry.parent.mkdir(parents=True, exist_ok=True)
                private_source = (private_root / run_id).resolve()
                if not private_source.is_relative_to(private_root.resolve()):
                    raise ValueError("Unsafe private retry source path")
                shutil.move(str(private_source), str(private_retry))
        print(f"\n[{position}/29] {run_id}", flush=True)
        states[run_id] = "running"
        save_json(public_root / "progress.json", states)
        try:
            run_one(config, data, cache, public_root, private_root)
            states[run_id] = "completed"
        except KeyboardInterrupt:
            directory.mkdir(parents=True, exist_ok=True)
            save_json(directory / "status.json", {"status": "interrupted", "ended_at": datetime.now(timezone.utc).isoformat()})
            states[run_id] = "interrupted"
            save_json(public_root / "progress.json", states)
            finish_comparisons(public_root, states, configurations)
            print("INTERRUPTED: partial epoch logs preserved", flush=True)
            raise
        except Exception:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            save_json(directory / "status.json", {"status": "failed", "ended_at": datetime.now(timezone.utc).isoformat()})
            states[run_id] = "failed"
            print("FAILED " + run_id + "\n" + traceback.format_exc(), flush=True)
        save_json(public_root / "progress.json", states)
        finish_comparisons(public_root, states, configurations)
    save_json(public_root / "completion.json", {"status": "completed" if all(s == "completed" for s in states.values()) else "incomplete",
                                               "ended_at": datetime.now(timezone.utc).isoformat(), "states": states})
    print("ALL RESULTS SAVED " + str(session), flush=True)
    if any(state != "completed" for state in states.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
