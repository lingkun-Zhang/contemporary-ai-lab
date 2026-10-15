"""Compare a fixed LR model before and after conservative metadata removal."""
import argparse
import hashlib
import json
import platform
import re
import shutil
import warnings
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


CONFIG = {"max_features": 10000, "C": 100, "penalty": "l2", "solver": "lbfgs",
          "max_iter": 1000, "tol": 1e-4, "random_state": 42,
          "class_weight": None, "training_threads": 2}
SOURCE_FIELDS = frozenset({
    "from", "sender", "reply-to", "organization", "nntp-posting-host", "lines",
    "date", "path", "message-id", "references", "xref", "newsgroups", "followup-to",
    "distribution", "approved", "return-path", "received", "expires", "originator",
    "nntp-posting-user", "x-newsreader", "x-mailer", "x-posting-host",
})
FIELD = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]*):")
GROUPS = (("original", "原文"), ("no_source_headers", "去指定来源头"),
          ("no_source_headers_signature", "再去标记签名"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def parse_header(text):
    """Recognize a mail prefix, never colon-containing lines in the message body."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    boundary = re.search(r"\n[ \t]*\n", normalized)
    if boundary is None:
        return None
    blocks = []
    for line in normalized[:boundary.start()].split("\n"):
        match = FIELD.match(line)
        if match:
            blocks.append([match.group(1).lower(), [line]])
        elif line.startswith((" ", "\t")) and blocks:
            blocks[-1][1].append(line)
        else:
            return None  # Leave ambiguous, non-standard prefixes intact.
    names = {name for name, _ in blocks}
    if not {"from", "subject"}.issubset(names):
        return None
    return blocks, normalized[boundary.end():]


def remove_marked_signature(body):
    """Only a literal, unquoted '-- ' delimiter; normal final paragraphs stay."""
    lines = body.splitlines(keepends=True)
    markers = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == "-- "]
    if not markers:
        return body, False
    return "".join(lines[:markers[-1]]), True


def clean_text(text, group):
    if group not in {name for name, _ in GROUPS}:
        raise ValueError(group)
    parsed = parse_header(text)
    stats = {"recognized_header": parsed is not None, "removed_fields": [],
             "signature_removed": False, "changed": False, "removed_characters": 0,
             "empty_after": False}
    if group == "original":
        result = text
    elif parsed is None:
        # An ambiguous prefix stays, but the independent signature rule can run.
        result = text
        if group == "no_source_headers_signature":
            result, stats["signature_removed"] = remove_marked_signature(result)
    else:
        blocks, body = parsed
        stats["removed_fields"] = [name for name, _ in blocks if name in SOURCE_FIELDS]
        kept = [line for name, lines in blocks if name not in SOURCE_FIELDS for line in lines]
        if group == "no_source_headers_signature":
            body, stats["signature_removed"] = remove_marked_signature(body)
        elif group != "no_source_headers":
            raise ValueError(group)
        result = "\n".join(kept) + "\n\n" + body
    stats.update(changed=result != text, removed_characters=max(0, len(text) - len(result)),
                 empty_after=not bool(result.strip()))
    return result, stats


def audit(texts, indices, group):
    transformed, rows = zip(*(clean_text(texts[int(i)], group) for i in indices))
    fields = Counter(field for row in rows for field in row["removed_fields"])
    def text_digest(values):
        digest = hashlib.sha256()
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        return digest.hexdigest()

    record = {"rows": len(indices), "recognized_headers": sum(row["recognized_header"] for row in rows),
              "changed_rows": sum(row["changed"] for row in rows),
              "signature_removed_rows": sum(row["signature_removed"] for row in rows),
              "empty_after": sum(row["empty_after"] for row in rows),
              "removed_characters": sum(row["removed_characters"] for row in rows),
              "removed_fields": dict(sorted(fields.items())),
              "text_hash_definition": "UTF-8 bytes, each preceded by its 8-byte little-endian byte length, in split row order",
              "original_text_sha256": text_digest(texts[int(i)] for i in indices),
              "transformed_text_sha256": text_digest(transformed),
              "changed_source_indices": [int(i) for i, row in zip(indices, rows) if row["changed"]],
              "signature_source_indices": [int(i) for i, row in zip(indices, rows) if row["signature_removed"]]}
    assert len(transformed) == len(indices) and record["empty_after"] == 0
    return list(transformed), record


def draw_comparison(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import PercentFormatter

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    labels = [label for _, label in GROUPS] if font.is_file() else ["Original", "No source headers", "+ marked signatures"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
    positions = np.arange(len(summaries))
    for phase, shift, color in (("train", -.18, "#2166AC"), ("val", .18, "#D6604D")):
        name = {"train": "训练", "val": "验证"}[phase] if font.is_file() else phase
        for ax, field in zip(axes, ("accuracy", "loss")):
            values = [row[f"{phase}_{field}"] for row in summaries]
            bars = ax.bar(positions + shift, values, width=.36, label=name, color=color)
            ax.bar_label(bars, labels=[f"{value:.2%}" if field == "accuracy" else f"{value:.4f}" for value in values], fontsize=8, padding=3)
    axes[0].set_ylim(0, 1.13)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("准确率" if font.is_file() else "Accuracy")
    axes[1].set_ylim(0, max(row["val_loss"] for row in summaries) * 1.25)
    axes[1].set_ylabel("平均交叉熵（不含L2）" if font.is_file() else "Mean cross-entropy (no L2)")
    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.17), ncol=2, frameon=False)
    fig.savefig(output / "comparison.png", dpi=200)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output", type=Path, default=root / "local_cleaning")
    parser.add_argument("--redraw", action="store_true", help="redraw saved metrics only; no dataset access or training")
    args = parser.parse_args()
    if args.redraw:
        saved = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
        old_figure = sha(args.output / "comparison.png")
        draw_comparison(args.output, saved["groups"])
        save_json(args.output / "figure_source.json", {
            "metrics_file": "summary.json", "metrics_sha256": sha(args.output / "summary.json"),
            "figure_file": "comparison.png", "figure_sha256": sha(args.output / "comparison.png"),
            "previous_figure_sha256": old_figure, "rendering_script_sha256": sha(__file__),
            "change": "move legends outside axes so no value labels are covered",
            "training_source": "source_snapshot.py", "new_training": False, "numeric_data_changed": False,
        })
        save_json(args.output / "checksums.json", {file.relative_to(args.output).as_posix(): sha(file)
                  for file in sorted(args.output.rglob("*")) if file.is_file() and file.name != "checksums.json"})
        print("Redrawn from saved metrics only; no training or dataset access.")
        return
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; existing results are never replaced.")
    data_file = args.data_dir / "train_data.csv"
    input_sha = sha(data_file)
    reference = root / "results" / "reproduction_20261014"
    old_session = json.loads((reference / "session.json").read_text(encoding="utf-8"))
    assert input_sha == old_session["hashes"]["train_data_sha256"], "Data differ from the recorded experiments."
    frame = pd.read_csv(data_file)
    assert len(frame) == 7368 and not frame[["text", "target"]].isna().any().any()
    texts, labels = frame.text.astype(str).tolist(), frame.target.to_numpy()
    assert np.array_equal(np.unique(labels), np.arange(10))
    fit, val = train_test_split(np.arange(len(labels)), test_size=.2, stratify=labels, random_state=42)
    assert array_sha(fit) == old_session["hashes"]["train_indices_sha256"]
    assert array_sha(val) == old_session["hashes"]["val_indices_sha256"]
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / "source_snapshot.py")
    started_at = datetime.now(timezone.utc).isoformat()
    save_json(args.output / "protocol.json", {
        "model": "LogisticRegression", "configuration": CONFIG,
        "source_fields_removed": sorted(SOURCE_FIELDS), "semantic_fields_retained": ["subject", "keywords", "summary"],
        "unknown_fields": "retained", "folded_header_lines": "removed or retained with their field",
        "header_guard": "first blank line; parsed header must contain From and Subject",
        "signature_rule": "last standalone literal '-- ' line and everything following it; no arbitrary tail truncation",
        "quotes": "no quote-line filtering; only unquoted standalone signature markers matched",
        "ambiguous_headers": "header not removed; independent signature rule can still apply",
        "sample_filtering": "none", "vocabulary_and_idf": "refit separately on each transformed training part only",
        "comparison": "same source rows and labels; no further hyperparameter search",
        "official_test_opened": False, "original_records_overwritten": False,
    })
    save_json(args.output / "split.json", {"train_source_indices": fit.tolist(), "validation_source_indices": val.tolist(),
                                           "train_sha256": array_sha(fit), "validation_sha256": array_sha(val)})
    summaries, saved_predictions = [], {}
    with (args.output / "console.log").open("w", encoding="utf-8") as log:
        def report(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        report("Fixed LR C=100, vocabulary cap=10000; fit=5894, validation=1474; original split, seed=42.")
        for group, display_name in GROUPS:
            directory = args.output / group
            directory.mkdir()
            fit_texts, train_audit = audit(texts, fit, group)
            val_texts, val_audit = audit(texts, val, group)
            vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1), sublinear_tf=False)
            start = perf_counter()
            X_fit = vectorizer.fit_transform(fit_texts)
            X_val = vectorizer.transform(val_texts)
            vectorization_seconds = perf_counter() - start
            model = LogisticRegression(C=100, penalty="l2", solver="lbfgs", max_iter=1000,
                                       tol=1e-4, random_state=42, class_weight=None)
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                warnings.simplefilter("always")
                start = perf_counter()
                model.fit(X_fit, labels[fit])
                fit_seconds = perf_counter() - start
                train_proba, val_proba = model.predict_proba(X_fit), model.predict_proba(X_val)
                train_pred, val_pred = model.predict(X_fit), model.predict(X_val)
            warning_records = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
            assert not any(issubclass(item.category, ConvergenceWarning) for item in caught), "Unconverged group; do not compare as final."
            for probability in (train_proba, val_proba):
                assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
            summary = {"group": group, "display_name": display_name,
                       "train_accuracy": float(accuracy_score(labels[fit], train_pred)),
                       "val_accuracy": float(accuracy_score(labels[val], val_pred)),
                       "train_loss": float(log_loss(labels[fit], train_proba, labels=model.classes_)),
                       "val_loss": float(log_loss(labels[val], val_proba, labels=model.classes_)),
                       "wrong_count": int(np.sum(val_pred != labels[val])),
                       "actual_features": int(X_fit.shape[1]), "parameter_count": int(model.coef_.size + model.intercept_.size),
                       "zero_feature_rows_train": int(np.sum(X_fit.getnnz(axis=1) == 0)),
                       "zero_feature_rows_validation": int(np.sum(X_val.getnnz(axis=1) == 0)),
                       "vocabulary_sha256": hashlib.sha256("\0".join(vectorizer.get_feature_names_out()).encode("utf-8")).hexdigest(),
                       "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
                       "optimizer_iterations": model.n_iter_.tolist(), "fit_seconds": fit_seconds,
                       "vectorization_seconds": vectorization_seconds, "warnings": warning_records}
            save_json(directory / "metrics.json", summary)
            save_json(directory / "cleaning_audit.json", {"train": train_audit, "validation": val_audit})
            save_json(directory / "classification_report.json", classification_report(labels[val], val_pred, output_dict=True, zero_division=0))
            save_json(directory / "confusion_matrix.json", {"classes": model.classes_.tolist(), "matrix": confusion_matrix(labels[val], val_pred).tolist()})
            save_json(directory / "validation_predictions.json", {
                "source_indices": val.tolist(), "true_labels": labels[val].tolist(), "predicted_labels": val_pred.tolist(),
                "classes": model.classes_.tolist(), "probabilities": val_proba.tolist()})
            saved_predictions[group] = val_pred
            summaries.append(summary)
            report(f"{display_name} | train={summary['train_accuracy']:.2%} | val={summary['val_accuracy']:.2%} | "
                   f"trainCE={summary['train_loss']:.4f} | valCE={summary['val_loss']:.4f} | wrong={summary['wrong_count']} | "
                   f"changed train/val={train_audit['changed_rows']}/{val_audit['changed_rows']} | "
                   f"signatures train/val={train_audit['signature_removed_rows']}/{val_audit['signature_removed_rows']}")
    pairs = []
    for before, after in (("original", "no_source_headers"), ("no_source_headers", "no_source_headers_signature"),
                          ("original", "no_source_headers_signature")):
        old, new = saved_predictions[before] == labels[val], saved_predictions[after] == labels[val]
        pairs.append({"before": before, "after": after,
                      "corrected_indices": val[~old & new].tolist(), "introduced_error_indices": val[old & ~new].tolist(),
                      "corrected_count": int(np.sum(~old & new)), "introduced_error_count": int(np.sum(old & ~new)),
                      "changed_predictions": int(np.sum(saved_predictions[before] != saved_predictions[after]))})
    old_baseline = json.loads((reference / "runs" / "lr_vocab10000" / "summary.json").read_text(encoding="utf-8"))
    baseline_checks = {field: bool(np.isclose(summaries[0][field], old_baseline[field], rtol=0, atol=1e-12))
                       for field in ("train_accuracy", "val_accuracy", "train_loss", "val_loss", "wrong_count")}
    baseline_checks.update({field: summaries[0][field] == old_baseline[field] for field in ("vocabulary_sha256", "idf_sha256")})
    save_json(args.output / "summary.json", {"groups": summaries, "paired_changes": pairs,
                                            "fresh_baseline_matches_recorded_run": baseline_checks})
    draw_comparison(args.output, summaries)
    assert sha(data_file) == input_sha
    save_json(args.output / "session.json", {
        "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "threadpoolctl")},
        "input_sha256": input_sha, "script_sha256": sha(__file__), "test_used": False,
        "fresh_baseline_matches_recorded_run": all(baseline_checks.values()),
        "all_three_groups_completed": True, "raw_texts_and_models_exported": False})
    save_json(args.output / "checksums.json", {file.name if file.parent == args.output else file.relative_to(args.output).as_posix(): sha(file)
                                             for file in sorted(args.output.rglob("*")) if file.is_file()})
    print("Complete. Records and comparison.png saved; official prediction.csv was not touched.", flush=True)


if __name__ == "__main__":
    main()
