"""Archive every available historical result, without selecting better runs.

The private archive preserves files byte-for-byte.  The separate public export
is explicitly transformed: text columns and feature dictionaries are excluded,
but no metric or prediction rows are selected or shortened.  Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


RESULT_ROOTS = ("mlp_results", "lr_results", "svm_results")
EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
TEXT_KEYS = {"text", "raw_text", "document_text", "sample_text", "original_text", "content"}
HASH_KEYS = {"script_sha256", "training_core_sha256", "current_training_function_sha256",
             "current_training_code_sha256", "wrapper_sha256", "helper_sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"CSV has extra unheaded fields: {path}")
    return columns, rows


def find_values(value, keys: set[str], prefix: str = "") -> list[dict]:
    matches = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{prefix}.{key}" if prefix else key
            if key in keys:
                matches.append({"key": current, "value": item})
            matches.extend(find_values(item, keys, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(find_values(item, keys, f"{prefix}[{index}]"))
    return matches


def privacy_findings(value, prefix: str = "") -> list[str]:
    findings = []
    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{prefix}.{key}" if prefix else key
            if str(key).lower() in TEXT_KEYS:
                findings.append(f"raw_text_key:{current}")
            findings.extend(privacy_findings(item, current))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(privacy_findings(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and EMAIL_PATTERN.search(value):
        findings.append(f"email_like_value:{prefix}")
    return findings


def normalize_lab_paths(value, lab_dir: Path):
    # Redacting a local path is documented as a transformation, not an original.
    if isinstance(value, dict):
        return {key: normalize_lab_paths(item, lab_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_lab_paths(item, lab_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(lab_dir), "<LAB_DIR>").replace(lab_dir.as_posix(), "<LAB_DIR>")
    return value


def export_public(source: Path, target: Path, lab_dir: Path) -> dict:
    suffix = source.suffix.lower()
    info = {"status": "excluded", "reason": "non-public artifact type"}
    target.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        shutil.copy2(source, target)
        return {"status": "byte_copy", "reason": "complete historical plot; no redraw or cropping"}
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        findings = privacy_findings(value)
        if findings:
            return {"status": "excluded", "reason": "JSON privacy audit failed", "findings": findings}
        write_json(target, normalize_lab_paths(value, lab_dir))
        return {"status": "transformed", "reason": "JSON audited; local lab paths normalized; no metric values removed"}
    if suffix == ".csv":
        columns, rows = read_csv(source)
        # Vocabulary tokens can reproduce identifying fragments.  Keep all of
        # those files privately, rather than selecting only innocent-looking tokens.
        if "feature" in columns or source.name in {"features.csv", "added_features.csv", "dropped_features.csv"}:
            return {"status": "excluded", "reason": "complete feature dictionary kept private", "row_count": len(rows)}
        dropped = [name for name in columns if name.lower() in TEXT_KEYS or name.lower().endswith("_text")]
        retained = [name for name in columns if name not in dropped]
        safe_rows = [normalize_lab_paths({key: row[key] for key in retained}, lab_dir) for row in rows]
        findings = privacy_findings(safe_rows)
        if findings:
            return {"status": "excluded", "reason": "retained CSV values failed privacy audit", "findings": findings,
                    "source_row_count": len(rows), "removed_columns": dropped}
        write_csv(target, safe_rows, retained)
        public_columns, public_rows = read_csv(target)
        if len(public_rows) != len(rows) or public_columns != retained:
            raise RuntimeError("Public CSV unexpectedly changed row count or retained columns.")
        return {"status": "transformed", "reason": "all rows retained; text columns removed; lab paths normalized",
                "source_row_count": len(rows), "public_row_count": len(public_rows),
                "removed_columns": dropped, "retained_columns": retained}
    if suffix == ".txt" and source.name in {"classification_report.txt", "convergence_warnings.txt"}:
        value = source.read_text(encoding="utf-8-sig")
        if EMAIL_PATTERN.search(value):
            return {"status": "excluded", "reason": "text artifact contains email-like value"}
        target.write_text(normalize_lab_paths(value, lab_dir), encoding="utf-8")
        return {"status": "transformed", "reason": "numeric classification/convergence report audited; lab paths normalized"}
    return info


def history_details(path: Path, expected_epochs) -> dict:
    if not path.exists():
        return {"exists": False, "status": "not_recorded", "row_count": 0}
    columns, rows = read_csv(path)
    epochs = [int(row["epoch"]) for row in rows] if "epoch" in columns else []
    expected = list(range(1, int(expected_epochs) + 1)) if expected_epochs is not None else None
    return {"exists": True, "row_count": len(rows), "columns": columns, "epochs": epochs,
            "status": "complete_recorded_epochs" if expected == epochs else "check_recorded_epoch_sequence",
            "expected_epochs": expected_epochs, "missing_epochs": sorted(set(expected or []) - set(epochs)),
            "duplicate_epochs": sorted({epoch for epoch in epochs if epochs.count(epoch) > 1}),
            "first_row": rows[0] if rows else None, "last_row": rows[-1] if rows else None}


def archive(lab_dir: Path, output: Path) -> dict:
    lab_dir = lab_dir.resolve(strict=True)
    output = output.resolve()
    if output == lab_dir or output.is_relative_to(lab_dir):
        raise ValueError("Archive output must not be the source lab directory or inside it.")
    private = output / "history_private"
    public = output / "history_public"
    if private.exists() or public.exists():
        raise FileExistsError("Archive already exists. Use a new output root; never overwrite historical records.")
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    inventory, run_index, csv_index, json_index, exports, groups = [], [], [], [], [], []
    for root_name in RESULT_ROOTS:
        result_root = lab_dir / root_name
        if not result_root.is_dir():
            groups.append({"result_root": root_name, "status": "source_root_absent"})
            continue
        for group in sorted(path for path in result_root.iterdir() if path.is_dir()):
            group_files = sorted(path for path in group.rglob("*") if path.is_file())
            groups.append({"result_root": root_name, "run_directory": group.name, "status": "archived_all_files",
                           "file_count": len(group_files),
                           "config_count": sum(path.name == "config.json" for path in group_files),
                           "history_count": sum(path.name == "history.csv" for path in group_files),
                           "validation_predictions_count": sum(path.name == "validation_predictions.csv" for path in group_files)})
        # Also include possible files directly below a result root.
        for source in sorted(path for path in result_root.rglob("*") if path.is_file()):
            if not source.resolve().is_relative_to(lab_dir):
                raise ValueError(f"Result symlink escapes the lab directory: {source}")
            relative = source.relative_to(lab_dir)
            private_target = private / "raw" / relative
            public_target = public / "raw" / relative
            private_target.parent.mkdir(parents=True, exist_ok=True)
            before = source.stat()
            original_hash = sha256(source)
            shutil.copy2(source, private_target)
            after = source.stat()
            copy_hash = sha256(private_target)
            if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or original_hash != copy_hash:
                raise RuntimeError(f"Source changed while archiving: {source}")
            entry = {"original_path": str(source), "relative_path": relative.as_posix(),
                     "size_bytes": before.st_size, "sha256": original_hash,
                     "private_path": str(private_target), "status": "byte_copy_verified"}
            inventory.append(entry)
            if source.suffix.lower() == ".csv":
                columns, rows = read_csv(private_target)
                csv_index.append({"relative_path": relative.as_posix(), "columns": columns,
                                  "row_count": len(rows), "sha256": original_hash})
            elif source.suffix.lower() == ".json":
                value = json.loads(private_target.read_text(encoding="utf-8-sig"))
                json_index.append({"relative_path": relative.as_posix(), "sha256": original_hash, "parameters": value})
                if source.name == "config.json":
                    hash_records = find_values(value, HASH_KEYS)
                    epochs = value.get("epochs")
                    reuse_note_path = source.parent / "reuse_note.json"
                    reuse_note = json.loads(reuse_note_path.read_text(encoding="utf-8-sig")) if reuse_note_path.exists() else None
                    run_index.append({"config_path": relative.as_posix(), "parameters": value,
                                      "recorded_source_hashes": hash_records,
                                      "source_provenance_status": "some_hashes_recorded_may_describe_current_or_wrapper_code"
                                      if hash_records else "original_training_source_hash_not_recorded",
                                      "history": history_details(private_target.parent / "history.csv", epochs),
                                      "history_requirement_note": "MLP records one full pass per epoch" if root_name == "mlp_results"
                                      else "fit-based solver did not expose an epoch log; do not fabricate epoch curves",
                                      "validation_predictions_recorded": (source.parent / "validation_predictions.csv").exists(),
                                      "baseline_original_config_recorded": (source.parent / "baseline_original_config.json").exists(),
                                      "reuse_note": reuse_note,
                                      "reuse_source_fields": find_values(value, {"source_run_dir", "reused"}),
                                      "reused": bool(value.get("reused") or value.get("summary", {}).get("reused")
                                                     or reuse_note_path.exists())})
            exported = export_public(private_target, public_target, lab_dir)
            export_entry = {"source_relative_path": relative.as_posix(), "source_sha256": original_hash, **exported}
            if exported["status"] != "excluded":
                export_entry.update({"public_relative_path": (Path("raw") / relative).as_posix(),
                                     "public_sha256": sha256(public_target), "public_size_bytes": public_target.stat().st_size})
            exports.append(export_entry)

    # Histories may sort after config.json, so evaluate the completed private
    # archive only after every source result file has been copied.
    for row in run_index:
        config_parent = private / "raw" / Path(row["config_path"]).parent
        row["history"] = history_details(config_parent / "history.csv", row["parameters"].get("epochs"))

    # Current code is a new snapshot.  It is never described as the original
    # source used for old runs whose historical source hash was not recorded.
    source_inventory = []
    for source in sorted([*lab_dir.glob("*.py"), *lab_dir.glob("requirements.txt")]):
        destination = private / "current_source_snapshot" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        original_hash, copy_hash = sha256(source), sha256(destination)
        if original_hash != copy_hash:
            raise RuntimeError(f"Source snapshot changed while copying: {source}")
        source_inventory.append({"original_path": str(source), "relative_path": source.name,
                                 "size_bytes": source.stat().st_size, "sha256": original_hash,
                                 "status": "current_source_byte_copy_verified_not_historical_training_source"})

    datasets = [{"filename": name, "status": "hash_only_not_copied", "sha256": sha256(lab_dir / name),
                 "size_bytes": (lab_dir / name).stat().st_size}
                for name in ("train_data.csv", "test_data_unlabeled.csv") if (lab_dir / name).is_file()]
    gaps = [
        {"experiment": "LR C=0.1/1/10, 5000 features", "status": "historical_file_log_not_found",
         "note": "Earlier conversation/screenshots contain final metrics only. Do not invent historical probabilities or optimizer trajectories."},
        {"experiment": "LR C=100, class7 relative weights 1/1.5/2", "status": "historical_file_log_not_found",
         "note": "Current Logistic_experiment.py prints these groups but does not persist their original outputs."},
        {"experiment": "all earlier unrecorded or unsaved attempts", "status": "not_recoverable_from_current_result_directories",
         "note": "A fresh rerun is a new reproduction, not a reconstruction of missing original measurements."},
    ]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_lab_directory": str(lab_dir),
               "historical_result_directory_count": sum("run_directory" in row for row in groups),
               "historical_file_count": len(inventory), "config_count": len(run_index),
               "history_count": sum(row["history"]["exists"] for row in run_index),
               "recorded_epoch_row_count": sum(row["history"]["row_count"] for row in run_index),
               "history_runs_missing_validation_predictions": sum(row["history"]["exists"] and not row["validation_predictions_recorded"] for row in run_index),
               "configs_without_recorded_source_hash": sum(not row["recorded_source_hashes"] for row in run_index),
               "csv_file_count": len(csv_index), "json_file_count": len(json_index),
               "all_source_files_archived": True, "public_exported_file_count": sum(row["status"] != "excluded" for row in exports),
               "public_excluded_file_count": sum(row["status"] == "excluded" for row in exports),
               "datasets": datasets, "known_history_gaps": gaps,
               "provenance_limit": "Recorded metrics and duplicated/reused runs are all retained. Some historical configs lack the original training-source hash; current source cannot establish exact historical byte-for-byte reproduction.",
               "public_limit": "No upload occurred. Public export removes raw text and complete feature dictionaries, never metric/prediction rows. PNGs are complete byte copies; JSON/CSV transformations have their own hashes and manifest."}
    write_json(private / "file_manifest.json", inventory)
    write_csv(private / "file_manifest.csv", inventory, ["original_path", "relative_path", "size_bytes", "sha256", "private_path", "status"])
    write_json(private / "run_index.json", run_index)
    write_json(private / "csv_index.json", csv_index)
    write_json(private / "json_parameters_index.json", json_index)
    write_json(private / "group_index.json", groups)
    write_json(private / "current_source_manifest.json", source_inventory)
    write_json(private / "archive_summary.json", summary)
    write_json(public / "public_export_manifest.json", exports)
    write_json(public / "archive_summary.json", normalize_lab_paths(summary, lab_dir))
    # Public indexes contain column names/counts, not the removed raw texts.
    write_json(public / "csv_index.json", csv_index)
    write_json(public / "group_index.json", groups)
    public_run_index = []
    for row in run_index:
        safe_row = {key: value for key, value in row.items() if key != "parameters"}
        findings = privacy_findings(row["parameters"])
        if findings:
            safe_row["parameters_status"] = "kept_private_due_to_privacy_audit"
            safe_row["parameters_privacy_findings"] = findings
        else:
            safe_row["parameters"] = row["parameters"]
        public_run_index.append(normalize_lab_paths(safe_row, lab_dir))
    write_json(public / "run_index.json", public_run_index)
    note = (
        "# Historical experimental records\n\n"
        "Every existing run directory is retained, including repeated runs, failed improvements, and cached baselines.\n"
        "The private raw archive is byte-verified against the source files. It is not intended for public publication.\n"
        "The public raw export is transformed: every row is retained, raw-text columns are removed, feature dictionaries are excluded, and local lab paths are normalized. See public_export_manifest.json.\n"
        "Teacher-provided datasets were not copied. Dataset hashes identify the input; public redistribution permission is unconfirmed.\n"
        "The current_source_snapshot is the current code, not retroactively the exact training source for old runs. Missing original logs remain marked as gaps.\n"
        "A rerun must live separately and be labeled as a new reproduction. Nothing in this archive publishes, refits a final model, or creates an official prediction submission.\n"
    )
    (private / "README.md").write_text(note, encoding="utf-8")
    (public / "README.md").write_text(note, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-dir", type=Path, required=True, help="含阶段结果的本地目录")
    parser.add_argument("--output", type=Path, required=True, help="新建归档根目录；history_private 不可公开")
    args = parser.parse_args()
    summary = archive(args.lab_dir, args.output)
    print(json.dumps({key: summary[key] for key in ("historical_result_directory_count", "historical_file_count", "config_count",
                                                  "history_count", "recorded_epoch_row_count", "public_exported_file_count", "public_excluded_file_count")}, indent=2))


if __name__ == "__main__":
    main()
