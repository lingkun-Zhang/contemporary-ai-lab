"""Summarize saved timing batches; does not run or overwrite any experiment."""

import csv
import hashlib
import json
import statistics
from pathlib import Path


def summarize_existing_timings(output_dir=None):
    root = Path(__file__).resolve().parent
    output = Path(output_dir) if output_dir is not None else root / "results_heap_tuning"
    output.mkdir(parents=True, exist_ok=True)
    rows, sources = [], []
    for relative, batch_field in (
        ("results/comparison.csv", "batch_average_ms"),
        ("results_supplementary/tie_break_comparison.csv", "total_batch_average_ms"),
        ("results_supplementary/cost_ratio.csv", "total_batch_average_ms"),
        ("results_supplementary/component_minimum.csv", "total_batch_average_ms"),
    ):
        source = root / relative
        sources.append(dict(path=relative, sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        with source.open(encoding="utf-8-sig", newline="") as stream:
            for record in csv.DictReader(stream):
                batches = json.loads(record[batch_field])
                quartiles = statistics.quantiles(batches, n=4, method="inclusive") if len(batches) > 1 else batches * 3
                rows.append(dict(
                    source=relative, case=record["case"], variant=record["variant"],
                    tie_break=record.get("tie_break", "low_h"), min_scope=record.get("min_scope", "global"),
                    batches=len(batches), median_ms=statistics.median(batches),
                    q1_ms=quartiles[0], q3_ms=quartiles[2], iqr_ms=quartiles[2]-quartiles[0],
                    min_ms=min(batches), max_ms=max(batches), raw_batch_average_ms=json.dumps(batches),
                ))
    with (output / "previous_timing_ranges.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    info = dict(
        method="descriptive summaries of existing batch averages; no new experiments",
        unit="milliseconds per complete query, per source timing protocol",
        limitations="IQR and min/max describe recorded timing batches, not confidence intervals or independent maps",
        configurations=len(rows), source_files=sources,
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "previous_timing_sources.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    info = summarize_existing_timings()
    print(f"Summarized {info['configurations']} saved timing records without rerunning them.")
