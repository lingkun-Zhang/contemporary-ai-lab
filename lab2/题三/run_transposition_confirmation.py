"""题三固定全层界感知缓存：新种子、新计时块，不重新挑选缓存参数。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from alphabeta_tree import validate_tree
from run_transposition_study import (
    csv_write, encode_graph, invoke, make_graph, quantiles, semantic_oracle,
)


BASE = Path(__file__).resolve().parent
SEEDS = (20261040, 20261041, 20261042)
PROBABILITIES = (0, .5, 1)
METHODS = ("original", "off", "exact", "bounds")
REFERENCES = ("original", "off", "exact")
TIMING_SEED = 20261040
PRIOR_FILES = (
    "inputs.json", "comparison.csv", "timing_batches.csv",
    "paired_differences.csv", "outcomes.json", "reproducibility.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(q1, q3):
    """A descriptive direction only; the IQR is not a confidence interval."""
    if q3 < 0:
        return "iqr_below_zero"
    if q1 > 0:
        return "iqr_above_zero"
    return "mixed"


def check_run(case, method, expected_stats=None):
    value, stats, elapsed = invoke(case, method, None)
    if value != case["expected"]:
        raise AssertionError((case["case"], method, "root value", value, case["expected"]))
    if stats["visited_nodes"] != stats["internal_nodes"] + stats["leaf_evaluations"] + stats["tt_returns"]:
        raise AssertionError((case["case"], method, "visit conservation"))
    if stats["tt_hits"] != stats["exact_hits"] + stats["bound_cutoffs"] + stats["noncutting_hits"]:
        raise AssertionError((case["case"], method, "hit conservation"))
    if not 0 < stats["visited_nodes"] <= case["expanded_nodes"]:
        raise AssertionError((case["case"], method, "invalid visit count"))
    if expected_stats is not None and stats != expected_stats:
        raise AssertionError((case["case"], method, "counters changed between calls"))
    return stats, elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "results_transposition_confirmation")
    parser.add_argument("--prior-results", type=Path, default=BASE / "results_transposition")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--batch", type=int, default=5)
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.batch < 1:
        parser.error("repeats至少3，batch至少1")

    prior = args.prior_results.resolve()
    if not all((prior / name).is_file() for name in PRIOR_FILES):
        parser.error("先提供完整的原探索结果目录；本确认不覆盖或重跑探索数据")
    prior_hashes = {name: sha(prior / name) for name in PRIOR_FILES}
    source_paths = [Path(__file__), BASE / "run_transposition_study.py",
                    BASE / "transposition_search.py", BASE / "alphabeta_tree.py"]
    source_hashes = {path.name: sha(path) for path in source_paths}

    cases, specifications = [], []
    for seed in SEEDS:
        for probability in PROBABILITIES:
            root = make_graph(seed, probability)
            validate_tree(root)
            value, nodes, leaves, digest = semantic_oracle(root)
            graph = encode_graph(root)
            graph_digest = hashlib.sha256(json.dumps(graph, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            role = "primary" if probability == 1 else "transfer" if probability == .5 else "no_reuse_control"
            case = dict(case=f"dag_s{seed}_p{probability:g}", seed=seed,
                        sharing_probability=probability, role=role, root=root,
                        expected=value, expanded_nodes=nodes, expanded_leaves=leaves,
                        unique_objects=len(graph["nodes"]),
                        ordered_expansion_sha256=digest, graph_sha256=graph_digest)
            cases.append(case)
            specifications.append({**{key: item for key, item in case.items() if key != "root"}, "graph": graph})
    for seed in SEEDS:
        same_seed = [case for case in cases if case["seed"] == seed]
        if len({case["ordered_expansion_sha256"] for case in same_seed}) != 1:
            raise AssertionError("sharing probability changed the ordered expanded input")
    if len({case["ordered_expansion_sha256"] for case in cases}) != 3:
        raise AssertionError("confirmation requires three distinct expanded semantic inputs")

    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        out = out / ("run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase="Before every invoke/warmup/timed solver call in this run.",
        target=dict(method="bounds", max_cache_depth="all"),
        references=[dict(method=method, max_cache_depth="all") for method in REFERENCES],
        methods=list(METHODS), max_cache_depth_candidates=["all"],
        graph_parameters=dict(branching=3, depth=8, width=12, seeds=list(SEEDS), sharing_probabilities=list(PROBABILITIES)),
        primary=dict(sharing_probability=1, target="bounds", reference="original",
                     cases=[case["case"] for case in cases if case["role"] == "primary"],
                     reporting="Report all three new-seed primary cases separately; do not choose a favorable case after timing."),
        controls=dict(no_reuse=0, transfer=.5),
        repeats=args.repeats, batch=args.batch, timing_seed=TIMING_SEED,
        counts=dict(object_graphs=9, distinct_ordered_expanded_inputs=3, configurations=36,
                    timing_batches=36 * args.repeats, timed_solver_calls=36 * args.repeats * args.batch),
        cases=specifications,
        source_sha256=source_hashes,
        prior_results=dict(path=str(prior), files_sha256=prior_hashes),
        descriptive_rule="Pair target minus reference within the same case and repeat. Q3<0: IQR below zero; Q1>0: IQR above zero; otherwise mixed. IQR is descriptive, not a confidence interval or significance test.",
        selection_rule="Always evaluate bounds at all layers; no winner reselection, no admission-depth tuning, and no case deletion based on confirmation results.",
        timing_boundary="Validated preconstructed input; each invoke creates a fresh table. Includes table allocation, lookup, stores, bound checks and full terminal search. Excludes generation, independent oracle, validation, trace, result checks and file output.",
    )
    protocol_path = out / "protocol.json"
    with protocol_path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    protocol_hash = sha(protocol_path)

    # First solver calls occur only after the immutable design was written.
    configs = []
    for case in cases:
        for method in METHODS:
            stats, _ = check_run(case, method)
            configs.append((case, method, stats))
    for case in cases:
        original = next(stats for item, method, stats in configs if item is case and method == "original")
        off = next(stats for item, method, stats in configs if item is case and method == "off")
        for field in ("visited_nodes", "internal_nodes", "leaf_evaluations", "cutoff_events", "skipped_child_roots", "bound_crossings"):
            if original[field] != off[field]:
                raise AssertionError((case["case"], "off/original mismatch", field))
    for seed in SEEDS:
        originals = [stats["visited_nodes"] for case, method, stats in configs if case["seed"] == seed and method == "original"]
        if len(set(originals)) != 1:
            raise AssertionError("sharing changed original Alpha-Beta visits")

    raw, rng = [], random.Random(TIMING_SEED)
    for repeat in range(args.repeats):
        order = list(enumerate(configs))
        rng.shuffle(order)
        for index, (case, method, stats) in order:
            elapsed = [check_run(case, method, stats)[1] for _ in range(args.batch)]
            raw.append(dict(config=index, case=case["case"], seed=case["seed"],
                            sharing_probability=case["sharing_probability"], role=case["role"],
                            method=method, max_cache_depth="all", repeat=repeat,
                            batch=args.batch, total_ms=statistics.mean(elapsed)))
        print(f"Confirmation block {repeat+1}/{args.repeats} verified", flush=True)

    if sha(protocol_path) != protocol_hash:
        raise AssertionError("confirmation protocol changed")
    if any(sha(path) != source_hashes[path.name] for path in source_paths):
        raise AssertionError("source changed during confirmation")
    if any(sha(prior / name) != prior_hashes[name] for name in PRIOR_FILES):
        raise AssertionError("prior exploratory results changed during confirmation")
    rows = []
    for index, (case, method, stats) in enumerate(configs):
        median, q1, q3 = quantiles([item["total_ms"] for item in raw if item["config"] == index])
        rows.append(dict(config=index, **{key: item for key, item in case.items() if key != "root"},
                         method=method, max_cache_depth="all", value=case["expected"], **stats,
                         median_total_ms=median, q1_total_ms=q1, q3_total_ms=q3))
    pairs = []
    for case in cases:
        target = next(row for row in rows if row["case"] == case["case"] and row["method"] == "bounds")
        for reference in REFERENCES:
            baseline = next(row for row in rows if row["case"] == case["case"] and row["method"] == reference)
            differences = []
            for repeat in range(args.repeats):
                target_time = next(item["total_ms"] for item in raw if item["config"] == target["config"] and item["repeat"] == repeat)
                reference_time = next(item["total_ms"] for item in raw if item["config"] == baseline["config"] and item["repeat"] == repeat)
                differences.append(target_time-reference_time)
            median, q1, q3 = quantiles(differences)
            pairs.append(dict(case=case["case"], seed=case["seed"], sharing_probability=case["sharing_probability"],
                              role=case["role"], target="bounds", reference=reference, max_cache_depth="all",
                              target_config=target["config"], reference_config=baseline["config"],
                              median_delta_ms=median, q1_delta_ms=q1, q3_delta_ms=q3,
                              faster_blocks=sum(delta < 0 for delta in differences),
                              slower_blocks=sum(delta > 0 for delta in differences),
                              equal_blocks=sum(delta == 0 for delta in differences),
                              descriptive_direction=classify(q1, q3)))
    csv_write(out / "comparison.csv", rows)
    csv_write(out / "timing_batches.csv", raw)
    csv_write(out / "paired_differences.csv", pairs)
    primary = [pair for pair in pairs if pair["role"] == "primary" and pair["reference"] == "original"]
    outcomes = dict(
        all_safe_values_correct=True, all_counters_stable=True, count_conservation=True,
        off_matches_original_visits=True, sharing_preserves_ordered_expansion=True,
        sharing_preserves_original_ab_counts=True, protocol_unchanged=True,
        source_unchanged=True, prior_results_unchanged=True,
        new_winner_selected=False, input_graphs=9, distinct_ordered_expanded_inputs=3,
        configurations=len(rows), timing_batches=len(raw), timed_solver_calls=len(raw)*args.batch,
        fixed_target=dict(method="bounds", max_cache_depth="all"),
        primary_comparisons=primary,
        primary_all_q3_below_zero=all(pair["q3_delta_ms"] < 0 for pair in primary),
        interpretation="All primary seeds and controls are retained. Descriptive paired IQR does not establish statistical significance or generalization to arbitrary game trees.",
    )
    (out / "outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version, platform=platform.platform(),
        protocol=dict(relative_path="protocol.json", sha256=protocol_hash,
                      declared_at_utc=protocol["declared_at_utc"]),
        source_sha256=source_hashes, prior_results=protocol["prior_results"],
        counts=protocol["counts"], parameters=dict(seeds=list(SEEDS), sharing_probabilities=list(PROBABILITIES),
                                                 methods=list(METHODS), max_cache_depth="all",
                                                 repeats=args.repeats, batch=args.batch, timing_seed=TIMING_SEED),
        result_files_sha256={name: sha(out/name) for name in ("comparison.csv", "timing_batches.csv", "paired_differences.csv", "outcomes.json")},
        timing_boundary=protocol["timing_boundary"],
        descriptive_rule=protocol["descriptive_rule"], selection_rule=protocol["selection_rule"],
    )
    (out / "reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} fixed configurations / {len(raw)} blocks / {len(raw)*args.batch} timed calls to {out}")
    return out


if __name__ == "__main__":
    main()
