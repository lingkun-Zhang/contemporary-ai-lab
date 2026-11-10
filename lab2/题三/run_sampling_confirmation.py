"""题三：预先固定q=8，在30棵新树上确认低点，不重新选择参数。"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from alphabeta_tree import validate_tree
from run_experiments import encode, execute, full_tree, oracle, quantiles, write_csv


BASE = Path(__file__).resolve().parent
SEEDS = tuple(range(20261101, 20261131))
BUDGETS = (0, 4, 8, 16)
TARGET = 8
REFERENCES = (0, 4, 16)
TIMING_SEED = 20261101


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_cases(seeds=SEEDS, branching=3, depth=6):
    cases, specifications = [], []
    for seed in seeds:
        root = full_tree(branching, depth, seed)
        validate_tree(root)
        value, _, nodes, leaves = oracle(root)
        tree = encode(root)
        fingerprint = hashlib.sha256(json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        case = dict(case=f"b{branching}_d{depth}_s{seed}", seed=seed, branching=branching,
                    depth=depth, root=root, expected=value, full_nodes=nodes,
                    full_leaves=leaves, input_sha256=fingerprint)
        cases.append(case)
        specifications.append({**{key: item for key, item in case.items() if key != "root"}, "tree": tree})
    return cases, specifications


def register_protocol(path, protocol):
    """Exclusive registration prevents silently replacing a prior design."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    return sha(path)


def check_run(case, budget, expected_stats=None, expected_counters=None):
    result, counters, total_ms, search_ms = execute(case, "alphabeta", f"sample_{budget}", True)
    stats = dataclasses.asdict(result.stats)
    if result.value != case["expected"]:
        raise AssertionError((case["case"], budget, "incorrect root value"))
    if stats["visited_nodes"] != stats["internal_nodes"] + stats["leaf_evaluations"]:
        raise AssertionError("visit conservation failed")
    if not 0 < stats["visited_nodes"] <= case["full_nodes"]:
        raise AssertionError("visit count outside complete tree")
    if budget == 0 and counters["ordering_probe_leaves"] != 0:
        raise AssertionError("q=0 must not inspect extra leaves")
    if expected_stats is not None and stats != expected_stats:
        raise AssertionError("search counters changed across repeats")
    if expected_counters is not None and counters != expected_counters:
        raise AssertionError("ordering counters changed across repeats")
    return stats, counters, total_ms, search_ms


def strict_lowpoint(counts, target=TARGET, neighbors=(4, 16)):
    """A within-tree hypothesis, separate from a pooled median ranking."""
    return all(counts[target] < counts[neighbor] for neighbor in neighbors)


def classification(q1, q3):
    if q3 < 0:
        return "iqr_below_zero"
    if q1 > 0:
        return "iqr_above_zero"
    return "mixed"


def summarize(rows, raw, repeats):
    lookup = {(row["case"], row["budget"]): row for row in rows}
    cases = sorted({row["case"] for row in rows})
    paired, flags = [], []
    for case in cases:
        target = lookup[(case, TARGET)]
        visits = {budget: lookup[(case, budget)]["visited_nodes"] for budget in BUDGETS}
        leaves = {budget: lookup[(case, budget)]["leaf_evaluations"] for budget in BUDGETS}
        flags.append(dict(case=case, seed=target["seed"], target_budget=TARGET,
                          visits_q0=visits[0], visits_q4=visits[4], visits_q8=visits[8], visits_q16=visits[16],
                          strict_internal_visit_lowpoint=strict_lowpoint(visits),
                          strict_internal_search_leaf_lowpoint=strict_lowpoint(leaves),
                          q8_visits_below_q0=visits[8] < visits[0]))
        for ref in REFERENCES:
            reference = lookup[(case, ref)]
            differences = []
            for repeat in range(repeats):
                target_ms = next(item["total_ms"] for item in raw if item["config"] == target["config"] and item["repeat"] == repeat)
                reference_ms = next(item["total_ms"] for item in raw if item["config"] == reference["config"] and item["repeat"] == repeat)
                differences.append(target_ms-reference_ms)
            med, q1, q3 = quantiles(differences)
            paired.append(dict(case=case, seed=target["seed"], target_budget=TARGET,
                               reference_budget=ref, target_config=target["config"], reference_config=reference["config"],
                               visit_delta=target["visited_nodes"]-reference["visited_nodes"],
                               search_leaf_delta=target["leaf_evaluations"]-reference["leaf_evaluations"],
                               probe_leaf_delta=target["ordering_probe_leaves"]-reference["ordering_probe_leaves"],
                               all_leaf_reads_delta=(target["leaf_evaluations"]+target["ordering_probe_leaves"]
                                                     -reference["leaf_evaluations"]-reference["ordering_probe_leaves"]),
                               median_paired_total_delta_ms=med, q1_paired_total_delta_ms=q1, q3_paired_total_delta_ms=q3,
                               faster_blocks=sum(delta < 0 for delta in differences),
                               slower_blocks=sum(delta > 0 for delta in differences),
                               descriptive_direction=classification(q1, q3)))
    aggregates = []
    for ref in REFERENCES:
        group = [item for item in paired if item["reference_budget"] == ref]
        for field in ("visit_delta", "search_leaf_delta", "probe_leaf_delta", "all_leaf_reads_delta", "median_paired_total_delta_ms"):
            data = [item[field] for item in group]
            med, q1, q3 = quantiles(data)
            aggregates.append(dict(target_budget=TARGET, reference_budget=ref, metric=field,
                                   independent_tree_count=len(group), median_delta=med, q1_delta=q1, q3_delta=q3,
                                   trees_target_lower=sum(item < 0 for item in data),
                                   trees_equal=sum(item == 0 for item in data),
                                   trees_target_higher=sum(item > 0 for item in data)))
    budget_summary = []
    for budget in BUDGETS:
        group = [row for row in rows if row["budget"] == budget]
        record = dict(budget=budget, independent_tree_count=len(group))
        for field in ("visited_nodes", "leaf_evaluations", "ordering_probe_leaves", "all_leaf_reads", "median_total_ms"):
            values = [(row["leaf_evaluations"]+row["ordering_probe_leaves"]) if field == "all_leaf_reads" else row[field] for row in group]
            med, q1, q3 = quantiles(values)
            record.update({f"median_{field}": med, f"q1_{field}": q1, f"q3_{field}": q3})
        budget_summary.append(record)
    return paired, flags, aggregates, budget_summary


def draw(output, summary, lowpoints, count):
    width, height = 1200, 830
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#293e4d">',
           '<text x="28" y="38" font-size="25" font-weight="bold">固定q=8：30棵新树上的独立确认</text>',
           f'<text x="28" y="69" font-size="14" fill="#566c80">q8严格少于q4和q16：{lowpoints}/{count}棵；不按本轮数据重新选参数。</text>']
    for panel, (field, title) in enumerate((("visited_nodes", "节点访问：跨30棵树的中位数及IQR"),
                                           ("median_total_ms", "完整时间：先取每树7批中位数，再跨树取中位数及IQR（ms）"))):
        y = 115+panel*350
        svg.append(f'<text x="28" y="{y}" font-size="17">{escape(title)}</text>')
        top, chart_height, left, chart_width = y+28, 190, 105, 1020
        maximum = max(item[f"q3_{field}"] for item in summary)*1.12 or 1
        for tick in range(5):
            py = top+chart_height-tick*chart_height/4
            svg.extend([f'<line x1="{left}" y1="{py}" x2="1140" y2="{py}" stroke="#dce5ed"/>',
                        f'<text x="90" y="{py+5}" text-anchor="end" font-size="13">{maximum*tick/4:.2f}</text>'])
        points = []
        for index, item in enumerate(summary):
            px = left+chart_width*(index+.5)/len(summary)
            value, low, high = item[f"median_{field}"], item[f"q1_{field}"], item[f"q3_{field}"]
            py = top+chart_height-value/maximum*chart_height
            y1, y2 = top+chart_height-low/maximum*chart_height, top+chart_height-high/maximum*chart_height
            points.append(f'{px},{py}')
            svg.extend([f'<path d="M{px},{y1}V{y2} M{px-5},{y1}h10 M{px-5},{y2}h10" stroke="#327f9b"/>',
                        f'<circle cx="{px}" cy="{py}" r="4" fill="#327f9b"/>',
                        f'<text x="{px}" y="{top+chart_height+28}" text-anchor="middle" font-size="15">q={item["budget"]}</text>'])
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#327f9b" stroke-width="2"/>')
        svg.append(f'<text x="105" y="{top+chart_height+60}" font-size="13" fill="#566c80">竖线描述树间分布；并非置信区间。所有策略包含额外取样与排序成本。</text>')
    (output/"sampling_confirmation.svg").write_text("\n".join(svg+["</g></svg>"]), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE/"results_sampling_confirmation")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.batch < 1:
        parser.error("repeats至少3，batch至少1")
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        output = output/("run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True, exist_ok=True)
    cases, specifications = prepare_cases()
    if len({case["input_sha256"] for case in cases}) != 30:
        raise AssertionError("confirmation requires30 distinct fixed new trees")
    sources = [Path(__file__), BASE/"run_experiments.py", BASE/"alphabeta_tree.py"]
    source_hashes = {path.name: sha(path) for path in sources}
    protocol = dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
                    registration_phase="Before first execute/warmup/timed call", target_budget=TARGET,
                    candidate_budgets=list(BUDGETS), reference_budgets=list(REFERENCES),
                    seeds=list(SEEDS), distribution="full_tree(3,6,seed); independent integer leaves uniform -100..100",
                    timing_seed=TIMING_SEED, repeats=args.repeats, batch=args.batch,
                    counts=dict(independent_trees=30, configurations=120, timing_batches=120*args.repeats,
                                timed_solver_calls=120*args.repeats*args.batch),
                    inputs=specifications, source_sha256=source_hashes,
                    target_metrics=["visited_nodes", "leaf_evaluations", "ordering_probe_leaves", "full total_ms"],
                    visit_lowpoint_rule="For EACH tree q8 visits must be strictly below both q4 and q16. Report count/30; a pooled median alone does not establish per-tree robustness.",
                    aggregate_rule="For work count differences use one difference per tree. For time first pair q8-reference within repeat, take median per tree, then summarize across30 trees.",
                    no_reselection="Always report q8 as the target, q0/4/16 as references; retain all30 seeds and do not switch to a new winner.",
                    timing_boundary="Prepared and prevalidated tree; execute includes order preparation, extra sampling, sorting and full alpha-beta. Excludes construction, oracle, validation, result checks, trace and file output.")
    protocol_hash = register_protocol(output/"protocol.json", protocol)
    configurations = []
    for case in cases:
        for budget in BUDGETS:
            stats, counters, _, _ = check_run(case, budget)
            configurations.append((case, budget, stats, counters))
    raw, rng = [], random.Random(TIMING_SEED)
    for repeat in range(args.repeats):
        order = list(enumerate(configurations)); rng.shuffle(order)
        for config, (case, budget, stats, counters) in order:
            times, search_times = [], []
            for _ in range(args.batch):
                _, _, total_ms, search_ms = check_run(case, budget, stats, counters)
                times.append(total_ms); search_times.append(search_ms)
            raw.append(dict(config=config, case=case["case"], seed=case["seed"], budget=budget,
                            repeat=repeat, batch=args.batch, total_ms=statistics.mean(times),
                            search_ms=statistics.mean(search_times)))
        print(f"Sampling confirmation block {repeat+1}/{args.repeats} verified", flush=True)
    if sha(output/"protocol.json") != protocol_hash:
        raise AssertionError("preregistered protocol changed")
    if any(sha(path) != source_hashes[path.name] for path in sources):
        raise AssertionError("source changed during confirmation")
    rows = []
    for config, (case, budget, stats, counters) in enumerate(configurations):
        med, q1, q3 = quantiles([item["total_ms"] for item in raw if item["config"] == config])
        rows.append(dict(config=config, **{key: item for key, item in case.items() if key != "root"},
                         budget=budget, value=case["expected"], **stats, **counters,
                         median_total_ms=med, q1_total_ms=q1, q3_total_ms=q3))
    paired, flags, aggregates, budget_summary = summarize(rows, raw, args.repeats)
    write_csv(output/"comparison.csv", rows)
    write_csv(output/"timing_batches.csv", raw)
    write_csv(output/"paired_differences.csv", paired)
    write_csv(output/"per_tree_lowpoints.csv", flags)
    write_csv(output/"aggregate_summary.csv", aggregates)
    write_csv(output/"budget_summary.csv", budget_summary)
    lowpoints = sum(item["strict_internal_visit_lowpoint"] for item in flags)
    draw(output, budget_summary, lowpoints, len(cases))
    outcomes = dict(all_values_correct=True, all_counters_stable=True, protocol_unchanged=True,
                    source_unchanged=True, target_budget=TARGET, new_winner_selected=False,
                    independent_trees=len(cases), strict_internal_visit_lowpoint_trees=lowpoints,
                    strict_internal_visit_lowpoint_fraction=lowpoints/len(cases),
                    q8_lower_than_q0_visit_trees=sum(item["q8_visits_below_q0"] for item in flags),
                    interpretation="Descriptive confirmation of a fixed target; evidence against a universal optimum is retained. IQR is not confidence/significance, and timing blocks are not independent input trees.")
    (output/"outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(), protocol=dict(relative_path="protocol.json", sha256=protocol_hash),
                    source_sha256=source_hashes, counts=protocol["counts"], seeds=list(SEEDS),
                    parameters=dict(target_budget=TARGET, candidates=list(BUDGETS), references=list(REFERENCES),
                                    repeats=args.repeats, batch=args.batch, timing_seed=TIMING_SEED),
                    result_files_sha256={name: sha(output/name) for name in ("comparison.csv", "timing_batches.csv", "paired_differences.csv", "per_tree_lowpoints.csv", "aggregate_summary.csv", "budget_summary.csv", "outcomes.json")},
                    interpretation=outcomes["interpretation"], aggregate_rule=protocol["aggregate_rule"],
                    timing_boundary=protocol["timing_boundary"])
    (output/"reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved120 fixed configurations / {len(raw)} batches / {len(raw)*args.batch} timed calls to {output}")
    return output


if __name__ == "__main__":
    main()
