"""题一独立计时确认：固定 k=1，比较 k=0/2，不重新调参。

三个原工作负载和全部查询从已保存调参元数据读取。协议先于预热写定，
每批独立建表并完整运行40查询，BFS与路径核验在计时外。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from run_experiments import Case, bfs_distance, svg_start, write_csv
from run_landmark_tuning import distribution, run_batch, validate_batch


CANDIDATES = (0, 1, 2)
CASES = ("long_wall", "perfect_maze", "open")
REFERENCES = (0, 2)
DEFAULT_REPEATS = 15
DEFAULT_SEED = 20261030
COLORS = {"long_wall": "#327c99", "perfect_maze": "#c48136", "open": "#69915c"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_output(requested):
    """首跑用指定目录；复跑另存，既有确认结果不覆盖。"""
    output = Path(requested)
    if output.exists() and any(output.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        output = output / ("rerun_" + stamp)
    output.mkdir(parents=True, exist_ok=False if output.exists() is False else True)
    return output


def load_workloads(source):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    for name, expected in data.get("source_sha256", {}).items():
        current_source = (root / name).resolve()
        if not current_source.is_relative_to(root):
            raise ValueError("历史来源的源码路径必须位于本题目录")
        if current_source.is_file() and sha256(current_source) != expected:
            raise ValueError(f"历史调参源码与当前文件不一致，不能称同实现确认：{name}")
    lookup = {row["case"]: row for row in data["workloads"]}
    workloads = []
    for slug in CASES:
        saved = lookup[slug]
        pairs = [tuple(tuple(point) for point in pair) for pair in saved["queries"]]
        if len(pairs) != 40 or len(set(pairs)) != 40 or any(a == b for a, b in pairs):
            raise ValueError("确认实验须精确复用三组原40个不同的有序查询")
        grid = [list(row) for row in saved["grid"]]
        case = Case(slug, saved["case_name"], grid, pairs[0][0], pairs[0][1])
        selected = tuple(tuple(node) for node in saved["max_k_landmarks"])
        if len(selected) < max(CANDIDATES):
            raise ValueError("来源须包含至少两个固定地标，供独立建表前缀核验")
        workloads.append((case, pairs, selected))
    return data, workloads


def protocol_design(source, workloads, repeats, seed):
    root = Path(__file__).resolve().parent
    names = ("maze_astar.py", "landmark_heuristic.py", "run_experiments.py",
             "run_landmark_tuning.py", "run_landmark_confirmation.py")
    return dict(
        candidates=list(CANDIDATES), target_k=1, references=list(REFERENCES),
        repeats=repeats, seed=seed, queries_per_map=40, cases=list(CASES),
        primary_comparisons="paired whole-batch delta T(k=1)-T(k=0), and T(k=1)-T(k=2)",
        descriptive_rules={
            "lower_is_better": True,
            "median_below_zero": "negative paired median favors the fixed k=1 candidate",
            "both_iqr_below_zero": "Q3 of both paired differences below zero describes consistent local-low-point support within this workload",
            "overlap_or_reversal": "retain near ties, losses and ranking changes; do not replace target k=1 after seeing confirmation",
            "inference_limit": "timing repetition IQR, not a confidence interval or independent-map significance test",
        },
        timing=dict(total="one complete wall-clock interval from independent build(k) through all 40 A* results",
                    warmup_batches_per_candidate=1, order="seeded shuffle within each repetition block",
                    oracle_and_validation="outside every timed interval", gc="interpreter default"),
        reference_source=dict(path=str(Path(source).resolve().relative_to(root))
                              if Path(source).resolve().is_relative_to(root) else str(Path(source).resolve()),
                              sha256=sha256(source)),
        source_sha256={name: sha256(root / name) for name in names},
        workloads=[dict(case=case.slug, case_name=case.name, grid=case.grid,
                        queries=pairs, expected_landmark_prefix=selected[:max(CANDIDATES)])
                   for case, pairs, selected in workloads],
    )


def write_protocol(path, design):
    """独占创建协议，绝不事后覆盖已登记的候选或比较规则。"""
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
                    registration_phase="before any confirmation warmup or timed solver call", design=design)
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
    return sha256(path)


def fixed_pair_summary(raw):
    pairs = []
    for slug in dict.fromkeys(row["case"] for row in raw):
        group = [row for row in raw if row["case"] == slug]
        lookup = {(row["repeat"], row["k"]): row["total_ms"] for row in group}
        blocks = sorted({row["repeat"] for row in group})
        for reference in REFERENCES:
            differences = [lookup[(block, 1)] - lookup[(block, reference)] for block in blocks]
            row = dict(case=slug, target_k=1, reference_k=reference, repeats=len(blocks),
                       meaning="target k1 whole-batch ms minus fixed reference whole-batch ms",
                       target_faster_blocks=sum(value < 0 for value in differences),
                       target_slower_blocks=sum(value > 0 for value in differences),
                       equal_blocks=sum(value == 0 for value in differences),
                       raw_delta_ms=json.dumps(differences),
                       **{key + "_delta_ms": value for key, value in distribution(differences).items()})
            row["paired_iqr_entirely_below_zero"] = row["q3_delta_ms"] < 0
            row["paired_iqr_crosses_zero"] = row["q1_delta_ms"] <= 0 <= row["q3_delta_ms"]
            pairs.append(row)
    return pairs


def fixed_candidate_summary(raw):
    output = []
    for slug in dict.fromkeys(row["case"] for row in raw):
        group = [row for row in raw if row["case"] == slug]
        baseline = {row["repeat"]: row["total_ms"] for row in group if row["k"] == 0}
        for k in CANDIDATES:
            rows = [row for row in group if row["k"] == k]
            item = dict(case=slug, case_name=rows[0]["case_name"], k=k, repeats=len(rows),
                        queries=40, expanded_nodes_sum=rows[0]["expanded_nodes_sum"],
                        distance_table_entries=rows[0]["distance_table_entries"])
            for name, values in (
                ("total_ms", [row["total_ms"] for row in rows]),
                ("preprocess_ms", [row["preprocess_ms"] for row in rows]),
                ("paired_speedup", [baseline[row["repeat"]] / row["total_ms"] for row in rows]),
            ):
                item.update({name + "_" + key: value for key, value in distribution(values).items()})
            output.append(item)
    return output


def confirmation_flags(summary, paired):
    rows = []
    for slug in CASES:
        values = {row["k"]: row for row in summary if row["case"] == slug}
        comparisons = [row for row in paired if row["case"] == slug]
        rows.append(dict(
            case=slug, fixed_target_k=1,
            target_total_median_below_both_refs=all(values[1]["total_ms_median"] < values[k]["total_ms_median"] for k in REFERENCES),
            target_paired_median_below_both_refs=all(row["median_delta_ms"] < 0 for row in comparisons),
            both_paired_iqr_below_zero=all(row["paired_iqr_entirely_below_zero"] for row in comparisons),
            any_paired_iqr_crosses_zero=any(row["paired_iqr_crosses_zero"] for row in comparisons),
            interpretation="fixed candidate confirmation on unchanged workloads; no winner reselection",
        ))
    return rows


def draw_totals(path, summary):
    width, panel_h, top, left, right = 1200, 245, 110, 95, 45
    parts = svg_start(width, top + panel_h * len(CASES) + 30, "题一独立计时确认：预先固定 k=1，与两侧候选比较")
    repeats = summary[0]["repeats"]
    parts.append(f'<text class="note" x="24" y="60">三组地图与原40查询全部不变；点为{repeats}批中位数，竖线为IQR；没有根据确认结果重新选冠军。</text>')
    for index, slug in enumerate(CASES):
        group = [row for row in summary if row["case"] == slug]
        ymax = max(1, max(row["total_ms_q3"] for row in group) * 1.15)
        ytop, plot_h, plot_w = top + index * panel_h, 155, width - left - right
        xpos = lambda k: left + plot_w * (k + .5) / 3
        ypos = lambda value: ytop + plot_h * (1 - value / ymax)
        parts.append(f'<text class="label" x="24" y="{ytop-18}">{escape(group[0]["case_name"])}：整批总耗时 ms（越低越好）；横轴固定地标数</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = ypos(value)
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e0e7ec"/>')
            parts.append(f'<text class="note" x="{left-10}" y="{y+4}" text-anchor="end">{value:.2f}</text>')
        points = []
        for row in group:
            x, y = xpos(row["k"]), ypos(row["total_ms_median"])
            low, high = ypos(row["total_ms_q1"]), ypos(row["total_ms_q3"])
            color = COLORS[slug]
            points.append((x, y))
            parts.append(f'<line x1="{x}" y1="{low}" x2="{x}" y2="{high}" stroke="{color}" stroke-width="2"/>')
            for cap in (low, high):
                parts.append(f'<line x1="{x-5}" y1="{cap}" x2="{x+5}" y2="{cap}" stroke="{color}"/>')
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            parts.append(f'<text class="note" x="{x}" y="{ytop+plot_h+22}" text-anchor="middle">{row["k"]}</text>')
        parts.append(f'<polyline fill="none" stroke="{COLORS[slug]}" stroke-width="2" points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + '"/>')
        parts.append(f'<text class="note" x="{left}" y="{ytop+plot_h+47}">按预注册比较分析 k=1 相对0和2的逐块差；IQR是计时描述，不是地图置信区间。</text>')
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def run(output, repeats=DEFAULT_REPEATS, seed=DEFAULT_SEED, source=None):
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats须为正整数")
    if type(seed) is not int:
        raise ValueError("seed须为整数")
    root = Path(__file__).resolve().parent
    source = Path(source) if source is not None else root / "results/landmark_tuning/reproducibility.json"
    historical, workloads = load_workloads(source)
    actual_output = choose_output(output)
    design = protocol_design(source, workloads, repeats, seed)
    protocol_path = actual_output / "protocol.json"
    protocol_hash = write_protocol(protocol_path, design)
    raw, details, orders = [], [], {}
    executions = 0
    for case_index, (case, pairs, selected) in enumerate(workloads):
        # 参考BFS重新计算，不能仅相信原调参文件中的expected_costs。
        oracle = [bfs_distance(Case(case.slug, case.name, case.grid, start, goal)) for start, goal in pairs]
        if any(cost < 0 for cost in oracle):
            raise AssertionError("原固定工作负载应全部可达")
        rng = random.Random(seed + 10000 * (case_index + 1))
        warmup = list(CANDIDATES)
        rng.shuffle(warmup)
        if not protocol_path.is_file() or sha256(protocol_path) != protocol_hash:
            raise AssertionError("确认协议必须在首个预热前写定且未变")
        for k in warmup:
            batch = run_batch(case.grid, pairs, k)
            validate_batch(case, pairs, oracle, batch, selected[:k])
        orders[case.slug] = dict(warmup=warmup, repeats=[])
        signatures = {}
        for repeat in range(1, repeats + 1):
            order = list(CANDIDATES)
            rng.shuffle(order)
            orders[case.slug]["repeats"].append(order)
            for k in order:
                batch = run_batch(case.grid, pairs, k)
                validate_batch(case, pairs, oracle, batch, selected[:k])
                results = batch["results"]
                signature = [(answer.cost, answer.path, answer.stats) for answer in results]
                if k in signatures and signature != signatures[k]:
                    raise AssertionError("固定k的路径/统计在确认重复块间不稳定")
                signatures[k] = signature
                executions += len(pairs)
                raw.append(dict(case=case.slug, case_name=case.name, repeat=repeat, k=k, queries=len(pairs),
                                total_ms=batch["total_ms"], preprocess_ms=batch["preprocess_ms"],
                                distance_table_entries=batch["distance_table_entries"],
                                expanded_nodes_sum=sum(answer.stats["expanded_nodes"] for answer in results),
                                generated_nodes_sum=sum(answer.stats["generated_nodes"] for answer in results)))
                if repeat == 1:
                    for qid, ((start, goal), expected, answer) in enumerate(zip(pairs, oracle, results), 1):
                        details.append(dict(case=case.slug, k=k, query_id=qid, start=json.dumps(start),
                                            goal=json.dumps(goal), expected_cost=expected, returned_cost=answer.cost,
                                            path=json.dumps(answer.path), **answer.stats))
        print(f"独立确认：{case.name}，固定k=0/1/2，{repeats}整批×40查询均核验。", flush=True)
    if sha256(protocol_path) != protocol_hash:
        raise AssertionError("确认协议在采集后发生变化")
    summary, paired = fixed_candidate_summary(raw), fixed_pair_summary(raw)
    flags = confirmation_flags(summary, paired)
    write_csv(actual_output / "batch_raw.csv", raw)
    write_csv(actual_output / "candidate_summary.csv", summary)
    write_csv(actual_output / "fixed_pairs.csv", paired)
    write_csv(actual_output / "query_details.csv", details)
    write_csv(actual_output / "confirmation_flags.csv", flags)
    draw_totals(actual_output / "confirmation_total.svg", summary)
    outcome = dict(input_maps=len(workloads), queries_per_map=40, candidates=list(CANDIDATES),
                   fixed_target_k=1, fixed_references=list(REFERENCES), repeats=repeats,
                   timed_batches=len(raw), measured_query_executions=executions,
                   unique_map_query_k_configurations=len(details),
                   all_shortest_costs_and_paths_verified=True, same_inputs_as_prior_tuning=True,
                   no_post_confirmation_candidate_reselection=True, protocol_unchanged=True,
                   path_and_stats_stable_across_repeats=True, confirmation_by_case=flags)
    (actual_output / "outcomes.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(), processor=platform.processor(),
                    protocol=dict(relative_path="protocol.json", sha256=protocol_hash),
                    parameters=dict(candidates=CANDIDATES, fixed_target_k=1, references=REFERENCES,
                                    repeats=repeats, seed=seed, cases=CASES, queries=40),
                    reference_source=design["reference_source"], source_sha256=design["source_sha256"],
                    timing=design["timing"], timing_orders=orders, workloads=design["workloads"],
                    prior_study_generated_at_utc=historical.get("generated_at_utc"),
                    evidence_scope="new timing blocks for unchanged fixed workloads; not independent new maps, not global parameter validation")
    (actual_output / "reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成{len(raw)}个独立确认整批、{executions}次查询核验；结果：{actual_output.resolve()}", flush=True)
    return outcome


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=positive, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source-metadata", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results/landmark_confirmation")
    args = parser.parse_args()
    try:
        run(args.output_dir, args.repeats, args.seed, args.source_metadata)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
