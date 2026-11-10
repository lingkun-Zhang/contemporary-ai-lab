"""BFS 性能对照：普通队列、匹配堆内核与现有 A* 的完整调用成本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from bfs_baseline import bfs_search, uniform_search
from maze_astar import astar_search
from run_experiments import Case, bfs_distance, build_cases, mark_endpoints, validate_result, write_csv


VARIANTS = (
    ("bfs_deque", "普通队列 BFS", "#437eaa"),
    ("uniform_heap", "匹配内核 / 堆", "#8674aa"),
    ("astar_zero", "原 A* / h=0", "#7b8c99"),
    ("astar_manhattan_fifo", "曼哈顿 / FIFO", "#3e9f91"),
    ("astar_manhattan_low_h", "曼哈顿 / low_h", "#d38a32"),
)


def solve(case, variant):
    args = case.start, case.goal, case.grid
    if variant == "bfs_deque":
        return bfs_search(*args)
    if variant == "uniform_heap":
        return uniform_search(*args, frontier="heap")
    if variant == "astar_zero":
        return astar_search(*args, heuristic="zero", tie_break="fifo")
    return astar_search(*args, heuristic="manhattan",
                        tie_break="low_h" if variant.endswith("low_h") else "fifo")


def make_cases(seeds, seed):
    cases = [("base", case) for case in build_cases(seed)]
    for n in (30, 60, 100):
        for p in (.15, .30):
            for index in range(seeds):
                case_seed = seed + n*100000 + round(p*100)*1000 + index
                rng = random.Random(case_seed)
                grid = [["X" if rng.random() < p else "." for _ in range(n)] for _ in range(n)]
                start, goal = (0, 0), (n-1, n-1)
                case = Case(f"bfs_scale_{n}_{round(p*100)}_{index}", f"{n}×{n} / p={p}",
                            mark_endpoints(grid, start, goal), start, goal, case_seed, p)
                cases.append(("scale", case))
    return cases


def quartiles(data):
    q1, _, q3 = statistics.quantiles(data, n=4, method="inclusive") if len(data)>1 else data*3
    return {"mean": statistics.mean(data), "median": statistics.median(data), "q1": q1, "q3": q3,
            "iqr": q3-q1, "min": min(data), "max": max(data)}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["suite"], row["rows"], row["density_requested"], row["reachable"], row["variant"])
        groups[key].append(row)
    summary = []
    for key, group in groups.items():
        item = dict(zip(("suite", "rows", "density_requested", "reachable", "variant"), key))
        item["n_maps"] = len(group)
        for metric in ("expanded_nodes", "max_frontier", "time_median_ms", "time_ratio_to_bfs", "node_ratio_to_bfs"):
            item.update({metric+"_"+name: value for name, value in quartiles([r[metric] for r in group]).items()})
        summary.append(item)
    return summary


def save_summary_svg(path, summary, metric, title):
    panels = []
    for reachable in (True, False):
        keys = sorted({(r["rows"], float(r["density_requested"])) for r in summary
                       if r["suite"] == "scale" and r["reachable"] == reachable})
        if keys:
            panels.append((reachable, keys))
    width, ph = 1220, 320
    height = 105 + len(panels)*ph
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;fill:#27394a}.title{font-size:22px;font-weight:700}.label{font-size:13px}.note{font-size:12px;fill:#536575}</style>',
             f'<rect width="{width}" height="{height}" fill="white"/>',
             f'<text class="title" x="24" y="34">{escape(title)}</text>',
             '<text class="note" x="24" y="60">同一地图逐图除以 BFS，再汇总中位数；竖线为地图间 Q1–Q3，不是计时置信区间。</text>']
    lookup = {(r["rows"], float(r["density_requested"]) if r["density_requested"] != "" else -1,
               r["reachable"], r["variant"]): r for r in summary if r["suite"] == "scale"}
    for panel_index, (reachable, keys) in enumerate(panels):
        top, left, plot_w, plot_h = 105+panel_index*ph, 74, 1110, 180
        parts.append(f'<text class="label" x="24" y="{top-18}">{"有解" if reachable else "无解"}地图；每组有效样本量见 CSV</text>')
        maximum = max(1, max(lookup[(n,p,reachable,key)][metric+"_q3"]
                              for n,p in keys for key,_,_ in VARIANTS))*1.12
        y = lambda value: top + plot_h*(1-value/maximum)
        for tick in range(5):
            value = maximum*tick/4
            parts.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{left+plot_w}" y2="{y(value):.2f}" stroke="#e1e8ed"/>')
            parts.append(f'<text class="note" x="{left-8}" y="{y(value)+4:.2f}" text-anchor="end">{value:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{y(1):.2f}" x2="{left+plot_w}" y2="{y(1):.2f}" stroke="#77899b" stroke-dasharray="4 4"/>')
        group_width = plot_w/len(keys)
        for group_index, (n,p) in enumerate(keys):
            cx = left + group_width*(group_index+.5)
            parts.append(f'<text class="note" x="{cx:.2f}" y="{top+plot_h+21}" text-anchor="middle">{n}×{n}, p={p:.2f}</text>')
            for variant_index, (key,_,color) in enumerate(VARIANTS):
                row = lookup[(n,p,reachable,key)]
                bx = cx + (variant_index-2)*17
                median = row[metric+"_median"]
                parts.append(f'<rect x="{bx-6:.2f}" y="{y(median):.2f}" width="12" height="{plot_h-(y(median)-top):.2f}" fill="{color}"/>')
                parts.append(f'<line x1="{bx:.2f}" x2="{bx:.2f}" y1="{y(row[metric+"_q1"]):.2f}" y2="{y(row[metric+"_q3"]):.2f}" stroke="#243746"/>')
        for i, (_,label,color) in enumerate(VARIANTS):
            x = 45+i*235
            parts.append(f'<rect x="{x}" y="{top+plot_h+48}" width="12" height="12" fill="{color}"/>')
            parts.append(f'<text class="note" x="{x+18}" y="{top+plot_h+59}">{escape(label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def run(output=None, seeds=20, repeats=5, batch=5, seed=20261028):
    output = Path(output) if output is not None else Path(__file__).resolve().parent / "results/bonus_bfs"
    output.mkdir(parents=True, exist_ok=True)
    cases, rows, maps = make_cases(seeds, seed), [], []
    for number, (suite, case) in enumerate(cases, 1):
        expected = bfs_distance(case)
        order = [key for key,_,_ in VARIANTS]
        random.Random(seed+number).shuffle(order)
        case_rows, results = [], {}
        for variant in order:
            result = solve(case, variant)
            validate_result(case, result, expected)
            solve(case, variant)
            samples = []
            for _ in range(repeats):
                started = perf_counter_ns()
                for _ in range(batch):
                    timed = solve(case, variant)
                samples.append((perf_counter_ns()-started)/batch/1e6)
                if timed.cost != expected or timed.path != result.path:
                    raise AssertionError("重复计时结果改变")
            results[variant] = result
            case_rows.append({"suite": suite, "case": case.slug, "case_name": case.name,
                              "rows": len(case.grid), "cols": len(case.grid[0]),
                              "density_requested": case.density if case.density is not None else "",
                              "seed": case.seed if case.seed is not None else "", "variant": variant,
                              "reachable": expected >= 0, "optimal_steps": expected, "bfs_verified": True,
                              **result.stats, "time_median_ms": statistics.median(samples),
                              "time_min_ms": min(samples), "timing_batches_ms": json.dumps(samples)})
        matched = results["uniform_heap"]
        bfs = results["bfs_deque"]
        if bfs.path != matched.path or bfs.stats != matched.stats:
            raise AssertionError("受控容器实验搜索行为不一致")
        if bfs.path != results["astar_zero"].path or bfs.stats != results["astar_zero"].stats:
            raise AssertionError("本题单位代价下 UCS/FIFO 与 BFS 行为不一致")
        baseline_time = next(r["time_median_ms"] for r in case_rows if r["variant"] == "bfs_deque")
        for row in case_rows:
            row["time_ratio_to_bfs"] = row["time_median_ms"]/baseline_time
            row["node_ratio_to_bfs"] = row["expanded_nodes"]/bfs.stats["expanded_nodes"] if bfs.stats["expanded_nodes"] else 1
        rows.extend(case_rows)
        maps.append({"suite": suite, "case": case.slug, "grid": ["".join(row) for row in case.grid],
                     "start": case.start, "goal": case.goal, "seed": case.seed,
                     "density_requested": case.density, "bfs_steps": expected})
        if number % 20 == 0 or number == len(cases):
            print(f"BFS 性能对照：完成 {number}/{len(cases)} 张地图。", flush=True)
    summary = aggregate(rows)
    write_csv(output/"comparison.csv", rows)
    write_csv(output/"summary.csv", summary)
    save_summary_svg(output/"time_ratio.svg", summary, "time_ratio_to_bfs", "完整搜索耗时 / 普通队列 BFS（比值越小越省时）")
    save_summary_svg(output/"node_ratio.svg", summary, "node_ratio_to_bfs", "扩展节点 / 普通队列 BFS（比值越小搜索量越少）")
    source = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                     for name in ("maze_astar.py", "bfs_baseline.py", "run_bfs_study.py", "run_experiments.py")}
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "source_sha256": source_hashes,
                "parameters": {"seed": seed, "seeds": seeds, "repeats": repeats, "batch": batch},
                "methodology": {"unit_of_sample": "one map, not one timing call", "timing_unit": "ms per complete search",
                                "includes": "same grid validation; search; stats; complete path reconstruction; solve dispatch",
                                "excludes": "BFS reference, maps generation, plotting, file IO; trace disabled",
                                "warmups": 1, "timing_order": "seeded randomized variants per map", "gc": "default interpreter behavior",
                                "matched_control": "uniform_search shares all code except the frontier container adapter",
                                "ratios": "variant / deque BFS per map, then aggregate; below one means less",
                                "memory_metric": "frontier record count, not total bytes; no claim of causal isolation across different codebases"},
                "maps": maps}
    (output/"reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    basic = [row for row in rows if row["suite"] == "base" and row["case"] in ("sample", "open40", "detour", "unreachable")]
    scale_rows = [row for row in rows if row["suite"] == "scale"]
    outcomes = {"n_maps": len(cases), "n_configurations": len(rows), "n_extra_maps": len(cases)-14,
                "all_paths_bfs_verified": True, "matched_frontier_behavior_identical": True,
                "representative_results": basic, "summary": summary,
                "pooled_paired_time_ratios": {
                    str(reachable): {key: quartiles([row["time_ratio_to_bfs"] for row in scale_rows
                                                     if row["reachable"] == reachable and row["variant"] == key])
                                     for key,_,_ in VARIANTS
                                     if any(row["reachable"] == reachable and row["variant"] == key for row in scale_rows)}
                    for reachable in (True, False)}}
    (output/"outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"BFS study: {len(rows)} configurations passed; saved to {output.resolve()}", flush=True)
    return outcomes


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=positive, default=20)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--batch", type=positive, default=5)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"results/bonus_bfs")
    args = parser.parse_args()
    run(args.output_dir, args.seeds, args.repeats, args.batch, args.seed)


if __name__ == "__main__":
    main()
