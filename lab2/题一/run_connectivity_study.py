"""连通分量预检探究：静态快照、失败工作负载与预处理摊销。

所有方案每次均保留一次整图验证，最短路径由独立 BFS 核对。
累计耗时为实际建表时间加逐查询计时中位数之和，不是连续整段实测。
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from time import perf_counter_ns

from connectivity_index import build_connectivity_index
from maze_astar import astar_search
from run_experiments import Case, svg_start, validate_result, write_csv


LABELS = {"open": "全连通开阔图", "two_regions": "两大区域隔墙", "small_rooms": "许多小隔间"}


def make_maps(n):
    open_grid = [["."] * n for _ in range(n)]
    separated = [["."] * n for _ in range(n)]
    for row in range(n):
        separated[row][n // 2] = "X"
    rooms = [["X" if row % 4 == 3 or col % 4 == 3 else "." for col in range(n)] for row in range(n)]
    return {"open": open_grid, "two_regions": separated, "small_rooms": rooms}


def independent_distances(grid, start):
    """独立 oracle；不使用索引标签，不计入搜索时间。"""
    distances, frontier = {start: 0}, deque([start])
    while frontier:
        r, c = frontier.popleft()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nr, nc = neighbor
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != "X" and neighbor not in distances):
                distances[neighbor] = distances[(r, c)] + 1
                frontier.append(neighbor)
    return distances


def workload(index, queries, unsolved_fraction, seed):
    """通过真实分量抽样控制任务混合，不把任务改变归因于算法。"""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for node, component in index.labels.items():
        groups[component].append(node)
    groups = dict(groups)
    components = sorted(groups)
    same_components = [component for component in components if len(groups[component]) >= 2]
    failed = int(round(queries * unsolved_fraction))
    if failed and len(components) < 2:
        raise ValueError("单连通分量地图无法产生异分量查询")
    pairs = []
    for is_unsolved in [True] * failed + [False] * (queries - failed):
        if is_unsolved:
            first, second = rng.sample(components, 2)
            start, goal = rng.choice(groups[first]), rng.choice(groups[second])
        else:
            component = rng.choice(same_components)
            start, goal = rng.sample(groups[component], 2)
        pairs.append((start, goal, is_unsolved))
    rng.shuffle(pairs)
    return pairs


def timed(call, repeats, batch):
    result = call()
    call()  # 独立的一次无追踪预热。
    times = []
    for _ in range(repeats):
        before = perf_counter_ns()
        for _ in range(batch):
            final = call()
        times.append((perf_counter_ns() - before) / batch / 1e6)
        if final.cost != result.cost or final.path != result.path:
            raise AssertionError("同一配置重复运行产生不同结果")
    return result, {"query_median_ms": statistics.median(times), "query_min_ms": min(times),
                    "query_batches_ms": json.dumps(times)}


def line_svg(path, title, panels, note):
    width, panel_h = 1200, 300
    height = 90 + panel_h * len(panels) + 35
    parts = svg_start(width, height, title)
    parts.append(f'<text class="note" x="24" y="59">{escape(note)}</text>')
    for i, panel in enumerate(panels):
        ytop, left, right, plot_h = 112 + i * panel_h, 90, 35, 175
        plot_w = width - left - right
        xs, series = panel["x"], panel["series"]
        xmin, xmax = min(xs), max(xs)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
        values = [value for _, items, _ in series for value in items]
        ymax = max(max(values) * 1.10, .001)
        parts.append(f'<text class="label" x="24" y="{ytop-20}">{escape(panel["title"])}</text>')
        for tick in range(5):
            value = ymax * tick / 4
            y = ytop + plot_h * (1 - tick / 4)
            text = f"{value:.2f}" if ymax < 20 else f"{value:.0f}"
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e2e8ee"/>')
            parts.append(f'<text class="note" x="{left-8}" y="{y+4}" text-anchor="end">{text}</text>')
        for x in xs:
            px = left + (x - xmin) / (xmax - xmin) * plot_w
            parts.append(f'<text class="note" x="{px}" y="{ytop+plot_h+21}" text-anchor="middle">{x:g}</text>')
        for j, (name, ys, color) in enumerate(series):
            points = [(left + (x-xmin)/(xmax-xmin)*plot_w, ytop + plot_h*(1-y/ymax)) for x, y in zip(xs, ys)]
            coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
            lx, ly = 110 + j * 330, ytop + plot_h + 48
            parts.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+24}" y2="{ly}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="note" x="{lx+31}" y="{ly+4}">{escape(name)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def study(output, size=80, queries=100, repeats=5, batch=3, seed=20261028):
    output.mkdir(parents=True, exist_ok=True)
    raw, summaries, cumulative, workloads_meta = [], [], [], []
    maps = make_maps(size)
    checkpoints = sorted(set(q for q in (1, 5, 10, 20, 50, 100, queries) if q <= queries))
    for map_number, (name, grid) in enumerate(maps.items()):
        ratios = (0,) if name == "open" else (0, .25, .5, .75, 1)
        for ratio in ratios:
            # 每个混合负载独立新建并收费，不借用其他组已构建的索引。
            index = build_connectivity_index(grid)
            workload_seed = seed + map_number * 10000 + round(ratio * 100)
            pairs = workload(index, queries, ratio, workload_seed)
            workload_key = f"{name}_{round(ratio * 100)}"
            metadata = {"workload": workload_key, "map": name, "unsolved_fraction_requested": ratio,
                        "seed": workload_seed, "index": index.metadata(),
                        "pairs": [{"start": start, "goal": goal, "expected_unsolved": unsolved}
                                  for start, goal, unsolved in pairs]}
            workloads_meta.append(metadata)
            group = []
            order_rng = random.Random(workload_seed + 721)
            distance_cache = {}
            for query_id, (start, goal, expected_unsolved) in enumerate(pairs, 1):
                if start not in distance_cache:
                    distance_cache[start] = independent_distances(index.grid, start)
                expected = distance_cache[start].get(goal, -1)
                if (expected < 0) != expected_unsolved:
                    raise AssertionError("生成的混合负载与独立 BFS 不一致")
                case = Case(workload_key, LABELS[name], index.grid, start, goal)
                order = ["baseline", "indexed"]
                order_rng.shuffle(order)
                for variant in order:
                    if variant == "baseline":
                        call = lambda s=start, g=goal: astar_search(s, g, index.grid, tie_break="low_h")
                    else:
                        call = lambda s=start, g=goal: index.query(s, g)
                    result, times = timed(call, repeats, batch)
                    validate_result(case, result, expected)
                    row = {"workload": workload_key, "map": name, "map_name": LABELS[name],
                           "size": size, "query_id": query_id, "unsolved_fraction_requested": ratio,
                           "start": json.dumps(start), "goal": json.dumps(goal), "variant": variant,
                           "reachable": expected >= 0, "optimal_steps": expected, "bfs_verified": True,
                           "start_component_size": index.component_sizes[index.labels[start]],
                           "component_count": index.component_count, "label_entries": index.label_entries,
                           "preprocess_ms": index.preprocess_ms if variant == "indexed" else 0,
                           "single_query_total_ms": times["query_median_ms"] + (index.preprocess_ms if variant == "indexed" else 0),
                           **{key: result.stats[key] for key in ("expanded_nodes", "generated_nodes", "popped_nodes", "max_frontier")},
                           "precheck_rejected": result.stats.get("precheck_rejected", 0), **times}
                    raw.append(row)
                    group.append(row)
            baseline = sorted((row for row in group if row["variant"] == "baseline"), key=lambda row: row["query_id"])
            indexed = sorted((row for row in group if row["variant"] == "indexed"), key=lambda row: row["query_id"])
            first_crossing = ""
            for q in range(1, queries + 1):
                baseline_total = sum(row["query_median_ms"] for row in baseline[:q])
                indexed_query = sum(row["query_median_ms"] for row in indexed[:q])
                indexed_total = index.preprocess_ms + indexed_query
                if first_crossing == "" and indexed_total <= baseline_total:
                    first_crossing = q
                if q in checkpoints:
                    cumulative.append({"workload": workload_key, "map": name, "map_name": LABELS[name],
                                       "unsolved_fraction_requested": ratio, "queries": q,
                                       "preprocess_ms": index.preprocess_ms,
                                       "baseline_total_ms": baseline_total, "indexed_query_sum_ms": indexed_query,
                                       "indexed_total_ms": indexed_total,
                                       "indexed_over_baseline_time_ratio": indexed_total / baseline_total,
                                       "baseline_expanded_sum": sum(row["expanded_nodes"] for row in baseline[:q]),
                                       "indexed_expanded_sum": sum(row["expanded_nodes"] for row in indexed[:q]),
                                       "rejected_queries": sum(row["precheck_rejected"] for row in indexed[:q]),
                                       "indexed_faster_at_this_q": indexed_total <= baseline_total})
            for row in cumulative:
                if row["workload"] == workload_key:
                    row["first_observed_crossing_q_within_workload"] = first_crossing
            for reachable in (True, False):
                for variant, records in (("baseline", baseline), ("indexed", indexed)):
                    selected = [row for row in records if row["reachable"] == reachable]
                    summaries.append({"workload": workload_key, "map": name, "map_name": LABELS[name],
                                      "unsolved_fraction_requested": ratio, "reachable_group": reachable,
                                      "variant": variant, "n_queries": len(selected),
                                      "query_median_ms": statistics.median(row["query_median_ms"] for row in selected) if selected else "",
                                      "expanded_median": statistics.median(row["expanded_nodes"] for row in selected) if selected else "",
                                      "expanded_sum": sum(row["expanded_nodes"] for row in selected),
                                      "rejected_queries": sum(row["precheck_rejected"] for row in selected),
                                      "preprocess_ms": index.preprocess_ms if variant == "indexed" else 0})
            print(f"完成 {LABELS[name]}，无解目标比例 {ratio:.0%}：{queries} 对查询、两种方案。", flush=True)

    write_csv(output / "queries.csv", raw)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "cumulative.csv", cumulative)
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(),
                "parameters": {"size": size, "queries_per_workload": queries, "repeats": repeats, "batch": batch, "seed": seed},
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("connectivity_index.py", "run_connectivity_study.py", "maze_astar.py", "run_experiments.py")},
                "maps": {name: ["".join(row) for row in grid] for name, grid in maps.items()},
                "workloads": workloads_meta,
                "methodology": {"snapshot": "immutable tuple-of-tuples; both algorithms query this identical representation",
                                "query_validation": "exactly one full RC scan in both methods, including rejected indexed queries",
                                "timing": "one warmup, median/min of repeated batch-average ms; default garbage collection",
                                "timing_excludes": "oracle BFS, workload generation, trace, charts, file IO",
                                "cumulative_time": "one actual preprocessing run plus sum of per-query median estimates; not a continuous workload timing",
                                "unreachable_sampling": "controlled within/across components; query mix is a task parameter, not algorithm output",
                                "workload_limit": "all-connected map only supports 0% unreachable pairs of passable endpoints",
                                "independent_samples": "one generated pair; batches are timing repetitions; maps are deterministic case studies"}}
    (output / "reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    final_rows = [row for row in cumulative if row["queries"] == queries]
    outcomes = {"query_configurations": len(raw), "query_pairs": len(raw) // 2, "workloads": len(workloads_meta),
                "all_bfs_verified": True, "final_totals": final_rows, "summary": summaries,
                "full_validation_retained": True}
    (output / "outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    totals_panels, node_panels = [], []
    for name in maps:
        rows = sorted((row for row in final_rows if row["map"] == name), key=lambda row: row["unsolved_fraction_requested"])
        xs = [row["unsolved_fraction_requested"] * 100 for row in rows]
        totals_panels.append({"title": f"{LABELS[name]}：{queries} 次查询累计分量估计（ms）；横轴无解比例 %", "x": xs,
                              "series": [("原 A*", [row["baseline_total_ms"] for row in rows], "#487ba7"),
                                         ("连通索引＋A*（含建表）", [row["indexed_total_ms"] for row in rows], "#d48737")]})
        node_panels.append({"title": f"{LABELS[name]}：实际扩展数总计；横轴无解比例 %", "x": xs,
                            "series": [("原 A*", [row["baseline_expanded_sum"] for row in rows], "#487ba7"),
                                       ("连通索引＋A*", [row["indexed_expanded_sum"] for row in rows], "#d48737")]})
    line_svg(output / "workload_costs.svg", "连通预检：失败查询比例与完整成本", totals_panels,
             "每次查询均保留整图验证；单次建表收费；全连通图只能生成 0% 异分量查询。")
    line_svg(output / "expanded_work.svg", "连通预检：实际搜索工作量", node_panels,
             "异分量直接拒绝的查询不扩展节点，但仍验证地图；不把扩展数当完整运行时间。")
    chosen = [("two_regions", .0), ("two_regions", .5), ("two_regions", 1.), ("small_rooms", 1.)]
    panels = []
    for name, ratio in chosen:
        rows = sorted((row for row in cumulative if row["map"] == name and row["unsolved_fraction_requested"] == ratio), key=lambda row: row["queries"])
        panels.append({"title": f"{LABELS[name]}，无解 {ratio:.0%}：累计分量估计（ms）；横轴查询数", "x": [row["queries"] for row in rows],
                       "series": [("原 A*", [row["baseline_total_ms"] for row in rows], "#487ba7"),
                                  ("连通索引＋A*（含建表）", [row["indexed_total_ms"] for row in rows], "#d48737")]})
    line_svg(output / "amortization.svg", "连通预检：多次查询是否摊销建表", panels,
             "横轴按实际查询数比例布点；累计值由单次建表实测与各查询中位数相加，非连续整段计时。")
    print(f"全部 {len(raw)} 个配置通过独立 BFS 校验，保存至 {output.resolve()}", flush=True)
    return outcomes


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("参数必须为正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=positive, default=80)
    parser.add_argument("--queries", type=positive, default=100)
    parser.add_argument("--repeats", type=positive, default=5)
    parser.add_argument("--batch", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "bonus_connectivity")
    args = parser.parse_args()
    if args.size < 8:
        parser.error("size 至少为 8，以生成具有通行区的三类地图")
    study(args.output_dir, args.size, args.queries, args.repeats, args.batch, args.seed)


if __name__ == "__main__":
    main()
