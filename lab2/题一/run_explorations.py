"""题一深入实验：启发质量、地图结构、并列规则与地标预处理摊销。

只使用标准库。默认 300 张规模/密度地图，不按有解性筛选。
计时批次不是独立地图样本；独立 BFS 校验、画图与文件写入不计入查询时间。
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from maze_astar import DIRECTIONS, SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar_search
from run_experiments import Case, bfs_distance, mark_endpoints, svg_start, validate_result, write_csv


HEURISTICS = ("zero", "chebyshev", "euclidean", "manhattan")
COLORS = {"zero": "#708ba3", "chebyshev": "#a48ab3", "euclidean": "#45a6a1",
          "manhattan": "#d88c33", "landmark_1": "#437eb5", "landmark_2": "#5d9b56",
          "landmark_4": "#b65d72"}
LABELS = {"zero": "h=0", "chebyshev": "切比雪夫", "euclidean": "欧氏距离",
          "manhattan": "曼哈顿", "landmark_1": "曼哈顿+1地标",
          "landmark_2": "曼哈顿+2地标", "landmark_4": "曼哈顿+4地标"}


def heuristic_value(name, node, goal):
    if callable(name):
        return name(node, goal)
    dr, dc = abs(node[0] - goal[0]), abs(node[1] - goal[1])
    return {"zero": lambda: 0, "chebyshev": lambda: max(dr, dc),
            "euclidean": lambda: math.hypot(dr, dc), "manhattan": lambda: dr + dc}[name]()


def distance_field(grid, origin):
    """独立 BFS 距离场，只用于验证与生成可达查询，不作为 A* 的输入。"""
    height, width = len(grid), len(grid[0])
    distances, queue = {origin: 0}, deque([origin])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            node = row + dr, col + dc
            if (0 <= node[0] < height and 0 <= node[1] < width and
                    grid[node[0]][node[1]] != "X" and node not in distances):
                distances[node] = distances[(row, col)] + 1
                queue.append(node)
    return distances


def quality(case, heuristic, field):
    values = {node: heuristic_value(heuristic, node, case.goal)
              for row in range(len(case.grid)) for col in range(len(case.grid[0]))
              if case.grid[row][col] != "X" for node in [(row, col)]}
    over = sum(values[node] > true + 1e-9 for node, true in field.items())
    inconsistent = sum(values[node] > 1 + values[neighbor] + 1e-9
                       for node in values for dr, dc in DIRECTIONS
                       for neighbor in [(node[0] + dr, node[1] + dc)] if neighbor in values)
    if over or inconsistent or abs(values[case.goal]) > 1e-9:
        raise AssertionError(f"{case.slug}: 启发函数可采纳性/一致性验证失败")
    true_start, h_start = field.get(case.start, -1), values[case.start]
    return {"h_start": h_start, "h_start_over_optimal": h_start / true_start if true_start > 0 else "",
            "mean_heuristic_gap": statistics.mean(true - values[node] for node, true in field.items()),
            "quality_verified_finite_nodes": len(field), "admissibility_violations": over,
            "consistency_violations": inconsistent}


def measured(case, heuristic, repeats, batch, expected, tie="low_h", order=None, trace=False):
    result = astar_search(case.start, case.goal, case.grid, heuristic=heuristic,
                          tie_break=tie, neighbor_order=order, record_trace=trace)
    validate_result(case, result, expected)
    # 轨迹模式的运行不充当预热，始终额外执行一次相同计时模式的预热。
    astar_search(case.start, case.goal, case.grid, heuristic=heuristic,
                 tie_break=tie, neighbor_order=order, record_trace=False)
    samples = []
    for _ in range(repeats):
        before = perf_counter_ns()
        for _ in range(batch):
            timed = astar_search(case.start, case.goal, case.grid, heuristic=heuristic,
                                  tie_break=tie, neighbor_order=order, record_trace=False)
        samples.append((perf_counter_ns() - before) / batch / 1e6)
        if timed.cost != expected or timed.path != result.path:
            raise AssertionError(f"{case.slug}: 计时与记录模式不一致")
    return result, {"time_median_ms": statistics.median(samples), "time_min_ms": min(samples),
                    "time_batch_samples_ms": json.dumps(samples)}


def density(grid):
    return sum(row.count("X") for row in grid) / (len(grid) * len(grid[0]))


def base_row(case, result, timing, variant):
    return {"case": case.slug, "case_name": case.name, "rows": len(case.grid),
            "cols": len(case.grid[0]), "seed": case.seed if case.seed is not None else "",
            "density_requested": case.density if case.density is not None else "",
            "density_actual": density(case.grid), "start": json.dumps(case.start),
            "goal": json.dumps(case.goal), "variant": variant, "reachable": result.cost >= 0,
            "optimal_steps": result.cost, "bfs_verified": True, **result.stats, **timing}


def perfect_maze(n, seed):
    rng, grid = random.Random(seed), [["X"] * n for _ in range(n)]
    stack = [(1, 1)]
    grid[1][1] = "."
    while stack:
        row, col = stack[-1]
        choices = [(row + dr * 2, col + dc * 2, dr, dc) for dr, dc in DIRECTIONS
                   if 1 <= row + dr * 2 < n - 1 and 1 <= col + dc * 2 < n - 1
                   and grid[row + dr * 2][col + dc * 2] == "X"]
        if not choices:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(choices)
        grid[row + dr][col + dc] = grid[nr][nc] = "."
        stack.append((nr, nc))
    return grid


def wall_case(gap, n=51):
    grid = [["."] * n for _ in range(n)]
    for row in range(n):
        if row != gap:
            grid[row][n // 2] = "X"
    start, goal = (1, 1), (1, n - 2)
    return Case(f"wall_gap_{gap}", f"竖墙缺口行 {gap}", mark_endpoints(grid, start, goal), start, goal)


def structures(seed):
    n, start, goal = 51, (1, 1), (1, 49)
    def wrap(slug, name, grid):
        return Case(slug, name, mark_endpoints(grid, start, goal), start, goal, seed)
    longwall = wall_case(49)
    longwall.slug, longwall.name = "long_wall", "长墙绕行"
    zig = [["."] * n for _ in range(n)]
    for index, col in enumerate((10, 20, 30, 40)):
        for row in range(n):
            if row != (49 if index % 2 == 0 else 1):
                zig[row][col] = "X"
    blocked = [["."] * n for _ in range(n)]
    for row in range(n):
        blocked[row][25] = "X"
    scatter = [["."] * n for _ in range(n)]
    available = [(r, c) for r in range(n) for c in range(n) if (r, c) not in (start, goal)]
    for row, col in random.Random(seed).sample(available, 50):
        scatter[row][col] = "X"
    return [wrap("open", "开阔地图", [["."] * n for _ in range(n)]), longwall,
            wrap("zigzag", "交替缺口墙", zig),
            wrap("perfect_maze", "DFS 完美迷宫", perfect_maze(n, seed)),
            wrap("unreachable", "完整竖墙无解", blocked),
            wrap("scattered_50", "随机分散 50 障碍", scatter)]


def summaries(rows, keys):
    groups, result = defaultdict(list), []
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for values, group in groups.items():
        for reachable in (True, False):
            selected = [row for row in group if row["reachable"] == reachable]
            item = dict(zip(keys, values))
            item.update({"reachable_group": reachable, "n_maps_total": len(group),
                         "n_maps": len(selected),
                         "reachable_fraction": sum(row["reachable"] for row in group) / len(group)})
            for metric in ("expanded_nodes", "max_frontier", "time_median_ms", "optimal_steps"):
                data = [row[metric] for row in selected] if metric != "optimal_steps" or reachable else []
                if data:
                    quartiles = statistics.quantiles(data, n=4, method="inclusive") if len(data) > 1 else data * 3
                    item.update({metric + "_mean": statistics.mean(data), metric + "_median": statistics.median(data),
                                 metric + "_q1": quartiles[0], metric + "_q3": quartiles[2],
                                 metric + "_iqr": quartiles[2] - quartiles[0]})
                else:
                    item.update({metric + suffix: "" for suffix in ("_mean", "_median", "_q1", "_q3", "_iqr")})
            result.append(item)
    return result


def save_chart(path, title, panels, note, width=1200):
    """简洁折线/分组图；各面板明确标注横纵轴和单位。"""
    panel_h, top, left, right = 280, 110, 90, 35
    height = top + len(panels) * panel_h + 40
    parts = svg_start(width, height, title)
    parts.append(f'<text class="note" x="24" y="60">{escape(note)}</text>')
    for index, panel in enumerate(panels):
        ytop, plot_h = top + index * panel_h, 175
        xs, series = panel["x"], panel["series"]
        ymax = panel.get("ymax", max([value for _, values, _ in series for value in values if value is not None] or [1]) * 1.12)
        ymax = max(1, ymax)
        plot_w = width - left - right
        def xpos(j):
            if panel.get("numeric_x", False):
                return left + plot_w * (float(xs[j]) - float(min(xs))) / max(1e-12, float(max(xs)) - float(min(xs)))
            return left + plot_w * (j + .5) / len(xs)
        parts.append(f'<text class="label" x="24" y="{ytop - 20}">{escape(panel["title"])}</text>')
        for tick in range(5):
            y = ytop + plot_h * (1 - tick / 4)
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e2e8ee"/>')
            parts.append(f'<text class="note" x="{left-8}" y="{y+4}" text-anchor="end">{ymax*tick/4:.1f}</text>')
        for j, label in enumerate(xs):
            x = xpos(j)
            parts.append(f'<text class="note" x="{x}" y="{ytop+plot_h+19}" text-anchor="middle">{escape(str(label))}</text>')
        for series_index, (label, values, color) in enumerate(series):
            points, segments, segment = [], [], []
            for j, value in enumerate(values):
                if value is None:
                    if segment:
                        segments.append(segment)
                        segment = []
                    continue
                point = xpos(j), ytop + plot_h * (1 - value/ymax)
                points.append(point)
                segment.append(point)
            if segment:
                segments.append(segment)
            for segment in segments:
                point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
                parts.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            lx = 100 + series_index * 250
            parts.append(f'<line x1="{lx}" y1="{ytop+plot_h+45}" x2="{lx+20}" y2="{ytop+plot_h+45}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="note" x="{lx+27}" y="{ytop+plot_h+49}">{escape(label)}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def export_maps(cases):
    return [{"case": case.slug, "name": case.name, "start": case.start, "goal": case.goal,
             "grid": ["".join(row) for row in case.grid], "seed": case.seed,
             "density_requested": case.density, "density_actual": density(case.grid)} for case in cases]


def run(output, seeds=20, repeats=3, batch=3, seed=20261022):
    from landmark_heuristic import build_landmark_index

    output.mkdir(parents=True, exist_ok=True)
    maps = structures(seed)
    structure_rows, gap_rows, landmark_rows = [], [], []
    cached_fields = {}
    for case in maps:
        field = cached_fields[case.slug] = distance_field(case.grid, case.goal)
        expected = field.get(case.start, -1)
        for heuristic in HEURISTICS:
            result, timing = measured(case, heuristic, repeats, batch, expected)
            structure_rows.append({**base_row(case, result, timing, heuristic), **quality(case, heuristic, field)})
    write_csv(output / "structure.csv", structure_rows)
    for gap in (1, 10, 25, 40, 49):
        case = wall_case(gap)
        field = distance_field(case.grid, case.goal)
        for heuristic in HEURISTICS:
            result, timing = measured(case, heuristic, repeats, batch, field[case.start])
            gap_rows.append({**base_row(case, result, timing, heuristic), "gap_row": gap,
                             "obstacle_count": 50, **quality(case, heuristic, field)})
    write_csv(output / "wall_gap.csv", gap_rows)
    print("[1/5] 地图结构、启发质量与固定障碍数的缺口位置实验完成。", flush=True)

    random_cases, random_rows = [], []
    for n, p in itertools.product((20, 40, 60), (0, .1, .2, .3, .4)):
        for instance in range(seeds):
            case_seed = seed + n * 100000 + round(p * 100) * 1000 + instance
            rng = random.Random(case_seed)
            grid = [["X" if rng.random() < p else "." for _ in range(n)] for _ in range(n)]
            start, goal = (0, 0), (n-1, n-1)
            case = Case(f"random_{n}_{round(p*100)}_{instance}", "角落端点随机地图",
                        mark_endpoints(grid, start, goal), start, goal, case_seed, p)
            random_cases.append(case)
            expected = bfs_distance(case)
            # 每张地图轮换算法执行顺序，减轻固定顺序的热缓存/时钟漂移影响。
            order = list(HEURISTICS)
            rng.shuffle(order)
            for heuristic in order:
                result, timing = measured(case, heuristic, repeats, batch, expected)
                random_rows.append(base_row(case, result, timing, heuristic))
    write_csv(output / "density_raw.csv", random_rows)
    random_summary = summaries(random_rows, ("rows", "density_requested", "variant"))
    write_csv(output / "density_summary.csv", random_summary)
    baseline = {row["case"]: row["expanded_nodes"] for row in random_rows if row["variant"] == "zero"}
    paired = [{"case": row["case"], "rows": row["rows"], "density_requested": row["density_requested"],
               "variant": row["variant"], "reachable": row["reachable"],
               "baseline_nodes": baseline[row["case"]], "expanded_nodes": row["expanded_nodes"],
               "node_reduction_fraction": 1 - row["expanded_nodes"] / baseline[row["case"]]}
              for row in random_rows if row["variant"] != "zero"]
    write_csv(output / "density_paired.csv", paired)
    paired_groups, paired_summary = defaultdict(list), []
    for row in paired:
        paired_groups[(row["rows"], row["density_requested"], row["variant"], row["reachable"])].append(row["node_reduction_fraction"])
    for (n, p, variant, reachable), values in paired_groups.items():
        quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
        paired_summary.append({"rows": n, "density_requested": p, "variant": variant,
                               "reachable": reachable, "n_maps": len(values),
                               "node_reduction_fraction_mean": statistics.mean(values),
                               "node_reduction_fraction_median": statistics.median(values),
                               "node_reduction_fraction_q1": quartiles[0],
                               "node_reduction_fraction_q3": quartiles[2],
                               "node_reduction_fraction_iqr": quartiles[2] - quartiles[0]})
    write_csv(output / "density_paired_summary.csv", paired_summary)
    robust_cases, robust_rows = [], []
    for instance in range(seeds):
        case_seed = seed + 7000000 + instance
        rng = random.Random(case_seed)
        grid = [["X" if rng.random() < .25 else "." for _ in range(40)] for _ in range(40)]
        available = [(r, c) for r in range(40) for c in range(40) if grid[r][c] != "X"]
        start, goal = rng.sample(available, 2)
        case = Case(f"random_endpoints_{instance}", "随机可通行端点稳健性", mark_endpoints(grid, start, goal),
                    start, goal, case_seed, .25)
        robust_cases.append(case)
        expected = bfs_distance(case)
        for heuristic in HEURISTICS:
            result, timing = measured(case, heuristic, repeats, batch, expected)
            robust_rows.append(base_row(case, result, timing, heuristic))
    write_csv(output / "random_endpoints_raw.csv", robust_rows)
    write_csv(output / "random_endpoints_summary.csv", summaries(robust_rows, ("variant",)))
    print(f"[2/5] {len(random_cases)} 张规模/密度地图及 {seeds} 张随机端点地图完成，未筛选有解地图。", flush=True)

    corner_grid = mark_endpoints([["."] * 51 for _ in range(51)], (0, 0), (50, 50))
    tie_cases = [Case("sample", "PDF 样例", [list(row) for row in SAMPLE_GRID], SAMPLE_START, SAMPLE_GOAL),
                 Case("open_corner", "开阔对角端点", corner_grid, (0, 0), (50, 50))]
    tie_cases.extend(case for case in maps if case.slug in ("long_wall", "perfect_maze"))
    tie_rows, tie_summary = [], []
    direction_name = {(0, 1): "R", (1, 0): "D", (0, -1): "L", (-1, 0): "U"}
    for case in tie_cases:
        expected = bfs_distance(case)
        path_sets = defaultdict(set)
        for order in itertools.permutations(DIRECTIONS):
            for tie in ("fifo", "lifo", "low_h"):
                result, timing = measured(case, "manhattan", repeats, batch, expected, tie, order, True)
                path_sets[tie].add(tuple(result.path))
                expanded_trace = [item for item in result.trace if tuple(item["node"]) != case.goal]
                less = sum(item["f"] < expected and not math.isclose(item["f"], expected) for item in expanded_trace)
                equal = sum(math.isclose(item["f"], expected) for item in expanded_trace)
                greater = len(expanded_trace) - less - equal
                if greater or len(expanded_trace) != result.stats["expanded_nodes"]:
                    raise AssertionError("一致启发搜索出现 f>C* 扩展或统计口径不符")
                tie_rows.append({**base_row(case, result, timing, "manhattan"), "tie_break": tie,
                                 "neighbor_order": "".join(direction_name[node] for node in order),
                                 "expanded_f_less_cstar": less, "expanded_f_equal_cstar": equal,
                                 "expanded_f_greater_cstar": greater,
                                 "path_sha256": hashlib.sha256(repr(result.path).encode()).hexdigest()})
        for tie in ("fifo", "lifo", "low_h"):
            selected = [row for row in tie_rows if row["case"] == case.slug and row["tie_break"] == tie]
            tie_summary.append({"case": case.slug, "case_name": case.name, "tie_break": tie,
                                "n_direction_orders": len(selected), "distinct_shortest_paths": len(path_sets[tie]),
                                **{metric + suffix: method(row[metric] for row in selected)
                                   for metric in ("expanded_nodes", "max_frontier", "expanded_f_less_cstar", "expanded_f_equal_cstar")
                                   for suffix, method in (("_min", min), ("_median", statistics.median), ("_max", max))}})
    write_csv(output / "tie_break_raw.csv", tie_rows)
    write_csv(output / "tie_break_summary.csv", tie_summary)
    print("[3/5] 24 种邻居顺序 × 3 种并列规则完成，已区分 f<C* 与 f=C* 扩展。", flush=True)

    indexes = {}
    for case in maps:
        index = indexes[case.slug] = build_landmark_index(case.grid, max_landmarks=4)
        expected = cached_fields[case.slug].get(case.start, -1)
        for k in (0, 1, 2, 4):
            variant = "manhattan" if k == 0 else f"landmark_{k}"
            heuristic = "manhattan" if k == 0 else index.heuristic(k)
            result, timing = measured(case, heuristic, repeats, batch, expected)
            preprocess = 0 if k == 0 else index.cumulative_preprocess_ms[k-1]
            landmark_rows.append({**base_row(case, result, timing, variant), "landmarks": k,
                                  "preprocess_ms": preprocess, "single_query_total_ms": preprocess + timing["time_median_ms"],
                                  "distance_table_entries": sum(len(table) for table in index.distance_tables[:k]),
                                  **quality(case, heuristic, cached_fields[case.slug])})
    write_csv(output / "landmark_raw.csv", landmark_rows)
    print("[4/5] 1/2/4 地标与曼哈顿对照完成，单查询总成本包含预处理。", flush=True)

    query_rows, cumulative_rows, query_metadata = [], [], []
    for original in maps:
        if original.slug not in ("long_wall", "perfect_maze"):
            continue
        # 独立重新预处理；累计时间对应本组工作负载，不借用前一组的预处理结果。
        index = build_landmark_index(original.grid, max_landmarks=4)
        reachable = distance_field(original.grid, original.start)
        candidates = sorted(set(reachable) - set(index.landmarks))
        rng = random.Random(seed + (100 if original.slug == "long_wall" else 200))
        pairs = []
        while len(pairs) < 40:
            pair = tuple(rng.sample(candidates, 2))
            if pair not in pairs:
                pairs.append(pair)
        query_metadata.append({"case": original.slug, "landmarks": index.landmarks, "queries": pairs,
                               "cumulative_preprocess_ms": index.cumulative_preprocess_ms})
        for query_id, (start, goal) in enumerate(pairs, 1):
            case = Case(original.slug, original.name, original.grid, start, goal)
            expected = bfs_distance(case)
            if expected < 0:
                raise AssertionError("多查询抽样应处于同一连通分量")
            order = [0, 1, 2, 4]
            rng.shuffle(order)
            for k in order:
                heuristic = "manhattan" if k == 0 else index.heuristic(k)
                result, timing = measured(case, heuristic, repeats, batch, expected)
                query_rows.append({**base_row(case, result, timing, "manhattan" if k == 0 else f"landmark_{k}"),
                                   "query_id": query_id, "landmarks": k,
                                   "preprocess_ms": 0 if k == 0 else index.cumulative_preprocess_ms[k-1]})
        selected = [row for row in query_rows if row["case"] == original.slug]
        for k in (0, 1, 2, 4):
            variant_rows = sorted((row for row in selected if row["landmarks"] == k), key=lambda row: row["query_id"])
            preprocess = variant_rows[0]["preprocess_ms"]
            baseline_rows = sorted((row for row in selected if row["landmarks"] == 0), key=lambda row: row["query_id"])
            first_break_even = ""
            for q in range(1, 41):
                total = preprocess + sum(row["time_median_ms"] for row in variant_rows[:q])
                baseline_total = sum(row["time_median_ms"] for row in baseline_rows[:q])
                if k and first_break_even == "" and total <= baseline_total:
                    first_break_even = q
                if q in (1, 5, 10, 20, 40):
                    cumulative_rows.append({"case": original.slug, "variant": "manhattan" if k == 0 else f"landmark_{k}",
                                            "landmarks": k, "queries": q, "preprocess_ms": preprocess,
                                            "query_sum_ms": total-preprocess, "total_ms": total,
                                            "manhattan_total_ms": baseline_total,
                                            "total_speedup_over_manhattan": baseline_total / total,
                                            "expanded_nodes_sum": sum(row["expanded_nodes"] for row in variant_rows[:q]),
                                            "break_even_reached_at_this_q": k > 0 and total <= baseline_total})
            for row in cumulative_rows:
                if row["case"] == original.slug and row["landmarks"] == k:
                    row["first_observed_break_even_q_within_40"] = first_break_even
    write_csv(output / "multi_query_raw.csv", query_rows)
    write_csv(output / "multi_query_cumulative.csv", cumulative_rows)
    print("[5/5] 两张静态地图各 40 对查询完成，累计成本已含预处理。", flush=True)

    def series_from(rows, xvalues, xkey, ykey, variants=HEURISTICS):
        return [(LABELS[v], [next((r[ykey] if r[ykey] != "" else None for r in rows if r[xkey] == x and r["variant"] == v), None)
                             for x in xvalues], COLORS[v]) for v in variants]

    save_chart(output / "structure_heuristics.svg", "地图结构与启发函数：同为四方向单位代价", [{
        "title": "实际扩展节点数（不含终点）；全部地图为 51×51，端点为 (1,1)→(1,49)",
        "x": [case.name for case in maps],
        "series": series_from(structure_rows, [case.slug for case in maps], "case", "expanded_nodes")}],
        "long_wall 与 scattered_50 均为 50 个障碍；其他结构场景未控制障碍总数。")
    scale_panels = []
    for p in (.1, .3):
        selected = [row for row in random_summary if row["density_requested"] == p and row["reachable_group"]]
        scale_panels.append({"title": f"障碍概率 {p:.1f}：有解组扩展节点均值，横轴为边长；每组合生成 {seeds} 图，有解数见 CSV",
                             "x": [20, 40, 60], "series": series_from(selected, [20, 40, 60], "rows", "expanded_nodes_mean")})
    save_chart(output / "scale.svg", "规模增长实验", scale_panels,
               "所有随机图原样保留；此图仅绘有解组，原始数据与无解组见 CSV。时间重复批次不是独立样本。")
    selected = [row for row in random_summary if row["rows"] == 40 and row["reachable_group"]]
    ps = [0, .1, .2, .3, .4]
    save_chart(output / "density.svg", "障碍密度与可达性", [
        {"title": f"40×40：可达比例（0—1），每点 {seeds} 张未筛选地图", "x": ps, "ymax": 1,
         "series": [("独立 BFS 可达率", [next(row["reachable_fraction"] for row in selected if row["density_requested"] == p and row["variant"] == "zero") for p in ps], "#437eb5")]},
        {"title": "40×40：有解组扩展节点均值；没有有解样本时留空", "x": ps,
         "series": series_from(selected, ps, "density_requested", "expanded_nodes_mean")}],
        "密度增大同时影响连通性和最短路；节点数变化不能单独解释为算法更好。")
    tie_panels = []
    for case in tie_cases:
        selected = [row for row in tie_summary if row["case"] == case.slug]
        tie_panels.append({"title": f"{case.name}：24 种方向顺序下扩展节点的最小 / 中位 / 最大值", "x": ["最小值", "中位数", "最大值"],
                           "series": [(tie, [next(r["expanded_nodes" + suffix] for r in selected if r["tie_break"] == tie)
                                             for suffix in ("_min", "_median", "_max")], color)
                                      for tie, color in (("fifo", "#708ba3"), ("lifo", "#45a6a1"), ("low_h", "#d88c33"))]})
    save_chart(output / "tie_break_ranges.svg", "并列规则与邻居顺序的影响范围", tie_panels,
               "所有配置均以独立 BFS 校验最优步数；同长路径可以不同。f 平台分解和不同路径数见 CSV。")
    landmark_variants = ("manhattan", "landmark_1", "landmark_2", "landmark_4")
    save_chart(output / "landmark_nodes.svg", "地标启发函数：改进幅度与地图结构", [{
        "title": "实际扩展节点数；相同 low_h；预处理与查询成本单列于 landmark_raw.csv", "x": [case.name for case in maps],
        "series": series_from(landmark_rows, [case.slug for case in maps], "case", "expanded_nodes", landmark_variants)}],
        "h=max(曼哈顿, |d(L,n)−d(L,goal)|)；不会把反向 BFS 的验证距离场提供给 A*。")
    save_chart(output / "landmark_amortization.svg", "地标预处理：多查询累计成本估计", [{
        "title": f"{case_name}：累计总耗时（毫秒），横轴为已完成查询数", "x": [1, 5, 10, 20, 40], "numeric_x": True,
        "series": series_from([r for r in cumulative_rows if r["case"] == slug], [1, 5, 10, 20, 40], "queries", "total_ms", landmark_variants)}
        for slug, case_name in (("long_wall", "长墙绕行"), ("perfect_maze", "DFS 完美迷宫"))],
        "每条地标曲线从对应预处理成本开始；查询节点均排除地标。计时批次中位数相加，非实时在线负载保证。")

    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("maze_astar.py", "run_experiments.py", "run_explorations.py", "landmark_heuristic.py")}
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "processor": platform.processor(),
                "parameters": {"seed": seed, "seeds_per_scale_density": seeds, "repeats": repeats, "batch": batch,
                               "random_reachability_filtered": False, "sizes": [20, 40, 60], "densities": [0, .1, .2, .3, .4]},
                "source_sha256": source_hashes, "structure_maps": export_maps(maps),
                "structure_landmark_indexes": {slug: index.metadata() for slug, index in indexes.items()},
                "tie_break_maps": export_maps(tie_cases),
                "wall_gap_maps": export_maps([wall_case(gap) for gap in (1, 10, 25, 40, 49)]),
                "random_maps": export_maps(random_cases), "random_endpoint_maps": export_maps(robust_cases),
                "multi_query_workloads": query_metadata,
                "methodology": {"independent_sample": "one randomly generated map; not one timing batch",
                                "timing": "one warmup, repeated batches, median batch-average ms; default GC behavior",
                                "query_timing_includes": "complete astar_search including validation and frontier operations",
                                "query_timing_excludes": "BFS, heuristic verification, trace collection, plot/file IO, landmark preprocessing",
                                "preprocessing": "landmark selection and all BFS distance tables included; each prefix k uses cumulative cost",
                                "total_time": "preprocessing plus summed per-query median times; noisy machine-dependent estimate",
                                "no_path_cost_summary": "unreachable optimal_steps=-1 kept in raw data, excluded from path length summaries",
                                "structure_confounders": "all sizes/endpoints fixed; only long_wall vs scattered_50 additionally controls obstacle count; wall-gap experiment controls count and wall column",
                                "heuristic_quality": "goal-connected finite-distance nodes only for admissibility/gap; consistency checked over all open directed edges"}}
    (output / "reproducibility.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    outcomes = {"structure": structure_rows, "wall_gap": gap_rows, "tie_break": tie_summary,
                "random_endpoints": summaries(robust_rows, ("variant",)),
                "density_summary": random_summary, "density_paired_summary": paired_summary,
                "landmark": landmark_rows, "amortization": cumulative_rows,
                "counts": {"scale_density_maps": len(random_cases), "random_endpoint_maps": len(robust_cases),
                           "scale_density_configurations": len(random_rows), "tie_configurations": len(tie_rows),
                           "multi_query_configurations": len(query_rows)},
                "all_bfs_verified": True, "all_quality_checks_passed": True}
    (output / "outcomes.json").write_text(json.dumps(outcomes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部验证通过；结果保存至 {output.resolve()}", flush=True)
    return outcomes


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=positive, default=20)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--batch", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "exploration")
    args = parser.parse_args()
    run(args.output_dir, args.seeds, args.repeats, args.batch, args.seed)


if __name__ == "__main__":
    main()
