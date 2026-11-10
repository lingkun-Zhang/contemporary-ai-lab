"""题一可复现实验：独立 BFS 校验、搜索统计、重复计时及 SVG 可视化。

仅使用 Python 标准库。运行：python run_experiments.py
计时仅包含无轨迹的单次搜索，不包含 BFS、图表和文件读写。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns
from typing import Optional

from maze_astar import SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar_search


@dataclass
class Case:
    slug: str
    name: str
    grid: list[list[str]]
    start: tuple[int, int]
    goal: tuple[int, int]
    seed: Optional[int] = None
    density: Optional[float] = None


VARIANTS = (
    ("zero_fifo", "h=0 / FIFO", "zero", "fifo", "#6987a5"),
    ("manhattan_fifo", "曼哈顿 / FIFO", "manhattan", "fifo", "#4aa5a0"),
    ("manhattan_low_h", "曼哈顿 / 较小 h 优先", "manhattan", "low_h", "#e4a44d"),
)
FONT = "'Microsoft YaHei', 'Noto Sans CJK SC', Arial, sans-serif"


def mark_endpoints(grid, start, goal):
    """每个场景使用独立矩阵，S/E 与普通可行格一样可通行。"""
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"
    return grid


def build_cases(seed: int) -> list[Case]:
    cases = [Case("sample", "PDF 样例", [list(row) for row in SAMPLE_GRID],
                  SAMPLE_START, SAMPLE_GOAL)]
    grid = [["."] * 40 for _ in range(40)]
    cases.append(Case("open40", "开阔 40×40", mark_endpoints(grid, (0, 0), (39, 39)),
                      (0, 0), (39, 39)))

    grid = [["."] * 25 for _ in range(25)]
    for row in range(24):
        grid[row][12] = "X"  # 只能从最底行绕过竖墙。
    cases.append(Case("detour", "强制绕路", mark_endpoints(grid, (2, 2), (2, 22)),
                      (2, 2), (2, 22)))

    grid = [["."] * 15 for _ in range(15)]
    for row in range(15):
        grid[row][7] = "X"
    cases.append(Case("unreachable", "无解地图", mark_endpoints(grid, (7, 2), (7, 12)),
                      (7, 2), (7, 12)))

    grid = [["."] * 15 for _ in range(15)]
    for row in range(5, 10):
        for col in range(6, 9):
            grid[row][col] = "X"
    cases.append(Case("multiple_paths", "多条最短路径", mark_endpoints(grid, (7, 1), (7, 13)),
                      (7, 1), (7, 13)))

    # 不筛选随机地图是否有解，保留原始分布；无解同样计入比较。
    for density_index, density in enumerate((0.15, 0.25, 0.35)):
        for index in range(3):
            case_seed = seed + density_index * 100 + index
            rng = random.Random(case_seed)
            grid = [["X" if rng.random() < density else "." for _ in range(30)]
                    for _ in range(30)]
            cases.append(Case(
                f"random_{int(density * 100)}_{index + 1}",
                f"随机 {int(density * 100)}% #{index + 1}",
                mark_endpoints(grid, (0, 0), (29, 29)), (0, 0), (29, 29),
                case_seed, density,
            ))
    return cases


def bfs_distance(case: Case) -> int:
    """独立实现的单位代价 BFS，仅作为最短步数校验器。"""
    rows, cols = len(case.grid), len(case.grid[0])
    queue = deque([(case.start, 0)])
    visited = {case.start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == case.goal:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = row + dr, col + dc
            nr, nc = neighbor
            if (0 <= nr < rows and 0 <= nc < cols and
                    case.grid[nr][nc] != "X" and neighbor not in visited):
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


def validate_result(case, result, expected_cost):
    """同时验证最优步数、路径连续性、边界与障碍。"""
    if result.cost != expected_cost:
        raise AssertionError(f"{case.name}: 搜索代价 {result.cost} 与 BFS {expected_cost} 不符")
    if expected_cost < 0:
        if result.path:
            raise AssertionError(f"{case.name}: 无解时路径必须为 []")
        return
    path = result.path
    if not path or path[0] != case.start or path[-1] != case.goal:
        raise AssertionError(f"{case.name}: 路径端点错误")
    if len(path) - 1 != expected_cost:
        raise AssertionError(f"{case.name}: 路径长度与代价不符")
    for row, col in path:
        if not (0 <= row < len(case.grid) and 0 <= col < len(case.grid[0])):
            raise AssertionError(f"{case.name}: 路径越界")
        if case.grid[row][col] == "X":
            raise AssertionError(f"{case.name}: 路径经过障碍")
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            raise AssertionError(f"{case.name}: 路径存在非四方向移动")


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    """UTF-8 BOM 让中文表头/内容可直接在 Excel 中打开。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_start(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        f'<style>text{{font-family:{FONT};fill:#263746}} '
        '.title{font-size:22px;font-weight:700}.label{font-size:14px}'
        '.note{font-size:12px;fill:#536575}</style>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text class="title" x="24" y="34">{escape(title)}</text>',
    ]


def save_search_svg(output: Path, case: Case, results: dict):
    """三种设置按同一网格比例展示，蓝色仅表示实际扩展节点。"""
    rows, cols = len(case.grid), len(case.grid[0])
    cell = max(7, min(28, 370 // max(rows, cols)))
    panel_width = max(360, cols * cell + 32)
    grid_top, margin = 114, 24
    height = grid_top + rows * cell + 58
    width = panel_width * 3 + margin
    parts = svg_start(width, height, f"{case.name}：路径与搜索范围")
    parts.append('<text class="note" x="24" y="58">蓝色：实际扩展节点（不含终点）　'
                 '橙色：返回路径　深色：障碍　S：起点　E：终点</text>')
    for variant_index, (key, label, _, _, _) in enumerate(VARIANTS):
        result = results[key]
        left = margin + variant_index * panel_width
        parts.append(f'<text class="label" x="{left}" y="82">{escape(label)}</text>')
        parts.append(f'<text class="note" x="{left}" y="102">'
                     f'最短步数 {result.cost}；扩展 {result.stats["expanded_nodes"]} 个节点</text>')
        expanded, path = set(result.expanded_order), set(result.path)
        for row in range(rows):
            for col in range(cols):
                node = row, col
                color = ("#263746" if case.grid[row][col] == "X" else
                         "#f6c780" if node in path else
                         "#c9e4f2" if node in expanded else "#f8fafb")
                parts.append(f'<rect x="{left + col * cell}" y="{grid_top + row * cell}" '
                             f'width="{cell}" height="{cell}" fill="{color}" '
                             'stroke="#d6dfe5" stroke-width="0.5"/>')
        if len(result.path) > 1:
            points = " ".join(f"{left + (col + .5) * cell},{grid_top + (row + .5) * cell}"
                              for row, col in result.path)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#b96a13" '
                         f'stroke-width="{max(1, cell / 8):.2f}" stroke-linejoin="round"/>')
        for node, letter, fill in ((case.start, "S", "#19765d"), (case.goal, "E", "#b3414d")):
            row, col = node
            cx, cy = left + (col + .5) * cell, grid_top + (row + .5) * cell
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{cell * .43}" fill="{fill}"/>')
            parts.append(f'<text x="{cx}" y="{cy + cell * .18}" text-anchor="middle" '
                         f'style="font-size:{max(7, cell * .65)}px;fill:white;font-weight:700">{letter}</text>')
    parts.append(f'<text class="note" x="24" y="{height - 18}">'
                 '各策略都返回最短路径；并列时的路径或扩展范围可以不同。</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def save_nodes_svg(output: Path, cases: list[Case], rows: list[dict]):
    width, group_height = 1180, 53
    top, label_width, right = 109, 190, 75
    height = top + len(cases) * group_height + 62
    parts = svg_start(width, height, "各场景实际扩展节点数对比")
    for index, (_, label, _, _, color) in enumerate(VARIANTS):
        x = 24 + index * 325
        parts.append(f'<rect x="{x}" y="54" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text class="label" x="{x + 22}" y="67">{escape(label)}</text>')
    maximum = max(1, max(row["expanded_nodes"] for row in rows))
    axis_max = ((maximum + 199) // 200) * 200
    plot_width = width - label_width - right
    for tick in range(5):
        value = axis_max * tick / 4
        x = label_width + plot_width * tick / 4
        parts.append(f'<line x1="{x}" y1="{top - 13}" x2="{x}" '
                     f'y2="{top + len(cases) * group_height - 3}" stroke="#e3e9ed"/>')
        parts.append(f'<text class="note" x="{x}" y="{top - 23}" '
                     f'text-anchor="middle">{value:g}</text>')
    lookup = {(row["case"], row["variant"]): row for row in rows}
    for case_index, case in enumerate(cases):
        y = top + case_index * group_height
        parts.append(f'<text class="label" x="24" y="{y + 23}">{escape(case.name)}</text>')
        for variant_index, (key, _, _, _, color) in enumerate(VARIANTS):
            count = lookup[(case.slug, key)]["expanded_nodes"]
            bar_y = y + variant_index * 13
            bar_width = count / axis_max * plot_width
            parts.append(f'<rect x="{label_width}" y="{bar_y}" width="{bar_width:.2f}" '
                         f'height="10" rx="2" fill="{color}"/>')
            parts.append(f'<text class="note" x="{label_width + bar_width + 5:.2f}" '
                         f'y="{bar_y + 10}">{count}</text>')
    parts.append(f'<text class="note" x="24" y="{height - 24}">'
                 '扩展节点：有效出队后实际生成邻居的节点，终点不计入；无解地图同样计入比较。</text>')
    parts.append('</svg>')
    output.write_text("\n".join(parts), encoding="utf-8")


def run_experiments(output: Path, repeats: int, batch: int, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    cases = build_cases(seed)
    table, trace_rows, saved_results, metadata_cases = [], [], {}, []
    for case in cases:
        expected = bfs_distance(case)
        original_grid = [row[:] for row in case.grid]
        case_results = {}
        for key, label, heuristic, tie_break, _ in VARIANTS:
            # 完整轨迹只用于可视化；计时调用不保存轨迹。
            result = astar_search(case.start, case.goal, case.grid,
                                  heuristic=heuristic, tie_break=tie_break, record_trace=True)
            validate_result(case, result, expected)
            if case.grid != original_grid:
                raise AssertionError(f"{case.name}: 搜索修改了输入矩阵")
            case_results[key] = result
            astar_search(case.start, case.goal, case.grid, heuristic=heuristic,
                         tie_break=tie_break, record_trace=False)  # 一次预热。
            per_search_ms = []
            for _ in range(repeats):
                started = perf_counter_ns()
                for _ in range(batch):
                    timed_result = astar_search(case.start, case.goal, case.grid,
                                                heuristic=heuristic, tie_break=tie_break,
                                                record_trace=False)
                elapsed = perf_counter_ns() - started
                per_search_ms.append(elapsed / batch / 1_000_000)
                if timed_result.cost != expected or timed_result.path != result.path:
                    raise AssertionError(f"{case.name}: 计时模式与轨迹模式返回结果不一致")
            table.append({
                "case": case.slug, "case_name": case.name,
                "rows": len(case.grid), "cols": len(case.grid[0]),
                "seed": case.seed if case.seed is not None else "",
                "obstacle_density_requested": case.density if case.density is not None else "",
                "variant": key, "variant_name": label, "heuristic": heuristic,
                "tie_break": tie_break, "reachable": expected >= 0,
                "shortest_steps": result.cost, "bfs_steps": expected,
                "bfs_verified": True, "path_node_count": len(result.path),
                **result.stats,
                "time_median_ms": round(statistics.median(per_search_ms), 6),
                "time_min_ms": round(min(per_search_ms), 6),
                "timing_repeats": repeats, "searches_per_batch": batch,
                "timing_batches_ms_per_search": json.dumps([round(v, 6) for v in per_search_ms]),
            })
            if case.slug == "sample":
                for item in result.trace:
                    row, col = item["node"]
                    trace_rows.append({"variant": key, "step": item["step"],
                                       "row": row, "col": col, "g": item["g"],
                                       "h": item["h"], "f": item["f"],
                                       "is_goal": (row, col) == case.goal})
        saved_results[case.slug] = case_results
        metadata_cases.append({
            "case": case.slug, "name": case.name, "grid": ["".join(row) for row in case.grid],
            "start": case.start, "goal": case.goal, "seed": case.seed,
            "requested_density": case.density,
            "actual_obstacle_fraction": sum(row.count("X") for row in case.grid) /
                                        (len(case.grid) * len(case.grid[0])),
            "bfs_steps": expected,
            "paths": {key: result.path for key, result in case_results.items()},
        })

    write_csv(output / "comparison.csv", table)
    write_csv(output / "sample_trace.csv", trace_rows)
    source_hashes = {}
    for filename in ("maze_astar.py", "run_experiments.py"):
        source = Path(__file__).resolve().with_name(filename)
        if source.exists():
            source_hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
        "parameters": {"seed": seed, "repeats": repeats, "batch": batch,
                       "random_grid_size": [30, 30], "random_densities": [0.15, 0.25, 0.35],
                       "maps_per_density": 3, "random_maps_filtered_by_reachability": False},
        "timing": {"clock": "time.perf_counter_ns", "reported_unit": "ms per search",
                   "warmups_per_case_variant": 1, "trace_recording": False,
                   "includes": "search call and loop overhead; input validation included",
                   "excludes": "BFS verification, trace collection, plotting and file IO",
                   "gc": "default interpreter behavior; not disabled"},
        "metric_definitions": {
            "expanded_nodes": "有效出队后生成邻居的节点数，不含终点",
            "generated_nodes": "成功入队次数，含起点；重复改进后入队也计数",
            "max_frontier": "物理优先队列的最大长度，含尚未弹出的过期项",
            "stale_pops": "过期队列项出队次数",
            "popped_nodes": "全部出队次数，含有效终点和过期项",
            "reopened_nodes": "已扩展节点以更优 g 再次实际扩展的次数",
            "unique_reached": "已发现有效 g 值的不同节点数，含起点",
            "time_median_ms": "各批次平均单次搜索耗时的中位数，毫秒",
            "time_min_ms": "各批次平均单次搜索耗时的最小值，毫秒",
        },
        "source_sha256": source_hashes, "cases": metadata_cases, "results": table,
    }
    (output / "reproducibility.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for slug in ("sample", "open40", "detour"):
        case = next(case for case in cases if case.slug == slug)
        save_search_svg(output / f"{slug}_search_comparison.svg", case, saved_results[slug])
    save_nodes_svg(output / "expanded_nodes_comparison.svg", cases, table)

    print(f"完成 {len(cases)} 张地图 × {len(VARIANTS)} 种策略，全部通过独立 BFS 校验。")
    print("计时单位：毫秒/次；每个设置先预热 1 次，再按批次平均，报告中位数与最小值。")
    for case in cases[:5]:
        selected = [row for row in table if row["case"] == case.slug]
        counts = " / ".join(str(row["expanded_nodes"]) for row in selected)
        print(f"{case.name}：最短步数 {selected[0]['shortest_steps']}；扩展节点 {counts}")
    print("上述节点顺序：h=0/FIFO，曼哈顿/FIFO，曼哈顿/较小 h 优先。")
    print(f"数据与 SVG 已保存：{output.resolve()}")
    return table


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=positive_int, default=7, help="计时批次数，默认 7")
    parser.add_argument("--batch", type=positive_int, default=10, help="每批搜索次数，默认 10")
    parser.add_argument("--seed", type=int, default=20261022, help="随机地图基准种子")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    run_experiments(args.output_dir, args.repeats, args.batch, args.seed)


if __name__ == "__main__":
    main()
