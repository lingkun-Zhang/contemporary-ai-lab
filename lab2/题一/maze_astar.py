"""题一：用 A* 在四方向、单位代价的迷宫中寻找最短路径。

直接运行输出 PDF 样例的路径；本模块及实验脚本只使用 Python 标准库。
"""

from __future__ import annotations

import argparse
import ast
import heapq
import json
import math
import sys
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


Coordinate = Tuple[int, int]
Grid = Sequence[Sequence[str]]
Heuristic = Union[str, Callable[[Coordinate, Coordinate], float]]

SAMPLE_START: Coordinate = (0, 0)
SAMPLE_GOAL: Coordinate = (3, 4)
SAMPLE_GRID = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]

# 固定顺序便于复现；同长最短路径中的任意一条均符合题目要求。
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # 右、下、左、上


@dataclass
class SearchResult:
    """详细结果。cost 是移动步数，无解为 -1；统计口径见 README。"""

    path: List[Coordinate]
    cost: int
    stats: Dict[str, int]
    trace: List[Dict[str, Any]]
    expanded_order: List[Coordinate]


def manhattan(node: Coordinate, goal: Coordinate) -> int:
    """四方向单位代价下，忽略障碍得到的剩余代价下界。"""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def chebyshev(node: Coordinate, goal: Coordinate) -> int:
    """允许对角移动的松弛距离；本题仍只生成四方向后继。"""
    return max(abs(node[0] - goal[0]), abs(node[1] - goal[1]))


def euclidean(node: Coordinate, goal: Coordinate) -> float:
    """直线距离下界，用于研究启发信息与计算成本。"""
    return math.hypot(node[0] - goal[0], node[1] - goal[1])


def _coordinate(value: Sequence[int], name: str) -> Coordinate:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(type(part) is not int for part in value)
    ):
        raise ValueError(f"{name} 必须是两个整数构成的坐标 (row, col)")
    return value[0], value[1]


def _shape(grid: Grid) -> Tuple[int, int]:
    if not isinstance(grid, (tuple, list)):
        raise ValueError("地图必须是二维列表或等效的规则二维数组")
    if not grid:
        return 0, 0
    if any(not isinstance(row, (tuple, list, str)) for row in grid):
        raise ValueError("地图的每一行必须是序列")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("地图必须是矩形，每一行的长度应相同")
    if any(cell not in ("S", "E", ".", "X") for row in grid for cell in row):
        raise ValueError("题一地图只允许 S、E、.、X 四种字符")
    return len(grid), width


def _reconstruct(parent: Dict[Coordinate, Coordinate], end: Coordinate) -> List[Coordinate]:
    path = [end]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def astar_search(
    start: Coordinate,
    end: Coordinate,
    grid: Grid,
    heuristic: Heuristic = "manhattan",
    tie_break: str = "fifo",
    record_trace: bool = False,
    neighbor_order: Optional[Sequence[Coordinate]] = None,
) -> SearchResult:
    """执行 A*；h=0 时为一致代价搜索（UCS）。

    start/end 明确决定起终点，S/E 标记仅用于地图展示。
    不修改输入地图。非法输入抛出 ValueError；合法但无路返回空路径。
    low_h 只在 f 相同时优先较小 h，始终以 f=g+h 为第一关键字。
    自定义启发函数接收 (node, end)，应返回非负、有限的可采纳下界。
    neighbor_order 仅允许四个基本方向的排列，不能改变题目的移动规则。
    """
    heuristics = {
        "zero": lambda node, goal: 0,
        "chebyshev": chebyshev,
        "euclidean": euclidean,
        "manhattan": manhattan,
    }
    if isinstance(heuristic, str) and heuristic in heuristics:
        heuristic_function = heuristics[heuristic]
    elif callable(heuristic):
        heuristic_function = heuristic
    else:
        raise ValueError("heuristic 应为 zero/chebyshev/euclidean/manhattan 或启发函数")
    if tie_break not in ("fifo", "lifo", "low_h"):
        raise ValueError("tie_break 应为 'fifo'、'lifo' 或 'low_h'")
    if neighbor_order is None:
        directions = DIRECTIONS
    else:
        try:
            directions = tuple(_coordinate(item, "方向") for item in neighbor_order)
        except TypeError as exc:
            raise ValueError("neighbor_order 必须是四个基本方向的排列") from exc
        if len(directions) != 4 or set(directions) != set(DIRECTIONS):
            raise ValueError("neighbor_order 必须是右、下、左、上四个方向的排列")
    start = _coordinate(start, "起点")
    end = _coordinate(end, "终点")
    height, width = _shape(grid)
    stats = {
        "expanded_nodes": 0,
        "generated_nodes": 0,
        "popped_nodes": 0,
        "max_frontier": 0,
        "stale_pops": 0,
        "reopened_nodes": 0,
        "unique_reached": 0,
    }
    trace: List[Dict[str, Any]] = []
    expanded_order: List[Coordinate] = []
    if height == 0 or width == 0:
        return SearchResult([], -1, stats, trace, expanded_order)
    for name, node in (("起点", start), ("终点", end)):
        if not (0 <= node[0] < height and 0 <= node[1] < width):
            raise ValueError(f"{name}坐标 {node} 超出地图范围")
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return SearchResult([], -1, stats, trace, expanded_order)

    def estimate(node: Coordinate) -> float:
        return heuristic_function(node, end)

    sequence = count()
    initial_h = estimate(start)
    # (f, 同 f 时的次关键字, 入队顺序, 坐标, 入队时的 g)
    frontier = [(initial_h, initial_h if tie_break == "low_h" else 0, next(sequence), start, 0)]
    best_g = {start: 0}
    parent: Dict[Coordinate, Coordinate] = {}
    expanded_g: Dict[Coordinate, int] = {}
    stats["generated_nodes"] = stats["max_frontier"] = 1

    while frontier:
        current_f, _, _, current, current_g = heapq.heappop(frontier)
        stats["popped_nodes"] += 1
        # 不需要在堆中查找并修改旧项；旧 g 失效后直接跳过。
        if current_g != best_g[current]:
            stats["stale_pops"] += 1
            continue
        if record_trace:
            trace.append({
                "step": len(trace) + 1,
                "node": current,
                "g": current_g,
                "h": current_f - current_g,
                "f": current_f,
            })
        # 目标必须从堆中有效取出后才结束，不能在首次生成时结束。
        if current == end:
            stats["unique_reached"] = len(best_g)
            return SearchResult(_reconstruct(parent, end), current_g, stats, trace, expanded_order)

        if current in expanded_g:
            stats["reopened_nodes"] += 1
        expanded_g[current] = current_g
        stats["expanded_nodes"] += 1
        if record_trace:
            expanded_order.append(current)

        row, col = current
        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if not (0 <= nr < height and 0 <= nc < width) or grid[nr][nc] == "X":
                continue
            new_g = current_g + 1  # 题一每步均为 1，包括进入终点。
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                parent[neighbor] = current
                new_h = estimate(neighbor)
                secondary = new_h if tie_break == "low_h" else 0
                insertion_order = next(sequence)
                if tie_break == "lifo":
                    insertion_order = -insertion_order
                heapq.heappush(frontier, (new_g + new_h, secondary, insertion_order, neighbor, new_g))
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(frontier))

    stats["unique_reached"] = len(best_g)
    return SearchResult([], -1, stats, trace, expanded_order)


def astar(start: Coordinate, end: Coordinate, grid: Grid) -> List[Coordinate]:
    """符合 PDF 输入输出约定的简洁接口：只返回坐标列表，无解返回 []。"""
    return astar_search(start, end, grid).path


def load_problem(text: str) -> Tuple[Coordinate, Coordinate, Grid]:
    """读取 PDF 格式：起点一行、终点一行、随后是可跨行的地图列表。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("输入应依次包含起点、终点和二维地图")
    try:
        start = ast.literal_eval(lines[0])
        end = ast.literal_eval(lines[1])
        grid = ast.literal_eval("\n".join(lines[2:]))
    except (ValueError, SyntaxError) as exc:
        raise ValueError("输入格式错误，请参照 samples/pdf_sample.txt；字符串须加引号") from exc
    return start, end, grid


def main() -> None:
    parser = argparse.ArgumentParser(description="题一：A* 迷宫最短路径")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--stdin", action="store_true", help="从标准输入读取 PDF 格式数据")
    source.add_argument("--input", type=Path, metavar="FILE", help="从 UTF-8 文本文件读取数据")
    parser.add_argument("--heuristic", choices=("zero", "chebyshev", "euclidean", "manhattan"), default="manhattan")
    parser.add_argument("--tie-break", choices=("fifo", "lifo", "low_h"), default="fifo")
    parser.add_argument("--details", action="store_true", help="输出含搜索轨迹和统计的 JSON")
    args = parser.parse_args()
    try:
        if args.stdin:
            start, end, grid = load_problem(sys.stdin.read())
        elif args.input:
            start, end, grid = load_problem(args.input.read_text(encoding="utf-8-sig"))
        else:
            start, end, grid = SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID
        result = astar_search(start, end, grid, args.heuristic, args.tie_break, args.details)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.details:
        print(json.dumps({
            "start": start, "end": end, "heuristic": args.heuristic,
            "tie_break": args.tie_break, "path": result.path, "cost": result.cost,
            "stats": result.stats, "trace": result.trace,
        }, ensure_ascii=False, indent=2))
    else:
        print(result.path)


if __name__ == "__main__":
    main()
