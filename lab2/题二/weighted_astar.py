"""题二：进入目标格收费的加权 A*；指定终点进入代价为 0。

默认仅输出 PDF 要求的整数最小代价。详细接口保留路径、逐步费用和统计。
输入权重限定非负整数；X 障碍和普通 0 权格是支持无解/边界探究的拓展。
"""

from __future__ import annotations

import argparse
import ast
import heapq
import json
import sys
from dataclasses import dataclass
from itertools import accumulate, count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


Coordinate = Tuple[int, int]
Grid = Sequence[Sequence[Union[int, str]]]
Heuristic = Union[str, Callable[[Coordinate, Coordinate], float]]
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
SAMPLE_START = (0, 0)
SAMPLE_GOAL = (4, 4)
SAMPLE_GRID = [
    ["S", 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, "E"],
]
SAFE_HEURISTICS = ("zero", "corrected", "row_column", "row_column_tight")


def _coordinate(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or any(type(part) is not int for part in value)):
        raise ValueError(f"{name} 必须是两个整数构成的坐标 (row, col)")
    return value[0], value[1]


@dataclass(frozen=True)
class PreparedTerrain:
    costs: Tuple[Tuple[Optional[int], ...], ...]
    start: Coordinate
    end: Coordinate
    rows: int
    cols: int
    min_step_cost: int
    row_mins: Tuple[int, ...]
    col_mins: Tuple[int, ...]
    row_lower_bounds: Tuple[int, ...]
    col_lower_bounds: Tuple[int, ...]
    min_scope: str = "global"
    component_size: Optional[int] = None

    def neighbors(self, node: Coordinate):
        """返回有效后继；删除返回起点的边，避免给 S 人为指定再次收费。"""
        row, col = node
        for dr, dc in DIRECTIONS:
            neighbor = row + dr, col + dc
            nr, nc = neighbor
            if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and neighbor != self.start and self.costs[nr][nc] is not None):
                yield neighbor


@dataclass
class WeightedResult:
    path: List[Coordinate]
    cost: int
    stats: Dict[str, int]
    trace: List[Dict[str, Any]]
    path_costs: List[int]
    cumulative_costs: List[int]
    min_step_cost: int
    expanded_order: List[Coordinate]
    min_scope: str = "global"
    component_size: Optional[int] = None

    @property
    def steps(self) -> int:
        return len(self.path) - 1 if self.path else -1


def _axis_bounds(minima: Sequence[int], goal_index: int) -> Tuple[int, ...]:
    answer = [0] * len(minima)
    for index in range(goal_index - 1, -1, -1):
        answer[index] = answer[index + 1] + minima[index + 1]
    for index in range(goal_index + 1, len(minima)):
        answer[index] = answer[index - 1] + minima[index - 1]
    return tuple(answer)


def prepare_terrain(start: Coordinate, end: Coordinate, grid: Grid,
                    min_scope: str = "global") -> PreparedTerrain:
    """验证并建立本次查询的不可变权重快照，预处理时间包含在公开搜索调用中。

    起点初始代价为 0；指定 end 按题目规定始终免费。非负路径上的回到
    起点的环可以删掉，不会增加费用，故禁止重新进入起点不损失最优解。
    reachable 模式只在 S 可达区域内取 m；它完整遍历区域但不提前判无解，
    用来隔离下界改善的作用。其下界保证适用于本次搜索能进入的区域。
    """
    if min_scope not in ("global", "reachable"):
        raise ValueError("min_scope 应为 global 或 reachable")
    start, end = _coordinate(start, "起点"), _coordinate(end, "终点")
    if not isinstance(grid, (tuple, list)):
        raise ValueError("地图必须是二维列表或等效规则数组")
    if not grid:
        return PreparedTerrain((), start, end, 0, 0, 0, (), (), (), (),
                               min_scope, 0 if min_scope == "reachable" else None)
    if any(not isinstance(row, (tuple, list, str)) for row in grid):
        raise ValueError("地图的每行必须是序列")
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("地图必须是矩形")
    if cols == 0:
        return PreparedTerrain(tuple(() for _ in grid), start, end, rows, 0, 0,
                               tuple(0 for _ in grid), (), tuple(0 for _ in grid), (),
                               min_scope, 0 if min_scope == "reachable" else None)
    for name, node in (("起点", start), ("终点", end)):
        if not (0 <= node[0] < rows and 0 <= node[1] < cols):
            raise ValueError(f"{name}坐标 {node} 超出地图范围")
    costs = []
    for r, row in enumerate(grid):
        parsed = []
        for c, value in enumerate(row):
            node = (r, c)
            if isinstance(value, str):
                text = value.strip()
                if text == "X":
                    parsed.append(None)
                    continue
                if text in ("S", "E"):
                    expected = start if text == "S" else end
                    if node != expected:
                        raise ValueError(f"标记 {text} 必须位于指定坐标 {expected}")
                    weight = 0
                elif text.isdecimal():
                    weight = int(text)
                else:
                    raise ValueError(f"格子 {node} 必须为非负整数、S、E 或 X")
            elif type(value) is int and value >= 0:
                weight = value
            else:
                raise ValueError(f"格子 {node} 的权重必须为非负整数，不能是负数、小数或布尔值")
            parsed.append(0 if node in (start, end) else weight)
        costs.append(tuple(parsed))
    costs = tuple(costs)
    component_size = None
    if min_scope == "reachable":
        # 删除进入 S 的边不影响从 S 出发的可达集合；seen 也避免重复返回 S。
        seen = {start} if costs[start[0]][start[1]] is not None else set()
        pending, values = list(seen), []
        while pending:
            node = pending.pop()
            r, c = node
            if node not in (start, end):
                values.append(costs[r][c])
            for dr, dc in DIRECTIONS:
                other = r + dr, c + dc
                nr, nc = other
                if (0 <= nr < rows and 0 <= nc < cols and other not in seen
                        and costs[nr][nc] is not None):
                    seen.add(other)
                    pending.append(other)
        component_size = len(seen)
    else:
        values = [costs[r][c] for r in range(rows) for c in range(cols)
                  if costs[r][c] is not None and (r, c) not in (start, end)]
    minimum = min(values, default=0)
    row_mins = tuple(min((value for value in row if value is not None), default=0)
                     for row in costs)
    col_mins = tuple(min((costs[r][c] for r in range(rows) if costs[r][c] is not None), default=0)
                     for c in range(cols))
    return PreparedTerrain(costs, start, end, rows, cols, minimum, row_mins, col_mins,
                           _axis_bounds(row_mins, end[0]), _axis_bounds(col_mins, end[1]),
                           min_scope, component_size)


def heuristic_value(terrain: PreparedTerrain, node: Coordinate, name: str) -> int:
    """安全下界；reachable 的保证仅针对 S 可达区域，不用于其他分量。"""
    distance = abs(node[0] - terrain.end[0]) + abs(node[1] - terrain.end[1])
    if name == "zero":
        return 0
    corrected = terrain.min_step_cost * max(distance - 1, 0)
    if name == "corrected":
        return corrected
    if name in ("row_column", "row_column_tight"):
        bound = terrain.row_lower_bounds[node[0]] + terrain.col_lower_bounds[node[1]]
        if name == "row_column_tight" and node[0] != terrain.end[0] and node[1] != terrain.end[1]:
            # 两轴均未对齐时，首次进入目标行和目标列是不同动作，至少一次
            # 到达非终点；该次代价至少 m。E 免费只能免除最后一个动作。
            bound += terrain.min_step_cost
        return max(corrected, bound)
    raise ValueError(f"未知的安全启发函数：{name}")


def astar_search(start: Coordinate, end: Coordinate, grid: Grid,
                 heuristic: Heuristic = "corrected", tie_break: str = "fifo",
                 record_trace: bool = False, min_scope: str = "global") -> WeightedResult:
    """四方向加权 A*；自定义启发函数仅供探究，最优保证依赖其正确下界条件。"""
    if not (callable(heuristic) or isinstance(heuristic, str) and heuristic in SAFE_HEURISTICS):
        raise ValueError("heuristic 应为 zero/corrected/row_column/row_column_tight 或函数")
    if tie_break not in ("fifo", "low_h"):
        raise ValueError("tie_break 应为 fifo 或 low_h")
    terrain = prepare_terrain(start, end, grid, min_scope)
    start, end = terrain.start, terrain.end
    stats = dict(expanded_nodes=0, generated_nodes=0, popped_nodes=0,
                 max_frontier=0, stale_pops=0, reopened_nodes=0, unique_reached=0)
    trace, expanded_order = [], []

    def result(path, cost):
        ledger = [0] + [terrain.costs[r][c] for r, c in path[1:]] if path else []
        return WeightedResult(path, cost, stats, trace, ledger, list(accumulate(ledger)),
                              terrain.min_step_cost, expanded_order,
                              terrain.min_scope, terrain.component_size)

    if not terrain.rows or not terrain.cols:
        return result([], -1)
    if terrain.costs[start[0]][start[1]] is None or terrain.costs[end[0]][end[1]] is None:
        return result([], -1)

    def estimate(node):
        return heuristic(node, end) if callable(heuristic) else heuristic_value(terrain, node, heuristic)

    serial = count()
    initial_h = estimate(start)
    frontier = [(initial_h, initial_h if tie_break == "low_h" else 0, next(serial), start, 0)]
    best_g, parent, expanded_g = {start: 0}, {}, {}
    stats["generated_nodes"] = stats["max_frontier"] = 1
    while frontier:
        current_f, _, _, current, current_g = heapq.heappop(frontier)
        stats["popped_nodes"] += 1
        if current_g != best_g[current]:
            stats["stale_pops"] += 1
            continue
        if record_trace:
            trace.append(dict(step=len(trace) + 1, node=current, g=current_g,
                              h=current_f-current_g, f=current_f))
        if current == end:
            path = [end]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            stats["unique_reached"] = len(best_g)
            return result(path, current_g)
        stats["expanded_nodes"] += 1
        if current in expanded_g:
            stats["reopened_nodes"] += 1
        expanded_g[current] = current_g
        if record_trace:
            expanded_order.append(current)
        for neighbor in terrain.neighbors(current):
            nr, nc = neighbor
            new_g = current_g + terrain.costs[nr][nc]
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor], parent[neighbor] = new_g, current
                h = estimate(neighbor)
                heapq.heappush(frontier, (new_g+h, h if tie_break == "low_h" else 0,
                                          next(serial), neighbor, new_g))
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(frontier))
    stats["unique_reached"] = len(best_g)
    return result([], -1)


def astar_cost(start: Coordinate, end: Coordinate, grid: Grid) -> int:
    """符合 PDF 的正式接口：返回最小总代价，无解 -1。"""
    return astar_search(start, end, grid).cost


def load_problem(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("依次输入起点一行、终点一行、可跨行的二维地图")
    try:
        start, end = ast.literal_eval(lines[0]), ast.literal_eval(lines[1])
        grid = ast.parse("\n".join(lines[2:]), mode="eval").body
        # 题面直接写 S/E；仅将二维列表或元组的独立格子标记转为字符串。
        if isinstance(grid, (ast.List, ast.Tuple)):
            for row in grid.elts:
                if isinstance(row, (ast.List, ast.Tuple)):
                    for i, cell in enumerate(row.elts):
                        if isinstance(cell, ast.Name) and cell.id in {"S", "E", "X"}:
                            row.elts[i] = ast.copy_location(ast.Constant(cell.id), cell)
        return start, end, ast.literal_eval(grid)
    except (ValueError, SyntaxError) as exc:
        raise ValueError("输入格式不正确；请参照 samples/pdf_sample.txt，S/E/X 可带或不带引号") from exc


def main():
    parser = argparse.ArgumentParser(description="题二：加权地形 A* 最小代价（终点进入免费）")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--stdin", action="store_true")
    source.add_argument("--input", type=Path, metavar="FILE")
    parser.add_argument("--heuristic", choices=SAFE_HEURISTICS, default="corrected")
    parser.add_argument("--tie-break", choices=("fifo", "low_h"), default="fifo")
    parser.add_argument("--min-scope", choices=("global", "reachable"), default="global")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    try:
        if args.stdin:
            start, end, grid = load_problem(sys.stdin.read())
        elif args.input:
            start, end, grid = load_problem(args.input.read_text(encoding="utf-8-sig"))
        else:
            start, end, grid = SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID
        answer = astar_search(start, end, grid, args.heuristic, args.tie_break,
                              args.details, args.min_scope)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.details:
        print(json.dumps(dict(cost=answer.cost, path=answer.path, steps=answer.steps,
                             path_costs=answer.path_costs, cumulative_costs=answer.cumulative_costs,
                             min_step_cost=answer.min_step_cost, min_scope=answer.min_scope,
                             component_size=answer.component_size,
                             stats=answer.stats, trace=answer.trace),
                         ensure_ascii=False, indent=2))
    else:
        print(answer.cost)


if __name__ == "__main__":
    main()
