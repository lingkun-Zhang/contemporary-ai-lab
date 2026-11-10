"""题一正确性探究：统一内核仅改变是否允许已扩展状态重新扩展。

allow_reopen=False 是故意保留错误策略的消融对照，不是求解器选项。
正式答案继续使用 maze_astar.astar；本模块不改变题目的四方向单位代价。
"""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import count
from math import isfinite

from maze_astar import DIRECTIONS, SearchResult, _coordinate, _reconstruct, _shape, manhattan


def corridor_case(width=9, branch_end=4, cutoff=4):
    """顶层直达；下层只开放到 branch_end，制造较长的汇合路径。"""
    if not (type(width) is int and type(branch_end) is int and type(cutoff) is int):
        raise ValueError("走廊参数应为整数")
    if not (width >= 4 and 1 <= branch_end < width - 1 and 1 <= cutoff < width):
        raise ValueError("走廊参数超出范围")
    grid = [list("S" + "." * (width - 2) + "E"),
            ["." if col <= branch_end else "X" for col in range(width)]]
    start, goal = (0, 0), (0, width - 1)
    table = {(row, col): width - 1 - col if row == 0 and col < cutoff else 0
             for row in range(2) for col in range(width) if grid[row][col] != "X"}
    return grid, start, goal, table


def table_heuristic(table):
    """实验表只对应记录的固定终点；不能当成任意目标可复用的启发。"""
    return lambda node, goal: table[node]


def analyze_heuristic(grid, goal, heuristic, exact_distances):
    """独立距离由调用者的 BFS 提供；不可达节点不参与有限下界检验。"""
    values = {(r, c): heuristic((r, c), goal) for r, row in enumerate(grid)
              for c, cell in enumerate(row) if cell != "X"}
    invalid = sum(not isfinite(value) or value < 0 for value in values.values())
    overestimates = sum(values[node] > distance for node, distance in exact_distances.items())
    violations = []
    for node, value in values.items():
        for dr, dc in DIRECTIONS:
            neighbor = node[0] + dr, node[1] + dc
            if neighbor in values and value > 1 + values[neighbor]:
                violations.append({"from": node, "to": neighbor, "h_from": value,
                                   "h_to": values[neighbor]})
    return {"admissibility_violations": overestimates, "invalid_values": invalid,
            "goal_value": values[goal], "consistency_violation_edges": len(violations),
            "finite_distance_nodes_checked": len(exact_distances),
            "inconsistent_edges": violations}


def probe_astar(start, end, grid, heuristic=manhattan, allow_reopen=True, record_trace=False):
    """FIFO 并列的 A* 对照内核；唯一的策略开关是冻结已扩展节点。"""
    if not callable(heuristic):
        raise ValueError("heuristic 必须是接收(node, goal)的函数")
    start, end = _coordinate(start, "起点"), _coordinate(end, "终点")
    height, width = _shape(grid)
    stats = {"expanded_nodes": 0, "generated_nodes": 0, "popped_nodes": 0,
             "max_frontier": 0, "stale_pops": 0, "reopened_nodes": 0,
             "unique_reached": 0, "f_decrease_events": 0,
             "ignored_closed_improvements": 0}
    trace, expanded_order = [], []
    if not height or not width:
        return SearchResult([], -1, stats, trace, expanded_order)
    for node in (start, end):
        if not (0 <= node[0] < height and 0 <= node[1] < width):
            raise ValueError("坐标超出地图范围")
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return SearchResult([], -1, stats, trace, expanded_order)
    sequence = count()
    frontier = [(heuristic(start, end), next(sequence), start, 0)]
    best_g, parent, expanded_g = {start: 0}, {}, {}
    stats["generated_nodes"] = stats["max_frontier"] = 1
    previous_f = None
    while frontier:
        f, _, node, g = heappop(frontier)
        stats["popped_nodes"] += 1
        if g != best_g[node]:
            stats["stale_pops"] += 1
            continue
        if previous_f is not None and f < previous_f:
            stats["f_decrease_events"] += 1
        previous_f = f
        reopened = node in expanded_g
        if record_trace:
            trace.append({"step": len(trace) + 1, "node": node, "g": g,
                          "h": f - g, "f": f, "reopened": reopened,
                          "is_goal": node == end})
        if node == end:
            stats["unique_reached"] = len(best_g)
            return SearchResult(_reconstruct(parent, end), g, stats, trace, expanded_order)
        stats["expanded_nodes"] += 1
        stats["reopened_nodes"] += reopened
        expanded_g[node] = g
        if record_trace:
            expanded_order.append(node)
        for dr, dc in DIRECTIONS:
            neighbor = node[0] + dr, node[1] + dc
            nr, nc = neighbor
            if not (0 <= nr < height and 0 <= nc < width) or grid[nr][nc] == "X":
                continue
            new_g = g + 1
            # 错误消融必须先阻止闭集改进；不能只改父指针而不传播代价。
            if not allow_reopen and neighbor in expanded_g:
                if new_g < best_g[neighbor]:
                    stats["ignored_closed_improvements"] += 1
                continue
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor], parent[neighbor] = new_g, node
                heappush(frontier, (new_g + heuristic(neighbor, end), next(sequence), neighbor, new_g))
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(frontier))
    stats["unique_reached"] = len(best_g)
    return SearchResult([], -1, stats, trace, expanded_order)
