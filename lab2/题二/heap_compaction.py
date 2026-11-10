"""题二可选优化：按物理堆/有效 open 比例清理过期项，保留 A* 优先级。

清理只在一次有效扩展的全部松弛完成后检查；rho=inf 为同内核不清理。
它仍维护 open 集合，因此实验另与原 astar_search 同批计时，披露额外管理成本。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from itertools import accumulate, count
from typing import Dict

from weighted_astar import (SAFE_HEURISTICS, WeightedResult, heuristic_value,
                            prepare_terrain)


@dataclass
class CompactedResult(WeightedResult):
    compaction_stats: Dict[str, int] = field(default_factory=dict)
    rho: float = math.inf


def validate_rho(rho):
    if type(rho) not in (int, float) or rho <= 1 or (type(rho) is float and math.isnan(rho)):
        raise ValueError("rho 必须大于 1，或为正无穷 inf；不能是布尔值或 NaN")
    return rho


def compacted_astar_search(start, end, grid, heuristic="corrected", tie_break="fifo",
                           record_trace=False, min_scope="global", rho=math.inf):
    """进入格收费的精确 A*；仅删除 current_g != best_g 的旧堆项。

    所有保留项的 (f, secondary, serial, node, g) 元组不变，故有效出队、
    父指针和返回路径不因清理改变。自定义启发仍须满足原接口的正确性条件。
    max_frontier 包含触发清理前的实际瞬时峰值。扫描与 heapify 均计入调用。
    """
    rho = validate_rho(rho)
    if not (callable(heuristic) or isinstance(heuristic, str) and heuristic in SAFE_HEURISTICS):
        raise ValueError("heuristic 应为 zero/corrected/row_column/row_column_tight 或函数")
    if tie_break not in ("fifo", "low_h"):
        raise ValueError("tie_break 应为 fifo 或 low_h")
    terrain = prepare_terrain(start, end, grid, min_scope)
    start, end = terrain.start, terrain.end
    stats = dict(expanded_nodes=0, generated_nodes=0, popped_nodes=0,
                 max_frontier=0, stale_pops=0, reopened_nodes=0, unique_reached=0)
    compaction = dict(compactions=0, removed_stale=0, scanned_entries=0,
                      trigger_checks=0, peak_valid_open=0,
                      final_heap_size=0, final_valid_open=0)
    trace, expanded_order, frontier, open_nodes = [], [], [], set()

    def result(path, cost):
        compaction["final_heap_size"] = len(frontier)
        compaction["final_valid_open"] = len(open_nodes)
        ledger = [0] + [terrain.costs[r][c] for r, c in path[1:]] if path else []
        answer = CompactedResult(path, cost, stats, trace, ledger, list(accumulate(ledger)),
                                 terrain.min_step_cost, expanded_order, terrain.min_scope,
                                 terrain.component_size, compaction, rho)
        if stats["generated_nodes"] != stats["popped_nodes"] + compaction["removed_stale"] + len(frontier):
            raise AssertionError("入堆、弹出、清理和末尾堆记录必须守恒")
        return answer

    if not terrain.rows or not terrain.cols:
        return result([], -1)
    if terrain.costs[start[0]][start[1]] is None or terrain.costs[end[0]][end[1]] is None:
        return result([], -1)

    def estimate(node):
        return heuristic(node, end) if callable(heuristic) else heuristic_value(terrain, node, heuristic)

    serial = count()
    initial_h = estimate(start)
    frontier.append((initial_h, initial_h if tie_break == "low_h" else 0, next(serial), start, 0))
    open_nodes.add(start)
    best_g, parent, expanded_g = {start: 0}, {}, {}
    stats["generated_nodes"] = stats["max_frontier"] = compaction["peak_valid_open"] = 1
    while frontier:
        current_f, _, _, current, current_g = heapq.heappop(frontier)
        stats["popped_nodes"] += 1
        if current_g != best_g[current]:
            stats["stale_pops"] += 1
            continue
        # 旧项弹出不能删除同格仍在 open 中的更优记录。
        open_nodes.remove(current)
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
                open_nodes.add(neighbor)
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(frontier))
                compaction["peak_valid_open"] = max(compaction["peak_valid_open"], len(open_nodes))
        if rho != math.inf and frontier:
            compaction["trigger_checks"] += 1
            if len(frontier) > rho * len(open_nodes):
                before = len(frontier)
                compaction["scanned_entries"] += before
                frontier[:] = [entry for entry in frontier if entry[4] == best_g[entry[3]]]
                removed = before - len(frontier)
                if not removed:
                    raise AssertionError("触发阈值时应存在过期记录")
                heapq.heapify(frontier)
                compaction["compactions"] += 1
                compaction["removed_stale"] += removed
    stats["unique_reached"] = len(best_g)
    return result([], -1)
