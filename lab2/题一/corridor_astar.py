"""静态四邻接单位权网格的精确走廊收缩（Python 3.9+，标准库）。

只删除度为2的内部节点；保留分岔、端点、孤立点及纯环的一个锚点。
查询时将任意内部起终点接回图中。边保存原坐标序列，因此返回的仍是
逐格路径。不是完整的 Contraction Hierarchies 或矩形对称消除算法。
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from typing import Callable, Dict, List, Tuple, Union

from maze_astar import DIRECTIONS, SearchResult, _coordinate, _shape, manhattan

Coordinate = Tuple[int, int]
# (邻居坐标, 原走廊编号, 本端偏移, 另一端偏移)。两偏移之差是边权。
Arc = Tuple[Coordinate, int, int, int]
Heuristic = Union[str, Callable[[Coordinate, Coordinate], float]]


@dataclass(frozen=True)
class CorridorIndex:
    """建成后只读使用；grid快照防止调用者改图后悄悄得到错误答案。"""

    grid: Tuple[str, ...]
    shape: Tuple[int, int]
    adjacency: Dict[Coordinate, Tuple[Arc, ...]]
    paths: Tuple[Tuple[Coordinate, ...], ...]
    interiors: Dict[Coordinate, Tuple[int, int]]
    metadata: dict

    def search(self, start, end, heuristic: Heuristic = "manhattan",
               tie_break: str = "low_h", record_trace: bool = False) -> SearchResult:
        """A*搜索宏边图，端点接入及逐格路径还原包含在调用内部。

        expanded_nodes只统计展开的宏图顶点，不能当作逐格检查次数。
        搜索不进行连通分量提前拒绝，也不缓存跨查询答案。
        """
        if heuristic == "manhattan":
            estimate = manhattan
        elif heuristic == "zero":
            estimate = lambda node, goal: 0
        elif callable(heuristic):
            estimate = heuristic
        else:
            raise ValueError("heuristic应为manhattan、zero或可采纳函数")
        if tie_break not in ("fifo", "low_h"):
            raise ValueError("tie_break应为fifo或low_h")
        start, end = _coordinate(start, "起点"), _coordinate(end, "终点")
        stats = dict(expanded_nodes=0, generated_nodes=0, popped_nodes=0,
                     max_frontier=0, stale_pops=0, reopened_nodes=0,
                     unique_reached=0, relaxed_arcs=0, inserted_terminals=0,
                     split_corridors=0, output_steps=0)
        trace, expanded_order = [], []
        height, width = self.shape
        if not height or not width:
            return SearchResult([], -1, stats, trace, expanded_order)
        for node in (start, end):
            if not (0 <= node[0] < height and 0 <= node[1] < width):
                raise ValueError("起终点坐标超出地图范围")
        if any(self.grid[r][c] == "X" for r, c in (start, end)):
            return SearchResult([], -1, stats, trace, expanded_order)

        # 最多切分两条走廊；不复制整个图。自环两端偏移0/L仍各保留。
        cuts: Dict[int, set] = {}
        for node in set((start, end)):
            if node in self.interiors:
                edge, offset = self.interiors[node]
                cuts.setdefault(edge, set()).add(offset)
                stats["inserted_terminals"] += 1
        overlay: Dict[Coordinate, List[Arc]] = {}
        for edge, offsets in cuts.items():
            path = self.paths[edge]
            points = sorted({0, len(path) - 1} | offsets)
            for a, b in zip(points, points[1:]):
                overlay.setdefault(path[a], []).append((path[b], edge, a, b))
                overlay.setdefault(path[b], []).append((path[a], edge, b, a))
        stats["split_corridors"] = len(cuts)

        serial = count()
        h = estimate(start, end)
        queue = [(h, h if tie_break == "low_h" else 0, next(serial), start, 0)]
        best_g, parent, expanded_g = {start: 0}, {}, {}
        stats["generated_nodes"] = stats["max_frontier"] = 1
        while queue:
            f, _, _, node, g = heappop(queue)
            stats["popped_nodes"] += 1
            if best_g[node] != g:
                stats["stale_pops"] += 1
                continue
            if record_trace:
                trace.append(dict(step=len(trace) + 1, node=node, g=g, h=f-g, f=f))
            if node == end:
                segments, current = [], end
                while current in parent:
                    previous, edge, a, b = parent[current]
                    segments.append((edge, a, b))
                    current = previous
                path = [start]
                for edge, a, b in reversed(segments):
                    original = self.paths[edge]
                    step = 1 if b > a else -1
                    path.extend(original[i] for i in range(a + step, b + step, step))
                stats["output_steps"] = len(path) - 1
                stats["unique_reached"] = len(best_g)
                return SearchResult(path, g, stats, trace, expanded_order)
            if node in expanded_g:
                stats["reopened_nodes"] += 1
            expanded_g[node] = g
            stats["expanded_nodes"] += 1
            if record_trace:
                expanded_order.append(node)

            # 先列原边，再列端点切分边；只屏蔽被切分的原边编号。
            arcs = (arc for arc in self.adjacency.get(node, ()) if arc[1] not in cuts)
            for arc in _chain_arcs(arcs, overlay.get(node, ())):
                neighbor, edge, a, b = arc
                stats["relaxed_arcs"] += 1
                new_g = g + abs(b-a)
                if new_g < best_g.get(neighbor, float("inf")):
                    best_g[neighbor] = new_g
                    parent[neighbor] = (node, edge, a, b)
                    h = estimate(neighbor, end)
                    heappush(queue, (new_g+h, h if tie_break == "low_h" else 0,
                                     next(serial), neighbor, new_g))
                    stats["generated_nodes"] += 1
                    stats["max_frontier"] = max(stats["max_frontier"], len(queue))
        stats["unique_reached"] = len(best_g)
        return SearchResult([], -1, stats, trace, expanded_order)


def _chain_arcs(first, second):
    yield from first
    yield from second


def build_corridor_index(grid, contract: bool = True) -> CorridorIndex:
    """O(RC+V+E)建图；contract=False是同一实现的逐格图消融。

    每条原始无向边恰好属于一个宏边。保留多重边和自环，不把不同
    走廊误合并成一条。索引包含路径资料，存储仍为O(RC+V+E)。
    """
    if type(contract) is not bool:
        raise ValueError("contract必须为布尔值")
    shape = _shape(grid)
    snapshot = tuple("".join(row) for row in grid)
    neighbors = {}
    for r, row in enumerate(snapshot):
        for c, cell in enumerate(row):
            if cell != "X":
                neighbors[(r, c)] = tuple((r+dr, c+dc) for dr, dc in DIRECTIONS
                    if 0 <= r+dr < shape[0] and 0 <= c+dc < shape[1]
                    and snapshot[r+dr][c+dc] != "X")
    retained = {node for node, adjacent in neighbors.items()
                if not contract or len(adjacent) != 2}
    # 每个连通分量至少保留一个点，特别是全度2的纯环。
    seen = set()
    pure_cycles = 0
    for initial in neighbors:
        if initial in seen:
            continue
        pending, component, has_retained = [initial], [], False
        seen.add(initial)
        while pending:
            node = pending.pop()
            component.append(node)
            has_retained = has_retained or node in retained
            for other in neighbors[node]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
        if not has_retained:
            retained.add(min(component))
            pure_cycles += 1

    adjacency = {node: [] for node in neighbors if node in retained}
    visited_edges, paths, interiors = set(), [], {}

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    for initial in adjacency:
        for neighbor in neighbors[initial]:
            if edge_key(initial, neighbor) in visited_edges:
                continue
            path = [initial, neighbor]
            visited_edges.add(edge_key(initial, neighbor))
            previous, current = initial, neighbor
            while current not in retained:
                following = next(node for node in neighbors[current] if node != previous)
                visited_edges.add(edge_key(current, following))
                path.append(following)
                previous, current = current, following
            edge = len(paths)
            paths.append(tuple(path))
            length = len(path)-1
            adjacency[initial].append((current, edge, 0, length))
            adjacency[current].append((initial, edge, length, 0))
            for offset, node in enumerate(path[1:-1], 1):
                interiors[node] = (edge, offset)
    original_edges = sum(map(len, neighbors.values())) // 2
    if len(visited_edges) != original_edges:
        raise AssertionError("存在未分配的原始网格边")
    metadata = dict(contract=contract, walkable_cells=len(neighbors),
                    grid_edges=original_edges, retained_nodes=len(adjacency),
                    corridor_edges=len(paths), degree_two_cells=sum(
                        len(items) == 2 for items in neighbors.values()),
                    pure_cycle_anchors=pure_cycles,
                    stored_path_coordinates=sum(map(len, paths)),
                    max_corridor_length=max((len(path)-1 for path in paths), default=0),
                    removed_nodes=len(interiors))
    def first_step_order(arc):
        _, edge, a, b = arc
        path = paths[edge]
        following = path[a + (1 if b > a else -1)]
        current = path[a]
        return DIRECTIONS.index((following[0]-current[0], following[1]-current[1]))

    # identity消融保持正式网格的右下左上顺序；压缩后宏边也按首步排序。
    # 动态切分会改变宏边生成顺序，因此不宣称与逐格搜索轨迹完全相同。
    return CorridorIndex(snapshot, shape,
                         {node: tuple(sorted(arcs, key=first_step_order))
                          for node, arcs in adjacency.items()},
                         tuple(paths), interiors, metadata)
