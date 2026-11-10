"""题一的静态连通分量索引：快照语义与完整调用成本均明确。

索引不替代 A*。只有起终点属于不同连通分量时，才直接判定无解。
query 每次仍扫描整张快照进行与原 A* 相同的地图验证；标签比较本身
是 O(1)，但完整拒绝调用不是 O(1)。改变原地图后需要重新建索引。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

from maze_astar import Coordinate, DIRECTIONS, Grid, Heuristic, SearchResult, _coordinate, _shape, astar_search


def _empty_result(rejected: bool = False) -> SearchResult:
    stats = {name: 0 for name in ("expanded_nodes", "generated_nodes", "popped_nodes",
                                "max_frontier", "stale_pops", "reopened_nodes", "unique_reached")}
    stats["precheck_rejected"] = int(rejected)
    return SearchResult([], -1, stats, [], [])


@dataclass(frozen=True)
class ConnectedGridIndex:
    """不可变地图快照、只读分量标签及实际建表成本。"""

    grid: Tuple[Tuple[str, ...], ...]
    labels: Mapping[Coordinate, int]
    component_sizes: Tuple[int, ...]
    preprocess_ms: float
    rows: int
    cols: int

    @property
    def component_count(self) -> int:
        return len(self.component_sizes)

    @property
    def label_entries(self) -> int:
        return len(self.labels)

    def query(self, start: Coordinate, end: Coordinate,
              heuristic: Heuristic = "manhattan", tie_break: str = "low_h") -> SearchResult:
        """返回完整 SearchResult；地图和坐标验证规则与题一基础接口一致。

        同分量时交给原 A*（原函数扫描地图一次）；异分量/空图/端点阻塞
        时在直接返回前扫描地图一次。不会把省掉 O(RC) 校验算成预检收益。
        """
        if not (callable(heuristic) or isinstance(heuristic, str)
                and heuristic in ("zero", "chebyshev", "euclidean", "manhattan")):
            raise ValueError("heuristic 应为 zero/chebyshev/euclidean/manhattan 或启发函数")
        if tie_break not in ("fifo", "lifo", "low_h"):
            raise ValueError("tie_break 应为 'fifo'、'lifo' 或 'low_h'")
        start = _coordinate(start, "起点")
        end = _coordinate(end, "终点")
        if self.rows == 0 or self.cols == 0:
            _shape(self.grid)
            return _empty_result()
        for name, node in (("起点", start), ("终点", end)):
            if not (0 <= node[0] < self.rows and 0 <= node[1] < self.cols):
                raise ValueError(f"{name}坐标 {node} 超出地图范围")
        start_label, end_label = self.labels.get(start), self.labels.get(end)
        if start_label is None or end_label is None:
            _shape(self.grid)
            return _empty_result()
        if start_label != end_label:
            _shape(self.grid)
            return _empty_result(rejected=True)
        result = astar_search(start, end, self.grid, heuristic=heuristic, tie_break=tie_break)
        result.stats["precheck_rejected"] = 0
        return result

    def metadata(self) -> dict:
        return {"shape": [self.rows, self.cols], "component_count": self.component_count,
                "component_sizes": list(self.component_sizes), "label_entries": self.label_entries,
                "preprocess_ms": self.preprocess_ms,
                "snapshot_semantics": "immutable copy; rebuild after map changes",
                "query_validation": "one full shape/character scan on every query branch"}


def build_connectivity_index(grid: Grid) -> ConnectedGridIndex:
    """用四方向 BFS 标记所有通行格，时间包括验证、复制、选点和标号。"""
    before = perf_counter_ns()
    rows, cols = _shape(grid)
    snapshot = tuple(tuple(row) for row in grid)
    labels: Dict[Coordinate, int] = {}
    sizes = []
    for row in range(rows):
        for col in range(cols):
            source = (row, col)
            if snapshot[row][col] == "X" or source in labels:
                continue
            component = len(sizes)
            labels[source] = component
            frontier = deque([source])
            size = 0
            while frontier:
                current_r, current_c = frontier.popleft()
                size += 1
                for dr, dc in DIRECTIONS:
                    neighbor = current_r + dr, current_c + dc
                    nr, nc = neighbor
                    if (0 <= nr < rows and 0 <= nc < cols and snapshot[nr][nc] != "X"
                            and neighbor not in labels):
                        labels[neighbor] = component
                        frontier.append(neighbor)
            sizes.append(size)
    elapsed_ms = (perf_counter_ns() - before) / 1_000_000
    return ConnectedGridIndex(snapshot, MappingProxyType(labels), tuple(sizes), elapsed_ms, rows, cols)
