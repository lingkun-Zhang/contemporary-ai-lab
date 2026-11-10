"""用于题一拓展实验的地标启发函数（纯标准库，Python 3.9+）。

适用范围是静态、无向、四方向、单位边代价的网格；改变地图后必须重建。
每个地标 L 通过一次 BFS 保存 d(L, ·)，查询时使用三角不等式：

    h(n, t) = max(Manhattan(n, t), max_L |d(L, n) - d(L, t)|).

预处理只依赖地图，不依赖某次查询的起终点。偶然选中终点作为地标仍属
固定选点规则的结果，不是以该查询终点进行反向 BFS 的“预知答案”基线。
地标不覆盖 n 和 t 时跳过该项，不据此提前拒绝搜索。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Callable, Dict, List, Sequence, Tuple


Coordinate = Tuple[int, int]
DistanceTable = Dict[Coordinate, int]
Heuristic = Callable[[Coordinate, Coordinate], int]
_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass
class LandmarkIndex:
    """地标距离表及实测构建开销。

    cumulative_preprocess_ms[k-1] 是构建前 k 个地标的累计墙钟耗时，
    包括地图检查、选点、BFS 与维护下次选点所需的最短距离。
    distance_tables 的字典条目数用于可复现的存储规模比较，不冒充字节数。
    公开列表供实验检查；构建后应视为只读，避免使已经取得的启发函数失效。
    """

    landmarks: List[Coordinate]
    distance_tables: List[DistanceTable]
    cumulative_preprocess_ms: List[float]
    grid_shape: Tuple[int, int]
    walkable_cells: int

    def heuristic(self, k: int) -> Heuristic:
        """返回仅使用前 k 个地标的 h(node, goal)；k=0 为曼哈顿基线。"""
        if type(k) is not int or not 0 <= k <= len(self.landmarks):
            raise ValueError("k 必须是 0 到已构建地标数之间的整数")
        tables = tuple(self.distance_tables[:k])

        def estimate(node: Coordinate, goal: Coordinate) -> int:
            bound = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
            for distances in tables:
                node_distance = distances.get(node)
                goal_distance = distances.get(goal)
                if node_distance is not None and goal_distance is not None:
                    bound = max(bound, abs(node_distance - goal_distance))
            return bound

        return estimate

    def metadata(self) -> Dict[str, Any]:
        """返回便于写入 JSON 的信息，列出每个地标前缀的实际开销。"""
        entry_counts: List[int] = []
        entries = 0
        for distances in self.distance_tables:
            entries += len(distances)
            entry_counts.append(entries)
        return {
            "method": "manhattan_plus_landmark_triangle_bound",
            "applicability": "static_undirected_four_direction_unit_cost_grid",
            "selection": "lexicographic_first_then_farthest_with_uncovered_components_first",
            "grid_shape": list(self.grid_shape),
            "walkable_cells": self.walkable_cells,
            "landmarks": [list(node) for node in self.landmarks],
            "distance_entries": entries,
            "cumulative_distance_entries": entry_counts,
            "cumulative_preprocess_ms": list(self.cumulative_preprocess_ms),
        }


def _open_cells(grid: Sequence[Sequence[str]]) -> Tuple[List[Coordinate], Tuple[int, int]]:
    if not isinstance(grid, (tuple, list)):
        raise ValueError("地图必须是二维列表或等效的规则二维数组")
    if not grid:
        return [], (0, 0)
    if any(not isinstance(row, (tuple, list, str)) for row in grid):
        raise ValueError("地图的每一行必须是序列")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("地图必须是矩形，每一行的长度应相同")
    if any(cell not in ("S", "E", ".", "X") for row in grid for cell in row):
        raise ValueError("题一地图只允许 S、E、.、X 四种字符")
    cells = [(row, col) for row, line in enumerate(grid)
             for col, cell in enumerate(line) if cell != "X"]
    return cells, (len(grid), width)


def _bfs(source: Coordinate, passable: set[Coordinate]) -> DistanceTable:
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        row, col = frontier.popleft()
        next_distance = distances[(row, col)] + 1
        for dr, dc in _DIRECTIONS:
            neighbor = (row + dr, col + dc)
            if neighbor in passable and neighbor not in distances:
                distances[neighbor] = next_distance
                frontier.append(neighbor)
    return distances


def build_landmark_index(
    grid: Sequence[Sequence[str]], max_landmarks: int = 4
) -> LandmarkIndex:
    """确定性选择地标并用 BFS 建表，不读取任何查询的起终点。

    第一个地标为字典序最小的通行格。后续先覆盖尚无地标的连通分量，
    否则选择到已有地标最短图距离的最小值最大的通行格，坐标打破并列。
    max_landmarks=0 或无通行格时返回空索引，其 heuristic(0) 仍可使用。
    """
    if type(max_landmarks) is not int or max_landmarks < 0:
        raise ValueError("max_landmarks 必须为非负整数")
    started_ns = perf_counter_ns()
    cells, shape = _open_cells(grid)
    passable = set(cells)
    landmarks: List[Coordinate] = []
    distance_tables: List[DistanceTable] = []
    cumulative_ms: List[float] = []
    minimum_distance: DistanceTable = {}
    selected: set[Coordinate] = set()

    for _ in range(min(max_landmarks, len(cells))):
        # cells 原本就是行列字典序；优先处理未覆盖分量。
        landmark = next((cell for cell in cells if cell not in minimum_distance), None)
        if landmark is None:
            remaining = (cell for cell in cells if cell not in selected)
            landmark = min(remaining, key=lambda cell: (-minimum_distance[cell], cell))
        distances = _bfs(landmark, passable)
        landmarks.append(landmark)
        selected.add(landmark)
        distance_tables.append(distances)
        for cell, distance in distances.items():
            old_distance = minimum_distance.get(cell)
            if old_distance is None or distance < old_distance:
                minimum_distance[cell] = distance
        cumulative_ms.append((perf_counter_ns() - started_ns) / 1_000_000)

    return LandmarkIndex(landmarks, distance_tables, cumulative_ms, shape, len(cells))
