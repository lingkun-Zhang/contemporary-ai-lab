"""地标启发的独立 BFS 验证；运行 python -m unittest -v。"""

from __future__ import annotations

from collections import deque
import math
import random
import unittest

from landmark_heuristic import build_landmark_index
from maze_astar import astar_search


def reference_distances(grid, source):
    """测试侧独立计算真实最短距离，不复用地标模块或 A* 的搜索逻辑。"""
    height = len(grid)
    width = len(grid[0]) if grid else 0
    if not (0 <= source[0] < height and 0 <= source[1] < width):
        return {}
    if grid[source[0]][source[1]] == "X":
        return {}
    queue = deque([(source, 0)])
    answer = {}
    while queue:
        point, distance = queue.popleft()
        if point in answer:
            continue
        answer[point] = distance
        row, col = point
        for neighbor in ((row - 1, col), (row, col - 1), (row + 1, col), (row, col + 1)):
            nr, nc = neighbor
            if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] != "X":
                if neighbor not in answer:
                    queue.append((neighbor, distance + 1))
    return answer


class LandmarkHeuristicTests(unittest.TestCase):
    def assert_bounds(self, grid, max_landmarks=4):
        index = build_landmark_index(grid, max_landmarks)
        cells = [(r, c) for r, row in enumerate(grid)
                 for c, value in enumerate(row) if value != "X"]
        estimates = [index.heuristic(k) for k in range(len(index.landmarks) + 1)]
        for goal in cells:
            truth = reference_distances(grid, goal)
            for estimate in estimates:
                self.assertEqual(estimate(goal, goal), 0)
            for node in cells:
                bounds = [estimate(node, goal) for estimate in estimates]
                self.assertTrue(all(math.isfinite(value) and value >= 0 for value in bounds))
                self.assertEqual(bounds, sorted(bounds), "加地标后启发不应变弱")
                self.assertEqual(bounds[0], abs(node[0] - goal[0]) + abs(node[1] - goal[1]))
                if node in truth:
                    self.assertTrue(all(value <= truth[node] for value in bounds), "不能高估真实距离")
                row, col = node
                for neighbor in ((row + 1, col), (row, col + 1)):
                    if neighbor in cells:
                        for estimate in estimates:
                            self.assertLessEqual(estimate(node, goal), 1 + estimate(neighbor, goal))
                            self.assertLessEqual(estimate(neighbor, goal), 1 + estimate(node, goal))
        for landmark, table in zip(index.landmarks, index.distance_tables):
            self.assertEqual(table, reference_distances(grid, landmark))
        return index

    def test_admissibility_consistency_and_all_prefixes(self):
        grids = [
            ["....", ".XX.", "...."],
            [".X...", ".X.X.", "...X.", "XXXX.", "....."],
            ["..X..", "..X..", "XXXXX", ".X.X."],
            ["."],
        ]
        for grid in grids:
            with self.subTest(grid=grid):
                self.assert_bounds(grid, 6)

    def test_random_grid_bounds(self):
        rng = random.Random(20261022)
        for _ in range(12):
            grid = [["X" if rng.random() < 0.28 else "." for _ in range(5)] for _ in range(4)]
            self.assert_bounds(grid, 4)

    def test_selection_is_deterministic_and_farthest(self):
        grid = [".....", ".....", "....."]
        index = build_landmark_index(grid, 4)
        self.assertEqual(index.landmarks, [(0, 0), (2, 4), (0, 3), (2, 1)])
        other = build_landmark_index([list(row) for row in grid], 4)
        self.assertEqual(index.landmarks, other.landmarks)
        self.assertEqual(index.distance_tables, other.distance_tables)
        # 终点字符不参与选点，同一地图可重复服务于不同起终点。
        marked = [list(row) for row in grid]
        marked[1][1] = "S"
        marked[1][3] = "E"
        self.assertEqual(index.landmarks, build_landmark_index(marked, 4).landmarks)

    def test_uncovered_components_selected_before_farthest(self):
        grid = ["..X..", "..X..", "XXXXX", ".X.X."]
        index = build_landmark_index(grid, 6)
        self.assertEqual(index.landmarks[:5], [(0, 0), (0, 3), (3, 0), (3, 2), (3, 4)])
        self.assertEqual(index.landmarks[5], (1, 1))
        estimate = index.heuristic(6)
        # 两点不属于同一分量，返回有限下界，不通过预检拒绝查询。
        self.assertEqual(estimate((0, 0), (0, 3)), 3)
        result = astar_search((0, 0), (0, 3), grid, heuristic=estimate)
        self.assertEqual(result.path, [])
        self.assertEqual(result.cost, -1)
        self.assertGreater(result.stats["expanded_nodes"], 0)

    def test_stronger_than_manhattan_on_detour(self):
        grid = [".....", "XXXX.", "....."]
        index = build_landmark_index(grid, 1)
        self.assertEqual(index.landmarks, [(0, 0)])
        estimate = index.heuristic(1)
        self.assertEqual(estimate((0, 0), (2, 0)), 10)
        self.assertGreater(estimate((0, 0), (2, 0)), index.heuristic(0)((0, 0), (2, 0)))

    def test_astar_matches_bfs_for_all_prefixes_and_repeated_queries(self):
        rng = random.Random(415)
        for _ in range(16):
            grid = [["X" if rng.random() < 0.25 else "." for _ in range(7)] for _ in range(6)]
            cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value != "X"]
            index = build_landmark_index(grid, 4)
            for _ in range(4):
                start, goal = rng.choice(cells), rng.choice(cells)
                expected = reference_distances(grid, start).get(goal, -1)
                for k in range(len(index.landmarks) + 1):
                    for tie_break in ("fifo", "low_h"):
                        result = astar_search(start, goal, grid, index.heuristic(k), tie_break)
                        self.assertEqual(result.cost, expected)
                        if expected == -1:
                            self.assertEqual(result.path, [])
                        else:
                            self.assertEqual(len(result.path) - 1, expected)
                            self.assertEqual(result.path[0], start)
                            self.assertEqual(result.path[-1], goal)
                            for a, b in zip(result.path, result.path[1:]):
                                self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
                                self.assertNotEqual(grid[b[0]][b[1]], "X")

    def test_empty_blocked_zero_and_excess_landmark_counts(self):
        for grid in ([], [[]], ["XXX", "XXX"]):
            index = build_landmark_index(grid)
            self.assertEqual(index.landmarks, [])
            self.assertEqual(index.distance_tables, [])
            self.assertEqual(index.cumulative_preprocess_ms, [])
            self.assertEqual(index.heuristic(0)((0, 0), (3, 4)), 7)
            self.assertEqual(index.metadata()["distance_entries"], 0)
            self.assertEqual(astar_search((0, 0), (0, 0), grid, index.heuristic(0)).path, [])
        index = build_landmark_index([".."], 20)
        self.assertEqual(index.landmarks, [(0, 0), (0, 1)])
        self.assertEqual(len(index.distance_tables), 2)
        self.assertEqual(build_landmark_index([".."], 0).landmarks, [])

    def test_metadata_counts_and_real_cumulative_time(self):
        index = build_landmark_index(["..X.", "..X."], 4)
        meta = index.metadata()
        running = 0
        for k, table in enumerate(index.distance_tables):
            running += len(table)
            self.assertEqual(meta["cumulative_distance_entries"][k], running)
        self.assertEqual(meta["distance_entries"], running)
        self.assertEqual(meta["walkable_cells"], 6)
        self.assertEqual(meta["grid_shape"], [2, 4])
        self.assertEqual(len(index.cumulative_preprocess_ms), 4)
        self.assertTrue(all(value >= 0 for value in index.cumulative_preprocess_ms))
        self.assertEqual(index.cumulative_preprocess_ms, sorted(index.cumulative_preprocess_ms))

    def test_invalid_arguments(self):
        for bad_k in (-1, 1.5, True, "4"):
            with self.assertRaises(ValueError):
                build_landmark_index([".."], bad_k)
        index = build_landmark_index([".."], 1)
        for bad_k in (-1, 2, 0.5, True, "1"):
            with self.assertRaises(ValueError):
                index.heuristic(bad_k)
        for grid in (["..", "."], [".?"], "..", [123]):
            with self.assertRaises(ValueError):
                build_landmark_index(grid)


if __name__ == "__main__":
    unittest.main()
