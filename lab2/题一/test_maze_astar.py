"""用独立 BFS 核对最短性；覆盖 PDF 样例、边界与固定随机地图。"""

import copy
import random
import unittest
from collections import deque
from itertools import permutations
from pathlib import Path

from maze_astar import DIRECTIONS, SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar, astar_search, load_problem


def bfs_distance(start, end, grid):
    """仅作测试参照；不调用 A* 的邻居生成、启发函数或代价实现。"""
    if not grid or not grid[0] or grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return -1
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), distance = queue.popleft()
        if (r, c) == end:
            return distance
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if grid[nr][nc] != "X" and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    queue.append(((nr, nc), distance + 1))
    return -1


class AStarTests(unittest.TestCase):
    def assert_valid_path(self, path, start, end, grid, expected_distance):
        if expected_distance == -1:
            self.assertEqual(path, [])
            return
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], end)
        self.assertEqual(len(path) - 1, expected_distance)
        for r, c in path:
            self.assertTrue(0 <= r < len(grid) and 0 <= c < len(grid[0]))
            self.assertNotEqual(grid[r][c], "X")
        for a, b in zip(path, path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)

    def test_pdf_sample(self):
        expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
        self.assertEqual(astar(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID), expected)

    def test_no_path_and_diagonal_forbidden(self):
        self.assertEqual(astar((0, 0), (1, 1), [["S", "X"], ["X", "E"]]), [])

    def test_start_equals_goal(self):
        result = astar_search((0, 0), (0, 0), [["."]], record_trace=True)
        self.assertEqual(result.path, [(0, 0)])
        self.assertEqual(result.cost, 0)
        self.assertEqual(result.stats["expanded_nodes"], 0)
        self.assertEqual(result.stats["generated_nodes"], 1)
        self.assertEqual(result.trace[-1]["f"], 0)

    def test_adjacent_goal_costs_one(self):
        self.assertEqual(astar_search((0, 0), (0, 1), [["S", "E"]]).cost, 1)

    def test_empty_and_blocked(self):
        for grid in ([], [[]], [["X"]]):
            self.assertEqual(astar((0, 0), (0, 0), grid), [])
        self.assertEqual(astar((0, 0), (0, 1), [["S", "X"]]), [])

    def test_single_row_column_and_multiple_optima(self):
        for grid, start, end in [
            ([list("S..E")], (0, 0), (0, 3)),
            ([["S"], ["."], ["E"]], (0, 0), (2, 0)),
            ([list("..."), list("..."), list("...")], (0, 0), (2, 2)),
        ]:
            self.assert_valid_path(astar(start, end, grid), start, end, grid, bfs_distance(start, end, grid))

    def test_input_not_modified_and_trace(self):
        before = copy.deepcopy(SAMPLE_GRID)
        traced = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, record_trace=True)
        plain = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID)
        self.assertEqual(SAMPLE_GRID, before)
        self.assertEqual(traced.path, plain.path)
        self.assertEqual(traced.stats, plain.stats)
        self.assertEqual(len(traced.expanded_order), traced.stats["expanded_nodes"])
        self.assertEqual(len(traced.trace), traced.stats["expanded_nodes"] + 1)
        self.assertTrue(all(item["g"] + item["h"] == item["f"] for item in traced.trace))
        self.assertEqual(plain.trace, [])
        self.assertEqual(plain.expanded_order, [])

    def test_validation(self):
        for start, end, grid in [
            ((-1, 0), (0, 0), [["."]]),
            ((0, 0), (0, 2), [["."]]),
            ((0, 0), (0, 0), [["."], [".", "."]]),
            ((0, 0), (0, 0), [["?"]]),
            ((0, 0.5), (0, 0), [["."]]),
            ((True, 0), (0, 0), [["."]]),
        ]:
            with self.subTest(start=start, end=end, grid=grid):
                with self.assertRaises(ValueError):
                    astar(start, end, grid)
        for kwargs in ({"heuristic": "invalid"}, {"tie_break": "invalid"}):
            with self.assertRaises(ValueError):
                astar_search((0, 0), (0, 0), [["."]], **kwargs)

    def test_pdf_format_input(self):
        sample = Path(__file__).parent / "samples" / "pdf_sample.txt"
        start, end, grid = load_problem(sample.read_text(encoding="utf-8"))
        self.assertEqual(astar(start, end, grid), astar(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID))
        with self.assertRaises(ValueError):
            load_problem("(0,0)\n(1,1)")
        with self.assertRaises(ValueError):
            load_problem("(0,0)\n(1,1)\n__import__('os')")

    def test_seeded_random_maps_against_bfs(self):
        rng = random.Random(20261022)
        solved = unsolved = 0
        for case in range(300):
            rows, cols = rng.randint(2, 18), rng.randint(2, 18)
            density = rng.choice((0.0, 0.15, 0.3, 0.45))
            grid = [["X" if rng.random() < density else "." for _ in range(cols)] for _ in range(rows)]
            start = (rng.randrange(rows), rng.randrange(cols))
            end = (rng.randrange(rows), rng.randrange(cols))
            grid[start[0]][start[1]] = "S"
            grid[end[0]][end[1]] = "E"
            expected = bfs_distance(start, end, grid)
            solved += expected >= 0
            unsolved += expected == -1
            for heuristic, tie in (("zero", "fifo"), ("manhattan", "fifo"), ("manhattan", "low_h")):
                with self.subTest(case=case, heuristic=heuristic, tie=tie):
                    result = astar_search(start, end, grid, heuristic, tie)
                    self.assertEqual(result.cost, expected)
                    self.assert_valid_path(result.path, start, end, grid, expected)
                    self.assertEqual(result.stats["reopened_nodes"], 0)
        self.assertGreater(solved, 0)
        self.assertGreater(unsolved, 0)

    def test_all_neighbor_orders_and_ties_preserve_shortest_path(self):
        for order in permutations(DIRECTIONS):
            for tie in ("fifo", "lifo", "low_h"):
                with self.subTest(order=order, tie=tie):
                    result = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID,
                                          tie_break=tie, neighbor_order=order)
                    self.assert_valid_path(result.path, SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, 7)

    def test_extra_heuristics_against_independent_bfs(self):
        rng = random.Random(1882026)
        for case in range(60):
            grid = [["X" if rng.random() < .28 else "." for _ in range(15)] for _ in range(15)]
            start, end = (0, 0), (14, 14)
            grid[0][0], grid[14][14] = "S", "E"
            expected = bfs_distance(start, end, grid)
            for heuristic in ("zero", "chebyshev", "euclidean", "manhattan"):
                for tie in ("fifo", "lifo", "low_h"):
                    with self.subTest(case=case, heuristic=heuristic, tie=tie):
                        result = astar_search(start, end, grid, heuristic, tie)
                        self.assertEqual(result.cost, expected)
                        self.assert_valid_path(result.path, start, end, grid, expected)

    def test_custom_heuristic_and_invalid_directions(self):
        result = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID,
                              heuristic=lambda node, goal: 0)
        self.assertEqual(result.cost, 7)
        for order in ([], [(0, 1)] * 4, [(1, 1), (1, 0), (-1, 0), (0, -1)], 3):
            with self.subTest(order=order):
                with self.assertRaises(ValueError):
                    astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, neighbor_order=order)


if __name__ == "__main__":
    unittest.main(verbosity=2)
