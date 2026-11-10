"""普通 BFS 与共同搜索内核的独立最短性验证。"""

import copy
import random
import unittest
from collections import deque
from itertools import permutations

from bfs_baseline import bfs_search, uniform_search
from maze_astar import DIRECTIONS, SAMPLE_GRID, SAMPLE_START, SAMPLE_GOAL, astar_search


def reference(start, goal, grid):
    if not grid or not grid[0] or grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return -1
    queue, visited = deque([(start, 0)]), {start}
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X" and (nr, nc) not in visited:
                visited.add((nr, nc)); queue.append(((nr, nc), d+1))
    return -1


class BFSBaselineTests(unittest.TestCase):
    def test_pdf_sample(self):
        result = bfs_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID)
        self.assertEqual(result.cost, 7)
        self.assertEqual(result.path[0], SAMPLE_START)
        self.assertEqual(result.path[-1], SAMPLE_GOAL)

    def test_same_kernel_pop_order_all_directions(self):
        for order in permutations(DIRECTIONS):
            a = uniform_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, "deque", True, order)
            b = uniform_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, "heap", True, order)
            self.assertEqual(a, b)
            c = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, "zero", "fifo", True, order)
            self.assertEqual(a.path, c.path)
            self.assertEqual(a.expanded_order, c.expanded_order)
            self.assertEqual(a.stats, c.stats)

    def test_empty_unreachable_blocked_and_same_endpoint(self):
        for grid in ([], [[]], [["X"]]):
            self.assertEqual(bfs_search((0, 0), (0, 0), grid).path, [])
        self.assertEqual(bfs_search((0, 0), (0, 0), [["."]]).path, [(0, 0)])
        self.assertEqual(bfs_search((0, 0), (1, 1), [list("SX"), list("XE")]).path, [])
        self.assertEqual(bfs_search((0, 0), (0, 1), [list("SE")]).cost, 1)

    def test_validation(self):
        for start, goal, grid in [((-1, 0), (0, 0), [["."]]), ((0, 0), (0, 2), [["."]]),
                                  ((0, 0), (0, 0), [list(".."), ["."]]), ((0, 0), (0, 0), [["?"]])]:
            with self.assertRaises(ValueError):
                bfs_search(start, goal, grid)
        with self.assertRaises(ValueError):
            uniform_search((0, 0), (0, 0), [["."]], "invalid")
        with self.assertRaises(ValueError):
            bfs_search((0, 0), (0, 0), [["."]], neighbor_order=[(1, 1)]*4)

    def test_no_mutation_trace_independence(self):
        before = copy.deepcopy(SAMPLE_GRID)
        a = bfs_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, True)
        b = bfs_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID)
        self.assertEqual(before, SAMPLE_GRID)
        self.assertEqual(a.path, b.path)
        self.assertEqual(a.stats, b.stats)
        self.assertEqual(b.trace, [])
        self.assertEqual(len(a.trace), a.stats["expanded_nodes"]+1)

    def test_seeded_random_oracle_and_matched_metrics(self):
        rng = random.Random(20261028)
        for _ in range(200):
            h, w = rng.randint(2, 20), rng.randint(2, 20)
            grid = [["X" if rng.random() < .3 else "." for _ in range(w)] for _ in range(h)]
            start, goal = (rng.randrange(h), rng.randrange(w)), (rng.randrange(h), rng.randrange(w))
            grid[start[0]][start[1]] = grid[goal[0]][goal[1]] = "."
            expected = reference(start, goal, grid)
            a = uniform_search(start, goal, grid, "deque", True)
            b = uniform_search(start, goal, grid, "heap", True)
            self.assertEqual(a, b)
            self.assertEqual(a.cost, expected)
            if a.path:
                self.assertEqual(len(a.path)-1, expected)
                for x, y in zip(a.path, a.path[1:]):
                    self.assertEqual(abs(x[0]-y[0])+abs(x[1]-y[1]), 1)
                    self.assertNotEqual(grid[y[0]][y[1]], "X")


if __name__ == "__main__":
    unittest.main()
