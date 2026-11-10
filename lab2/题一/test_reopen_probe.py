"""有界格图上的重开正确性测试，不把故意错误的消融当成求解器。"""

import random
import unittest
from collections import deque

from maze_astar import astar_search, manhattan
from reopen_probe import analyze_heuristic, corridor_case, probe_astar, table_heuristic


def bfs_distances(grid, start):
    distances, queue = {start: 0}, deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and
                    grid[nr][nc] != "X" and (nr, nc) not in distances):
                distances[(nr, nc)] = distances[(r, c)] + 1
                queue.append((nr, nc))
    return distances


class ReopenProbeTests(unittest.TestCase):
    def test_canonical_counterexample(self):
        grid, start, goal, table = corridor_case()
        h = table_heuristic(table)
        analysis = analyze_heuristic(grid, goal, h, bfs_distances(grid, goal))
        self.assertEqual(analysis["admissibility_violations"], 0)
        self.assertGreater(analysis["consistency_violation_edges"], 0)
        self.assertEqual(h((0, 3), goal), 5)
        self.assertEqual(h((0, 4), goal), 0)
        correct = probe_astar(start, goal, grid, h, True, True)
        frozen = probe_astar(start, goal, grid, h, False, True)
        self.assertEqual((correct.cost, frozen.cost), (8, 10))
        self.assertEqual(correct.stats["reopened_nodes"], 3)
        self.assertEqual(frozen.stats["reopened_nodes"], 0)
        self.assertGreater(frozen.stats["ignored_closed_improvements"], 0)
        for result in (correct, frozen):
            self.assertEqual(len(result.path)-1, result.cost)
            for a, b in zip(result.path, result.path[1:]):
                self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]), 1)
        self.assertEqual(astar_search(start, goal, grid, h).cost, correct.cost)

    def test_consistent_controls_need_no_reopen(self):
        grid, start, goal, _ = corridor_case()
        for h in (lambda n, g: 0, manhattan):
            for allow in (False, True):
                result = probe_astar(start, goal, grid, h, allow)
                self.assertEqual(result.cost, 8)
                self.assertEqual(result.stats["reopened_nodes"], 0)

    def test_random_admissible_tables_match_independent_bfs_and_core(self):
        rng = random.Random(20261028)
        for case in range(80):
            grid = [["X" if rng.random() < .25 else "." for _ in range(10)] for _ in range(10)]
            start, goal = (0, 0), (9, 9)
            grid[0][0], grid[9][9] = "S", "E"
            table = {(r, c): 0 if rng.random() < .5 else manhattan((r, c), goal)
                     for r, row in enumerate(grid) for c, cell in enumerate(row) if cell != "X"}
            h = table_heuristic(table)
            truth = bfs_distances(grid, goal)
            analysis = analyze_heuristic(grid, goal, h, truth)
            self.assertEqual(analysis["admissibility_violations"], 0)
            correct = probe_astar(start, goal, grid, h)
            core = astar_search(start, goal, grid, h)
            self.assertEqual(correct.cost, truth.get(start, -1), case)
            self.assertEqual(correct.cost, core.cost)
            self.assertEqual(correct.path, core.path)
            frozen = probe_astar(start, goal, grid, h, False)
            if frozen.cost >= 0:
                self.assertEqual(len(frozen.path)-1, frozen.cost)
                self.assertGreaterEqual(frozen.cost, correct.cost)

    def test_edges_and_invalid_input(self):
        self.assertEqual(probe_astar((0,0),(0,0),[],manhattan).path, [])
        self.assertEqual(probe_astar((0,0),(0,0),[["."]]).path, [(0,0)])
        self.assertEqual(probe_astar((0,0),(1,1),[list(".X"),list("X.")]).path, [])
        for args in ((3, 1, 1), (9, 8, 4), (9, 4, 0)):
            with self.assertRaises(ValueError):
                corridor_case(*args)
        with self.assertRaises(ValueError):
            probe_astar((0,0),(0,0),[["."]],"manhattan")


if __name__ == "__main__":
    unittest.main()
