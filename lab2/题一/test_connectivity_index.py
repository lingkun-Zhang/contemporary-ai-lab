"""连通预检的独立可达性、最优性和快照语义验证。"""

from collections import deque
import random
import unittest
from unittest.mock import patch

from connectivity_index import build_connectivity_index
from maze_astar import astar_search


def independent_bfs(grid, start, goal):
    if not grid or not grid[0] or grid[start[0]][start[1]] == "X" or grid[goal[0]][goal[1]] == "X":
        return -1
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        point, distance = queue.popleft()
        if point == goal:
            return distance
        r, c = point
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nr, nc = neighbor
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != "X" and neighbor not in seen):
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return -1


class ConnectivityTests(unittest.TestCase):
    def test_component_labels_and_rejection(self):
        grid = ["..X..", "..X.."]
        index = build_connectivity_index(grid)
        self.assertEqual(index.component_count, 2)
        self.assertEqual(index.component_sizes, (4, 4))
        self.assertEqual(index.label_entries, 8)
        rejected = index.query((0, 0), (0, 3))
        self.assertEqual(rejected.path, [])
        self.assertEqual(rejected.cost, -1)
        self.assertEqual(rejected.stats["expanded_nodes"], 0)
        self.assertEqual(rejected.stats["precheck_rejected"], 1)
        solved = index.query((0, 0), (1, 1))
        self.assertEqual(solved.cost, 2)
        self.assertEqual(solved.stats["precheck_rejected"], 0)

    def test_snapshot_does_not_silently_change(self):
        grid = [list("S.X.E")]
        old = build_connectivity_index(grid)
        grid[0][2] = "."
        self.assertEqual(old.query((0, 0), (0, 4)).cost, -1)
        self.assertEqual(old.grid[0][2], "X")
        new = build_connectivity_index(grid)
        self.assertEqual(new.query((0, 0), (0, 4)).cost, 4)
        with self.assertRaises(TypeError):
            old.labels[(0, 0)] = 123
        with self.assertRaises(TypeError):
            old.grid[0][0] = "."

    def test_empty_blocked_same_point_and_illegal_inputs(self):
        for grid in ([], [[]], ["XXX"]):
            self.assertEqual(build_connectivity_index(grid).query((0, 0), (0, 0)).path, [])
        index = build_connectivity_index([".X."])
        self.assertEqual(index.query((0, 0), (0, 0)).path, [(0, 0)])
        self.assertEqual(index.query((0, 0), (0, 1)).path, [])
        for start in ((-1, 0), (0, 3), (True, 0), (0, .5)):
            with self.assertRaises(ValueError):
                index.query(start, (0, 0))
        for kwargs in ({"heuristic": "wrong"}, {"tie_break": "wrong"}):
            with self.assertRaises(ValueError):
                index.query((0, 0), (0, 2), **kwargs)
        for grid in (["..", "."], ["?"], "..", [None]):
            with self.assertRaises(ValueError):
                build_connectivity_index(grid)

    def test_exactly_one_full_validation_scan_on_each_branch(self):
        index = build_connectivity_index(["..X..", "..X.."])
        from maze_astar import _shape
        for goal, expected_index_calls, expected_astar_calls in (((0, 3), 1, 0), ((1, 1), 0, 1)):
            with patch("connectivity_index._shape", wraps=_shape) as index_shape:
                with patch("maze_astar._shape", wraps=_shape) as astar_shape:
                    index.query((0, 0), goal)
                    self.assertEqual(index_shape.call_count, expected_index_calls)
                    self.assertEqual(astar_shape.call_count, expected_astar_calls)

    def test_random_reachability_and_full_paths(self):
        rng = random.Random(20261028)
        for _ in range(60):
            rows, cols = rng.randrange(3, 16), rng.randrange(3, 16)
            grid = [["X" if rng.random() < .3 else "." for _ in range(cols)] for _ in range(rows)]
            index = build_connectivity_index(grid)
            self.assertEqual(sum(index.component_sizes), index.label_entries)
            cells = list(index.labels)
            if not cells:
                continue
            for _ in range(10):
                start, goal = rng.choice(cells), rng.choice(cells)
                expected = independent_bfs(grid, start, goal)
                self.assertEqual(index.labels[start] == index.labels[goal], expected >= 0)
                result = index.query(start, goal)
                self.assertEqual(result.cost, expected)
                self.assertEqual(result.cost, astar_search(start, goal, grid).cost)
                if expected >= 0:
                    self.assertEqual(len(result.path) - 1, expected)
                    self.assertEqual(result.path[0], start)
                    self.assertEqual(result.path[-1], goal)
                    for a, b in zip(result.path, result.path[1:]):
                        self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
                        self.assertNotEqual(grid[b[0]][b[1]], "X")
                else:
                    self.assertEqual(result.path, [])
                    self.assertEqual(result.stats["precheck_rejected"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
