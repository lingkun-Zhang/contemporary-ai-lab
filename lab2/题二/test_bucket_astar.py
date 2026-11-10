"""整数桶A*的独立最优费用、完整路径与同序控制检验。"""

import heapq
import itertools
import random
import unittest

from bucket_astar import queue_astar_search
from weighted_astar import SAFE_HEURISTICS, SAMPLE_GRID, SAMPLE_START, SAMPLE_GOAL, astar_search


COMMON_STATS = ("expanded_nodes", "generated_nodes", "popped_nodes", "max_frontier",
                "stale_pops", "reopened_nodes", "unique_reached")


def oracle(grid, start, end):
    """不调用待测模块的规范化、邻居和启发式；直接按原题进入费Dijkstra。"""
    if not grid or not grid[0] or any(grid[r][c] == "X" for r, c in (start, end)):
        return -1
    frontier, best = [(0, start)], {start: 0}
    while frontier:
        cost, current = heapq.heappop(frontier)
        if cost != best[current]:
            continue
        if current == end:
            return cost
        r, c = current
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "X":
                weight = 0 if (nr, nc) in (start, end) else int(grid[nr][nc])
                candidate = cost+weight
                if candidate < best.get((nr, nc), float("inf")):
                    best[(nr, nc)] = candidate
                    heapq.heappush(frontier, (candidate, (nr, nc)))
    return -1


def scaled_grid(grid, q):
    return [[value if value in ("S", "E", "X") else int(value)*q for value in row] for row in grid]


class BucketAstarTests(unittest.TestCase):
    def assert_result(self, grid, start, end, result, expected):
        self.assertEqual(result.cost, expected)
        if expected < 0:
            self.assertEqual(result.path, [])
            return
        self.assertEqual((result.path[0], result.path[-1]), (start, end))
        self.assertEqual(result.cumulative_costs[-1], expected)
        total = 0
        for i, (r, c) in enumerate(result.path):
            self.assertTrue(0 <= r < len(grid) and 0 <= c < len(grid[0]))
            self.assertNotEqual(grid[r][c], "X")
            weight = 0 if i == 0 or (r, c) == end else int(grid[r][c])
            total += weight
            self.assertEqual(result.path_costs[i], weight)
            self.assertEqual(result.cumulative_costs[i], total)
            if i:
                a, b = result.path[i-1], result.path[i]
                self.assertEqual(abs(a[0]-b[0])+abs(a[1]-b[1]), 1)
        self.assertEqual(total, expected)

    def check_case(self, grid, start, end, heuristics=SAFE_HEURISTICS, cap=32768):
        expected = oracle(grid, start, end)
        for heuristic in heuristics:
            for tie in ("fifo", "low_h"):
                original = astar_search(start, end, grid, heuristic, tie, True)
                matched = queue_astar_search(start, end, grid, heuristic, tie, "heap", True, cap)
                bucket = queue_astar_search(start, end, grid, heuristic, tie, "bucket", True, cap)
                normalized = queue_astar_search(start, end, grid, heuristic, tie, "bucket_gcd", True, cap)
                for result in (original, matched, bucket, normalized):
                    self.assert_result(grid, start, end, result, expected)
                self.assertEqual(original.path, matched.path)
                self.assertEqual(original.path, bucket.path)
                self.assertEqual(original.trace, matched.trace)
                self.assertEqual(original.trace, bucket.trace)
                self.assertEqual(original.path, normalized.path)
                self.assertEqual(original.trace, normalized.trace)
                for name in COMMON_STATS:
                    self.assertEqual(original.stats[name], matched.stats[name], name)
                    self.assertEqual(original.stats[name], bucket.stats[name], name)
                    self.assertEqual(original.stats[name], normalized.stats[name], name)

    def test_pdf_sample_all_safe_heuristics_and_ties(self):
        self.check_case(SAMPLE_GRID, SAMPLE_START, SAMPLE_GOAL)

    def test_exhaustive_two_by_three_zero_one_and_obstacle_cells(self):
        positions = ((0, 1), (0, 2), (1, 0), (1, 1))
        for values in itertools.product((0, 1, "X"), repeat=4):
            grid = [["S", 0, 0], [0, 0, "E"]]
            for (r, c), value in zip(positions, values):
                grid[r][c] = value
            self.check_case(grid, (0, 0), (1, 2))

    def test_new_random_maps_independent_optimality_and_exact_trace(self):
        for seed in range(202611110, 202611140):
            rng = random.Random(seed)
            n = 4+seed % 4
            grid = [["X" if rng.random() < .18 else rng.choice((0, 1, 2, 9, 37))
                     for _ in range(n)] for _ in range(n)]
            grid[0][0], grid[-1][-1] = "S", "E"
            self.check_case(grid, (0, 0), (n-1, n-1))

    def test_scaling_keeps_paths_nodes_and_normalized_heap_trace(self):
        for seed in range(202611141, 202611147):
            rng = random.Random(seed)
            grid = [[rng.choice((0, 1, 2, 8)) for _ in range(7)] for _ in range(7)]
            grid[0][0], grid[-1][-1] = "S", "E"
            for heuristic in SAFE_HEURISTICS:
                for tie in ("fifo", "low_h"):
                    base = astar_search((0, 0), (6, 6), grid, heuristic, tie, True)
                    for q in (1, 2, 4, 8, 16, 64, 256):
                        scaled = scaled_grid(grid, q)
                        for mode in ("heap", "bucket", "bucket_gcd"):
                            result = queue_astar_search((0, 0), (6, 6), scaled, heuristic,
                                                        tie, mode, True)
                            self.assert_result(scaled, (0, 0), (6, 6), result, base.cost*q)
                            self.assertEqual(result.path, base.path)
                            self.assertEqual([r["node"] for r in result.trace], [r["node"] for r in base.trace])
                            self.assertEqual([(r["g"], r["h"], r["f"]) for r in result.trace],
                                [(r["g"]*q, r["h"]*q, r["f"]*q) for r in base.trace])

    def test_zero_width_increment_and_zero_cost_cycles(self):
        grid = [["S", 0, 0], [0, 0, "E"]]
        result = queue_astar_search((0, 0), (1, 2), grid, "zero")
        self.assertEqual(result.cost, 0)
        self.assertEqual(result.stats["max_reduced_cost"], 0)
        self.assertEqual(result.stats["bucket_slots"], 1)
        self.assertEqual(result.stats["empty_bucket_scans"], 0)
        self.check_case(grid, (0, 0), (1, 2))

    def test_free_numeric_goal_and_singleton(self):
        self.check_case([["S", 4, 90000]], (0, 0), (0, 2))
        self.check_case([[7]], (0, 0), (0, 0))

    def test_empty_maps_and_blocked_endpoints(self):
        for grid in ([], [[]], [[], []]):
            for mode in ("heap", "bucket"):
                self.assertEqual(queue_astar_search((0, 0), (0, 0), grid, frontier=mode).cost, -1)
        for grid in ((["X", 2], [3, "E"]), (["S", 2], [3, "X"])):
            self.check_case(grid, (0, 0), (1, 1))

    def test_sparse_huge_integer_costs_use_exact_memory_guard(self):
        grid = [["S", 10**15, 1], [1, 1, "E"]]
        result = queue_astar_search((0, 0), (1, 2), grid, max_bucket_slots=128)
        self.assertEqual(result.cost, 2)
        self.assertEqual(result.stats["bucket_fallback"], 1)
        self.assertEqual(result.stats["bucket_slots"], 0)
        self.check_case(grid, (0, 0), (1, 2), cap=128)

    def test_cost_scaling_changes_queue_range_without_changing_search(self):
        grid = [["S", 3, 8], [1, 1, "E"]]
        small = queue_astar_search((0, 0), (1, 2), grid, "zero", "fifo", record_trace=True)
        large = queue_astar_search((0, 0), (1, 2), scaled_grid(grid, 16), "zero", "fifo", record_trace=True)
        self.assertEqual(large.stats["max_reduced_cost"], 16*small.stats["max_reduced_cost"])
        self.assertEqual(large.stats["bucket_slots"]-1, 16*(small.stats["bucket_slots"]-1))
        self.assertEqual(large.stats["empty_bucket_scans"], 16*small.stats["empty_bucket_scans"])
        for name in COMMON_STATS:
            self.assertEqual(large.stats[name], small.stats[name])

    def test_gcd_units_remove_scaling_sensitivity_without_altering_original_costs(self):
        grid = [["S", 3, 8], [1, 1, "E"]]
        base = queue_astar_search((0, 0), (1, 2), grid, "zero", "fifo", "bucket_gcd", True)
        scaled = queue_astar_search((0, 0), (1, 2), scaled_grid(grid, 256), "zero", "fifo", "bucket_gcd", True)
        self.assertEqual(scaled.cost, base.cost*256)
        self.assertEqual(scaled.stats["priority_quantum"], 256)
        self.assertEqual(scaled.stats["bucket_slots"], base.stats["bucket_slots"])
        self.assertEqual(scaled.stats["empty_bucket_scans"], base.stats["empty_bucket_scans"])
        self.assertEqual(scaled.path, base.path)

    def test_gcd_units_avoid_representation_only_memory_fallback(self):
        grid = [["S", 300, 800], [100, 100, "E"]]
        raw = queue_astar_search((0, 0), (1, 2), grid, "zero", frontier="bucket", max_bucket_slots=16)
        normalized = queue_astar_search((0, 0), (1, 2), grid, "zero", frontier="bucket_gcd", max_bucket_slots=16)
        self.assertEqual(raw.cost, normalized.cost)
        self.assertEqual(raw.path, normalized.path)
        self.assertEqual(raw.stats["bucket_fallback"], 1)
        self.assertEqual(normalized.stats["bucket_fallback"], 0)
        self.assertEqual(normalized.stats["bucket_slots"], 9)

    def test_rejects_unsupported_heuristics_and_invalid_queue_parameters(self):
        grid = [["S", "E"]]
        for heuristic in ("raw_M", lambda node, end: 1, None):
            with self.assertRaises(ValueError):
                queue_astar_search((0, 0), (0, 1), grid, heuristic=heuristic)
        for flag, value in (("frontier", "deque"), ("tie_break", "lifo"),
                            ("max_bucket_slots", 0), ("max_bucket_slots", True)):
            with self.assertRaises(ValueError):
                queue_astar_search((0, 0), (0, 1), grid, **{flag: value})

    def test_does_not_mutate_input(self):
        grid = [["S", 1], [8, "E"]]
        before = [row[:] for row in grid]
        self.check_case(grid, (0, 0), (1, 1))
        self.assertEqual(grid, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
