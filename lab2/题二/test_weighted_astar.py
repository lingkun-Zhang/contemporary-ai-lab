"""题二：用独立最短路与小图枚举核验代价、路径及启发函数条件。"""

import copy
import heapq
import random
import unittest

from weighted_astar import (
    SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar_cost, astar_search,
    heuristic_value, load_problem, prepare_terrain,
)


SAFE_HEURISTICS = ("zero", "corrected", "row_column", "row_column_tight")


def reference_costs(start, end, grid):
    """测试侧独立解释合法地图，不复用正式转换或邻居生成。"""
    rows = []
    for row in grid:
        converted = []
        for value in row:
            if value == "X":
                converted.append(None)
            elif value in ("S", "E"):
                converted.append(0)
            else:
                converted.append(int(value))
        rows.append(converted)
    if rows and rows[0]:
        if rows[start[0]][start[1]] is not None:
            rows[start[0]][start[1]] = 0
        if rows[end[0]][end[1]] is not None:
            rows[end[0]][end[1]] = 0
    return rows


def reference_neighbors(node, start, costs):
    r, c = node
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        other = (nr, nc)
        if (0 <= nr < len(costs) and 0 <= nc < len(costs[0])
                and costs[nr][nc] is not None and other != start):
            yield other


def independent_dijkstra(start, end, grid):
    if not grid or not grid[0]:
        return -1
    costs = reference_costs(start, end, grid)
    if costs[start[0]][start[1]] is None or costs[end[0]][end[1]] is None:
        return -1
    distance, heap = {start: 0}, [(0, start)]
    settled = set()
    while heap:
        total, node = heapq.heappop(heap)
        if node in settled:
            continue
        settled.add(node)
        if node == end:
            return total
        for other in reference_neighbors(node, start, costs):
            candidate = total + costs[other[0]][other[1]]
            if candidate < distance.get(other, float("inf")):
                distance[other] = candidate
                heapq.heappush(heap, (candidate, other))
    return -1


def independent_reachable(start, costs):
    """只用测试侧的格子和邻居解释，求从 S 出发的区域。"""
    if not costs or not costs[0] or costs[start[0]][start[1]] is None:
        return set()
    found, pending = {start}, [start]
    while pending:
        node = pending.pop(0)
        for other in reference_neighbors(node, start, costs):
            if other not in found:
                found.add(other)
                pending.append(other)
    return found


def independent_reverse_distances(start, end, grid):
    """反向边 v→u 的代价是原边 u→v 的进入代价 w(v)。"""
    costs = reference_costs(start, end, grid)
    if costs[end[0]][end[1]] is None:
        return {}
    distances, heap = {end: 0}, [(0, end)]
    while heap:
        total, node = heapq.heappop(heap)
        if total != distances[node]:
            continue
        # 原图删除所有进入 start 的边，所以反向图从 start 无后继。
        if node == start:
            continue
        r, c = node
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (0 <= nr < len(costs) and 0 <= nc < len(costs[0])
                    and costs[nr][nc] is not None):
                predecessor = (nr, nc)
                candidate = total + costs[r][c]
                if candidate < distances.get(predecessor, float("inf")):
                    distances[predecessor] = candidate
                    heapq.heappush(heap, (candidate, predecessor))
    return distances


def enumerate_simple_paths(start, end, grid):
    """小图穷举简单路径；非负代价的最优路径可删除环。"""
    costs = reference_costs(start, end, grid)
    best = float("inf")

    def visit(node, total, seen):
        nonlocal best
        if node == end:
            best = min(best, total)
            return
        if total >= best:
            return
        for other in reference_neighbors(node, start, costs):
            if other not in seen:
                visit(other, total + costs[other[0]][other[1]], seen | {other})

    visit(start, 0, {start})
    return -1 if best == float("inf") else best


class WeightedAStarTests(unittest.TestCase):
    def assert_valid_result(self, result, start, end, grid, expected):
        self.assertEqual(result.cost, expected)
        if expected == -1:
            self.assertEqual(result.path, [])
            self.assertEqual(result.path_costs, [])
            self.assertEqual(result.cumulative_costs, [])
            return
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], end)
        self.assertEqual(len(set(result.path)), len(result.path))
        self.assertEqual(result.path_costs[0], 0)
        self.assertEqual(result.cumulative_costs[0], 0)
        costs = reference_costs(start, end, grid)
        ledger, accumulated = [0], [0]
        for a, b in zip(result.path, result.path[1:]):
            self.assertEqual(abs(a[0] - b[0]) + abs(a[1] - b[1]), 1)
            r, c = b
            self.assertTrue(0 <= r < len(costs) and 0 <= c < len(costs[0]))
            self.assertIsNotNone(costs[r][c])
            self.assertNotEqual(b, start)
            ledger.append(costs[r][c])
            accumulated.append(accumulated[-1] + costs[r][c])
        self.assertEqual(result.path_costs, ledger)
        self.assertEqual(result.cumulative_costs, accumulated)
        self.assertEqual(sum(ledger), expected)

    def test_pdf_sample_and_ledger(self):
        self.assertEqual(astar_cost(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID), 7)
        for name in SAFE_HEURISTICS:
            result = astar_search(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, heuristic=name)
            self.assert_valid_result(result, SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID, 7)
            self.assertEqual(result.path_costs[-1], 0)

    def test_adjacent_goal_is_free_and_numeric_goal_overridden(self):
        for grid in ([["S", "E"]], [[5, 900]], [["S", "999"]]):
            self.assert_valid_result(astar_search((0, 0), (0, 1), grid), (0, 0), (0, 1), grid, 0)
        self.assertEqual(astar_cost((0, 0), (0, 2), [[9, 5, 999]]), 5)

    def test_start_equals_end(self):
        for grid in ([[999]], [["S"]], [["E"]]):
            result = astar_search((0, 0), (0, 0), grid, record_trace=True)
            self.assert_valid_result(result, (0, 0), (0, 0), grid, 0)
            self.assertEqual(result.path, [(0, 0)])
            self.assertEqual(result.stats["expanded_nodes"], 0)

    def test_minimum_steps_is_not_minimum_cost(self):
        grid = [["S", 9, "E"], [1, 1, 1]]
        result = astar_search((0, 0), (0, 2), grid)
        self.assert_valid_result(result, (0, 0), (0, 2), grid, 3)
        self.assertEqual(len(result.path) - 1, 4)

    def test_zero_cost_terrain_and_raw_manhattan_counterexample(self):
        grid = [["S", 1, "E"], [0, 0, 0]]
        for name in SAFE_HEURISTICS:
            result = astar_search((0, 0), (0, 2), grid, heuristic=name)
            self.assert_valid_result(result, (0, 0), (0, 2), grid, 0)
        raw = lambda node, goal: abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        self.assertEqual(astar_search((0, 0), (0, 2), grid, heuristic=raw).cost, 1)
        all_zero = [["S", 0, 0], [0, 0, "E"]]
        self.assertEqual(astar_cost((0, 0), (1, 2), all_zero), 0)

    def test_row_column_can_remain_informative_when_global_minimum_is_zero(self):
        grid = [["S", 1, 5, 1, 1], [1, 0, 5, 1, 1], [1, 1, 5, 1, "E"]]
        terrain = prepare_terrain((0, 0), (2, 4), grid)
        self.assertEqual(terrain.min_step_cost, 0)
        self.assertEqual(heuristic_value(terrain, (0, 0), "corrected"), 0)
        self.assertGreater(heuristic_value(terrain, (0, 0), "row_column"), 0)
        self.assertEqual(astar_cost((0, 0), (2, 4), grid), independent_dijkstra((0, 0), (2, 4), grid))

    def test_tight_row_column_counts_only_one_free_goal_entry(self):
        grid = [[9 if r in (5, 10, 15) else 1 for _ in range(21)] for r in range(21)]
        grid[0][0], grid[20][20] = "S", "E"
        start, end = (0, 0), (20, 20)
        terrain = prepare_terrain(start, end, grid)
        self.assertEqual(heuristic_value(terrain, start, "row_column"), 62)
        self.assertEqual(heuristic_value(terrain, start, "row_column_tight"), 63)
        self.assertEqual(independent_dijkstra(start, end, grid), 63)
        result = astar_search(start, end, grid, heuristic="row_column_tight", tie_break="low_h")
        self.assert_valid_result(result, start, end, grid, 63)

    def test_empty_blocked_and_unreachable(self):
        for grid, end in (([], (0, 0)), ([[]], (0, 0)), ([["X"]], (0, 0)),
                          ([["S", "X", "E"]], (0, 2))):
            for scope in ("global", "reachable"):
                result = astar_search((0, 0), end, grid, min_scope=scope)
                self.assert_valid_result(result, (0, 0), end, grid, -1)

    def test_reachable_minimum_ignores_only_disconnected_zero(self):
        grid = [list(row) + ["X", "X"] for row in SAMPLE_GRID]
        grid[2][6] = 0
        before = copy.deepcopy(grid)
        ordinary = astar_search(SAMPLE_START, SAMPLE_GOAL, grid, tie_break="low_h")
        repaired = astar_search(SAMPLE_START, SAMPLE_GOAL, grid, tie_break="low_h",
                                 min_scope="reachable")
        self.assert_valid_result(ordinary, SAMPLE_START, SAMPLE_GOAL, grid, 7)
        self.assert_valid_result(repaired, SAMPLE_START, SAMPLE_GOAL, grid, 7)
        self.assertEqual(ordinary.min_step_cost, 0)
        self.assertEqual(repaired.min_step_cost, 1)
        self.assertEqual(repaired.component_size, 25)
        self.assertEqual(repaired.min_scope, "reachable")
        self.assertIsNone(ordinary.component_size)
        self.assertLess(repaired.stats["expanded_nodes"], ordinary.stats["expanded_nodes"])
        self.assertEqual(grid, before)
        # 同一零格如果通过走廊接到 S 区域，必须继续把 m 保守地取为 0。
        grid[2][5] = 1
        connected = astar_search(SAMPLE_START, SAMPLE_GOAL, grid, min_scope="reachable")
        self.assertEqual(connected.min_step_cost, 0)
        self.assertEqual(connected.component_size, 27)
        self.assert_valid_result(connected, SAMPLE_START, SAMPLE_GOAL, grid,
                                 independent_dijkstra(SAMPLE_START, SAMPLE_GOAL, grid))

    def test_reachable_minimum_does_not_short_circuit_unreachable(self):
        grid = [["S", 2, "X", "E"], [3, 2, "X", 0]]
        result = astar_search((0, 0), (0, 3), grid, min_scope="reachable")
        self.assert_valid_result(result, (0, 0), (0, 3), grid, -1)
        self.assertEqual(result.component_size, 4)
        self.assertEqual(result.min_step_cost, 2)
        self.assertEqual(result.stats["expanded_nodes"], 4)
        for small, end, size in (([], (0, 0), 0), ([[]], (0, 0), 0),
                                 ([["X"]], (0, 0), 0), ([[9]], (0, 0), 1),
                                 ([["S", "E"]], (0, 1), 2)):
            candidate = astar_search((0, 0), end, small, min_scope="reachable")
            self.assertEqual(candidate.component_size, size)
            self.assertEqual(candidate.min_step_cost, 0)
            self.assertEqual(candidate.cost, -1 if size == 0 else 0)

    def test_numeric_strings_and_input_not_modified(self):
        grid = [["S", "7", "E"], ["1", "2", "1"]]
        before = copy.deepcopy(grid)
        traced = astar_search((0, 0), (0, 2), grid, record_trace=True)
        plain = astar_search((0, 0), (0, 2), grid)
        self.assertEqual(grid, before)
        self.assertEqual(traced.path, plain.path)
        self.assertEqual(traced.stats, plain.stats)
        self.assertEqual(plain.trace, [])
        self.assertEqual(len(traced.trace), traced.stats["expanded_nodes"] + 1)
        self.assertTrue(all(item["g"] + item["h"] == item["f"] for item in traced.trace))

    def test_load_problem_accepts_pdf_markers_and_quoted_form(self):
        coordinates = "(0, 0)\n(4, 4)\n"
        bare = """[
            [S, 1, 1, 8, 1],
            [1, 8, 3, 1, 1],
            [1, 8, 1, 8, 1],
            [1, 1, 1, 8, 1],
            [8, 8, 1, 1, E]
        ]"""
        quoted = repr(SAMPLE_GRID)
        for grid in (bare, quoted):
            with self.subTest(grid=grid):
                self.assertEqual(astar_cost(*load_problem(coordinates + grid)), 7)
        self.assertEqual(astar_cost(*load_problem("(0, 0)\n(0, 2)\n[[S, X, 'E']]")), -1)

    def test_load_problem_rejects_nonliteral_grid_expressions(self):
        for invalid in ("[[S, UNKNOWN, E]]", "[[S, str(1), E]]", "[[S, X + '1', E]]"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    load_problem("(0, 0)\n(0, 2)\n" + invalid)

    def test_invalid_inputs(self):
        invalid = [
            ((0, 0), (0, 1), [["S", -1]]),
            ((0, 0), (0, 1), [["S", True]]),
            ((0, 0), (0, 1), [["S", 1.5]]),
            ((0, 0), (0, 1), [["S", "2.0"]]),
            ((0, 0), (0, 1), [["S", "."]]),
            ((0, 0), (0, 1), [["S", "?"]]),
            ((0, 0), (0, 1), [["S", 1], [1]]),
            ((-1, 0), (0, 0), [[1]]),
            ((0, 0), (0, 1), [[1]]),
            ((False, 0), (0, 0), [[1]]),
            ((0, 0), (0, 2), [[1, "S", "E"]]),
            ((0, 0), (0, 2), [["S", "E", 1]]),
        ]
        for start, goal, grid in invalid:
            with self.subTest(start=start, goal=goal, grid=grid):
                with self.assertRaises(ValueError):
                    astar_search(start, goal, grid)
        for kwargs in ({"heuristic": "bad"}, {"tie_break": "bad"}, {"min_scope": "bad"}):
            with self.assertRaises(ValueError):
                astar_search((0, 0), (0, 1), [["S", "E"]], **kwargs)

    def test_random_maps_against_independent_dijkstra(self):
        rng = random.Random(20261028)
        solved, unsolved = 0, 0
        for case in range(200):
            rows, cols = rng.randint(2, 12), rng.randint(2, 12)
            probability = rng.choice((0, .15, .35))
            grid = [["X" if rng.random() < probability else rng.choice((0, 1, 2, 3, 9))
                     for _ in range(cols)] for _ in range(rows)]
            start = rng.randrange(rows), rng.randrange(cols)
            end = rng.randrange(rows), rng.randrange(cols)
            grid[start[0]][start[1]] = "S"
            grid[end[0]][end[1]] = "E"
            expected = independent_dijkstra(start, end, grid)
            solved += expected >= 0
            unsolved += expected == -1
            for name in SAFE_HEURISTICS:
                for tie in ("fifo", "low_h"):
                    for scope in ("global", "reachable"):
                        with self.subTest(case=case, heuristic=name, tie=tie, scope=scope):
                            result = astar_search(start, end, grid, heuristic=name,
                                                  tie_break=tie, min_scope=scope)
                            self.assert_valid_result(result, start, end, grid, expected)
                            self.assertEqual(result.stats["reopened_nodes"], 0)
        self.assertGreater(solved, 0)
        self.assertGreater(unsolved, 0)

    def test_safe_heuristics_are_admissible_and_consistent(self):
        rng = random.Random(18820261028)
        positive_cases = zero_cases = 0
        for case in range(48):
            rows, cols = rng.randint(2, 6), rng.randint(2, 7)
            weights = (1, 2, 7) if case % 2 == 0 else (0, 1, 2, 7)
            grid = [["X" if rng.random() < .2 else rng.choice(weights)
                     for _ in range(cols)] for _ in range(rows)]
            start, end = (0, 0), (rows - 1, cols - 1)
            grid[0][0], grid[-1][-1] = "S", "E"
            if case % 2:
                grid[0][1] = 0  # 显式覆盖 m=0 的退化分支。
            terrain = prepare_terrain(start, end, grid)
            positive_cases += terrain.min_step_cost > 0
            zero_cases += terrain.min_step_cost == 0
            costs = reference_costs(start, end, grid)
            truth = independent_reverse_distances(start, end, grid)
            reachable = independent_reachable(start, costs)
            all_nodes = {(r, c) for r in range(rows) for c in range(cols)
                         if costs[r][c] is not None}
            for scope in ("global", "reachable"):
                terrain = prepare_terrain(start, end, grid, min_scope=scope)
                domain = all_nodes if scope == "global" else reachable
                if scope == "reachable":
                    self.assertEqual(terrain.component_size, len(reachable))
                    expected_m = min((costs[r][c] for r, c in reachable
                                      if (r, c) not in (start, end)), default=0)
                    self.assertEqual(terrain.min_step_cost, expected_m)
                for name in SAFE_HEURISTICS:
                    for node in domain:
                        h = heuristic_value(terrain, node, name)
                        self.assertGreaterEqual(h, 0)
                        if node in truth:
                            self.assertLessEqual(h, truth[node], (case, name, scope, node))
                        for other in reference_neighbors(node, start, costs):
                            bound = costs[other[0]][other[1]] + heuristic_value(terrain, other, name)
                            self.assertLessEqual(h, bound, (case, name, scope, node, other))
                    self.assertEqual(heuristic_value(terrain, end, name), 0)
                for node in domain:
                    self.assertGreaterEqual(heuristic_value(terrain, node, "row_column"),
                                            heuristic_value(terrain, node, "corrected"))
                    self.assertGreaterEqual(heuristic_value(terrain, node, "row_column_tight"),
                                            heuristic_value(terrain, node, "row_column"))
                    if terrain.min_step_cost == 0:
                        self.assertEqual(heuristic_value(terrain, node, "row_column_tight"),
                                         heuristic_value(terrain, node, "row_column"))
        self.assertEqual(positive_cases, 24)
        self.assertEqual(zero_cases, 24)

    def test_small_graphs_against_simple_path_enumeration(self):
        rng = random.Random(227)
        for case in range(40):
            grid = [[rng.choice((0, 1, 2, 8)) for _ in range(3)] for _ in range(3)]
            grid[0][0], grid[2][2] = "S", "E"
            expected = enumerate_simple_paths((0, 0), (2, 2), grid)
            self.assertEqual(expected, independent_dijkstra((0, 0), (2, 2), grid))
            for name in SAFE_HEURISTICS:
                self.assertEqual(astar_search((0, 0), (2, 2), grid, heuristic=name).cost, expected,
                                 (case, name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
