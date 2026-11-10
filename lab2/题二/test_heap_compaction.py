"""堆清理必须保持有效搜索顺序；物理堆记录通过守恒核验。"""

import copy
import argparse
import math
import random
import unittest

from heap_compaction import compacted_astar_search
from run_experiments import band_grid, mark
from test_weighted_astar import independent_dijkstra, independent_reverse_distances
from weighted_astar import SAFE_HEURISTICS, SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar_search


STABLE_STATS = ("expanded_nodes", "generated_nodes", "reopened_nodes", "unique_reached")


class HeapCompactionTests(unittest.TestCase):
    def assert_same_search(self, answer, baseline):
        for name in ("path", "cost", "trace", "expanded_order", "path_costs", "cumulative_costs",
                     "min_step_cost", "min_scope", "component_size"):
            self.assertEqual(getattr(answer, name), getattr(baseline, name), name)
        for name in STABLE_STATS:
            self.assertEqual(answer.stats[name], baseline.stats[name], name)
        extra = answer.compaction_stats
        self.assertEqual(answer.stats["generated_nodes"], answer.stats["popped_nodes"]
                         + extra["removed_stale"] + extra["final_heap_size"])
        self.assertGreaterEqual(extra["final_heap_size"], extra["final_valid_open"])
        self.assertLessEqual(answer.stats["max_frontier"], baseline.stats["max_frontier"])

    def test_infinite_threshold_preserves_original_fields_and_statistics(self):
        cases = [(SAMPLE_START, SAMPLE_GOAL, SAMPLE_GRID),
                 ((0, 0), (0, 2), [["S", 1, "E"], [0, 0, 0]]),
                 ((0, 0), (0, 2), [["S", "X", "E"]]),
                 ((0, 0), (0, 0), [[8]]), ((0, 0), (0, 0), []),
                 ((0, 0), (0, 0), [["X"]])]
        for start, end, grid in cases:
            for heuristic in SAFE_HEURISTICS:
                for scope in ("global", "reachable"):
                    for tie in ("fifo", "low_h"):
                        baseline = astar_search(start, end, grid, heuristic, tie, True, scope)
                        answer = compacted_astar_search(start, end, grid, heuristic, tie, True, scope)
                        self.assert_same_search(answer, baseline)
                        self.assertEqual(answer.stats, baseline.stats)
                        self.assertEqual(answer.compaction_stats["compactions"], 0)

    def test_finite_cleanup_preserves_banded_search_and_removes_stale(self):
        grid = mark(band_grid(31), (0, 0), (30, 30))
        baseline = astar_search((0, 0), (30, 30), grid, "corrected", "low_h", True)
        self.assertGreater(baseline.stats["stale_pops"], 0)
        total_compactions = 0
        for rho in (1.05, 1.2, 1.5, 2, 4, 8):
            answer = compacted_astar_search((0, 0), (30, 30), grid, "corrected", "low_h", True, rho=rho)
            self.assert_same_search(answer, baseline)
            total_compactions += answer.compaction_stats["compactions"]
            if answer.compaction_stats["compactions"]:
                self.assertGreater(answer.compaction_stats["removed_stale"], 0)
            self.assertGreaterEqual(answer.compaction_stats["scanned_entries"], answer.compaction_stats["removed_stale"])
        self.assertGreater(total_compactions, 0)

    def test_no_stale_control_does_not_rebuild(self):
        grid = mark([[1] * 31 for _ in range(31)], (0, 0), (30, 30))
        baseline = astar_search((0, 0), (30, 30), grid, "row_column_tight", "low_h", True)
        for rho in (1.0001, 1.5, 8):
            answer = compacted_astar_search((0, 0), (30, 30), grid, "row_column_tight", "low_h", True, rho=rho)
            self.assert_same_search(answer, baseline)
            self.assertEqual(answer.compaction_stats["compactions"], 0)
            self.assertEqual(answer.compaction_stats["removed_stale"], 0)

    def test_empty_valid_open_clears_remaining_old_records(self):
        grid = [["S", 9, "X", "E"], [1, 1, "X", 1]]
        heuristic = lambda node, goal: 100 if node in ((1, 0), (1, 1)) else 0
        baseline = astar_search((0, 0), (0, 3), grid, heuristic, record_trace=True)
        answer = compacted_astar_search((0, 0), (0, 3), grid, heuristic, record_trace=True, rho=1.05)
        self.assert_same_search(answer, baseline)
        self.assertEqual(answer.cost, -1)
        self.assertGreater(answer.compaction_stats["removed_stale"], 0)
        self.assertEqual(answer.compaction_stats["final_valid_open"], 0)
        self.assertEqual(answer.compaction_stats["final_heap_size"], 0)

    def test_inconsistent_admissible_reopening_preserves_effective_order(self):
        grid = [["S", 9, 1, "X", "X"], [1, 1, 100, 1, "E"]]
        heuristic = lambda node, goal: 100 if node == (1, 0) else 0
        truths = independent_reverse_distances((0, 0), (1, 4), grid)
        self.assertTrue(all(heuristic(node, (1, 4)) <= cost for node, cost in truths.items()))
        baseline = astar_search((0, 0), (1, 4), grid, heuristic, record_trace=True)
        self.assertEqual(baseline.cost, 103)
        self.assertGreater(baseline.stats["reopened_nodes"], 0)
        for rho in (1.05, 1.5, 8, math.inf):
            answer = compacted_astar_search((0, 0), (1, 4), grid, heuristic, record_trace=True, rho=rho)
            self.assert_same_search(answer, baseline)

    def test_random_maps_match_independent_oracle_and_unchanged_priority_order(self):
        rng = random.Random(20261029)
        for instance in range(60):
            rows, cols = rng.randint(3, 7), rng.randint(3, 7)
            weights = (1, 2, 7) if instance % 2 == 0 else (0, 1, 2, 7)
            grid = [["X" if rng.random() < .18 else rng.choice(weights) for _ in range(cols)] for _ in range(rows)]
            start, end = (rng.randrange(rows), rng.randrange(cols)), (rng.randrange(rows), rng.randrange(cols))
            grid[start[0]][start[1]], grid[end[0]][end[1]] = "S", "E"
            snapshot = copy.deepcopy(grid)
            expected = independent_dijkstra(start, end, grid)
            for heuristic in SAFE_HEURISTICS:
                for tie in ("fifo", "low_h"):
                    for scope in ("global", "reachable"):
                        baseline = astar_search(start, end, grid, heuristic, tie, True, scope)
                        self.assertEqual(baseline.cost, expected)
                        for rho in (1.05, 1.5, 4, math.inf):
                            with self.subTest(instance=instance, heuristic=heuristic, tie=tie, scope=scope, rho=rho):
                                answer = compacted_astar_search(start, end, grid, heuristic, tie, True, scope, rho)
                                self.assert_same_search(answer, baseline)
                                self.assertEqual(answer.cost, expected)
            self.assertEqual(grid, snapshot)

    def test_invalid_parameters_retain_original_validation(self):
        for bad in (None, True, False, 0, 1, -1, -math.inf, math.nan, "2", [], {}):
            with self.subTest(rho=bad):
                with self.assertRaises(ValueError):
                    compacted_astar_search((0, 0), (0, 1), [["S", "E"]], rho=bad)
        for kwargs in ({"heuristic": "bad"}, {"tie_break": "bad"}, {"min_scope": "bad"}):
            with self.assertRaises(ValueError):
                compacted_astar_search((0, 0), (0, 1), [["S", "E"]], **kwargs)
        with self.assertRaises(ValueError):
            compacted_astar_search((0, 0), (0, 1), [["S", -1]])

    def test_parameter_grid_requires_no_clean_reference_and_valid_candidates(self):
        from run_heap_tuning import parse_rhos, rho_label
        self.assertEqual(parse_rhos("inf,2,1.05"), (1.05, 2.0, math.inf))
        self.assertNotEqual(rho_label(1.0500001), rho_label(1.0500002))
        for text in ("1.05,2", "1.05,1.05,inf", "1,inf", "nan,inf", "-inf,inf", "bad,inf"):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_rhos(text)

    def test_parameter_selection_reports_boundary_and_descriptive_overlap(self):
        from run_heap_tuning import selection_rows
        def row(rho, median, q1, q3, implementation="compacted"):
            return dict(case="fixed",case_name="fixed",rho=rho,implementation=implementation,
                        time_median_ms=median,time_q1_ms=q1,time_q3_ms=q3)
        rows = [row("1.05",3,2.5,3.5), row("1.2",2,1.5,2.5), row("2",3,2.6,3.4),
                row("inf",4,3,5), row("original",2.2,2,2.5,"original")]
        selection = selection_rows(rows, (1.05,1.2,2,math.inf))[0]
        self.assertEqual(selection["best_tested_rho"], "1.2")
        self.assertTrue(selection["best_is_internal"])
        self.assertFalse(selection["best_at_lower_boundary"])
        self.assertIn('"1.05"', selection["iqr_overlap_with_best"])
        rows[0] = row("1.05",1,.8,1.2)
        boundary = selection_rows(rows, (1.05,1.2,2,math.inf))[0]
        self.assertTrue(boundary["best_at_lower_boundary"])
        self.assertFalse(boundary["best_is_internal"])
        rows[0] = row("1.05",3,2.5,3.5)
        upper = selection_rows([r for r in rows if r['rho'] != '2'], (1.05,1.2,math.inf))[0]
        self.assertTrue(upper["best_at_upper_boundary"])
        self.assertFalse(upper["best_is_internal"])


if __name__ == "__main__":
    unittest.main()
