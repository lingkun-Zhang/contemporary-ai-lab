"""核验补充实验的模型预测与工作量口径，不重复正式算法单测。"""

import unittest
from types import SimpleNamespace

from run_experiments import reference_dijkstra
from run_supplementary_experiments import (COMPONENT_SIZES, HIGH_COSTS, component_cases,
                                           platform_counts, ratio_cases, reachable_nodes)


class SupplementaryExperimentTests(unittest.TestCase):
    def test_platform_excludes_goal_and_no_path_has_no_numeric_threshold(self):
        answer = SimpleNamespace(trace=[dict(node=(0, 0),f=2), dict(node=(0, 1),f=3),
                                        dict(node=(0, 2),f=3)], stats=dict(expanded_nodes=2))
        counts = platform_counts(answer, (0, 2), 3)
        self.assertEqual(counts, dict(expanded_f_below_optimum=1, expanded_f_equal_optimum=1,
                                      expanded_f_above_optimum=0))
        self.assertTrue(all(v is None for v in platform_counts(answer, (0, 2), -1).values()))
        with self.assertRaises(AssertionError):
            platform_counts(answer, (0, 2), 1)

    def test_cost_ratio_exact_threshold_and_unchanged_locations(self):
        variants = ratio_cases()
        self.assertEqual([k for _, k in variants], list(HIGH_COSTS))
        for case, high in variants:
            self.assertEqual(reference_dijkstra(case).get(case.goal,-1), min(18+high,29))
            self.assertEqual(case.grid[5][10], 1)
            self.assertEqual(sum(case.grid[r][10] == high for r in range(21) if r != 5), 20)
            self.assertEqual((case.start,case.goal), ((10,0),(10,20)))

    def test_isolated_weight_does_not_change_reachable_problem(self):
        cases = component_cases()
        for shape,n in [("bands",n) for n in COMPONENT_SIZES]+[("uniform",n) for n in (31,101)]:
            pair = [(c,e) for c,e in cases if e["base_size"]==n and e["terrain_structure"]==shape
                    and e["zero_location"] in ("isolated","none")]
            self.assertEqual(len(pair), 2)
            a,b = [c for c,_ in pair]
            reached = reachable_nodes(a)
            self.assertEqual(reached, reachable_nodes(b))
            self.assertEqual(len(reached), n*n)
            self.assertNotIn((n//2,n+1), reached)
            self.assertEqual(reference_dijkstra(a).get(a.goal,-1), reference_dijkstra(b).get(b.goal,-1))
            self.assertTrue(all(a.grid[r][c]==b.grid[r][c] for r,c in reached))

    def test_reachability_walk_continues_past_goal_and_retains_zero(self):
        from run_experiments import Case
        line = Case("walk", "walk", [["S","E",0]], (0,0),(0,1))
        self.assertEqual(reachable_nodes(line), {(0,0),(0,1),(0,2)})
        case = next(c for c,_ in component_cases() if c.slug=="reachable_zero_31")
        self.assertIn((2,3), reachable_nodes(case))
        self.assertEqual(case.grid[2][3],0)
        unreachable = next(c for c,_ in component_cases() if c.slug=="component_unreachable")
        self.assertEqual(reference_dijkstra(unreachable).get(unreachable.goal,-1),-1)
        self.assertEqual(len(reachable_nodes(unreachable)),4)


if __name__ == "__main__":
    unittest.main()
