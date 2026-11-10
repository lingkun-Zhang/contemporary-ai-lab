import json
import tempfile
import unittest
from pathlib import Path

from run_sampling_confirmation import (
    BUDGETS, REFERENCES, SEEDS, TARGET, check_run, prepare_cases, register_protocol,
    strict_lowpoint, summarize,
)


class SamplingConfirmationTests(unittest.TestCase):
    def test_fixed_new_workloads_and_candidates(self):
        self.assertEqual(SEEDS, tuple(range(20261101, 20261131)))
        self.assertEqual(BUDGETS, (0, 4, 8, 16))
        self.assertEqual((TARGET, REFERENCES), (8, (0, 4, 16)))
        cases, specs = prepare_cases(SEEDS[:2], branching=2, depth=3)
        again, _ = prepare_cases(SEEDS[:2], branching=2, depth=3)
        self.assertEqual([c["input_sha256"] for c in cases], [c["input_sha256"] for c in again])
        self.assertNotEqual(cases[0]["input_sha256"], cases[1]["input_sha256"])
        self.assertEqual((cases[0]["full_nodes"], cases[0]["full_leaves"]), (15, 8))
        self.assertEqual(specs[0]["seed"], SEEDS[0])

    def test_q0_has_no_extra_probe_and_target_keeps_extra_cost(self):
        case = prepare_cases(SEEDS[:1], branching=2, depth=4)[0][0]
        _, baseline, _, _ = check_run(case, 0)
        stats, sampled, _, _ = check_run(case, 8)
        self.assertEqual(baseline["ordering_probe_leaves"], 0)
        self.assertGreater(sampled["ordering_probe_leaves"], 0)
        self.assertGreater(stats["leaf_evaluations"]+sampled["ordering_probe_leaves"], 0)
        self.assertEqual(sampled["oracle_leaf_reads"], 0)

    def test_local_lowpoint_requires_both_sides_in_same_tree(self):
        self.assertTrue(strict_lowpoint({0: 500, 4: 300, 8: 250, 16: 280}))
        self.assertFalse(strict_lowpoint({0: 500, 4: 200, 8: 250, 16: 280}))
        self.assertFalse(strict_lowpoint({0: 500, 4: 300, 8: 250, 16: 250}))
        self.assertFalse(strict_lowpoint({0: 100, 4: 300, 8: 250, 16: 200}))

    def test_protocol_is_exclusive_and_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/"protocol.json"
            digest = register_protocol(path, {"target_budget": 8, "seeds": list(SEEDS)})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                register_protocol(path, {"target_budget": 16})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original)["target_budget"], 8)
            self.assertEqual(len(digest), 64)

    def test_aggregation_uses_each_tree_once_after_paired_repeat_median(self):
        rows, raw = [], []
        for tree in range(2):
            for budget in BUDGETS:
                config = len(rows)
                visits = {0: 100, 4: 70, 8: 60, 16: 80}[budget]+tree
                rows.append(dict(config=config, case=f"case{tree}", seed=tree, budget=budget,
                                 visited_nodes=visits, leaf_evaluations=visits//2,
                                 ordering_probe_leaves=0 if budget == 0 else budget,
                                 median_total_ms=float(budget+tree)))
                for repeat in range(3):
                    raw.append(dict(config=config, repeat=repeat, total_ms=budget+tree+repeat))
        pairs, flags, aggregates, budget_summary = summarize(rows, raw, 3)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(sum(f["strict_internal_visit_lowpoint"] for f in flags), 2)
        self.assertTrue(all(item["independent_tree_count"] == 2 for item in aggregates))
        time_pair = next(item for item in aggregates if item["reference_budget"] == 4 and item["metric"] == "median_paired_total_delta_ms")
        self.assertEqual(time_pair["median_delta"], 4)
        self.assertEqual([item["budget"] for item in budget_summary], list(BUDGETS))


if __name__ == "__main__":
    unittest.main()
