"""固定比较、协议保护和原输入复用的非计时口径测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from run_landmark_confirmation import (
    CANDIDATES, REFERENCES, choose_output, confirmation_flags,
    fixed_candidate_summary, fixed_pair_summary, load_workloads, sha256, write_protocol,
)


class LandmarkConfirmationTests(unittest.TestCase):
    def test_exact_three_original_workloads_and_uniform_queries_are_reused(self):
        path = Path(__file__).resolve().parent / "results/landmark_tuning/reproducibility.json"
        saved, workloads = load_workloads(path)
        prior = {item["case"]: item for item in saved["workloads"]}
        self.assertEqual(tuple(case.slug for case, _, _ in workloads), ("long_wall", "perfect_maze", "open"))
        for case, queries, selected in workloads:
            self.assertEqual(len(queries), 40)
            self.assertEqual([[list(a), list(b)] for a, b in queries], prior[case.slug]["queries"])
            self.assertEqual(case.grid, [list(row) for row in prior[case.slug]["grid"]])
            self.assertEqual([list(point) for point in selected], prior[case.slug]["max_k_landmarks"])
        corrupt = dict(saved)
        corrupt["source_sha256"] = dict(saved["source_sha256"], **{"run_landmark_tuning.py": "0" * 64})
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root, prefix=".confirmation_test_") as folder:
            self.assertEqual(Path(folder).resolve().parent, test_root)
            changed_source = Path(folder) / "changed_source.json"
            changed_source.write_text(json.dumps(corrupt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "同实现确认"):
                load_workloads(changed_source)

    def test_protocol_is_exclusive_and_records_phase_before_warmup(self):
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root, prefix=".confirmation_test_") as folder:
            self.assertEqual(Path(folder).resolve().parent, test_root)
            path = Path(folder) / "protocol.json"
            expected = dict(candidates=[0, 1, 2], target=1, refs=[0, 2])
            digest = write_protocol(path, expected)
            self.assertEqual(sha256(path), digest)
            protocol = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(protocol["design"], expected)
            self.assertIn("before any", protocol["registration_phase"])
            with self.assertRaises(FileExistsError):
                write_protocol(path, dict(candidates=[0, 2, 4]))
            self.assertEqual(sha256(path), digest)

    def test_rerun_uses_fresh_directory_without_overwriting_previous_results(self):
        test_root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=test_root, prefix=".confirmation_test_") as folder:
            self.assertEqual(Path(folder).resolve().parent, test_root)
            path = Path(folder) / "confirmation"
            first = choose_output(path)
            marker = first / "protocol.json"
            marker.write_text("preserved", encoding="utf-8")
            second = choose_output(path)
            self.assertNotEqual(first, second)
            self.assertEqual(second.parent, first)
            self.assertTrue(second.name.startswith("rerun_"))
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserved")

    def test_fixed_pair_deltas_do_not_reselect_the_fastest_candidate(self):
        self.assertEqual(CANDIDATES, (0, 1, 2))
        self.assertEqual(REFERENCES, (0, 2))
        raw = []
        for repeat, totals in enumerate(((10, 5, 6), (100, 90, 75), (200, 180, 170)), 1):
            for k, value in enumerate(totals):
                raw.append(dict(case="long_wall", case_name="x", repeat=repeat, k=k, queries=40,
                                total_ms=value, preprocess_ms=k, expanded_nodes_sum=100-k,
                                distance_table_entries=k*10))
        paired = fixed_pair_summary(raw)
        against_zero = next(row for row in paired if row["reference_k"] == 0)
        against_two = next(row for row in paired if row["reference_k"] == 2)
        self.assertEqual(against_zero["median_delta_ms"], -10)
        self.assertEqual(against_two["median_delta_ms"], 10)
        self.assertEqual((against_two["target_faster_blocks"], against_two["target_slower_blocks"]), (1, 2))
        self.assertTrue(all(row["target_k"] == 1 for row in paired))
        summary = fixed_candidate_summary(raw)
        self.assertTrue(all("best_k_by_median" not in row for row in summary))

    def test_confirm_flags_preserve_loss_and_near_tie_instead_of_forcing_extremum(self):
        summaries, paired = [], []
        for slug in ("long_wall", "perfect_maze", "open"):
            for k, value in ((0, 10), (1, 12), (2, 11)):
                summaries.append(dict(case=slug, k=k, total_ms_median=value))
            for reference in (0, 2):
                paired.append(dict(case=slug, reference_k=reference, median_delta_ms=1,
                                   paired_iqr_entirely_below_zero=False, paired_iqr_crosses_zero=True))
        flags = confirmation_flags(summaries, paired)
        self.assertTrue(all(row["fixed_target_k"] == 1 for row in flags))
        self.assertTrue(all(not row["target_total_median_below_both_refs"] for row in flags))
        self.assertTrue(all(not row["both_paired_iqr_below_zero"] for row in flags))
        self.assertTrue(all(row["any_paired_iqr_crosses_zero"] for row in flags))


if __name__ == "__main__":
    unittest.main()
