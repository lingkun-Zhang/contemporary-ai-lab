"""预声明输入、计时前落盘和固定配对统计的独立口径测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_heap_confirmation import (FIXED_CASES, FIXED_RHOS, TARGET, REFERENCES,
                                   delta_summary, fixed_pair_rows, fresh_output_dir,
                                   prepare_protocol, run_confirmation, write_protocol)


def local_temporary_directory():
    # Keep test writes in the task workspace rather than an inaccessible Windows app-container temp directory.
    parent = Path(__file__).resolve().parent
    directory = tempfile.TemporaryDirectory(prefix="_confirmation_test_", dir=parent)
    if Path(directory.name).resolve().parent != parent:
        raise RuntimeError("Unexpected temporary test path")
    return directory


class HeapConfirmationTests(unittest.TestCase):
    def test_protocol_uses_exact_prior_workloads_and_fixed_parameters(self):
        with local_temporary_directory() as tmp:
            out,protocol,digest=prepare_protocol(Path(tmp)/"run")
            self.assertEqual(protocol["parameters"]["rhos"],["1.05","1.2","1.5","2","inf"])
            self.assertEqual(tuple(item["case"] for item in protocol["input_cases"]),FIXED_CASES)
            self.assertEqual(protocol["counts"],dict(groups=4,distinct_grid_endpoint_inputs=3,
                                                    configurations=24,timing_batches=360,timed_solver_calls=1080))
            self.assertEqual(protocol["primary"]["target_rho"],"1.5")
            self.assertEqual(protocol["primary"]["comparison"],"original")
            self.assertEqual(len(digest),64)
            self.assertEqual(json.loads((out/"protocol.json").read_text(encoding="utf-8")),protocol)

    def test_protocol_is_written_before_first_runner_call(self):
        class StopBeforeTiming(Exception):pass
        with local_temporary_directory() as tmp:
            out=Path(tmp)/"run"
            def inspect(output,**kwargs):
                self.assertTrue((Path(output)/"protocol.json").is_file())
                self.assertEqual(kwargs["rhos"],FIXED_RHOS)
                self.assertEqual(kwargs["selected_cases"],list(FIXED_CASES))
                raise StopBeforeTiming()
            with patch("run_heap_confirmation.run_tuning",side_effect=inspect):
                with self.assertRaises(StopBeforeTiming):run_confirmation(out)

    def test_existing_results_are_preserved_and_protocol_never_overwritten(self):
        with local_temporary_directory() as tmp:
            root=Path(tmp);existing=root/"protocol.json"
            write_protocol(root,{"frozen":True})
            before=existing.read_bytes()
            with self.assertRaises(FileExistsError):write_protocol(root,{"changed":True})
            fresh=fresh_output_dir(root)
            self.assertNotEqual(fresh,root)
            self.assertEqual(fresh.parent,root)
            self.assertTrue(fresh.name.startswith("run_"))
            self.assertEqual(existing.read_bytes(),before)

    def test_paired_delta_is_not_difference_of_marginal_medians(self):
        # Candidate median minus reference median is -49, but median paired delta is +1.
        deltas=[candidate-reference for candidate,reference in zip((50,51,102),(0,100,101))]
        result=delta_summary(deltas)
        self.assertEqual(result["delta_ms_median"],1)
        self.assertEqual(result["direction"],"mixed_or_unresolved")
        self.assertEqual((result["faster_blocks"],result["slower_blocks"]),(1,2))
        self.assertEqual(delta_summary([-5,-4,-3])["direction"],"directionally_consistent_improvement")
        self.assertEqual(delta_summary([1,2,3])["direction"],"directionally_consistent_slowdown")

    def test_fixed_target_remains_one_point_five_even_when_neighbor_is_faster(self):
        batches=[]
        for repeat in range(1,4):
            for rho,value in (("original",10),("inf",9),("1.2",2),("1.5",7),("2",8)):
                batches.append(dict(case="uniform_101",rho=rho,repeat=repeat,batch_average_ms=value+repeat))
        comparison=[dict(case="uniform_101",rho=TARGET,case_name="uniform",compactions=0)]
        paired=fixed_pair_rows(batches,comparison)
        self.assertEqual(len(paired),4)
        self.assertEqual(tuple(row["reference"] for row in paired),REFERENCES)
        self.assertTrue(all(row["target_rho"]=="1.5" for row in paired))
        self.assertTrue(all(not row["cleanup_action_present"] for row in paired))
        against_left=next(row for row in paired if row["reference"]=="1.2")
        self.assertEqual(against_left["delta_ms_median"],5)
        self.assertEqual(against_left["direction"],"directionally_consistent_slowdown")


if __name__=="__main__":unittest.main()
