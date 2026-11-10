"""整批计时边界、共同查询及参数汇总的独立口径检查。"""

import unittest
from unittest.mock import patch

from landmark_heuristic import build_landmark_index
from run_experiments import Case, bfs_distance
from run_landmark_tuning import (load_legacy_reference, make_queries, prepare_workload,
                                  run_batch, selected_cases, summarize, validate_batch)


class LandmarkTuningTests(unittest.TestCase):
    def test_original_workloads_are_reused_without_resampling_for_new_landmarks(self):
        _,legacy=load_legacy_reference()
        originals={item['case']:item for item in legacy['multi_query_workloads']}
        for case in selected_cases(legacy,('long_wall','perfect_maze')):
            pairs,selected,info=prepare_workload(case,8,40,123456,legacy)
            expected=[tuple(tuple(point) for point in pair) for pair in originals[case.slug]['queries']]
            self.assertEqual(pairs,expected)
            self.assertEqual(info['query_source'],'exact_original_40_queries')
            self.assertEqual(selected[:4],tuple(tuple(point) for point in originals[case.slug]['landmarks']))
            self.assertEqual(info['newly_added_after_four_endpoint_occurrences'],
                             sum(point in selected[4:] for pair in pairs for point in pair))

    def test_query_selection_excludes_maximum_prefix_and_is_reproducible(self):
        case=Case('tiny','tiny',[list('S...'),list('....'),list('...E')],(0,0),(2,3))
        pairs,landmarks=make_queries(case,3,10,7)
        self.assertEqual((pairs,landmarks),make_queries(case,3,10,7))
        self.assertEqual(len(set(pairs)),10)
        self.assertTrue(all(a!=b and a not in landmarks and b not in landmarks for a,b in pairs))
        self.assertTrue(all(bfs_distance(Case('t','t',case.grid,a,b))>=0 for a,b in pairs))
        with self.assertRaises(ValueError):make_queries(case,11,2,7)

    def test_each_batch_builds_requested_index_and_timer_encloses_whole_query_list(self):
        grid=[list('S..'),list('..E')];pairs=[((0,0),(1,2)),((1,0),(0,2))]
        case=Case('tiny','tiny',grid,(0,0),(1,2))
        expected=[bfs_distance(Case('t','t',grid,a,b)) for a,b in pairs]
        landmarks=tuple(build_landmark_index(grid,2).landmarks)
        with patch('run_landmark_tuning.build_landmark_index',wraps=build_landmark_index) as build:
            with patch('run_landmark_tuning.perf_counter_ns',side_effect=[100,1100,4100]):
                first=run_batch(grid,pairs,2)
            with patch('run_landmark_tuning.perf_counter_ns',side_effect=[100,2100,7100]):
                second=run_batch(grid,pairs,2)
            self.assertEqual(build.call_count,2)
        self.assertAlmostEqual(first['total_ms'],.004)
        self.assertAlmostEqual(first['preprocess_ms'],.001)
        self.assertAlmostEqual(second['total_ms'],.007)
        validate_batch(case,pairs,expected,first,landmarks)
        validate_batch(case,pairs,expected,second,landmarks)

    def test_zero_candidate_never_builds_landmark_index(self):
        grid=[list('S.E')];pairs=[((0,0),(0,2))]
        with patch('run_landmark_tuning.build_landmark_index') as build:
            with patch('run_landmark_tuning.perf_counter_ns',side_effect=[100,5100]):
                result=run_batch(grid,pairs,0)
        build.assert_not_called()
        self.assertAlmostEqual(result['total_ms'],.005)
        self.assertEqual(result['preprocess_ms'],0)
        self.assertEqual(result['landmarks'],())
        self.assertEqual(result['distance_table_entries'],0)

    def test_summary_uses_whole_batch_values_and_marks_only_tested_interior_minimum(self):
        raw=[]
        for k,values in ((0,[90,100,110]),(1,[40,50,60]),(2,[60,70,80])):
            for repeat,value in enumerate(values,1):
                raw.append(dict(case='x',case_name='x',k=k,repeat=repeat,total_ms=value,
                                preprocess_ms=k*2,queries=40,distance_table_entries=k*10,
                                expanded_nodes_sum=100-k,generated_nodes_sum=150,
                                max_frontier_per_query_max=5))
        summary=summarize(raw,(0,1,2))
        best=next(r for r in summary if r['k']==1)
        self.assertEqual(best['total_ms_median'],50)
        self.assertEqual((best['total_ms_q1'],best['total_ms_q3']),(45,55))
        self.assertEqual(best['paired_speedup_median'],2)
        self.assertTrue(best['best_is_interior'])
        self.assertEqual(best['best_k_by_median'],1)


if __name__=='__main__':unittest.main()
