"""Structural/analysis tests for the robustness study, not another benchmark."""
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from run_robustness_study import (FAMILY_SEEDS,VALUES,build_cases,connected_components,
                                  check_result,full_horizon_regret,generate_case,register_protocol)
from reference_tools import dijkstra_matrix,root_values
from treasure_game import game_search,sample_problem


class RobustnessStudyTests(unittest.TestCase):
    def test_generation_is_reproducible_with_fixed_attempt_log(self):
        for family,seeds in FAMILY_SEEDS.items():
            first,spec1=generate_case(family,seeds[0]);second,spec2=generate_case(family,seeds[0])
            self.assertEqual(spec1,spec2)
            self.assertEqual(first['problem'],second['problem'])
            self.assertGreaterEqual(spec1['attempts'],1)

    def test_all_named_fees_are_one_and_endpoints_connected(self):
        cases,specifications=build_cases()
        self.assertEqual(len(cases),30)
        self.assertEqual(len({s['input_sha256'] for s in specifications}),30)
        for case,spec in zip(cases,specifications):
            problem=case['problem'];components=connected_components(problem.grid)
            containing=next(c for c in components if problem.points[0] in c)
            self.assertGreaterEqual(len(containing),20)
            self.assertTrue(set(problem.points)<=containing)
            self.assertEqual(problem.values,VALUES)
            self.assertEqual(len(set(problem.points)),6)
            self.assertTrue(all(problem.grid[r][c]==1 for r,c in problem.points))
            self.assertEqual(len(containing),spec['largest_component_size'])

    def test_families_have_registered_shapes_and_detour_separates_starts(self):
        cases,_=build_cases()
        self.assertEqual({f:sum(c['family']==f for c in cases) for f in FAMILY_SEEDS},{f:10 for f in FAMILY_SEEDS})
        for case in cases:
            problem=case['problem'];size=7 if case['family']=='open_weighted' else 9
            self.assertEqual((len(problem.grid),len(problem.grid[0])),(size,size))
            if case['family']=='open_weighted':self.assertTrue(all(v is not None for row in problem.grid for v in row))
            if case['family']=='detour':
                self.assertLess(problem.points[0][1],4);self.assertGreater(problem.points[1][1],4)
                self.assertEqual(sum(problem.grid[r][4] is not None for r in range(9)),1)

    def test_regret_uses_actual_selected_action_even_for_shallow_ties(self):
        case,_=generate_case('open_weighted',FAMILY_SEEDS['open_weighted'][0])
        full={0:5,1:2,2:5,3:-1}
        self.assertEqual(full_horizon_regret(case['problem'],full,'T1'),(0,5,5))
        self.assertEqual(full_horizon_regret(case['problem'],full,'T2'),(3,2,5))
        self.assertEqual(full_horizon_regret(case['problem'],full,'T3'),(0,5,5))
        self.assertEqual(full_horizon_regret(case['problem'],full,'T4'),(6,-1,5))

    def test_protocol_registration_cannot_overwrite_prior_design(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'protocol.json'
            digest=register_protocol(path,{'target':'fixed','inputs':30})
            with self.assertRaises(FileExistsError):register_protocol(path,{'target':'changed'})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'target':'fixed','inputs':30})
            self.assertEqual(len(digest),64)

    def test_validation_checks_the_returned_line_not_only_the_root_value(self):
        problem=sample_problem();matrix=dijkstra_matrix(problem)
        case=dict(problem=problem,reference_matrix=matrix,reference_roots={4:root_values(matrix,problem.values,4)})
        config=dict(phase='game',depth=4)
        result=game_search(problem,matrix,4)
        counters=dataclasses.asdict(result.stats)
        check_result(case,config,result,counters)
        with self.assertRaises(AssertionError):
            check_result(case,config,dataclasses.replace(result,principal_variation=('T3','T1')),counters)
        with self.assertRaises(AssertionError):
            check_result(case,config,dataclasses.replace(result,principal_variation=('T3','T4','T2','T1')),counters)


if __name__=='__main__':unittest.main()
