"""Contract and pairing checks for the new experiment runner, not timing tests."""
import unittest

from reference_tools import dijkstra_matrix, root_values
from treasure_game import build_distance_matrix
from run_lazy_study import (
    ENGINES, build_configurations, build_inputs, invoke, make_pairs,
    signature, summarize, validate,
)


class LazyStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases,cls.specs,cls.route_names=build_inputs()
        cls.sample=cls.cases[0]
        cls.sample['reference_matrix']=dijkstra_matrix(cls.sample['problem'])
        cls.sample['reference_roots']={h:root_values(cls.sample['reference_matrix'],cls.sample['problem'].values,h) for h in (2,4)}
        cls.sample['prepared']=build_distance_matrix(cls.sample['problem'])

    def test_input_and_configuration_contract(self):
        configs=build_configurations(self.cases,self.route_names)
        self.assertEqual(len(configs),1620)
        self.assertEqual(sum(c['phase']=='routing' for _,c in configs),504)
        self.assertEqual(len(self.route_names),7)
        expected={'pdf_sample'}
        for family,seed in [('open_weighted',202611101),('detour',202611201),('random_obstacles',202611301)]:
            expected.update((f'{family}_s{seed}',f'{family}_s{seed+1}'))
        self.assertEqual(self.route_names,expected)
        # Selection is based on fixed identifiers, without any solution fields.
        self.assertTrue(all('reference_roots' not in c for c in self.cases[1:]))

    def test_same_traversal_and_controls(self):
        for algorithm in ('alphabeta','minimax'):
            for order in ('label','reverse','gain'):
                expected=None;lookups=[]
                for engine in ENGINES:
                    config=dict(phase='search_only',depth=4,algorithm=algorithm,order=order,engine=engine)
                    result,stats,times,seen,emitted=invoke(self.sample,config)
                    expected=signature(result) if expected is None else expected
                    validate(self.sample,config,result,stats,seen,emitted,expected,check_ledger=True)
                    self.assertEqual(times['total_ms'],times['search_ms'])
                    lookups.append(stats['distance_lookups'])
                if algorithm=='minimax' or order=='gain':
                    self.assertEqual(len(set(lookups)),1)
                else:self.assertLess(lookups[2],lookups[1])

    def test_complete_matrix_contract_is_not_decision_contract(self):
        for strategy in ('precompute','on_demand','repeated'):
            for mode in ('decision_only','complete_task'):
                for engine in ENGINES:
                    config=dict(phase='routing',depth=4,algorithm='alphabeta',order='label',
                                engine=engine,strategy=strategy,output_mode=mode)
                    result,stats,times,seen,emitted=invoke(self.sample,config)
                    validate(self.sample,config,result,stats,seen,emitted,check_ledger=True)
                    self.assertAlmostEqual(times['total_ms'],times['prepare_ms']+times['search_ms']+times['completion_ms'],places=9)
                    self.assertEqual(emitted is None,mode=='decision_only')
                    if mode=='complete_task' and strategy!='repeated':self.assertEqual(stats['astar_calls'],30)

    def test_pairing_uses_same_repetition_before_median(self):
        common=dict(phase='search_only',case='synthetic',family='test',depth=2,algorithm='alphabeta',
                    order='label',strategy='prepared',output_mode='decision_only',
                    visited_nodes=10,leaf_evaluations=4,astar_calls=0)
        rows=[dict(common,config=i,engine=engine,distance_lookups=12 if i<2 else 9,
                   median_total_ms=2 if i<2 else 3) for i,engine in enumerate(ENGINES)]
        # Paired median +2 differs from median(target)-median(reference)=+1.
        values={0:[1,100,2],1:[1,100,2],2:[3,2,4]}
        raw=[dict(config=i,repeat=j,total_ms=value) for i,series in values.items() for j,value in enumerate(series)]
        pairs,blocks=make_pairs(rows,raw,3)
        self.assertEqual(len(pairs),2);self.assertEqual(len(blocks),6)
        self.assertTrue(all(p['median_delta_ms']==2 for p in pairs))
        summary=summarize(pairs)
        self.assertTrue(all(row['map_count']==1 for row in summary))
        self.assertTrue(all(row['median_lookup_ratio']==.75 for row in summary))


if __name__=='__main__':unittest.main()
