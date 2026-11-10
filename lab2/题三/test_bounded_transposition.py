"""Capacity and bound correctness, checked against independent full-tree values."""
import math
import random
import unittest

from alphabeta_tree import TreeNode, search, sample_tree
from bounded_transposition import BoundedStats, _LRUTable, bounded_search
from transposition_search import cached_search


def full_values(root, player=True):
    values, pending = {}, [(root, player, 'root', False)]
    while pending:
        node, role, path, done = pending.pop()
        if not node.children:
            values[path] = node.value
        elif done:
            children = sorted(values[f'{path}/{i}'] for i in range(len(node.children)))
            values[path] = children[-1 if role else 0]
        else:
            pending.append((node, role, path, True))
            pending.extend((c, not role, f'{path}/{i}', False) for i,c in enumerate(node.children))
    return values


def small_dag(rng):
    layer=[TreeNode(rng.choice([-8,-3,0,0,4,11])) for _ in range(4)]
    for _ in range(3):
        layer=[TreeNode(children=[rng.choice(layer) for _ in range(rng.randint(1,4))]) for _ in range(4)]
    return TreeNode(children=[rng.choice(layer) for _ in range(3)])


class BoundedTableTests(unittest.TestCase):
    def assert_counts(self,result,capacity):
        s=result.stats
        self.assertEqual(s.visited_nodes,s.internal_nodes+s.leaf_evaluations+s.tt_returns)
        self.assertEqual(s.tt_hits,s.exact_hits+s.bound_cutoffs+s.noncutting_hits)
        self.assertEqual(s.tt_returns,s.exact_hits+s.bound_cutoffs)
        self.assertEqual(s.stores,s.new_insertions+s.entry_updates)
        self.assertEqual(s.table_final_entries,s.new_insertions-s.evictions)
        if capacity is not None: self.assertLessEqual(s.table_peak_entries,capacity)

    def assert_bound(self,value,kind,true):
        if kind=='exact': self.assertEqual(value,true)
        elif kind=='lower': self.assertLessEqual(value,true)
        elif kind=='upper': self.assertGreaterEqual(value,true)
        else: self.fail(kind)

    def test_sample_and_unlimited_match_existing_search_semantics(self):
        root=sample_tree()
        original=search(root)
        self.assertEqual(original.value,5)
        for cap in (0,1,4,None):
            r=bounded_search(root,capacity=cap)
            self.assertEqual(r.value,5)
            self.assert_counts(r,cap)
        old=cached_search(root)
        unlimited=bounded_search(root)
        for field in vars(old.stats):
            self.assertEqual(getattr(old.stats,field),getattr(unlimited.stats,field))

    def test_random_dags_preserve_values_for_capacities_and_players(self):
        rng=random.Random(20261301)
        for case in range(36):
            root=small_dag(rng)
            for role in (True,False):
                true=full_values(root,role)['root']
                for cap in (0,1,2,4,8,None):
                    with self.subTest(case=case,role=role,cap=cap):
                        result=bounded_search(root,capacity=cap,is_maximizing_player=role)
                        self.assertEqual(result.value,true)
                        self.assertEqual(result.bound,'exact')
                        self.assert_counts(result,cap)

    def test_narrow_windows_keep_all_returned_and_stored_bounds_safe(self):
        rng=random.Random(20261304)
        for case in range(10):
            root=small_dag(rng)
            for role in (True,False):
                true=full_values(root,role)
                for low,high in ((-12,-5),(-2,2),(4,8)):
                    for cap in (1,4,None):
                        r=bounded_search(root,capacity=cap,alpha=low,beta=high,
                                         is_maximizing_player=role,trace=True)
                        self.assert_bound(r.value,r.bound,true['root'])
                        for event in r.trace:
                            if event['event']=='return':
                                self.assert_bound(event['value'],event['returned_bound'],true[event['path']])
                            if event['event']=='store':
                                self.assert_bound(event['value'],event['stored_bound'],true[event['path']])
                        self.assert_counts(r,cap)

    def test_player_key_and_cutoff_counterexample_survive_eviction(self):
        x=TreeNode(children=[TreeNode(6),TreeNode(9)])
        counter=TreeNode(children=[TreeNode(children=[TreeNode(5),x]),TreeNode(children=[x,TreeNode(8)])])
        shared=TreeNode(children=[TreeNode(1),TreeNode(9)])
        different_roles=TreeNode(children=[TreeNode(children=[shared,TreeNode(4)]),shared])
        for cap in (0,1,2,4,8,None):
            self.assertEqual(bounded_search(counter,capacity=cap).value,8)
            self.assertEqual(bounded_search(different_roles,capacity=cap).value,4)

    def test_each_call_starts_empty_and_preserves_the_input(self):
        a,b=TreeNode(1),TreeNode(9)
        shared=TreeNode(children=[a,b])
        root=TreeNode(children=[shared,shared])
        children=[tuple(root.children),tuple(shared.children)]
        first=bounded_search(root,capacity=4)
        self.assertEqual(first.value,1)
        a.value=12
        self.assertEqual(bounded_search(root,capacity=4).value,9)
        self.assertEqual([tuple(root.children),tuple(shared.children)],children)

    def test_lru_hit_refreshes_even_a_nonclosing_bound(self):
        stats=BoundedStats();table=_LRUTable(2,stats)
        a,b,c=('a',True),('b',True),('c',False)
        table.store(a,6,'lower');table.store(b,9,'exact')
        # A lower bound of 6 does not close beta=10, but this probe is an access.
        hit=table.probe(a)
        self.assertEqual((hit.value,hit.bound),(6,'lower'))
        self.assertLess(hit.value,10)
        accepted,evicted=table.store(c,-2,'upper')
        self.assertTrue(accepted);self.assertEqual(evicted,b)
        self.assertEqual(list(table.entries),[a,c])
        self.assertEqual(stats.evictions,1)

    def test_exact_and_stronger_bounds_are_not_replaced_by_weaker_entries(self):
        stats=BoundedStats();table=_LRUTable(3,stats)
        table.store(('x',True),8,'exact')
        table.store(('l',True),6,'lower')
        table.store(('u',True),3,'upper')
        self.assertEqual(table.store(('x',True),6,'lower'),(False,None))
        self.assertEqual(table.store(('l',True),5,'lower'),(False,None))
        self.assertEqual(table.store(('u',True),4,'upper'),(False,None))
        self.assertEqual(stats.stores,3)
        self.assertTrue(table.store(('l',True),7,'lower')[0])
        self.assertEqual(stats.entry_updates,1)
        self.assertEqual(stats.stores,stats.new_insertions+stats.entry_updates)

    def test_zero_disables_table_and_small_capacities_obey_conservation(self):
        leaf=TreeNode(3)
        root=TreeNode(children=[leaf,leaf,leaf])
        off=bounded_search(root,capacity=0)
        self.assertEqual((off.stats.tt_probes,off.stats.stores,off.stats.evictions,off.stats.table_peak_entries),(0,0,0,0))
        self.assertEqual(off.stats.visited_nodes,4)
        for cap in (1,2,4):
            r=bounded_search(sample_tree(),capacity=cap)
            self.assert_counts(r,cap)
            self.assertGreater(r.stats.evictions,0)
            self.assertEqual(r.stats.table_final_entries,cap)

    def test_invalid_capacity_player_and_windows_are_rejected(self):
        root=TreeNode(1)
        invalid=[dict(capacity=x) for x in (-1,1.5,True,'4')]
        invalid.extend([dict(alpha=5,beta=5),dict(alpha=6,beta=5),dict(alpha=math.nan),
                        dict(beta=math.nan),dict(alpha=True),dict(is_maximizing_player=1)])
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError): bounded_search(root,**kwargs)

    def test_invalid_tree_and_cycles_are_rejected(self):
        cycle=TreeNode(children=[TreeNode(2)]);cycle.children.append(cycle)
        for root in (cycle,TreeNode(),TreeNode(True),TreeNode(math.inf),TreeNode(2,[TreeNode(1)])):
            with self.assertRaises(ValueError): bounded_search(root)


if __name__=='__main__': unittest.main()
