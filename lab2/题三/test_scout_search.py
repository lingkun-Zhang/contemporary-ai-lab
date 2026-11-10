"""Scout tests use an independent full-tree minimax and predicate oracle."""

import itertools
import math
import random
import unittest

from alphabeta_tree import TreeNode, sample_tree, search
from scout_search import scout_search


def oracle(node, maximizing=True):
    if not node.children:
        return node.value
    values = [oracle(child, not maximizing) for child in node.children]
    return max(values) if maximizing else min(values)


def node(*children):
    return TreeNode(children=[child if isinstance(child, TreeNode) else TreeNode(child)
                              for child in children])


class ScoutCorrectnessTests(unittest.TestCase):
    def check(self, root, maximizing=True, order="left"):
        expected = oracle(root, maximizing)
        for mode in ("scout", "alphabeta"):
            result = scout_search(root, mode=mode, order=order,
                                  is_maximizing_player=maximizing)
            self.assertEqual(result.value, expected)
            self.assertEqual(result.bound, "exact")
            s = result.stats
            self.assertEqual(s.exact_visits, s.exact_internal_visits+s.exact_leaf_reads)
            self.assertEqual(s.test_visits, s.test_internal_visits+s.test_leaf_reads)
            self.assertEqual(s.total_visits, s.exact_visits+s.test_visits)
            self.assertEqual(s.total_leaf_reads, s.exact_leaf_reads+s.test_leaf_reads)
            self.assertEqual(s.probe_calls, s.test_true+s.test_false)
            self.assertEqual(s.research_calls, s.test_true)
            if mode == "alphabeta":
                reference = search(root, order=order, is_maximizing_player=maximizing)
                self.assertEqual(s.total_visits, reference.stats.visited_nodes)
                self.assertEqual(s.total_leaf_reads, reference.stats.leaf_evaluations)
                self.assertEqual(s.test_visits+s.probe_calls+s.research_calls, 0)

    def test_exhaustive_binary_depth_two_and_three(self):
        count = 0
        for leaves in (4, 8):
            alphabet = (-1, 0, 1) if leaves == 4 else (0, 1)
            for scores in itertools.product(alphabet, repeat=leaves):
                layer = [TreeNode(score) for score in scores]
                while len(layer) > 1:
                    layer = [TreeNode(children=layer[i:i+2]) for i in range(0, len(layer), 2)]
                for maximizing in (True, False):
                    for order in ("left", "right"):
                        self.check(layer[0], maximizing, order)
                        count += 2
        self.assertEqual(count, (81+256)*8)

    def test_pdf_and_nonuniform_tree(self):
        for root in (sample_tree(), node(1, node(-4, 2, node(8)), node(node(1, -2), 3))):
            self.check(root)
            self.check(root, False, "right")

    def test_fractional_values_and_very_close_float(self):
        nearby = math.nextafter(0.5, math.inf)
        for root in (node(0.0, node(.8, .2)), node(.5, nearby, -1.2),
                     node(node(-.2, -.8), node(nearby, .5))):
            self.check(root)
            self.check(root, False)

    def test_large_integers_and_negative_values(self):
        self.check(node(2**100+1, 2**100, -2**120))
        self.check(node(node(-9, -2), node(-4, -5)))

    def test_ties_never_trigger_research(self):
        root = node(node(7, 7, 7), node(7, 7, 7), node(7, 7, 7))
        result = scout_search(root)
        self.assertEqual(result.value, 7)
        self.assertEqual(result.stats.research_calls, 0)
        self.assertGreater(result.stats.test_false, 0)

    def test_late_improvements_count_repeated_leaf_reads(self):
        root = node(1, 2, 3)
        result = scout_search(root)
        self.assertEqual(result.value, 3)
        self.assertEqual(result.stats.exact_leaf_reads, 3)
        self.assertEqual(result.stats.test_leaf_reads, 2)
        self.assertEqual(result.stats.total_leaf_reads, 5)
        self.assertEqual(result.stats.research_calls, 2)

    def test_shared_objects_count_visits_not_unique_nodes(self):
        child = node(6, 9)
        self.check(node(node(5, child), node(child, 8)))

    def test_fixed_callback_order(self):
        root = sample_tree()
        callback = lambda current, maximizing: list(range(len(current.children)-1, -1, -1))
        self.check(root, order=callback)

    def test_trace_is_optional(self):
        root = node(1, 2, 3)
        plain, traced = scout_search(root), scout_search(root, trace=True)
        self.assertEqual(plain.value, traced.value)
        self.assertEqual(plain.stats, traced.stats)
        self.assertEqual(plain.trace, [])
        self.assertTrue(any(r['event'] == 'research' for r in traced.trace))

    def test_invalid_modes_roles_order_and_tree(self):
        root = node(1, 2)
        for kwargs in ({'mode':'pvs'}, {'is_maximizing_player':1}, {'trace':1},
                       {'order':'random'}, {'order':42}, {'order':lambda n,m:[0,0]}):
            with self.assertRaises(ValueError):scout_search(root, **kwargs)
        for invalid in (TreeNode(), TreeNode(float('nan')), TreeNode(float('inf')),
                        TreeNode(True), TreeNode(1, children=[TreeNode(1)])):
            with self.assertRaises(ValueError):scout_search(invalid)
        cyclic = node(1)
        cyclic.children.append(cyclic)
        with self.assertRaises(ValueError):scout_search(cyclic)

    def test_random_nonuniform_trees(self):
        rng = random.Random(20261110)
        def make(depth):
            if depth == 0 or rng.random() < .2:
                return TreeNode(rng.randint(-100, 100)/8)
            return TreeNode(children=[make(depth-1) for _ in range(rng.randint(1, 4))])
        for _ in range(60):
            tree = make(4)
            self.check(tree, rng.choice((True, False)), rng.choice(("left", "right")))


if __name__ == "__main__":
    unittest.main()
