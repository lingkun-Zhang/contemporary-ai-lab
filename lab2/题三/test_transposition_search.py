"""Independent checks for full-terminal Alpha-Beta caching on shared DAGs.

The oracle expands occurrences with path keys; it never calls either production
searcher, and does not assume an object's value is independent of player role.
"""

import math
import random
import unittest

from alphabeta_tree import TreeNode, sample_tree, search
from transposition_search import cached_search


def occurrence_values(root, maximizing=True):
    values = {}
    pending = [(root, maximizing, "root", False)]
    while pending:
        node, role, path, finished = pending.pop()
        if not node.children:
            values[path] = node.value
        elif finished:
            scores = sorted(values[f"{path}/{i}"] for i in range(len(node.children)))
            values[path] = scores[-1 if role else 0]
        else:
            pending.append((node, role, path, True))
            pending.extend((child, not role, f"{path}/{i}", False)
                           for i, child in enumerate(node.children))
    return values


def random_dag(rng, depth=4, width=5):
    layer = [TreeNode(rng.choice([-12, -5, -1, 0, 0, 3, 8, 14]))
             for _ in range(width)]
    for _ in range(depth-1):
        layer = [TreeNode(children=[rng.choice(layer)
                                   for _ in range(rng.randint(1, 4))])
                 for _ in range(width)]
    return TreeNode(children=[rng.choice(layer) for _ in range(4)])


def random_tree(rng, level=0):
    if level >= 5 or (level and rng.random() < 0.3):
        return TreeNode(rng.randint(-40, 40))
    return TreeNode(children=[random_tree(rng, level+1)
                              for _ in range(rng.randint(1, 4))])


def shared_bound_example(right_first=8):
    shared = TreeNode(children=[TreeNode(6), TreeNode(9)])
    return TreeNode(children=[
        TreeNode(children=[TreeNode(5), shared]),
        TreeNode(children=[shared, TreeNode(right_first)]),
    ])


def snapshots(root):
    found, pending = {}, [root]
    while pending:
        node = pending.pop()
        if id(node) not in found:
            found[id(node)] = (node, node.value, tuple(node.children))
            pending.extend(node.children)
    return found


class TranspositionSearchTests(unittest.TestCase):
    def assert_conservation(self, result):
        stats = result.stats
        self.assertEqual(stats.visited_nodes,
                         stats.internal_nodes + stats.leaf_evaluations + stats.tt_returns)
        self.assertEqual(stats.tt_hits,
                         stats.exact_hits + stats.bound_cutoffs + stats.noncutting_hits)
        self.assertEqual(stats.tt_returns, stats.exact_hits + stats.bound_cutoffs)
        self.assertLessEqual(stats.tt_hits, stats.tt_probes)

    def assert_bound(self, value, kind, expected):
        if kind == "exact":
            self.assertEqual(value, expected)
        elif kind == "lower":
            self.assertLessEqual(value, expected)
        elif kind == "upper":
            self.assertGreaterEqual(value, expected)
        else:
            self.fail(f"Unknown bound {kind!r}")

    def test_pdf_sample_and_disabled_cache_match_original_counts(self):
        root = sample_tree()
        original = search(root)
        off = cached_search(root, mode="off")
        self.assertEqual(off.value, 5)
        for field in ("visited_nodes", "internal_nodes", "leaf_evaluations",
                      "cutoff_events", "skipped_child_roots", "bound_crossings"):
            self.assertEqual(getattr(off.stats, field), getattr(original.stats, field))
        self.assertEqual((off.stats.visited_nodes, off.stats.leaf_evaluations), (14, 7))
        self.assertEqual((off.stats.tt_probes, off.stats.tt_hits, off.stats.stores), (0, 0, 0))
        for mode in ("off", "exact", "bounds"):
            result = cached_search(root, mode=mode)
            self.assertEqual(result.value, 5)
            self.assertEqual(result.bound, "exact")
            self.assert_conservation(result)

    def test_repeated_exact_subtree_saves_actual_work(self):
        shared = TreeNode(children=[TreeNode(1), TreeNode(9)])
        root = TreeNode(children=[shared, shared, shared])
        off = cached_search(root, mode="off")
        self.assertEqual((off.value, off.stats.visited_nodes, off.stats.leaf_evaluations),
                         (1, 8, 4))
        for mode in ("exact", "bounds"):
            result = cached_search(root, mode=mode)
            self.assertEqual((result.value, result.stats.visited_nodes,
                              result.stats.leaf_evaluations, result.stats.exact_hits),
                             (1, 6, 2, 2))
            self.assert_conservation(result)

    def test_cutoff_lower_bound_is_not_reused_as_exact_value(self):
        root = shared_bound_example()
        self.assertEqual(occurrence_values(root)["root"], 8)
        for mode in ("exact", "bounds"):
            result = cached_search(root, mode=mode, trace=True)
            self.assertEqual(result.value, 8)
            self.assertEqual(result.bound, "exact")
            self.assertTrue(any(e["event"] == "return" and e["path"] == "root/0/1"
                                and e["value"] == 6 and e["returned_bound"] == "lower"
                                for e in result.trace))
            if mode == "bounds":
                self.assertGreaterEqual(result.stats.noncutting_hits, 1)
            self.assert_conservation(result)

    def test_decisive_lower_bound_can_return_without_expanding(self):
        shared = TreeNode(children=[TreeNode(6), TreeNode(9)])
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(5), shared]),
            TreeNode(children=[TreeNode(6), shared]),
        ])
        result = cached_search(root, trace=True)
        self.assertEqual(result.value, 6)
        self.assertGreaterEqual(result.stats.bound_cutoffs, 1)
        self.assertTrue(any(e["event"] == "return" and e["path"] == "root/1/1"
                            and e["returned_bound"] == "lower" and e["source"] == "cache"
                            for e in result.trace))
        self.assert_conservation(result)

    def test_decisive_upper_bound_for_min_root_is_sound(self):
        shared = TreeNode(children=[TreeNode(-6), TreeNode(-9)])
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(-5), shared]),
            TreeNode(children=[TreeNode(-6), shared]),
        ])
        result = cached_search(root, is_maximizing_player=False, trace=True)
        self.assertEqual(result.value, -6)
        self.assertGreaterEqual(result.stats.bound_cutoffs, 1)
        self.assertTrue(any(e["event"] == "return" and e["path"] == "root/1/1"
                            and e["returned_bound"] == "upper" and e["source"] == "cache"
                            for e in result.trace))
        self.assert_conservation(result)

    def test_same_object_at_different_player_roles_has_distinct_values(self):
        shared = TreeNode(children=[TreeNode(1), TreeNode(9)])
        root = TreeNode(children=[
            TreeNode(children=[shared, TreeNode(4)]), shared,
        ])
        true = occurrence_values(root)
        self.assertEqual((true["root/0/0"], true["root/1"], true["root"]), (9, 1, 4))
        for mode in ("off", "exact", "bounds"):
            result = cached_search(root, mode=mode)
            self.assertEqual(result.value, 4)
            self.assert_conservation(result)

    def test_fixed_random_dags_agree_for_modes_roles_and_depth_limits(self):
        rng = random.Random(20261030)
        for case in range(48):
            root = random_dag(rng)
            for role in (True, False):
                expected = occurrence_values(root, role)["root"]
                for mode in ("off", "exact", "bounds"):
                    for limit in (0, 1, 2, 4, None):
                        with self.subTest(case=case, role=role, mode=mode, limit=limit):
                            result = cached_search(root, mode=mode,
                                                   max_cache_depth=limit,
                                                   is_maximizing_player=role)
                            self.assertEqual(result.value, expected)
                            self.assertEqual(result.bound, "exact")
                            self.assert_conservation(result)

    def test_unshared_irregular_trees_keep_the_same_exact_result(self):
        rng = random.Random(71981)
        for case in range(90):
            root = random_tree(rng)
            expected = occurrence_values(root)["root"]
            for mode in ("off", "exact", "bounds"):
                with self.subTest(case=case, mode=mode):
                    result = cached_search(root, mode=mode)
                    self.assertEqual(result.value, expected)
                    self.assertEqual(result.stats.tt_hits, 0)
                    self.assert_conservation(result)

    def test_narrow_windows_and_internal_returns_obey_independent_bounds(self):
        rng = random.Random(87042)
        for case in range(12):
            root = random_dag(rng)
            for role in (True, False):
                true = occurrence_values(root, role)
                for alpha, beta in ((-20, -10), (-4, 2), (4, 8), (-math.inf, math.inf)):
                    for mode in ("exact", "bounds"):
                        with self.subTest(case=case, role=role, alpha=alpha, mode=mode):
                            result = cached_search(root, mode=mode, trace=True,
                                                   is_maximizing_player=role,
                                                   alpha=alpha, beta=beta)
                            self.assert_bound(result.value, result.bound, true["root"])
                            for event in result.trace:
                                if event["event"] == "return":
                                    self.assert_bound(event["value"], event["returned_bound"],
                                                      true[event["path"]])
                                if event["event"] == "store":
                                    self.assert_bound(event["value"], event["stored_bound"],
                                                      true[event["path"]])
                            self.assert_conservation(result)

    def test_cache_depth_is_admission_limit_not_search_truncation(self):
        shared = TreeNode(children=[TreeNode(1), TreeNode(9)])
        root = TreeNode(children=[shared, shared, shared])
        root_only = cached_search(root, max_cache_depth=0)
        one_level = cached_search(root, max_cache_depth=1)
        all_levels = cached_search(root, max_cache_depth=None)
        self.assertEqual([r.value for r in (root_only, one_level, all_levels)], [1, 1, 1])
        self.assertEqual(root_only.stats.tt_hits, 0)
        self.assertEqual(root_only.stats.table_peak_entries, 1)
        self.assertGreater(one_level.stats.exact_hits, 0)
        self.assertLess(one_level.stats.leaf_evaluations, root_only.stats.leaf_evaluations)
        self.assertGreaterEqual(all_levels.stats.table_peak_entries,
                                one_level.stats.table_peak_entries)

    def test_new_call_does_not_reuse_a_stale_value_after_leaf_mutation(self):
        first, second = TreeNode(1), TreeNode(2)
        root = TreeNode(children=[first, second])
        for mode in ("exact", "bounds"):
            second.value = 2
            self.assertEqual(cached_search(root, mode=mode).value, 2)
            second.value = -3
            result = cached_search(root, mode=mode)
            self.assertEqual(result.value, 1)
            self.assertEqual(result.stats.tt_hits, 0)

    def test_search_does_not_modify_values_or_children(self):
        root = random_dag(random.Random(9142))
        before = snapshots(root)
        for mode in ("off", "exact", "bounds"):
            cached_search(root, mode=mode, trace=True)
        for node, value, children in before.values():
            self.assertEqual(node.value, value)
            self.assertEqual(tuple(node.children), children)

    def test_invalid_mode_depth_role_and_windows_are_rejected(self):
        root = TreeNode(children=[TreeNode(1), TreeNode(9)])
        invalid = [dict(mode="naive"), dict(mode=None),
                   *[dict(max_cache_depth=x) for x in (-1, 1.5, True, "2")],
                   dict(alpha=5, beta=5), dict(alpha=8, beta=5),
                   dict(alpha=math.nan), dict(beta=math.nan),
                   dict(alpha=True), dict(is_maximizing_player=1)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    cached_search(root, **kwargs)

    def test_invalid_structure_and_cycles_are_rejected(self):
        loop = TreeNode(children=[TreeNode(1)])
        loop.children.append(loop)
        invalid = [loop, TreeNode(), TreeNode(True), TreeNode(math.inf),
                   TreeNode(2, [TreeNode(3)]), TreeNode(children=[None])]
        for root in invalid:
            with self.subTest(root=root):
                with self.assertRaises((ValueError, TypeError)):
                    cached_search(root)

    def test_negative_float_and_large_integer_scores_are_not_approximated(self):
        for scores in ((-2.5, -0.5, -9.25), (-(10**100), 10**100, 0)):
            shared = TreeNode(children=[TreeNode(v) for v in scores])
            root = TreeNode(children=[shared, shared])
            for role in (True, False):
                expected = occurrence_values(root, role)["root"]
                result = cached_search(root, is_maximizing_player=role)
                self.assertEqual(result.value, expected)
                self.assert_conservation(result)

    def test_left_to_right_order_is_preserved_and_trace_can_be_disabled(self):
        root = TreeNode(children=[TreeNode(-2), TreeNode(4), TreeNode(1)])
        result = cached_search(root, trace=True)
        children = [e["path"] for e in result.trace if e["event"] == "enter"
                    and e["depth"] == 1]
        self.assertEqual(children, ["root/0", "root/1", "root/2"])
        self.assertEqual(cached_search(root, trace=False).trace, [])


if __name__ == "__main__":
    unittest.main()
