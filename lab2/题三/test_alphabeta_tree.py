"""Independent correctness and counting checks for the course game-tree task.

The reference evaluator below uses iterative postorder traversal and path keys.
It neither calls the production minimax nor merges shared object occurrences.
"""

import itertools
import math
import random
import unittest

from alphabeta_tree import (
    TreeNode,
    alpha_beta,
    minimax,
    parse_tree,
    search,
    validate_tree,
)


def independent_values(root):
    values = {}
    counts = {"nodes": 0, "leaves": 0}
    pending = [(root, True, (), False)]
    while pending:
        node, is_max, path, finishing = pending.pop()
        if finishing:
            child_values = [values[path + (i,)] for i in range(len(node.children))]
            values[path] = sorted(child_values)[-1 if is_max else 0]
        else:
            counts["nodes"] += 1
            if not node.children:
                counts["leaves"] += 1
                values[path] = node.value
            else:
                pending.append((node, is_max, path, True))
                pending.extend(
                    (child, not is_max, path + (i,), False)
                    for i, child in enumerate(node.children)
                )
    return values, counts


def sample_tree(shared=False):
    first = TreeNode(children=[TreeNode(3), TreeNode(5)])
    middle = TreeNode(children=[TreeNode(6), TreeNode(9)])
    other_middle = middle if shared else TreeNode(children=[TreeNode(6), TreeNode(9)])
    last = TreeNode(children=[TreeNode(1), TreeNode(2)])
    return TreeNode(children=[
        TreeNode(children=[first, middle]),
        TreeNode(children=[other_middle, last]),
    ])


def random_tree(rng, depth=0):
    if depth >= 5 or (depth and rng.random() < 0.34):
        return TreeNode(rng.choice([-11, -4, -1, 0, 0, 1, 2, 7, 12]))
    return TreeNode(children=[
        random_tree(rng, depth + 1) for _ in range(rng.randint(1, 4))
    ])


class AlphaBetaTreeTests(unittest.TestCase):
    def test_official_sample_value_and_minimax_backups(self):
        root = sample_tree()
        values, counts = independent_values(root)
        self.assertEqual([values[(0, 0)], values[(0, 1)],
                          values[(1, 0)], values[(1, 1)]], [5, 9, 9, 2])
        self.assertEqual([values[(0,)], values[(1,)], values[()]], [5, 2, 5])
        self.assertEqual(counts, {"nodes": 15, "leaves": 8})
        self.assertEqual(minimax(root), 5)
        self.assertEqual(alpha_beta(root), 5)

    def test_left_to_right_sample_counts_and_actual_skipped_leaf(self):
        root = sample_tree()
        plain = search(root, algorithm="minimax", trace=True)
        pruned = search(root, trace=True)
        self.assertEqual(plain.stats.visited_nodes, 15)
        self.assertEqual(plain.stats.leaf_evaluations, 8)
        self.assertEqual(pruned.stats.visited_nodes, 14)
        self.assertEqual(pruned.stats.internal_nodes, 7)
        self.assertEqual(pruned.stats.leaf_evaluations, 7)
        leaf_paths = [event["path"] for event in pruned.trace if event["event"] == "leaf"]
        self.assertEqual(leaf_paths, [
            "root/0/0/0", "root/0/0/1", "root/0/1/0",
            "root/1/0/0", "root/1/0/1", "root/1/1/0", "root/1/1/1",
        ])
        self.assertNotIn("root/0/1/1", leaf_paths)
        self.assertEqual(pruned.stats.skipped_child_roots, 1)

    def test_shared_example_counts_occurrences_without_implicit_cache(self):
        shared, expanded = sample_tree(shared=True), sample_tree(shared=False)
        validate_tree(shared)
        for algorithm in ("minimax", "alphabeta"):
            a = search(shared, algorithm=algorithm)
            b = search(expanded, algorithm=algorithm)
            self.assertEqual(a.value, b.value)
            self.assertEqual(a.stats.visited_nodes, b.stats.visited_nodes)
            self.assertEqual(a.stats.leaf_evaluations, b.stats.leaf_evaluations)
        self.assertIs(shared.children[0].children[1], shared.children[1].children[0])
        self.assertEqual(independent_values(shared)[1]["nodes"], 15)

    def test_single_leaf_has_one_visit_and_zero_internal_nodes(self):
        for value in (-10, 0, 2.5):
            with self.subTest(value=value):
                result = search(TreeNode(value), trace=True)
                self.assertEqual(result.value, value)
                self.assertEqual(result.stats.visited_nodes, 1)
                self.assertEqual(result.stats.leaf_evaluations, 1)
                self.assertEqual(result.stats.internal_nodes, 0)
                self.assertEqual(result.stats.skipped_child_roots, 0)

    def test_negative_fractional_and_unbalanced_tree(self):
        root = TreeNode(children=[
            TreeNode(-4.5),
            TreeNode(children=[TreeNode(-3.25), TreeNode(-7.75)]),
            TreeNode(children=[TreeNode(children=[TreeNode(-2.5)]), TreeNode(0)]),
        ])
        expected = independent_values(root)[0][()]
        self.assertEqual(expected, -2.5)
        for order in ("left", "right"):
            for algorithm in ("minimax", "alphabeta"):
                self.assertEqual(search(root, algorithm=algorithm, order=order).value, expected)

    def test_equality_cutoff_keeps_root_value_and_saves_work(self):
        root = TreeNode(children=[
            TreeNode(5), TreeNode(children=[TreeNode(5), TreeNode(-100)])
        ])
        equal = search(root, cutoff_on_equal=True)
        strict = search(root, cutoff_on_equal=False)
        self.assertEqual(equal.value, 5)
        self.assertEqual(strict.value, 5)
        self.assertEqual(equal.stats.visited_nodes, 4)
        self.assertEqual(strict.stats.visited_nodes, 5)
        self.assertEqual(equal.stats.leaf_evaluations, 2)
        self.assertEqual(strict.stats.leaf_evaluations, 3)

    def test_random_trees_match_independent_iterative_oracle(self):
        rng = random.Random(188030)
        for case in range(180):
            root = random_tree(rng)
            values, counts = independent_values(root)
            for order in ("left", "right"):
                with self.subTest(case=case, order=order):
                    full = search(root, algorithm="minimax", order=order)
                    bounded = search(root, order=order)
                    self.assertEqual(full.value, values[()])
                    self.assertEqual(bounded.value, values[()])
                    self.assertEqual(full.stats.visited_nodes, counts["nodes"])
                    self.assertEqual(full.stats.leaf_evaluations, counts["leaves"])
                    self.assertLessEqual(bounded.stats.visited_nodes, full.stats.visited_nodes)
                    self.assertLessEqual(bounded.stats.leaf_evaluations, full.stats.leaf_evaluations)
                    self.assertEqual(bounded.stats.visited_nodes,
                                     bounded.stats.internal_nodes + bounded.stats.leaf_evaluations)

    def test_custom_permutations_change_work_but_preserve_value(self):
        root = TreeNode(children=[
            TreeNode(children=[TreeNode(8), TreeNode(7), TreeNode(6)]),
            TreeNode(children=[TreeNode(2), TreeNode(-1), TreeNode(9)]),
            TreeNode(children=[TreeNode(5), TreeNode(4), TreeNode(3)]),
        ])
        expected = independent_values(root)[0][()]
        visits = set()
        for root_order in itertools.permutations(range(3)):
            for inner_order in itertools.permutations(range(3)):
                def ordering(node, is_max, ro=root_order, io=inner_order):
                    return ro if node is root else io
                result = search(root, order=ordering)
                self.assertEqual(result.value, expected)
                visits.add(result.stats.visited_nodes)
        self.assertGreater(len(visits), 1)

    def test_ordering_never_mutates_tree(self):
        root = sample_tree(shared=True)
        snapshots = {}
        pending = [root]
        while pending:
            node = pending.pop()
            if id(node) not in snapshots:
                snapshots[id(node)] = (node, node.value, tuple(node.children))
                pending.extend(node.children)
        def backwards(node, is_max):
            return list(reversed(range(len(node.children))))
        for order in ("right", backwards):
            self.assertEqual(search(root, order=order).value, 5)
        for node, value, children in snapshots.values():
            self.assertEqual(node.value, value)
            self.assertEqual(tuple(node.children), children)

    def test_strict_and_equal_cutoff_agree_on_random_repeated_values(self):
        rng = random.Random(30188)
        for case in range(80):
            root = random_tree(rng)
            expected = independent_values(root)[0][()]
            with self.subTest(case=case):
                strict = search(root, cutoff_on_equal=False)
                equal = search(root, cutoff_on_equal=True)
                self.assertEqual(strict.value, expected)
                self.assertEqual(equal.value, expected)
                self.assertLessEqual(equal.stats.visited_nodes, strict.stats.visited_nodes)

    def test_cycle_is_rejected_but_shared_leaf_is_allowed(self):
        leaf = TreeNode(3)
        root = TreeNode(children=[leaf, leaf])
        validate_tree(root)
        self.assertEqual(search(root, algorithm="minimax").stats.visited_nodes, 3)
        direct = TreeNode(children=[TreeNode(1)])
        direct.children.append(direct)
        first, second = TreeNode(), TreeNode()
        first.children = [second]
        second.children = [first]
        for invalid in (direct, first):
            with self.assertRaises((ValueError, TypeError)):
                validate_tree(invalid)

    def test_nonfinite_bool_and_missing_leaf_scores_are_rejected(self):
        for value in (True, False, math.inf, -math.inf, math.nan, "3", None):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    search(TreeNode(value))

    def test_internal_score_conflicts_and_invalid_children_are_rejected(self):
        invalid_roots = [
            TreeNode(3, [TreeNode(1)]),
            TreeNode(children=[None]),
            TreeNode(children=[3]),
        ]
        for root in invalid_roots:
            with self.subTest(root=root):
                with self.assertRaises((ValueError, TypeError)):
                    validate_tree(root)

    def test_json_accepts_direct_and_wrapped_nodes(self):
        data = {"is_maximizing_player": True, "children": [
            {"is_maximizing_player": False, "children": [{"value": 2}, {"value": 8}]},
            {"value": 5},
        ]}
        for payload in (data, {"tree_structure": data}):
            root = parse_tree(payload)
            self.assertEqual(search(root).value, 5)

    def test_json_rejects_inconsistent_player_roles(self):
        for data in (
            {"is_maximizing_player": False, "value": 1},
            {"is_maximizing_player": True, "children": [
                {"is_maximizing_player": True, "value": 1}
            ]},
            {"is_maximizing_player": 1, "value": 1},
        ):
            with self.subTest(data=data):
                with self.assertRaises((ValueError, TypeError)):
                    parse_tree(data)

    def test_json_rejects_missing_or_conflicting_structure(self):
        for data in ({}, {"children": []}, {"value": 1, "children": [{"value": 2}]},
                     {"children": "wrong"}, [], {"value": True}):
            with self.subTest(data=data):
                with self.assertRaises((ValueError, TypeError)):
                    parse_tree(data)

    def test_invalid_order_callback_cannot_drop_or_duplicate_children(self):
        root = TreeNode(children=[TreeNode(1), TreeNode(2)])
        for indices in ([0], [0, 0], [0, 2], [-1, 0], [True, False]):
            with self.subTest(indices=indices):
                with self.assertRaises((ValueError, TypeError)):
                    search(root, order=lambda node, role, selected=indices: selected)

    def test_invalid_algorithm_or_order_is_rejected(self):
        root = sample_tree()
        with self.assertRaises((ValueError, TypeError)):
            search(root, algorithm="greedy")
        with self.assertRaises((ValueError, TypeError)):
            search(root, order="unknown")

    def test_public_windows_must_be_open_and_increasing(self):
        root = TreeNode(children=[TreeNode(1), TreeNode(9)])
        for alpha, beta in ((5, 5), (6, 5)):
            with self.subTest(alpha=alpha, beta=beta):
                with self.assertRaises(ValueError):
                    alpha_beta(root, alpha=alpha, beta=beta)
                for equal_cutoff in (True, False):
                    with self.assertRaises(ValueError):
                        search(root, alpha=alpha, beta=beta,
                               cutoff_on_equal=equal_cutoff)

    def test_narrow_windows_return_honest_bounds_instead_of_forced_exact_values(self):
        root = TreeNode(children=[TreeNode(1), TreeNode(9)])
        truth = independent_values(root)[0][()]
        for alpha, beta, expected_kind in ((-10, 0, "lower"), (5, 8, "lower"),
                                            (10, 20, "upper"), (0, 10, "exact")):
            with self.subTest(alpha=alpha, beta=beta):
                result = search(root, alpha=alpha, beta=beta)
                self.assertEqual(result.bound, expected_kind)
                if result.bound == "lower":
                    self.assertLessEqual(result.value, truth)
                elif result.bound == "upper":
                    self.assertGreaterEqual(result.value, truth)
                else:
                    self.assertEqual(result.value, truth)
                self.assertEqual(alpha_beta(root, alpha=alpha, beta=beta), result.value)
        # The official sample's pruned MAX(6,9) returns 6, a lower bound on 9.
        sample = sample_tree()
        true_subtree_values = independent_values(sample)[0]
        returns = {event["path"]: event for event in search(sample, trace=True).trace
                   if event["event"] == "return"}
        cut_return = returns["root/0/1"]
        self.assertEqual(cut_return["value"], 6)
        self.assertEqual(cut_return["returned_bound"], "lower")
        self.assertEqual(true_subtree_values[(0, 1)], 9)
        self.assertLessEqual(cut_return["value"], true_subtree_values[(0, 1)])

    def test_random_narrow_window_and_subtree_bounds_match_full_tree_truth(self):
        rng = random.Random(18803031)
        windows = ((-10, -5), (-2, 2), (5, 10), (-20, 20))
        for case in range(40):
            root = random_tree(rng)
            values, _ = independent_values(root)
            for alpha, beta in windows:
                for order in ("left", "right"):
                    with self.subTest(case=case, alpha=alpha, beta=beta, order=order):
                        result = search(root, alpha=alpha, beta=beta, order=order, trace=True)
                        returns = [event for event in result.trace if event["event"] == "return"]
                        for event in returns:
                            path = tuple(int(part) for part in event["path"].split("/")[1:])
                            bound, returned, truth = event["returned_bound"], event["value"], values[path]
                            self.assertIn(bound, ("lower", "upper", "exact"))
                            if bound == "lower":
                                self.assertLessEqual(returned, truth)
                            elif bound == "upper":
                                self.assertGreaterEqual(returned, truth)
                            else:
                                self.assertEqual(returned, truth)
                        if result.bound == "lower":
                            self.assertLessEqual(result.value, values[()])
                        elif result.bound == "upper":
                            self.assertGreaterEqual(result.value, values[()])
                        else:
                            self.assertEqual(result.value, values[()])


if __name__ == "__main__":
    unittest.main()
