"""增量缓存的独立完整枚举、账本和错误抽象反例。"""
import math
import random
import unittest

from incremental_cache_search import cache_search
from reference_tools import enumerate_game
from treasure_game import (Problem, State, apply_action, build_distance_matrix,
                           game_search, initial_state, sample_problem)


def line_problem(n, seed=0, values=None):
    rng = random.Random(seed)
    values = tuple(rng.randrange(-3, 19) for _ in range(n)) if values is None else tuple(values)
    problem = Problem(((1,)*(n+2),), ('A', 'B', *(f'T{i+1}' for i in range(n))),
                      ((0, 0), (0, n+1), *((0, i+1) for i in range(n))), values)
    matrix = tuple(tuple(abs(a[1]-b[1]) for b in problem.points) for a in problem.points)
    return problem, matrix


def oracle(problem, matrix, depth, state=None):
    s = initial_state(problem) if state is None else state
    remaining = [i for i in range(len(problem.values)) if s.remaining & (1 << i)]
    return enumerate_game(matrix, problem.values, depth, s.pos_a, s.pos_b,
                          remaining, s.score_a, s.score_b, s.turn)[0]


class IncrementalCacheTests(unittest.TestCase):
    def assert_pv(self, problem, matrix, depth, result, state=None):
        s = initial_state(problem) if state is None else state
        pa, pb, sa, sb, turn = s.pos_a, s.pos_b, s.score_a, s.score_b, s.turn
        remaining = {i for i in range(len(problem.values)) if s.remaining & (1 << i)}
        self.assertLessEqual(len(result.principal_variation), depth)
        self.assertEqual(result.best_target, result.principal_variation[0]
                         if result.principal_variation else None)
        for label in result.principal_variation:
            action = problem.labels.index(label)-2
            self.assertIn(action, remaining)
            position = pa if turn == 'A' else pb
            cost = matrix[position][action+2]
            self.assertTrue(math.isfinite(cost))
            gain = problem.values[action]-cost
            if turn == 'A':
                sa += gain; pa = action+2; turn = 'B'
            else:
                sb += gain; pb = action+2; turn = 'A'
            remaining.remove(action)
        self.assertEqual(sa-sb, result.best_value)
        if len(result.principal_variation) < depth and remaining:
            position = pa if turn == 'A' else pb
            self.assertFalse(any(math.isfinite(matrix[position][a+2]) for a in remaining))

    def test_official_h2_h4_all_modes_and_orders(self):
        p = sample_problem(); m = build_distance_matrix(p)
        for h, target, value in ((2, 'T2', -1), (4, 'T3', 3)):
            for algorithm in ('minimax', 'alphabeta'):
                for order in ('label', 'gain', 'reverse'):
                    for mode in ('none', 'fullscore', 'incremental'):
                        r = cache_search(p, m, h, algorithm, order, mode)
                        self.assertEqual((r.best_target, r.best_value), (target, value))
                        self.assert_pv(p, m.matrix, h, r)

    def test_four_treasures_have_no_nonterminal_transpositions(self):
        p, m = line_problem(4, 81)
        for algorithm in ('minimax', 'alphabeta'):
            r = cache_search(p, m, 4, algorithm)
            self.assertEqual(r.stats.cache_hits, 0)
            original = game_search(p, m, 4, algorithm)
            self.assertEqual(r.stats.visited_nodes, original.stats.visited_nodes)

    def test_incremental_minimax_reuses_real_nonterminal_states(self):
        p, m = line_problem(6, values=(9, 6, 7, 7, 8, 5))
        r = cache_search(p, m, 6, 'minimax')
        control = cache_search(p, m, 6, 'minimax', cache_mode='none')
        self.assertGreater(r.stats.cache_hits, 0)
        self.assertLess(r.stats.internal_nodes, control.stats.internal_nodes)
        self.assertEqual(r.best_value, oracle(p, m, 6))
        self.assert_pv(p, m, 6, r)

    def test_scorefree_absolute_value_cache_counterexample(self):
        p, m = line_problem(6, values=(9, 6, 7, 7, 8, 5))
        states = []
        for history in ((0, 1, 2, 3), (1, 0, 2, 3)):
            state = initial_state(p)
            for action in history:
                state = apply_action(p, state, action, m)
            states.append(state)
        first, second = states
        key = lambda s: (s.pos_a, s.pos_b, s.remaining, s.turn, 2)
        self.assertEqual(key(first), key(second))
        self.assertEqual((first.utility, second.utility), (7, 3))
        a, b = [cache_search(p, m, 2, state=s) for s in states]
        self.assertEqual((a.best_value, b.best_value), (10, 6))
        self.assertEqual((a.best_value-first.utility, b.best_value-second.utility), (3, 3))
        wrong = {key(first): a.best_value}
        self.assertNotEqual(wrong[key(second)], oracle(p, m, 2, second))

    def test_random_full_enumeration_all_modes_algorithms_depths(self):
        for seed in range(8):
            p, m = line_problem(6, 20261110+seed)
            for depth in (0, 2, 4, 6):
                expected = oracle(p, m, depth)
                for algorithm in ('minimax', 'alphabeta'):
                    for order in ('label', 'gain'):
                        for mode in ('none', 'fullscore', 'incremental'):
                            r = cache_search(p, m, depth, algorithm, order, mode)
                            self.assertEqual(r.best_value, expected,
                                (seed, depth, algorithm, order, mode))
                            self.assert_pv(p, m, depth, r)

    def test_relative_bound_returns_translate_by_current_utility(self):
        hit_kinds = set()
        for seed in range(6):
            p, m = line_problem(7, 20261200+seed)
            r = cache_search(p, m, 7, order='gain', record_trace=True)
            self.assertEqual(r.best_value, oracle(p, m, 7))
            self.assert_pv(p, m, 7, r)
            for event in r.trace:
                if event['event'] == 'cache_return':
                    hit_kinds.add(event['kind'])
                    self.assertEqual(event['stored_value']+event['utility'], event['absolute_value'])
                    if event['kind'] == 'lower':
                        self.assertGreaterEqual(event['absolute_value'], event['beta'])
                    elif event['kind'] == 'upper':
                        self.assertLessEqual(event['absolute_value'], event['alpha'])
        self.assertIn('lower', hit_kinds)
        self.assertIn('upper', hit_kinds)

    def test_score_shift_and_noninitial_players(self):
        p, m = line_problem(5, 113)
        state = State(2, 3, 0b11100, 13, -6, 'B')
        shifted = State(2, 3, 0b11100, 53, -6, 'B')
        for h in (1, 2, 3, 6):
            for mode in ('fullscore', 'incremental'):
                a = cache_search(p, m, h, cache_mode=mode, state=state)
                b = cache_search(p, m, h, cache_mode=mode, state=shifted)
                self.assertEqual(b.best_value-a.best_value, 40)
                self.assertEqual(a.best_value, oracle(p, m, h, state))
                self.assert_pv(p, m, h, b, shifted)

    def test_fresh_cache_per_call_depth_and_problem(self):
        p, m = line_problem(6, 53)
        changed, _ = line_problem(6, values=(500, 0, -2, 0, 7, 8))
        for p2, h in ((p, 6), (p, 2), (changed, 6), (p, 6)):
            r = cache_search(p2, m, h)
            self.assertEqual(r.best_value, oracle(p2, m, h))

    def test_unreachable_action_ends_branch_without_pass(self):
        p, m = line_problem(4)
        blocked = tuple(tuple(0 if i == j else math.inf for j in range(6)) for i in range(6))
        for mode in ('none', 'fullscore', 'incremental'):
            r = cache_search(p, blocked, 4, cache_mode=mode)
            self.assertEqual((r.best_value, r.best_target, r.principal_variation), (0, None, ()))

    def test_no_input_mutation_and_reproducible_calls(self):
        p, m = line_problem(6, 3)
        before = repr((p, m))
        a = cache_search(p, m, 6)
        b = cache_search(p, m, 6)
        self.assertEqual(repr((p, m)), before)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_zero_depth_zero_treasures_and_large_existing_scores(self):
        p, m = line_problem(0)
        s = State(0, 1, 0, 10**30, -(10**30), 'B')
        for h in (0, 4):
            r = cache_search(p, m, h, state=s)
            self.assertEqual((r.best_value, r.principal_variation), (2*10**30, ()))

    def test_invalid_options_and_exact_integer_scope(self):
        p, m = line_problem(4)
        for kwargs in ({'depth':-1}, {'depth':True}, {'depth':4, 'algorithm':'bad'},
                       {'depth':4,'order':'bad'}, {'depth':4,'cache_mode':'bad'}):
            with self.assertRaises(ValueError):
                cache_search(p, m, **kwargs)
        pfloat = Problem(p.grid, p.labels, p.points, (1.5, 2, 3, 4))
        with self.assertRaises(ValueError):
            cache_search(pfloat, m, 4)
        with self.assertRaises(ValueError):
            cache_search(p, m, 4, state=State(0, 1, 15, .5, 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
