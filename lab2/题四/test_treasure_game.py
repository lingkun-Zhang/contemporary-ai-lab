"""Independent path and dynamic-game verification for Lab2 question four."""
import heapq
import math
import random
import unittest

from treasure_game import (
    Problem, State, apply_action, astar_search, build_distance_matrix,
    game_search, initial_state, legal_actions, parse_problem,
    root_action_values, sample_problem,
)


RAW_GRID = (
    (1, 1, 1, None, 2, 1, 1),
    (1, None, 6, None, 1, None, 1),
    (1, 1, 1, 1, 1, 1, 1),
    (None, 1, None, 8, None, 1, None),
    (1, 1, 1, 1, 1, 1, 1),
    (1, None, 9, None, 1, None, 1),
    (1, 1, 1, None, 1, 1, 1),
)
POINTS = ((0, 0), (6, 6), (0, 6), (2, 2), (4, 4), (6, 0))
ROOT_VALUES = {
    1: (-1, 2, -1, -1),
    2: (-4, -1, -2, -4),
    3: (-4, 2, -1, -4),
    4: (-5, 1, 3, -7),
}


def independent_dijkstra(grid, source, destination):
    """Uses only raw numeric cells; no formal normalization/neighbour helpers."""
    costs = {source: 0}
    pending = [(0, source)]
    height, width = len(grid), len(grid[0])
    while pending:
        distance, point = heapq.heappop(pending)
        if distance != costs[point]:
            continue
        if point == destination:
            return distance
        row, column = point
        for r, c in ((row-1, column), (row+1, column), (row, column-1), (row, column+1)):
            if not (0 <= r < height and 0 <= c < width) or grid[r][c] is None:
                continue
            candidate = distance + grid[r][c]
            if candidate < costs.get((r, c), math.inf):
                costs[(r, c)] = candidate
                heapq.heappush(pending, (candidate, (r, c)))
    return math.inf


def independent_matrix(problem):
    return tuple(tuple(independent_dijkstra(problem.grid, p, q) for q in problem.points)
                 for p in problem.points)


def state_tuple(state):
    return state.pos_a, state.pos_b, state.remaining, state.score_a, state.score_b, state.turn


def independent_step(values, distances, state, action):
    """Deliberately separate transition implementation for the game oracle."""
    pa, pb, mask, sa, sb, turn = state
    destination = action + 2
    if turn == 'A':
        return destination, pb, mask ^ (1 << action), sa + values[action] - distances[pa][destination], sb, 'B'
    return pa, destination, mask ^ (1 << action), sa, sb + values[action] - distances[pb][destination], 'A'


def independent_game(values, distances, state, depth):
    pa, pb, mask, sa, sb, turn = state
    if depth == 0 or mask == 0:
        return sa - sb
    source = pa if turn == 'A' else pb
    candidates = [i for i in range(len(values)) if mask & (1 << i) and math.isfinite(distances[source][i+2])]
    if not candidates:
        return sa - sb
    returns = [independent_game(values, distances,
                                independent_step(values, distances, state, action), depth-1)
               for action in candidates]
    return (max if turn == 'A' else min)(returns)


def independent_actions(problem, distances, state, depth):
    if depth == 0:
        return {}
    pa, pb, mask, _, _, turn = state
    source = pa if turn == 'A' else pb
    return {
        problem.labels[i+2]: independent_game(problem.values, distances,
                                               independent_step(problem.values, distances, state, i), depth-1)
        for i in range(len(problem.values))
        if mask & (1 << i) and math.isfinite(distances[source][i+2])
    }


def small_problem(grid, points, values):
    labels = ('A', 'B') + tuple(f'T{i+1}' for i in range(len(values)))
    return Problem(tuple(tuple(row) for row in grid), labels, tuple(points), tuple(values))


class PathTests(unittest.TestCase):
    def assert_path(self, grid, start, goal, result, expected):
        self.assertEqual(result.cost, expected)
        self.assertGreaterEqual(result.stats.generated_nodes, result.stats.unique_reached)
        self.assertLessEqual(result.stats.unique_reached, sum(cell is not None for row in grid for cell in row))
        if math.isinf(expected):
            self.assertEqual(tuple(result.path), ())
            self.assertEqual(tuple(result.ledger), ())
            return
        path = tuple(tuple(point) for point in result.path)
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertGreaterEqual(result.stats.unique_reached, len(path))
        self.assertEqual(len(result.ledger), len(path))
        self.assertEqual(result.ledger[0]['enter_cost'], 0)
        self.assertEqual(result.ledger[0]['cumulative_cost'], 0)
        total = 0
        for step, (before, after) in enumerate(zip(path, path[1:]), 1):
            self.assertEqual(abs(before[0]-after[0]) + abs(before[1]-after[1]), 1)
            self.assertIsNotNone(grid[after[0]][after[1]])
            total += grid[after[0]][after[1]]
            entry = result.ledger[step]
            self.assertEqual(entry['step'], step)
            self.assertEqual(tuple(entry['position']), after)
            self.assertEqual(entry['enter_cost'], grid[after[0]][after[1]])
            self.assertEqual(entry['cumulative_cost'], total)
        self.assertEqual(total, expected)

    def test_pdf_grid_positions_and_treasure_values_are_exact(self):
        problem = sample_problem()
        self.assertEqual(problem.grid, RAW_GRID)
        self.assertEqual(problem.points, POINTS)
        self.assertEqual(problem.labels, ('A', 'B', 'T1', 'T2', 'T3', 'T4'))
        self.assertEqual(problem.values, (9, 6, 7, 7))

    def test_entering_fee_includes_goal_and_excludes_start(self):
        grid = ((7, 3, 9),)
        forward = astar_search(grid, (0, 0), (0, 2))
        backward = astar_search(grid, (0, 2), (0, 0))
        self.assert_path(grid, (0, 0), (0, 2), forward, 12)
        self.assert_path(grid, (0, 2), (0, 0), backward, 10)

    def test_identical_start_and_goal_needs_no_entering_payment(self):
        result = astar_search(((9,),), (0, 0), (0, 0))
        self.assertEqual(result.cost, 0)
        self.assertEqual(tuple(result.path), ((0, 0),))

    def test_walls_cannot_be_crossed_diagonally(self):
        grid = ((1, None), (None, 1))
        self.assert_path(grid, (0, 0), (1, 1), astar_search(grid, (0, 0), (1, 1)), math.inf)

    def test_fixed_random_positive_cost_maps_match_dijkstra(self):
        rng = random.Random(40101)
        for index in range(60):
            grid = [[None if rng.random() < .20 else rng.randint(1, 9) for _ in range(5)] for _ in range(4)]
            grid[0][0], grid[3][4] = 7, 3
            expected = independent_dijkstra(grid, (0, 0), (3, 4))
            for heuristic in ('zero', 'manhattan'):
                with self.subTest(index=index, heuristic=heuristic):
                    self.assert_path(grid, (0, 0), (3, 4), astar_search(grid, (0, 0), (3, 4), heuristic=heuristic), expected)

    def test_zero_cost_maps_keep_heuristic_safe(self):
        rng = random.Random(40102)
        for index in range(40):
            grid = [[None if rng.random() < .15 else rng.randint(0, 4) for _ in range(4)] for _ in range(3)]
            grid[0][0], grid[2][3] = 1, 0
            expected = independent_dijkstra(grid, (0, 0), (2, 3))
            with self.subTest(index=index):
                self.assert_path(grid, (0, 0), (2, 3), astar_search(grid, (0, 0), (2, 3)), expected)

    def test_invalid_grid_costs_coordinates_and_heuristic_rejected(self):
        for grid in ((), ((1,), (1, 2)), ((1, -1),), ((1, True),), ((1, math.nan),)):
            with self.subTest(grid=grid), self.assertRaises((ValueError, TypeError)):
                astar_search(grid, (0, 0), (0, 0))
        for start, goal in (((-1, 0), (0, 0)), ((0, 0), (1, 0))):
            with self.subTest(start=start, goal=goal), self.assertRaises((ValueError, TypeError)):
                astar_search(((1,),), start, goal)
        with self.assertRaises((ValueError, TypeError)):
            astar_search(((1,),), (0, 0), (0, 0), heuristic='unknown')


class MatrixAndStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = sample_problem()
        cls.matrix = build_distance_matrix(cls.problem)
        cls.independent = independent_matrix(cls.problem)

    def test_all_sample_matrix_entries_match_independent_paths(self):
        self.assertEqual(self.matrix.labels, self.problem.labels)
        self.assertEqual(self.matrix.matrix, self.independent)
        self.assertEqual(len(self.matrix.queries), 30)
        for i, p in enumerate(POINTS):
            self.assertEqual(self.matrix.matrix[i][i], 0)
            for j, q in enumerate(POINTS):
                if i != j:
                    result = self.matrix.queries[(i, j)]
                    self.assertEqual(tuple(result.path)[0], p)
                    self.assertEqual(tuple(result.path)[-1], q)
                    self.assertEqual(sum(RAW_GRID[r][c] for r, c in tuple(result.path)[1:]), self.independent[i][j])

    def test_general_matrix_can_be_asymmetric(self):
        problem = small_problem(((1, 5, 2),), ((0, 0), (0, 2), (0, 1)), (4,))
        result = build_distance_matrix(problem).matrix
        self.assertEqual(result, independent_matrix(problem))
        self.assertEqual(result[0][1], 7)
        self.assertEqual(result[1][0], 6)
        self.assertNotEqual(result[0][2], result[2][0])

    def test_zero_and_manhattan_matrices_agree(self):
        zero = build_distance_matrix(self.problem, heuristic='zero')
        self.assertEqual(zero.matrix, self.matrix.matrix)

    def test_initial_state_contains_all_required_components(self):
        state = initial_state(self.problem)
        self.assertEqual(state_tuple(state), (0, 1, 15, 0, 0, 'A'))
        self.assertEqual(tuple(legal_actions(self.problem, state, self.matrix)), (0, 1, 2, 3))

    def test_A_move_only_updates_A_and_keeps_original_state(self):
        state = initial_state(self.problem)
        after = apply_action(self.problem, state, 1, self.matrix)
        self.assertEqual(state_tuple(after), independent_step(self.problem.values, self.independent, state_tuple(state), 1))
        self.assertEqual((after.pos_a, after.pos_b, after.remaining, after.score_a, after.score_b, after.turn), (3, 1, 13, 2, 0, 'B'))
        self.assertEqual(state_tuple(state), (0, 1, 15, 0, 0, 'A'))

    def test_B_move_updates_own_score_and_only_B_position(self):
        state = apply_action(self.problem, initial_state(self.problem), 1, self.matrix)
        after = apply_action(self.problem, state, 2, self.matrix)
        self.assertEqual(state_tuple(after), independent_step(self.problem.values, self.independent, state_tuple(state), 2))
        self.assertEqual(after.pos_a, state.pos_a)
        self.assertEqual(after.pos_b, 4)
        self.assertEqual(after.turn, 'A')

    def test_passing_another_treasure_does_not_collect_it(self):
        problem = small_problem(((1, 1, 1, 1, 1),), ((0, 0), (0, 4), (0, 1), (0, 3)), (4, 8))
        matrix = build_distance_matrix(problem)
        after = apply_action(problem, initial_state(problem), 1, matrix)
        self.assertEqual(after.remaining, 1)
        self.assertEqual(after.score_a, 5)
        self.assertEqual(tuple(matrix.queries[(0, 3)].path), ((0, 0), (0, 1), (0, 2), (0, 3)))

    def test_players_and_collected_tiles_do_not_block_static_paths(self):
        problem = small_problem(((1, 1, 1, 1, 1),), ((0, 0), (0, 2), (0, 1), (0, 4)), (3, 5))
        matrix = build_distance_matrix(problem)
        after_a = apply_action(problem, initial_state(problem), 0, matrix)
        self.assertEqual(after_a.remaining, 2)
        after_b = apply_action(problem, after_a, 1, matrix)
        self.assertEqual(after_b.remaining, 0)
        # A's original route to T2 crosses both T1 and B's starting tile.
        self.assertEqual(matrix.matrix[0][3], 4)
        self.assertEqual(tuple(matrix.queries[(0, 3)].path), tuple((0, c) for c in range(5)))
        self.assertEqual(build_distance_matrix(problem).matrix, matrix.matrix)

    def test_invalid_problem_and_conflicting_json_roles_are_rejected(self):
        bad = [
            ((), ('A', 'B'), ((0, 0), (0, 1)), ()),
            (((1, 1),), ('A', 'A'), ((0, 0), (0, 1)), ()),
            (((1, None, 1),), ('A', 'B', 'T1'), ((0, 0), (0, 2), (0, 1)), (1,)),
            (((1, 1, 1),), ('A', 'B', 'T1'), ((0, 0), (0, 2), (0, 1)), (True,)),
        ]
        for arguments in bad:
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                build_distance_matrix(Problem(*arguments))
        with self.assertRaises((ValueError, TypeError)):
            parse_problem({'grid': [['A', 'B', 'T1']], 'treasures': [{'id': 'T1', 'position': [0, 1], 'value': 1}]})

    def test_invalid_actions_and_out_of_domain_states_are_rejected(self):
        state = apply_action(self.problem, initial_state(self.problem), 0, self.matrix)
        for action in (0, -1, 4, True):
            with self.subTest(action=action), self.assertRaises((ValueError, TypeError)):
                apply_action(self.problem, state, action, self.matrix)
        invalid = (
            State(0, 1, 16, 0, 0, 'A'), State(-1, 1, 15, 0, 0, 'A'),
            State(0, 1, -1, 0, 0, 'A'), State(0, 1, 15, 0, 0, 'C'),
            State(0, 1, 15, math.inf, 0, 'A'),
        )
        for state in invalid:
            with self.subTest(state=state), self.assertRaises((ValueError, TypeError)):
                game_search(self.problem, self.matrix, 2, state=state)


class GameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = sample_problem()
        cls.matrix = build_distance_matrix(cls.problem)
        cls.independent = independent_matrix(cls.problem)

    def assert_solution(self, problem, matrix, depth, state=None):
        state = state or initial_state(problem)
        distances = independent_matrix(problem)
        expected = independent_game(problem.values, distances, state_tuple(state), depth)
        action_values = independent_actions(problem, distances, state_tuple(state), depth)
        optimal = {label for label, value in action_values.items() if value == expected}
        for algorithm in ('minimax', 'alphabeta'):
            for order in ('label', 'reverse', 'gain'):
                result = game_search(problem, matrix, depth, algorithm=algorithm, order=order, state=state)
                self.assertEqual(result.best_value, expected)
                self.assertIn(result.best_target, optimal if optimal else {None})
                if algorithm == 'minimax' and order == 'label':
                    baseline_visits = result.stats.visited_nodes
                if algorithm == 'alphabeta':
                    self.assertLessEqual(result.stats.visited_nodes, baseline_visits)

    def test_sample_depths_and_each_root_action_use_single_action_horizon(self):
        for depth, expected_values in ROOT_VALUES.items():
            with self.subTest(depth=depth):
                expected = dict(zip(self.problem.labels[2:], expected_values))
                independent = independent_actions(self.problem, self.independent, state_tuple(initial_state(self.problem)), depth)
                self.assertEqual(independent, expected)
                self.assertEqual(root_action_values(self.problem, self.matrix, depth), expected)
                self.assert_solution(self.problem, self.matrix, depth)
        self.assertEqual(game_search(self.problem, self.matrix, 2).best_target, 'T2')
        self.assertEqual(game_search(self.problem, self.matrix, 4).best_target, 'T3')

    def test_principal_variation_has_valid_per_player_ledger(self):
        for depth in (2, 4):
            for algorithm in ('minimax', 'alphabeta'):
                for order in ('label', 'reverse', 'gain'):
                    result = game_search(self.problem, self.matrix, depth, algorithm=algorithm, order=order, record_trace=True)
                    state = state_tuple(initial_state(self.problem))
                    self.assertEqual(len(result.principal_variation), depth)
                    self.assertEqual(result.principal_variation[0], result.best_target)
                    for label in result.principal_variation:
                        action = self.problem.labels.index(label) - 2
                        self.assertTrue(state[2] & (1 << action))
                        state = independent_step(self.problem.values, self.independent, state, action)
                    self.assertEqual(state[3]-state[4], result.best_value)
        full = game_search(self.problem, self.matrix, 4)
        self.assertEqual(full.principal_variation, ('T3', 'T1', 'T2', 'T4'))

    def test_zero_horizon_and_nonzero_scores_evaluate_current_U(self):
        state = State(0, 1, 15, 11, -4, 'B')
        for algorithm in ('minimax', 'alphabeta'):
            result = game_search(self.problem, self.matrix, 0, algorithm=algorithm, state=state)
            self.assertEqual(result.best_value, 15)
            self.assertIsNone(result.best_target)
            self.assertEqual(tuple(result.principal_variation), ())
        self.assertEqual(root_action_values(self.problem, self.matrix, 0, state=state), {})

    def test_empty_remaining_and_horizon_above_remaining(self):
        state = State(3, 4, 0, -2, 8, 'A')
        result = game_search(self.problem, self.matrix, 4, state=state)
        self.assertEqual(result.best_value, -10)
        self.assertIsNone(result.best_target)
        self.assertEqual(game_search(self.problem, self.matrix, 20).best_value,
                         game_search(self.problem, self.matrix, 4).best_value)
        empty = small_problem(((1, 1),), ((0, 0), (0, 1)), ())
        self.assert_solution(empty, build_distance_matrix(empty), 5)
        disconnected = small_problem(((1, None, 1, 1),), ((0, 0), (0, 3), (0, 2)), (5,))
        disconnected_matrix = build_distance_matrix(disconnected)
        self.assertEqual(tuple(legal_actions(disconnected, initial_state(disconnected), disconnected_matrix)), ())
        # No reachable choice terminates at current U; it does not silently pass.
        self.assert_solution(disconnected, disconnected_matrix, 2)

    def test_negative_gain_is_still_a_mandatory_action(self):
        problem = small_problem(((1, 10, 1, 1),), ((0, 0), (0, 3), (0, 2)), (1,))
        matrix = build_distance_matrix(problem)
        result = game_search(problem, matrix, 1)
        self.assertEqual(result.best_target, 'T1')
        self.assertEqual(result.best_value, -10)
        self.assertEqual(tuple(result.principal_variation), ('T1',))

    def test_fixed_random_games_match_independent_dynamic_enumeration(self):
        rng = random.Random(40401)
        for case in range(20):
            grid = tuple(tuple(rng.randint(1, 5) for _ in range(4)) for _ in range(3))
            points = rng.sample([(r, c) for r in range(3) for c in range(4)], 6)
            values = tuple(rng.randint(-3, 15) for _ in range(4))
            problem = small_problem(grid, points, values)
            matrix = build_distance_matrix(problem)
            with self.subTest(case=case):
                self.assertEqual(matrix.matrix, independent_matrix(problem))
                for depth in (1, 2, 3, 4):
                    self.assert_solution(problem, matrix, depth)

    def test_custom_substates_with_B_turn_and_previous_scores(self):
        rng = random.Random(40402)
        for case in range(24):
            state = State(rng.randrange(6), rng.randrange(6), rng.randrange(16),
                          rng.randint(-10, 10), rng.randint(-10, 10), rng.choice(('A', 'B')))
            with self.subTest(case=case, state=state):
                self.assert_solution(self.problem, self.matrix, 2, state=state)

    def test_invalid_search_options_and_horizons_are_rejected(self):
        for depth in (-1, True, 1.5):
            with self.subTest(depth=depth), self.assertRaises((ValueError, TypeError)):
                game_search(self.problem, self.matrix, depth)
        with self.assertRaises((ValueError, TypeError)):
            game_search(self.problem, self.matrix, 2, algorithm='greedy')
        with self.assertRaises((ValueError, TypeError)):
            game_search(self.problem, self.matrix, 2, order='oracle')


if __name__ == '__main__':
    unittest.main()
