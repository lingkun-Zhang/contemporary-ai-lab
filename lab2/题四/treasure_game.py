"""Lab2 Q4: enter-cost A* distances and alternating treasure allocation.

The official sample uses MAX A, MIN B and one action per depth unit.  Markers
A/B/T cost 1; reaching a treasure does not make its cell free or collect any
other treasure passed along the path.  This module uses only the standard library.
"""
from __future__ import annotations

import argparse
import dataclasses
import heapq
import itertools
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and (
        not isinstance(value, float) or math.isfinite(value))


def _coordinate(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(
        isinstance(x, int) and not isinstance(x, bool) for x in value)


def _numeric_grid(grid):
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError('grid must be a nonempty rectangular sequence')
    rows = []
    width = None
    for row in grid:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError('grid rows must be nonempty sequences')
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError('grid must be rectangular')
        if any(value is not None and (not _number(value) or value < 0) for value in row):
            raise ValueError('grid costs must be nonnegative finite numbers, or None walls')
        rows.append(tuple(row))
    return tuple(rows)


def _check_point(grid, point):
    if not _coordinate(point):
        raise ValueError('a point must contain two integer coordinates')
    row, col = point
    if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] is None:
        raise ValueError('point is outside the grid or on a wall')


@dataclass(frozen=True)
class Problem:
    grid: tuple
    labels: tuple
    points: tuple
    values: tuple

    def __post_init__(self):
        object.__setattr__(self, 'grid', _numeric_grid(self.grid))
        object.__setattr__(self, 'labels', tuple(self.labels))
        if not all(_coordinate(p) for p in self.points):
            raise ValueError('points must contain integer coordinate pairs')
        object.__setattr__(self, 'points', tuple(tuple(p) for p in self.points))
        object.__setattr__(self, 'values', tuple(self.values))
        validate_problem(self)


def validate_problem(problem):
    if not isinstance(problem, Problem):
        raise ValueError('expected Problem')
    if len(problem.labels) < 2 or problem.labels[:2] != ('A', 'B'):
        raise ValueError('labels must start with A and B')
    if any(not isinstance(label, str) or not label for label in problem.labels):
        raise ValueError('labels must be nonempty strings')
    if len(set(problem.labels)) != len(problem.labels):
        raise ValueError('point labels must be unique')
    if len(problem.points) != len(problem.labels) or len(problem.values) != len(problem.labels)-2:
        raise ValueError('labels, points and treasure values have incompatible sizes')
    if len(set(problem.points)) != len(problem.points):
        raise ValueError('named points must occupy different cells')
    for point in problem.points:
        _check_point(problem.grid, point)
    if any(not _number(value) for value in problem.values):
        raise ValueError('treasure values must be finite numbers, not bool')


def parse_problem(payload):
    """Parse grid plus treasures; optional starts supports unmarked numeric grids.

    Treasures are [{"id":"T1", "position":[r,c], "value":9}, ...].
    Marker cells are assigned fee 1, while explicitly numeric cells retain their
    input costs.  Treasure identifiers follow input order for default ties.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get('grid'), list):
        raise ValueError('input must provide a grid list')
    treasures = payload.get('treasures')
    if not isinstance(treasures, list):
        raise ValueError('treasures must be a list')
    ids, positions, values = [], [], []
    for item in treasures:
        if not isinstance(item, dict):
            raise ValueError('each treasure must be an object')
        identity = item.get('id')
        position = item.get('position')
        if not isinstance(identity, str) or not identity or identity in ('A', 'B'):
            raise ValueError('treasure id must be a nonempty label distinct from A/B')
        if identity in ids or not _coordinate(position):
            raise ValueError('treasure ids must be unique and positions must be coordinates')
        ids.append(identity); positions.append(tuple(position)); values.append(item.get('value'))
    marker_names = {'A', 'B', *ids}
    found, grid = {}, []
    for row_index, raw_row in enumerate(payload['grid']):
        if isinstance(raw_row, str):
            raw_row = raw_row.split()
        if not isinstance(raw_row, (list, tuple)):
            raise ValueError('each grid row must be a sequence or whitespace-separated string')
        row = []
        for col_index, token in enumerate(raw_row):
            if isinstance(token, str) and token in marker_names:
                if token in found:
                    raise ValueError(f'duplicate marker {token}')
                found[token] = (row_index, col_index); row.append(1)
            elif token is None or token == 'X':
                row.append(None)
            elif isinstance(token, str):
                try:
                    number = int(token)
                except ValueError as error:
                    raise ValueError(f'unknown grid token {token}') from error
                row.append(number)
            else:
                row.append(token)
        grid.append(row)
    starts = payload.get('starts', {})
    if not isinstance(starts, dict):
        raise ValueError('starts must be an object')
    initial = []
    for label in ('A', 'B'):
        explicit = starts.get(label)
        if explicit is not None and not _coordinate(explicit):
            raise ValueError('start positions must be coordinates')
        position = tuple(explicit) if explicit is not None else found.get(label)
        if position is None:
            raise ValueError(f'missing start {label}')
        if label in found and found[label] != position:
            raise ValueError(f'start {label} disagrees with marker')
        initial.append(position)
    for label, position in zip(ids, positions):
        if label in found and found[label] != position:
            raise ValueError(f'{label} position disagrees with marker')
    return Problem(grid, ('A', 'B', *ids), (*initial, *positions), values)


@dataclass
class PathStats:
    expanded_nodes: int = 0
    generated_nodes: int = 0
    popped_nodes: int = 0
    stale_pops: int = 0
    reopened_nodes: int = 0
    unique_reached: int = 0
    max_frontier: int = 0


@dataclass
class PathResult:
    cost: object
    path: tuple
    stats: PathStats
    trace: list = field(default_factory=list)
    ledger: tuple = ()

    def to_dict(self):
        return _json_safe(dataclasses.asdict(self))


def astar_search(grid, start, goal, heuristic='manhattan', record_trace=False):
    """Find minimum entering fee; start is free, every entered cell is charged.

    The global minimum passable fee times Manhattan is a consistent heuristic
    for this four-neighbor graph.  If a zero fee exists it safely degenerates.
    Valid target pop, not first generation, ends search.  FIFO breaks f ties.
    """
    grid = _numeric_grid(grid)
    _check_point(grid, start); _check_point(grid, goal)
    start, goal = tuple(start), tuple(goal)
    if heuristic not in ('manhattan', 'zero'):
        raise ValueError('heuristic must be manhattan or zero')
    minimum = min(value for row in grid for value in row if value is not None)
    def estimate(point):
        return minimum*(abs(point[0]-goal[0])+abs(point[1]-goal[1])) if heuristic == 'manhattan' else 0
    stats, trace = PathStats(), []
    distances, parents, expanded = {start: 0}, {}, set()
    serial = itertools.count()
    frontier = [(estimate(start), next(serial), 0, start)]
    stats.generated_nodes = stats.max_frontier = stats.unique_reached = 1
    while frontier:
        priority, _, paid, point = heapq.heappop(frontier)
        stats.popped_nodes += 1
        if paid != distances[point]:
            stats.stale_pops += 1
            continue
        if record_trace:
            trace.append(dict(node=point, g=paid, h=estimate(point), f=priority))
        if point == goal:
            path = [goal]
            while path[-1] != start:
                path.append(parents[path[-1]])
            path = tuple(reversed(path))
            cumulative, ledger = 0, [dict(step=0, position=start, enter_cost=0, cumulative_cost=0)]
            for step, (row, col) in enumerate(path[1:], 1):
                cumulative += grid[row][col]
                ledger.append(dict(step=step, position=(row, col), enter_cost=grid[row][col], cumulative_cost=cumulative))
            return PathResult(paid, path, stats, trace, tuple(ledger))
        stats.expanded_nodes += 1
        expanded.add(point)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = point[0]+dr, point[1]+dc
            row, col = neighbor
            if not (0 <= row < len(grid) and 0 <= col < len(grid[0])) or grid[row][col] is None:
                continue
            candidate = paid+grid[row][col]
            if candidate < distances.get(neighbor, math.inf):
                if neighbor in expanded:
                    stats.reopened_nodes += 1
                distances[neighbor] = candidate; parents[neighbor] = point
                stats.unique_reached = len(distances)
                heapq.heappush(frontier, (candidate+estimate(neighbor), next(serial), candidate, neighbor))
                stats.generated_nodes += 1
                stats.max_frontier = max(stats.max_frontier, len(frontier))
    return PathResult(math.inf, (), stats, trace)


@dataclass(frozen=True)
class DistanceMatrix:
    labels: tuple
    matrix: tuple
    queries: dict


def build_distance_matrix(problem, heuristic='manhattan', record_trace=False):
    """Run A* for every ordered distinct endpoint pair, never assuming symmetry."""
    validate_problem(problem)
    size = len(problem.points)
    matrix, queries = [], {}
    for source in range(size):
        row = []
        for target in range(size):
            if source == target:
                row.append(0)
            else:
                result = astar_search(problem.grid, problem.points[source], problem.points[target], heuristic, record_trace)
                queries[(source, target)] = result
                row.append(result.cost)
        matrix.append(tuple(row))
    return DistanceMatrix(problem.labels, tuple(matrix), queries)


@dataclass(frozen=True)
class State:
    pos_a: int
    pos_b: int
    remaining: int
    score_a: object = 0
    score_b: object = 0
    turn: str = 'A'

    @property
    def utility(self):
        return self.score_a-self.score_b


def initial_state(problem):
    validate_problem(problem)
    return State(0, 1, (1 << len(problem.values))-1)


def validate_state(problem, state):
    if not isinstance(state, State):
        raise ValueError('expected State')
    size = len(problem.points)
    for value in (state.pos_a, state.pos_b):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
            raise ValueError('player position index is invalid')
    if isinstance(state.remaining, bool) or not isinstance(state.remaining, int) or not 0 <= state.remaining < (1 << len(problem.values)):
        raise ValueError('remaining mask is invalid')
    if not _number(state.score_a) or not _number(state.score_b) or state.turn not in ('A', 'B'):
        raise ValueError('state scores/turn are invalid')


def _matrix_data(problem, distances):
    if hasattr(distances, 'labels') and tuple(distances.labels) != problem.labels:
        raise ValueError('distance labels disagree with problem')
    matrix = distances.matrix if hasattr(distances, 'matrix') else distances
    if len(matrix) != len(problem.points) or any(len(matrix[i]) != len(problem.points) for i in range(len(problem.points))):
        raise ValueError('distance matrix has wrong shape')
    return matrix


def _distance(matrix, source, target):
    value = matrix[source][target]
    if value == math.inf:
        return value
    if not _number(value) or value < 0:
        raise ValueError('distance must be a nonnegative finite number or +inf')
    return value


def legal_actions(problem, state, distances):
    validate_state(problem, state)
    matrix = _matrix_data(problem, distances)
    position = state.pos_a if state.turn == 'A' else state.pos_b
    return tuple(index for index in range(len(problem.values))
                 if state.remaining & (1 << index) and math.isfinite(_distance(matrix, position, index+2)))


def _transition(state, index, gain):
    mask = state.remaining & ~(1 << index)
    if state.turn == 'A':
        return State(index+2, state.pos_b, mask, state.score_a+gain, state.score_b, 'B')
    return State(state.pos_a, index+2, mask, state.score_a, state.score_b+gain, 'A')


def apply_action(problem, state, action, distances):
    validate_state(problem, state)
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(problem.values) or not state.remaining & (1 << action):
        raise ValueError('action must select an available treasure index')
    matrix = _matrix_data(problem, distances)
    position = state.pos_a if state.turn == 'A' else state.pos_b
    cost = _distance(matrix, position, action+2)
    if not math.isfinite(cost):
        raise ValueError('selected treasure is unreachable')
    return _transition(state, action, problem.values[action]-cost)


@dataclass
class GameStats:
    visited_nodes: int = 0
    internal_nodes: int = 0
    leaf_evaluations: int = 0
    cutoff_events: int = 0
    skipped_child_roots: int = 0
    bound_crossings: int = 0
    generated_actions: int = 0
    distance_lookups: int = 0
    sorting_calls: int = 0
    sorted_actions: int = 0


@dataclass
class GameResult:
    best_target: object
    best_value: object
    stats: GameStats
    principal_variation: tuple
    trace: list = field(default_factory=list)

    def to_dict(self):
        return _json_safe(dataclasses.asdict(self))


def game_search(problem, distances, depth, algorithm='alphabeta', order='label', state=None, record_trace=False):
    """Minimax/Alpha-Beta with one selected treasure per depth unit.

    Every gain is value minus actual distance, even when negative.  Leaves use
    current A score minus B score.  No legal reachable action terminates the
    current branch (the official map is connected, so this is an extension).
    Default ties select the first input treasure label.  Other visit orders
    may select a different equally optimal first target; strict improvement
    prevents a cutoff bound tying an earlier exact root value from replacing it.
    Trace child returns may be bounds; use root_action_values for exact values
    of ALL root actions.  No cache or oracle child values are used here.
    """
    validate_problem(problem)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    if algorithm not in ('minimax', 'alphabeta') or order not in ('label', 'reverse', 'gain'):
        raise ValueError('invalid algorithm or action order')
    matrix = _matrix_data(problem, distances)
    state = initial_state(problem) if state is None else state
    validate_state(problem, state)
    stats, trace = GameStats(), []

    def emit(event, path, current, **details):
        if record_trace:
            trace.append(dict(event=event, path=path, state=dataclasses.asdict(current), **details))

    def visit(current, remaining_depth, alpha, beta, path):
        stats.visited_nodes += 1
        alpha_in, beta_in = alpha, beta
        emit('enter', path, current, depth_left=remaining_depth, alpha=alpha, beta=beta)
        candidates = []
        if remaining_depth and current.remaining:
            position = current.pos_a if current.turn == 'A' else current.pos_b
            for index in range(len(problem.values)):
                if current.remaining & (1 << index):
                    stats.distance_lookups += 1
                    cost = _distance(matrix, position, index+2)
                    if math.isfinite(cost):
                        candidates.append((index, problem.values[index]-cost, cost))
            stats.generated_actions += len(candidates)
        if not candidates:
            stats.leaf_evaluations += 1
            emit('return', path, current, value=current.utility, returned_bound='exact', terminal=True)
            return current.utility, ()
        stats.internal_nodes += 1
        if order == 'reverse':
            candidates.reverse()
        elif order == 'gain':
            stats.sorting_calls += 1; stats.sorted_actions += len(candidates)
            candidates.sort(key=lambda item: (-item[1], item[0]))
        maximizing = current.turn == 'A'
        best, best_line = (-math.inf if maximizing else math.inf), ()
        for position, (index, gain, cost) in enumerate(candidates):
            label = problem.labels[index+2]
            following = _transition(current, index, gain)
            candidate, line = visit(following, remaining_depth-1, alpha, beta, f'{path}/{label}')
            if (maximizing and candidate > best) or (not maximizing and candidate < best):
                best, best_line = candidate, (label, *line)
            if algorithm == 'alphabeta':
                if maximizing: alpha = max(alpha, best)
                else: beta = min(beta, best)
            emit('update', path, current, target=label, travel_cost=cost, gain=gain,
                 child_value=candidate, value=best, alpha=alpha, beta=beta)
            if algorithm == 'alphabeta' and alpha >= beta:
                stats.bound_crossings += 1
                remaining = candidates[position+1:]
                if remaining:
                    stats.cutoff_events += 1; stats.skipped_child_roots += len(remaining)
                    emit('cutoff', path, current, alpha=alpha, beta=beta,
                         skipped_targets=[problem.labels[item[0]+2] for item in remaining])
                break
        kind = 'exact'
        if algorithm == 'alphabeta':
            if best <= alpha_in: kind = 'upper'
            elif best >= beta_in: kind = 'lower'
        emit('return', path, current, value=best, returned_bound=kind, terminal=False)
        return best, best_line

    value, line = visit(state, depth, -math.inf, math.inf, 'root')
    return GameResult(line[0] if line else None, value, stats, line, trace)


def root_action_values(problem, distances, depth, state=None):
    """Exact value of each root move via independent full-window continuation."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    current = initial_state(problem) if state is None else state
    validate_state(problem, current)
    if depth == 0:
        return {}
    result = {}
    for action in legal_actions(problem, current, distances):
        child = apply_action(problem, current, action, distances)
        result[problem.labels[action+2]] = game_search(problem, distances, depth-1, algorithm='minimax', state=child).best_value
    return result


def sample_problem():
    return parse_problem(_SAMPLE)


_SAMPLE = {
    'grid': [
        ['A',1,1,'X',2,1,'T1'],
        [1,'X',6,'X',1,'X',1],
        [1,1,'T2',1,1,1,1],
        ['X',1,'X',8,'X',1,'X'],
        [1,1,1,1,'T3',1,1],
        [1,'X',9,'X',1,'X',1],
        ['T4',1,1,'X',1,1,'B'],
    ],
    'treasures': [
        {'id':'T1','position':[0,6],'value':9},
        {'id':'T2','position':[2,2],'value':6},
        {'id':'T3','position':[4,4],'value':7},
        {'id':'T4','position':[6,0],'value':7},
    ],
}


def _json_safe(value):
    if isinstance(value, float) and math.isinf(value):
        return 'inf' if value > 0 else '-inf'
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--depth', type=int)
    parser.add_argument('--algorithm', choices=('minimax','alphabeta'))
    parser.add_argument('--order', choices=('label','reverse','gain'), default='label')
    parser.add_argument('--heuristic', choices=('manhattan','zero'), default='manhattan')
    parser.add_argument('--details', action='store_true')
    args = parser.parse_args(argv)
    try:
        problem = parse_problem(json.loads(args.input.read_text(encoding='utf-8-sig'))) if args.input else sample_problem()
        distances = build_distance_matrix(problem, args.heuristic, args.details)
        data = dict(labels=problem.labels, distance_matrix=distances.matrix, results=[])
        for depth in (args.depth,) if args.depth is not None else (2,4):
            for algorithm in (args.algorithm,) if args.algorithm else ('minimax','alphabeta'):
                result = game_search(problem, distances, depth, algorithm, args.order, record_trace=args.details)
                summary = result.to_dict() if args.details else dict(best_target=result.best_target,best_value=result.best_value)
                data['results'].append(dict(depth=depth,algorithm=algorithm,order=args.order,**summary))
        if args.details:
            data['path_queries'] = [dict(source=problem.labels[s],target=problem.labels[t],**r.to_dict()) for (s,t),r in distances.queries.items()]
        print(json.dumps(_json_safe(data), ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, TypeError, RecursionError) as error:
        parser.exit(2, f'输入错误：{error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
