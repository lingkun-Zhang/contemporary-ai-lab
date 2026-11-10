"""Optional Q4 successor generation without querying pruned target distances.

The official implementation remains in treasure_game.py.  This module changes
when a candidate is prepared, never the game rules, horizon, or visit order.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from treasure_game import (
    GameResult, GameStats, _distance, _matrix_data, _transition,
    initial_state, validate_problem, validate_state,
)


@dataclass
class LazyGameStats(GameStats):
    """Generation work is distinct from recursive visits and legal children.

    scheduled_candidates counts raw remaining-target slots listed at visited
    nonterminal states. considered_candidates equals actual distance_lookups;
    unreachable_candidates is the subset whose cost is +inf.
    materialized_actions counts transitions actually followed, hence visits-1.
    unqueried_candidates counts slots whose costs were never requested.

    generated_actions retains the prepared-action meaning: eager/gain counts
    every prepared legal action; lazy label/reverse counts only actual legal
    actions visited.  Thus its old eager conservation formula must not be used
    for the latter. skipped_candidates and skipped_child_roots count remaining
    slots at a cutoff; in lazy mode their reachability is intentionally unknown.
    They are exact skipped legal-child counts only when every target is reachable.
    """

    scheduled_candidates: int = 0
    considered_candidates: int = 0
    unreachable_candidates: int = 0
    materialized_actions: int = 0
    unqueried_candidates: int = 0
    skipped_candidates: int = 0


def lazy_game_search(problem, distances, depth, algorithm='alphabeta',
                     order='label', state=None, record_trace=False,
                     generation='lazy'):
    """Return the same game value/PV while delaying distance queries.

    label/reverse with generation='lazy' look up each target only immediately
    before visiting it. order='gain' necessarily prepares all available gains;
    it is an eager boundary control. generation='eager' supplies a matched
    baseline in this same instrumented kernel for every order.

    No cache, oracle ordering, pass action, depth convention, or leaf evaluation
    is introduced. A no-reachable-action state ends at its current utility, as
    in the original extension. Default infinite root windows produce exact
    values; trace return events inside cutoffs can remain only bounds.

    Original and lazy searches have equal best value, first action, PV, visits,
    and leaf evaluations for a fixed order. On connected inputs their cutoff
    counts also agree. On disconnected inputs skipped target slots in lazy
    mode are not claimed to be confirmed legal children.
    """
    validate_problem(problem)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    if algorithm not in ('minimax', 'alphabeta') or order not in ('label', 'reverse', 'gain'):
        raise ValueError('invalid algorithm or action order')
    if generation not in ('lazy', 'eager'):
        raise ValueError('generation must be lazy or eager')
    matrix = _matrix_data(problem, distances)
    state = initial_state(problem) if state is None else state
    validate_state(problem, state)
    stats, trace = LazyGameStats(), []
    eager = generation == 'eager' or order == 'gain'

    def emit(event, path, current, **details):
        if record_trace:
            trace.append(dict(event=event, path=path,
                              state=dataclasses.asdict(current), **details))

    def obtain(current, index, path):
        stats.distance_lookups += 1
        stats.considered_candidates += 1
        position = current.pos_a if current.turn == 'A' else current.pos_b
        cost = _distance(matrix, position, index+2)
        if not math.isfinite(cost):
            stats.unreachable_candidates += 1
            if record_trace:
                emit('candidate', path, current, target=problem.labels[index+2],
                     travel_cost=cost, reachable=False)
            return None
        gain = problem.values[index]-cost
        if record_trace:
            emit('candidate', path, current, target=problem.labels[index+2],
                 travel_cost=cost, gain=gain, reachable=True)
        return index, gain, cost

    def leaf(current, path):
        stats.leaf_evaluations += 1
        emit('return', path, current, value=current.utility,
             returned_bound='exact', terminal=True)
        return current.utility, ()

    def visit(current, remaining_depth, alpha, beta, path):
        stats.visited_nodes += 1
        alpha_in, beta_in = alpha, beta
        emit('enter', path, current, depth_left=remaining_depth,
             alpha=alpha, beta=beta, generation='eager' if eager else 'lazy')
        if not remaining_depth or not current.remaining:
            return leaf(current, path)

        indices = [index for index in range(len(problem.values))
                   if current.remaining & (1 << index)]
        stats.scheduled_candidates += len(indices)
        if order == 'reverse' and not eager:
            indices.reverse()
        if eager:
            candidates = []
            for index in indices:
                prepared = obtain(current, index, path)
                if prepared is not None:
                    candidates.append(prepared)
            stats.generated_actions += len(candidates)
            if order == 'reverse':
                candidates.reverse()
            elif order == 'gain' and candidates:
                stats.sorting_calls += 1
                stats.sorted_actions += len(candidates)
                candidates.sort(key=lambda item: (-item[1], item[0]))
            items = candidates
        else:
            items = indices

        maximizing = current.turn == 'A'
        best, best_line = (-math.inf if maximizing else math.inf), ()
        has_legal_child = False
        for position, item in enumerate(items):
            prepared = item if eager else obtain(current, item, path)
            if prepared is None:
                continue
            index, gain, cost = prepared
            if not has_legal_child:
                stats.internal_nodes += 1
                has_legal_child = True
            if not eager:
                stats.generated_actions += 1
            stats.materialized_actions += 1
            label = problem.labels[index+2]
            following = _transition(current, index, gain)
            candidate, line = visit(following, remaining_depth-1,
                                    alpha, beta, f'{path}/{label}')
            if (maximizing and candidate > best) or (not maximizing and candidate < best):
                best, best_line = candidate, (label, *line)
            if algorithm == 'alphabeta':
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
            emit('update', path, current, target=label, travel_cost=cost,
                 gain=gain, child_value=candidate, value=best,
                 alpha=alpha, beta=beta)
            if algorithm == 'alphabeta' and alpha >= beta:
                stats.bound_crossings += 1
                remaining = items[position+1:]
                if remaining:
                    count = len(remaining)
                    stats.cutoff_events += 1
                    stats.skipped_child_roots += count
                    stats.skipped_candidates += count
                    if not eager:
                        stats.unqueried_candidates += count
                    skipped_indices = [value[0] for value in remaining] if eager else remaining
                    emit('cutoff', path, current, alpha=alpha, beta=beta,
                         skipped_targets=[problem.labels[index+2] for index in skipped_indices],
                         skipped_reachability='known_legal' if eager else 'not_queried',
                         skipped_count_kind='legal_children' if eager else 'target_slots')
                break
        if not has_legal_child:
            return leaf(current, path)
        kind = 'exact'
        if algorithm == 'alphabeta':
            if best <= alpha_in:
                kind = 'upper'
            elif best >= beta_in:
                kind = 'lower'
        emit('return', path, current, value=best,
             returned_bound=kind, terminal=False)
        return best, best_line

    value, line = visit(state, depth, -math.inf, math.inf, 'root')
    return GameResult(line[0] if line else None, value, stats, line, trace)
