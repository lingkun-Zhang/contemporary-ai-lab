"""宝藏博弈的收益增量缓存；正式题四入口不变（Python 3.9+，标准库）。

F(state,h)=U(state)+V(posA,posB,remaining,turn,h)。缓存V而非最终F，
允许不同历史收益下的相同后续局面复用。仅支持整数费用/价值/已有分数，
避免浮点加减的重排误差。每次顶层调用新建缓存，不跨地图、深度任务复用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from treasure_game import (GameResult, GameStats, _distance, _matrix_data,
                           _transition, initial_state, validate_problem,
                           validate_state)


@dataclass
class CacheStats(GameStats):
    cache_probes: int = 0
    cache_hits: int = 0
    cache_exact_hits: int = 0
    cache_lower_hits: int = 0
    cache_upper_hits: int = 0
    cache_unusable_bounds: int = 0
    cache_stores: int = 0
    cache_peak_entries: int = 0


@dataclass(frozen=True)
class Entry:
    value: int
    kind: str
    line: tuple


def cache_search(problem, distances, depth, algorithm='alphabeta', order='label',
                 cache_mode='incremental', state=None, record_trace=False):
    """同内核对照：none、fullscore、incremental；均返回逐动作PV。

    fullscore键包含双方各自累计分数；incremental键去掉两分数，
    存储F-U。两者都含双方位置、剩余集合、玩家和精确剩余深度。
    只缓存非终止状态，h=0或宝藏耗尽的U直接返回，不把廉价终局命中
    充当减少非终止子问题的证据。遇到不可达且无合法动作也直接终止。

    界条目只有能直接截断时才返回，否则按原输入窗口继续搜索，不用
    缓存收紧窗口。relative界在比较/返回前加当前U，保存时减当前U。
    """
    validate_problem(problem)
    if type(depth) is not int or depth < 0:
        raise ValueError('depth must be a nonnegative integer')
    if algorithm not in ('minimax', 'alphabeta'):
        raise ValueError('invalid algorithm')
    if order not in ('label', 'reverse', 'gain'):
        raise ValueError('invalid order')
    if cache_mode not in ('none', 'fullscore', 'incremental'):
        raise ValueError('invalid cache_mode')
    matrix = _matrix_data(problem, distances)
    state = initial_state(problem) if state is None else state
    validate_state(problem, state)
    if any(type(v) is not int for v in (*problem.values, state.score_a, state.score_b)):
        raise ValueError('incremental experiment requires integer values and scores')
    for row in matrix:
        for value in row:
            if value != math.inf and (type(value) is not int or value < 0):
                raise ValueError('integer nonnegative distances or +inf required')
    stats, trace, table = CacheStats(), [], {}

    def emit(event, current, left, **details):
        if record_trace:
            trace.append(dict(event=event, pos_a=current.pos_a, pos_b=current.pos_b,
                remaining=current.remaining, turn=current.turn, depth_left=left,
                utility=current.utility, **details))

    def visit(current, left, alpha, beta):
        stats.visited_nodes += 1
        alpha_in, beta_in = alpha, beta
        emit('enter', current, left, alpha=alpha, beta=beta)
        if not left or not current.remaining:
            stats.leaf_evaluations += 1
            return current.utility, ()

        key = None
        offset = current.utility if cache_mode == 'incremental' else 0
        if cache_mode != 'none':
            key = (current.pos_a, current.pos_b, current.remaining, current.turn, left)
            if cache_mode == 'fullscore':
                key += (current.score_a, current.score_b)
            stats.cache_probes += 1
            entry = table.get(key)
            if entry is not None:
                absolute = entry.value + offset
                reusable = (entry.kind == 'exact' or
                    (entry.kind == 'lower' and absolute >= beta) or
                    (entry.kind == 'upper' and absolute <= alpha))
                if reusable:
                    stats.cache_hits += 1
                    field = 'cache_' + entry.kind + '_hits'
                    setattr(stats, field, getattr(stats, field)+1)
                    emit('cache_return', current, left, kind=entry.kind,
                         stored_value=entry.value, offset=offset, absolute_value=absolute,
                         alpha=alpha, beta=beta)
                    return absolute, entry.line
                stats.cache_unusable_bounds += 1

        position = current.pos_a if current.turn == 'A' else current.pos_b
        candidates = []
        for index in range(len(problem.values)):
            if current.remaining & (1 << index):
                stats.distance_lookups += 1
                cost = _distance(matrix, position, index+2)
                if math.isfinite(cost):
                    candidates.append((index, problem.values[index]-cost))
        stats.generated_actions += len(candidates)
        if not candidates:
            stats.leaf_evaluations += 1
            return current.utility, ()
        stats.internal_nodes += 1
        if order == 'reverse':
            candidates.reverse()
        elif order == 'gain':
            stats.sorting_calls += 1
            stats.sorted_actions += len(candidates)
            candidates.sort(key=lambda item: (-item[1], item[0]))

        maximizing = current.turn == 'A'
        best = -math.inf if maximizing else math.inf
        best_line = ()
        for rank, (index, gain) in enumerate(candidates):
            following = _transition(current, index, gain)
            candidate, line = visit(following, left-1, alpha, beta)
            if (maximizing and candidate > best) or (not maximizing and candidate < best):
                best = candidate
                best_line = (problem.labels[index+2], *line)
            if algorithm == 'alphabeta':
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    stats.bound_crossings += 1
                    skipped = len(candidates)-rank-1
                    if skipped:
                        stats.cutoff_events += 1
                        stats.skipped_child_roots += skipped
                    break
        kind = 'exact'
        if algorithm == 'alphabeta':
            if best <= alpha_in:
                kind = 'upper'
            elif best >= beta_in:
                kind = 'lower'
        if key is not None:
            table[key] = Entry(best-offset, kind, best_line)
            stats.cache_stores += 1
            stats.cache_peak_entries = max(stats.cache_peak_entries, len(table))
            emit('cache_store', current, left, kind=kind, stored_value=best-offset,
                 offset=offset, absolute_value=best, alpha=alpha_in, beta=beta_in)
        return best, best_line

    value, line = visit(state, depth, -math.inf, math.inf)
    return GameResult(line[0] if line else None, value, stats, line, trace)
