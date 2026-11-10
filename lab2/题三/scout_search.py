"""Exact Scout threshold tests and an Alpha-Beta control using shared helpers.

Pearl's TEST answers a Boolean comparison, never an exact subtree score.
No epsilon or integer-only null window is used, so finite int/float leaves
keep the original question's semantics.  No depth truncation or table cache.
"""

import math
from dataclasses import asdict, dataclass

from alphabeta_tree import TreeNode, _ordered_indices, validate_tree


@dataclass
class ScoutStats:
    exact_visits: int = 0
    exact_internal_visits: int = 0
    exact_leaf_reads: int = 0
    test_visits: int = 0
    test_internal_visits: int = 0
    test_leaf_reads: int = 0
    probe_calls: int = 0
    research_calls: int = 0
    test_true: int = 0
    test_false: int = 0
    exact_cutoffs: int = 0
    test_short_circuits: int = 0
    total_visits: int = 0
    total_leaf_reads: int = 0


@dataclass
class ScoutResult:
    value: object
    stats: ScoutStats
    trace: list
    bound: str = "exact"

    def to_dict(self):
        return asdict(self)


def scout_search(root, *, mode="scout", order="left", validate=True,
                 is_maximizing_player=True, trace=False):
    """Return the exact full-tree value, with every repeated visit counted.

    mode='alphabeta' disables Scout tests within the same lightweight module;
    it is a control for the cost of changing implementation and instrumentation.
    The public original solver is retained as a separate practical baseline.
    order accepts the original left/right/permutation callback contract.
    Validation visits are outside search counters, as in the original solver.
    """
    if mode not in ("scout", "alphabeta"):
        raise ValueError("mode must be scout or alphabeta")
    if not isinstance(is_maximizing_player, bool):
        raise ValueError("is_maximizing_player must be bool")
    if not isinstance(trace, bool):
        raise ValueError("trace must be bool")
    if not isinstance(order, str) and not callable(order):
        raise ValueError("order must be left/right or an ordering callback")
    if isinstance(order, str) and order not in ("left", "right"):
        raise ValueError("order must be left/right or an ordering callback")
    if validate:
        validate_tree(root)
    stats, events = ScoutStats(), []

    def test(node, maximizing, reference, greater):
        """Whether true minimax(node) > reference, or < reference.

        For >: MAX is any(children), MIN is all(children).
        For <: MAX is all(children), MIN is any(children).
        The comparison is strict, so ties never trigger exact re-search.
        """
        stats.test_visits += 1
        if not node.children:
            stats.test_leaf_reads += 1
            return node.value > reference if greater else node.value < reference
        stats.test_internal_visits += 1
        any_mode = maximizing == greater
        indices = _ordered_indices(node, maximizing, order)
        count_children = len(node.children)
        for position, index in enumerate(indices):
            result = test(node.children[index], not maximizing, reference, greater)
            if result == any_mode:
                if position + 1 < count_children:
                    stats.test_short_circuits += 1
                return any_mode
        return not any_mode

    def exact(node, maximizing, alpha, beta):
        stats.exact_visits += 1
        if not node.children:
            stats.exact_leaf_reads += 1
            return node.value
        stats.exact_internal_visits += 1
        indices = _ordered_indices(node, maximizing, order)
        best = -math.inf if maximizing else math.inf
        if mode == "alphabeta":
            for position, index in enumerate(indices):
                child_value = exact(node.children[index], not maximizing, alpha, beta)
                best = max(best, child_value) if maximizing else min(best, child_value)
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    if position+1 < len(node.children):
                        stats.exact_cutoffs += 1
                    break
            return best

        for position, index in enumerate(indices):
            child = node.children[index]
            if position == 0:
                # Scout EVAL always asks for an exact value, with no inherited
                # narrow-window bound that could masquerade as that value.
                best = exact(child, not maximizing, -math.inf, math.inf)
                continue
            stats.probe_calls += 1
            reference = best
            improves = test(child, not maximizing, reference, maximizing)
            if improves:
                stats.test_true += 1
                stats.research_calls += 1
                if trace:
                    events.append(dict(event="research", reference=reference,
                        maximizing=maximizing, child_index=index))
                best = exact(child, not maximizing, -math.inf, math.inf)
            else:
                stats.test_false += 1
            if trace:
                events.append(dict(event="probe", reference=reference, improves=improves,
                    maximizing=maximizing, child_index=index))
        return best

    value = exact(root, is_maximizing_player, -math.inf, math.inf)
    stats.total_visits = stats.exact_visits + stats.test_visits
    stats.total_leaf_reads = stats.exact_leaf_reads + stats.test_leaf_reads
    return ScoutResult(value, stats, events)
