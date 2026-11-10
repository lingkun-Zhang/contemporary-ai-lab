"""Left-to-right Alpha-Beta with a fresh, bound-aware transposition table.

This optional experiment does not change the official Q3 implementation.  Nodes
are fully searched to terminal leaves: ``max_cache_depth`` only gates table
access by absolute depth from the root, and never truncates the game search.
The tree must remain unchanged during a call.  Tables never survive a call.
"""

import math
from dataclasses import asdict, dataclass

from alphabeta_tree import TreeNode, validate_tree


@dataclass
class CacheStats:
    """Visits count calls; a cache return does not count as an actual leaf read."""

    visited_nodes: int = 0
    internal_nodes: int = 0
    leaf_evaluations: int = 0
    tt_returns: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    exact_hits: int = 0
    bound_cutoffs: int = 0
    noncutting_hits: int = 0
    stores: int = 0
    table_peak_entries: int = 0
    table_final_entries: int = 0
    cutoff_events: int = 0
    skipped_child_roots: int = 0
    bound_crossings: int = 0


@dataclass
class CacheResult:
    value: object
    stats: CacheStats
    trace: list
    bound: str = "exact"

    def to_dict(self):
        """Make infinite windows standard-JSON-safe, without losing score ints."""
        def safe(item):
            if isinstance(item, float) and math.isinf(item):
                return "inf" if item > 0 else "-inf"
            if isinstance(item, dict):
                return {key: safe(value) for key, value in item.items()}
            if isinstance(item, (tuple, list)):
                return [safe(value) for value in item]
            return item
        return safe(asdict(self))


@dataclass(frozen=True)
class _Entry:
    value: object
    bound: str


def _bound_type(value, alpha, beta):
    if value <= alpha:
        return "upper"
    if value >= beta:
        return "lower"
    return "exact"


def cached_search(root: TreeNode, *, mode="bounds", max_cache_depth=None,
                  validate=True, trace=False, alpha=-math.inf, beta=math.inf,
                  is_maximizing_player=True):
    """Return the full terminal-search value, counters, and optional events.

    ``mode`` is ``off`` (matched kernel baseline), ``exact`` (only exact
    entries), or ``bounds`` (exact/lower/upper entries).  At a hit, return only
    an exact value or a bound that closes the current window: lower >= beta or
    upper <= alpha.  A nonclosing bound is counted but never tightens the
    window.  Entries are classified using the original entry window.

    Keys include node identity and current player.  Since every call searches
    to terminal leaves, there is no remaining search horizon in the key.  A
    future depth-limited implementation must revisit that assumption.  The
    nonnegative integer ``max_cache_depth`` is only an admission/probe policy;
    None permits every depth and 0 only the root.  No preprocessing computes
    subtree values or heights.  Public root windows require alpha < beta.
    The default infinite root window returns an exact value; a supplied narrow
    root window may return only the bound identified by ``result.bound``.

    visited_nodes = internal_nodes + leaf_evaluations + tt_returns.  Validation
    is outside those counters; timed callers must state whether they include
    it.  No existing child list or leaf value is mutated.
    """
    if mode not in ("off", "exact", "bounds"):
        raise ValueError("mode must be 'off', 'exact', or 'bounds'")
    if max_cache_depth is not None and (
            isinstance(max_cache_depth, bool)
            or not isinstance(max_cache_depth, int)
            or max_cache_depth < 0):
        raise ValueError("max_cache_depth must be a nonnegative integer or None")
    if not isinstance(is_maximizing_player, bool):
        raise ValueError("is_maximizing_player must be bool")
    for name, value in (("alpha", alpha), ("beta", beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if isinstance(value, float) and math.isnan(value):
            raise ValueError(f"{name} cannot be NaN")
    if alpha >= beta:
        raise ValueError("alpha must be strictly less than beta")
    if validate:
        validate_tree(root)

    stats, events, table = CacheStats(), [], {}

    def emit(event, path, maximizing, depth, **details):
        if trace:
            events.append(dict(event=event, path=path,
                               is_maximizing_player=maximizing,
                               depth=depth, **details))

    def store(key, value, bound, path, maximizing, depth):
        if mode == "exact" and bound != "exact":
            return
        previous = table.get(key)
        accepted = True
        if previous is not None:
            # Retain exact values and stronger bounds of the same kind.
            if previous.bound == "exact":
                accepted = False
            elif previous.bound == bound == "lower" and previous.value >= value:
                accepted = False
            elif previous.bound == bound == "upper" and previous.value <= value:
                accepted = False
        if accepted:
            table[key] = _Entry(value, bound)
            stats.stores += 1
            stats.table_peak_entries = max(stats.table_peak_entries, len(table))
        emit("store", path, maximizing, depth, value=value,
             stored_bound=bound, accepted=accepted, node_id=key[0])

    def visit(node, maximizing, low, high, path, depth):
        stats.visited_nodes += 1
        alpha_in, beta_in = low, high
        emit("enter", path, maximizing, depth, alpha=low, beta=high,
             node_id=id(node))
        eligible = mode != "off" and (max_cache_depth is None or depth <= max_cache_depth)
        key = (id(node), maximizing)
        if eligible:
            stats.tt_probes += 1
            entry = table.get(key)
            if entry is not None:
                stats.tt_hits += 1
                can_return = (entry.bound == "exact"
                              or (entry.bound == "lower" and entry.value >= high)
                              or (entry.bound == "upper" and entry.value <= low))
                emit("probe", path, maximizing, depth, hit=True,
                     cached_value=entry.value, cached_bound=entry.bound,
                     can_return=can_return, node_id=id(node), alpha=low, beta=high)
                if can_return:
                    stats.tt_returns += 1
                    if entry.bound == "exact":
                        stats.exact_hits += 1
                    else:
                        stats.bound_cutoffs += 1
                    emit("tt_return", path, maximizing, depth, value=entry.value,
                         returned_bound=entry.bound, node_id=id(node))
                    emit("return", path, maximizing, depth, value=entry.value,
                         returned_bound=entry.bound, source="cache")
                    return entry.value
                stats.noncutting_hits += 1
            else:
                emit("probe", path, maximizing, depth, hit=False,
                     cached_value=None, cached_bound=None, can_return=False,
                     node_id=id(node), alpha=low, beta=high)

        if not node.children:
            stats.leaf_evaluations += 1
            emit("leaf", path, maximizing, depth, value=node.value)
            if eligible:
                store(key, node.value, "exact", path, maximizing, depth)
            emit("return", path, maximizing, depth, value=node.value,
                 returned_bound="exact", source="search")
            return node.value

        stats.internal_nodes += 1
        best = -math.inf if maximizing else math.inf
        for index, child in enumerate(node.children):
            child_path = f"{path}/{index}"
            child_value = visit(child, not maximizing, low, high, child_path, depth+1)
            previous = best
            best = max(best, child_value) if maximizing else min(best, child_value)
            if maximizing:
                low = max(low, best)
            else:
                high = min(high, best)
            emit("update", path, maximizing, depth, child_index=index,
                 child_path=child_path, child_value=child_value,
                 previous_value=previous, value=best, alpha=low, beta=high)
            if low >= high:
                stats.bound_crossings += 1
                remaining = len(node.children)-index-1
                if remaining:
                    stats.cutoff_events += 1
                    stats.skipped_child_roots += remaining
                    emit("cutoff", path, maximizing, depth, value=best,
                         alpha=low, beta=high,
                         remaining_child_indices=list(range(index+1, len(node.children))),
                         skipped_children=[f"{path}/{j}" for j in range(index+1, len(node.children))])
                break
        kind = _bound_type(best, alpha_in, beta_in)
        if eligible:
            store(key, best, kind, path, maximizing, depth)
        emit("return", path, maximizing, depth, value=best,
             returned_bound=kind, source="search")
        return best

    value = visit(root, is_maximizing_player, alpha, beta, "root", 0)
    stats.table_final_entries = len(table)
    kind = "exact" if not root.children else _bound_type(value, alpha, beta)
    return CacheResult(value, stats, events, kind)
