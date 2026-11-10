"""Full terminal Alpha-Beta with a fresh, capacity-limited LRU bound table.

Capacity limits stored entries, never search depth.  Object identity and the
current player form the key.  Inputs must remain unchanged during one call.
"""
import math
from collections import OrderedDict
from dataclasses import dataclass

from alphabeta_tree import TreeNode, validate_tree
from transposition_search import CacheResult, CacheStats, _Entry, _bound_type


@dataclass
class BoundedStats(CacheStats):
    new_insertions: int = 0
    entry_updates: int = 0
    evictions: int = 0


class _LRUTable:
    """A hit refreshes recency even when the bound cannot close a window."""
    def __init__(self, capacity, stats):
        self.capacity, self.stats, self.entries = capacity, stats, OrderedDict()

    def probe(self, key):
        item = self.entries.get(key)
        if item is not None:
            self.entries.move_to_end(key)
        return item

    def store(self, key, value, bound):
        previous = self.entries.get(key)
        if previous is not None and (
            previous.bound == 'exact'
            or previous.bound == bound == 'lower' and previous.value >= value
            or previous.bound == bound == 'upper' and previous.value <= value
        ):
            return False, None
        evicted = None
        if previous is None:
            if self.capacity is not None and len(self.entries) >= self.capacity:
                evicted, _ = self.entries.popitem(last=False)
                self.stats.evictions += 1
            self.stats.new_insertions += 1
        else:
            self.stats.entry_updates += 1
        self.entries[key] = _Entry(value, bound)
        self.entries.move_to_end(key)
        self.stats.stores += 1
        self.stats.table_peak_entries = max(self.stats.table_peak_entries, len(self.entries))
        return True, evicted


def bounded_search(root: TreeNode, *, capacity=None, validate=True, trace=False,
                   alpha=-math.inf, beta=math.inf, is_maximizing_player=True):
    """Search left-to-right, returning CacheResult with BoundedStats.

    None means unlimited LRU storage; 0 disables all probes and stores in this
    kernel.  A positive capacity evicts the least recently probed/written entry.
    Keys are (id(node), current player).  No state or table survives this call.
    Bound hits return only for EXACT or LOWER>=beta or UPPER<=alpha; other hits
    merely refresh LRU recency and search normally.  No window tightening occurs.
    Every search reaches terminal leaves where required: capacity never truncates
    a tree.  With default infinite root bounds the returned root score is exact.
    """
    if capacity is not None and (isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0):
        raise ValueError('capacity must be a nonnegative integer or None')
    if not isinstance(is_maximizing_player, bool):
        raise ValueError('is_maximizing_player must be bool')
    for name, value in (('alpha', alpha), ('beta', beta)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and math.isnan(value):
            raise ValueError(f'{name} must be numeric and not NaN')
    if alpha >= beta:
        raise ValueError('alpha must be strictly less than beta')
    if validate:
        validate_tree(root)
    stats, events = BoundedStats(), []
    table = _LRUTable(capacity, stats)

    def emit(event, path, maximizing, depth, **details):
        if trace:
            events.append(dict(event=event, path=path, is_maximizing_player=maximizing, depth=depth, **details))

    def store(key, value, bound, path, maximizing, depth):
        accepted, evicted = table.store(key, value, bound)
        if evicted is not None:
            emit('evict', path, maximizing, depth, evicted_node_id=evicted[0], evicted_player=evicted[1])
        emit('store', path, maximizing, depth, value=value, stored_bound=bound,
             accepted=accepted, node_id=key[0], table_entries=len(table.entries))

    def visit(node, maximizing, low, high, path, depth):
        stats.visited_nodes += 1
        alpha_in, beta_in = low, high
        emit('enter', path, maximizing, depth, alpha=low, beta=high, node_id=id(node))
        key = (id(node), maximizing)
        if capacity != 0:
            stats.tt_probes += 1
            entry = table.probe(key)
            if entry is not None:
                stats.tt_hits += 1
                can_return = entry.bound == 'exact' or entry.bound == 'lower' and entry.value >= high or entry.bound == 'upper' and entry.value <= low
                emit('probe', path, maximizing, depth, hit=True, cached_value=entry.value,
                     cached_bound=entry.bound, can_return=can_return, node_id=id(node), alpha=low, beta=high)
                if can_return:
                    stats.tt_returns += 1
                    if entry.bound == 'exact': stats.exact_hits += 1
                    else: stats.bound_cutoffs += 1
                    emit('tt_return', path, maximizing, depth, value=entry.value, returned_bound=entry.bound, node_id=id(node))
                    emit('return', path, maximizing, depth, value=entry.value, returned_bound=entry.bound, source='cache')
                    return entry.value
                stats.noncutting_hits += 1
            else:
                emit('probe', path, maximizing, depth, hit=False, cached_value=None, cached_bound=None, can_return=False, node_id=id(node), alpha=low, beta=high)
        if not node.children:
            stats.leaf_evaluations += 1
            emit('leaf', path, maximizing, depth, value=node.value)
            if capacity != 0: store(key, node.value, 'exact', path, maximizing, depth)
            emit('return', path, maximizing, depth, value=node.value, returned_bound='exact', source='search')
            return node.value
        stats.internal_nodes += 1
        best = -math.inf if maximizing else math.inf
        for index, child in enumerate(node.children):
            child_path = f'{path}/{index}'
            child_value = visit(child, not maximizing, low, high, child_path, depth+1)
            previous = best
            best = max(best, child_value) if maximizing else min(best, child_value)
            if maximizing: low = max(low, best)
            else: high = min(high, best)
            emit('update', path, maximizing, depth, child_index=index, child_path=child_path,
                 child_value=child_value, previous_value=previous, value=best, alpha=low, beta=high)
            if low >= high:
                stats.bound_crossings += 1
                remaining = len(node.children)-index-1
                if remaining:
                    stats.cutoff_events += 1
                    stats.skipped_child_roots += remaining
                    emit('cutoff', path, maximizing, depth, value=best, alpha=low, beta=high,
                         remaining_child_indices=list(range(index+1, len(node.children))),
                         skipped_children=[f'{path}/{j}' for j in range(index+1, len(node.children))])
                break
        kind = _bound_type(best, alpha_in, beta_in)
        if capacity != 0: store(key, best, kind, path, maximizing, depth)
        emit('return', path, maximizing, depth, value=best, returned_bound=kind, source='search')
        return best

    value = visit(root, is_maximizing_player, alpha, beta, 'root', 0)
    stats.table_final_entries = len(table.entries)
    bound = 'exact' if not root.children else _bound_type(value, alpha, beta)
    return CacheResult(value, stats, events, bound)
