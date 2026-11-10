"""Minimax and fail-soft Alpha-Beta for the finite tree in Lab 2, question 3.

The default root is MAX and turns alternate along each edge.  Child order is
left-to-right unless explicitly changed.  Shared node objects are traversed per
occurrence; this module does not silently turn the search into a cached DAG.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union


Number = Union[int, float]


class TreeNode:
    """A leaf has a finite value; an internal node has children and value=None."""

    def __init__(self, value: Optional[Number] = None, children=None):
        self.value = value
        self.children = list(children) if children is not None else []


@dataclass
class SearchStats:
    """Counts refer to recursive visits, not different Python object identities."""

    visited_nodes: int = 0
    internal_nodes: int = 0
    leaf_evaluations: int = 0
    cutoff_events: int = 0
    skipped_child_roots: int = 0
    bound_crossings: int = 0


@dataclass
class SearchResult:
    value: Number
    stats: SearchStats
    trace: List[Dict]
    bound: str = "exact"

    def to_dict(self):
        """Return standard-JSON-safe data, including textual infinite bounds."""
        return _json_safe(asdict(self))


def _json_safe(value):
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _valid_number(value):
    return (isinstance(value, int) and not isinstance(value, bool)) or (
        isinstance(value, float) and math.isfinite(value)
    )


def validate_tree(root: TreeNode):
    """Check structure once, accepting shared objects and rejecting cycles.

    This validation is separate from search counts.  Timed experiments may call
    it beforehand and pass validate=False to time the search kernel alone.
    """
    active = set()
    finished = set()

    def visit(node, path):
        if not isinstance(node, TreeNode):
            raise ValueError(f"{path}: expected TreeNode")
        identity = id(node)
        if identity in active:
            raise ValueError(f"{path}: cycle detected")
        if identity in finished:
            return
        if not isinstance(node.children, (list, tuple)):
            raise ValueError(f"{path}: children must be a list or tuple")
        if node.children:
            if node.value is not None:
                raise ValueError(f"{path}: an internal node cannot also have a value")
        elif not _valid_number(node.value):
            raise ValueError(f"{path}: leaf value must be a finite number, not bool")
        active.add(identity)
        for index, child in enumerate(node.children):
            visit(child, f"{path}/{index}")
        active.remove(identity)
        finished.add(identity)

    visit(root, "root")


def parse_tree(payload):
    """Parse the PDF JSON wrapper or a direct node, checking optional roles."""
    if isinstance(payload, dict) and "tree_structure" in payload:
        payload = payload["tree_structure"]

    def build(item, maximizing, path):
        if not isinstance(item, dict):
            raise ValueError(f"{path}: JSON node must be an object")
        if "is_maximizing_player" in item:
            role = item["is_maximizing_player"]
            if not isinstance(role, bool) or role != maximizing:
                raise ValueError(f"{path}: player role must alternate from the MAX root")
        children = item.get("children", [])
        if not isinstance(children, list):
            raise ValueError(f"{path}: JSON children must be a list")
        return TreeNode(
            value=item.get("value"),
            children=[build(child, not maximizing, f"{path}/{index}")
                      for index, child in enumerate(children)],
        )

    root = build(payload, True, "root")
    validate_tree(root)
    return root


def _ordered_indices(node, maximizing, order):
    length = len(node.children)
    if order == "left":
        return range(length)
    if order == "right":
        return range(length - 1, -1, -1)
    indices = list(order(node, maximizing))
    if (len(indices) != length
            or any(not isinstance(index, int) or isinstance(index, bool) for index in indices)
            or sorted(indices) != list(range(length))):
        raise ValueError("ordering callback must return each child index exactly once")
    return indices


def _bound_type(value, alpha_in, beta_in):
    # A fail-low return is an upper bound; a fail-high return is a lower bound.
    if value <= alpha_in:
        return "upper"
    if value >= beta_in:
        return "lower"
    return "exact"


def search(root: TreeNode, *, algorithm="alphabeta", order="left", trace=False,
           cutoff_on_equal=True, validate=True, is_maximizing_player=True,
           alpha=-math.inf, beta=math.inf):
    """Search and return value, counters and optional recursive events.

    The supplied window must satisfy alpha < beta.  With the default infinite
    window the root result is exact.  Internal
    Alpha-Beta returns (and a root searched with a narrow window) may be bounds,
    not exact subtree values.  Equal bounds may prune under the default rule.
    cutoff_events counts only cases with unvisited siblings; bound_crossings
    also counts a crossing reached after the final child.  Validation is outside
    visited_nodes, so its cost must be included or explicitly excluded in timing.
    A callback order(node, maximizing) returns a permutation of child indices;
    its work takes place inside the search call.
    """
    if algorithm not in ("minimax", "alphabeta"):
        raise ValueError("algorithm must be 'minimax' or 'alphabeta'")
    if not isinstance(order, str) and not callable(order):
        raise ValueError("order must be left, right, or a callable")
    if isinstance(order, str) and order not in ("left", "right"):
        raise ValueError("order must be left, right, or a callable")
    if not isinstance(is_maximizing_player, bool):
        raise ValueError("is_maximizing_player must be bool")
    if not isinstance(cutoff_on_equal, bool):
        raise ValueError("cutoff_on_equal must be bool")
    for name, bound in (("alpha", alpha), ("beta", beta)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise ValueError(f"{name} must be a number")
        if isinstance(bound, float) and math.isnan(bound):
            raise ValueError(f"{name} cannot be NaN")
    if alpha >= beta:
        raise ValueError("alpha must be strictly less than beta")
    if validate:
        validate_tree(root)
    stats = SearchStats()
    events = []

    def emit(event, path, maximizing, **details):
        if trace:
            events.append(dict(event=event, path=path,
                               is_maximizing_player=maximizing, **details))

    def visit(node, maximizing, low, high, path):
        stats.visited_nodes += 1
        alpha_in, beta_in = low, high
        emit("enter", path, maximizing, alpha=low, beta=high)
        if not node.children:
            stats.leaf_evaluations += 1
            emit("leaf", path, maximizing, value=node.value)
            emit("return", path, maximizing, value=node.value, returned_bound="exact")
            return node.value
        stats.internal_nodes += 1
        best = -math.inf if maximizing else math.inf
        indices = _ordered_indices(node, maximizing, order)
        for position, index in enumerate(indices):
            child_path = f"{path}/{index}"
            child_value = visit(node.children[index], not maximizing, low, high, child_path)
            previous = best
            best = max(best, child_value) if maximizing else min(best, child_value)
            if algorithm == "alphabeta":
                if maximizing:
                    low = max(low, best)
                else:
                    high = min(high, best)
            emit("update", path, maximizing, child_index=index, child_path=child_path,
                 child_value=child_value, previous_value=previous, value=best,
                 alpha=low, beta=high)
            crossing = low >= high if cutoff_on_equal else low > high
            if algorithm == "alphabeta" and crossing:
                stats.bound_crossings += 1
                remaining = list(indices)[position + 1:]
                if remaining:
                    stats.cutoff_events += 1
                    stats.skipped_child_roots += len(remaining)
                    emit("cutoff", path, maximizing, value=best, alpha=low, beta=high,
                         remaining_child_indices=remaining,
                         skipped_children=[f"{path}/{item}" for item in remaining],
                         returned_bound="lower" if maximizing else "upper")
                break
        kind = "exact" if algorithm == "minimax" else _bound_type(best, alpha_in, beta_in)
        emit("return", path, maximizing, value=best, returned_bound=kind)
        return best

    result = visit(root, is_maximizing_player, alpha, beta, "root")
    kind = "exact" if algorithm == "minimax" else _bound_type(result, alpha, beta)
    # A root leaf is known exactly even when it lies outside a supplied window.
    if not root.children:
        kind = "exact"
    return SearchResult(result, stats, events, kind)


def minimax(root, is_maximizing_player=True, *, order="left", validate=True):
    """Return the exact minimax value (no pruning)."""
    return search(root, algorithm="minimax", order=order, validate=validate,
                  is_maximizing_player=is_maximizing_player).value


def alpha_beta(root, alpha=-math.inf, beta=math.inf,
               is_maximizing_player=True, *, order="left", validate=True,
               cutoff_on_equal=True):
    """Return a fail-soft value; require alpha < beta (default window is exact)."""
    return search(root, algorithm="alphabeta", order=order, validate=validate,
                  is_maximizing_player=is_maximizing_player, alpha=alpha, beta=beta,
                  cutoff_on_equal=cutoff_on_equal).value


def sample_tree():
    """The complete JSON sample: four MAX pairs, including repeated (6,9)."""
    pairs = [TreeNode(children=[TreeNode(a), TreeNode(b)])
             for a, b in ((3, 5), (6, 9), (6, 9), (1, 2))]
    return TreeNode(children=[TreeNode(children=pairs[:2]),
                              TreeNode(children=pairs[2:])])


def main(argv=None):
    parser = argparse.ArgumentParser(description="题三：Minimax 与 Alpha-Beta；默认按左到右输出样例最佳得分。")
    parser.add_argument("--input", type=Path, help="PDF 格式 JSON 文件；省略则使用正式样例")
    parser.add_argument("--algorithm", choices=("minimax", "alphabeta"), default="alphabeta")
    parser.add_argument("--order", choices=("left", "right"), default="left")
    parser.add_argument("--details", action="store_true", help="输出统计和逐步回传/剪枝轨迹 JSON")
    args = parser.parse_args(argv)
    try:
        root = parse_tree(json.loads(args.input.read_text(encoding="utf-8-sig"))) if args.input else sample_tree()
        result = search(root, algorithm=args.algorithm, order=args.order, trace=args.details)
        if args.details:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
        else:
            print(result.value)
    except (ValueError, OSError, RecursionError) as error:
        parser.exit(2, f"输入错误：{error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
