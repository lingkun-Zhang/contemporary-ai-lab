"""题二的整数优先队列消融：相同内核的堆与循环 f 桶（Python 3.9+）。

只接受原模块中已证明一致、非负整数的四种安全下界；不改变进入格收费、
E免费与禁止回到S的约定。所有初始化与D检查都在公开调用内部。
桶保留FIFO或low_h的完整优先级；超出预设槽数上限时精确回退二叉堆。
"""

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from itertools import accumulate, count
from math import gcd

from weighted_astar import (SAFE_HEURISTICS, WeightedResult, heuristic_value,
                            prepare_terrain)

DEFAULT_MAX_BUCKET_SLOTS = 32768


class _Heap:
    def __init__(self, initial_f, maximum_increment, tie_break, stats):
        self.records = []

    def push(self, record):
        heappush(self.records, record)

    def pop(self):
        return heappop(self.records)

    def __len__(self):
        return len(self.records)


class _CircularBuckets:
    def __init__(self, initial_f, maximum_increment, tie_break, stats):
        # 每槽按需创建容器；槽数组本身及其初始化成本仍纳入时间与统计。
        self.slots = [None] * (maximum_increment + 1)
        self.quantum = stats["priority_quantum"]
        self.cursor = initial_f // self.quantum
        self.maximum_increment = maximum_increment
        self.tie_break = tie_break
        self.stats = stats
        self.count = self.active = 0
        stats["bucket_slots"] = len(self.slots)

    def push(self, record):
        if record[0] % self.quantum:
            raise AssertionError("优先级不能被已核验费用单位整除")
        f = record[0] // self.quantum
        if not self.cursor <= f <= self.cursor+self.maximum_increment:
            raise AssertionError("一致整数下界的f增量超出已核验范围")
        slot = f % len(self.slots)
        bucket = self.slots[slot]
        if bucket is None:
            bucket = deque() if self.tie_break == "fifo" else []
            self.slots[slot] = bucket
        if not bucket:
            self.active += 1
            self.stats["max_active_buckets"] = max(self.stats["max_active_buckets"], self.active)
        if self.tie_break == "fifo":
            # 同f下全局入队序号单增，append/popleft与原堆的FIFO完全同序。
            bucket.append(record)
        else:
            # f在此桶固定，只对(h,序号,位置,g)排序；h最小值无需单调。
            heappush(bucket, record[1:])
        self.count += 1

    def pop(self):
        if not self.count:
            raise IndexError("空队列")
        while not self.slots[self.cursor % len(self.slots)]:
            self.cursor += 1
            self.stats["empty_bucket_scans"] += 1
        bucket = self.slots[self.cursor % len(self.slots)]
        if self.tie_break == "fifo":
            record = bucket.popleft()
            if record[0] != self.cursor*self.quantum:
                raise AssertionError("循环桶混入了不同轮次的f")
        else:
            record = (self.cursor*self.quantum,) + heappop(bucket)
        self.count -= 1
        if not bucket:
            self.active -= 1
        return record

    def __len__(self):
        return self.count


def queue_astar_search(start, end, grid, heuristic="row_column_tight",
                       tie_break="low_h", frontier="bucket", record_trace=False,
                       max_bucket_slots=DEFAULT_MAX_BUCKET_SLOTS):
    """完整查询，返回原WeightedResult；frontier为heap/bucket/bucket_gcd。

    两种容器共用：输入规范化、h表、合法边D=max(c(v)+h(v)-h(u))、
    松弛、入队序号、过期检查与路径账本。matched heap也承担D检查，
    用于隔离容器效果；原weighted_astar另作端到端实用基线。

    max_bucket_slots仅为确定性的内存保护，不是近似参数：超过后完全
    使用同内核heap，stats.bucket_fallback=1，不能称为纯桶结果。
    bucket_gcd只将桶索引按正进入费的最大公约数换单位，不改g/h/f、
    输出账本和优先级顺序；全部费用为0时单位取1。
    """
    if not isinstance(heuristic, str) or heuristic not in SAFE_HEURISTICS:
        raise ValueError("只接受zero/corrected/row_column/row_column_tight安全整数下界")
    if tie_break not in ("fifo", "low_h"):
        raise ValueError("tie_break应为fifo或low_h")
    if frontier not in ("heap", "bucket", "bucket_gcd"):
        raise ValueError("frontier应为heap、bucket或bucket_gcd")
    if type(max_bucket_slots) is not int or max_bucket_slots < 1:
        raise ValueError("max_bucket_slots应为正整数")
    terrain = prepare_terrain(start, end, grid)
    start, end = terrain.start, terrain.end
    stats = dict(expanded_nodes=0, generated_nodes=0, popped_nodes=0,
                 max_frontier=0, stale_pops=0, reopened_nodes=0, unique_reached=0,
                 h_evaluations=0, reduced_edges_checked=0, max_reduced_cost=0,
                 bucket_slots=0, empty_bucket_scans=0, max_active_buckets=0,
                 bucket_fallback=0, cost_quantum=1, priority_quantum=1,
                 bucket_priority_increment=0)
    trace, expanded_order = [], []

    def result(path, cost):
        ledger = [0]+[terrain.costs[r][c] for r, c in path[1:]] if path else []
        return WeightedResult(path, cost, stats, trace, ledger, list(accumulate(ledger)),
                              terrain.min_step_cost, expanded_order)

    if not terrain.rows or not terrain.cols:
        return result([], -1)
    if terrain.costs[start[0]][start[1]] is None or terrain.costs[end[0]][end[1]] is None:
        return result([], -1)

    estimates, cost_quantum = {}, 0
    for r, row in enumerate(terrain.costs):
        for c, weight in enumerate(row):
            if weight is not None:
                cost_quantum = gcd(cost_quantum, weight)
                h = heuristic_value(terrain, (r, c), heuristic)
                if type(h) is not int or h < 0:
                    raise AssertionError("桶队列要求非负整数h")
                estimates[(r, c)] = h
    stats["h_evaluations"] = len(estimates)
    stats["cost_quantum"] = cost_quantum or 1
    maximum_increment = 0
    for node, h in estimates.items():
        for other in terrain.neighbors(node):
            nr, nc = other
            delta = terrain.costs[nr][nc]+estimates[other]-h
            stats["reduced_edges_checked"] += 1
            if delta < 0:
                raise AssertionError("所选启发函数在合法边上不一致")
            maximum_increment = max(maximum_increment, delta)
    stats["max_reduced_cost"] = maximum_increment
    initial_h = estimates[start]
    quantum = stats["cost_quantum"] if frontier == "bucket_gcd" else 1
    stats["priority_quantum"] = quantum
    if any(value % quantum for value in estimates.values()) or maximum_increment % quantum:
        raise AssertionError("安全整数下界或增量没有保持公共费用单位")
    priority_increment = maximum_increment // quantum
    stats["bucket_priority_increment"] = priority_increment
    if frontier != "heap" and priority_increment+1 <= max_bucket_slots:
        queue = _CircularBuckets(initial_h, priority_increment, tie_break, stats)
    else:
        queue = _Heap(initial_h, maximum_increment, tie_break, stats)
        stats["bucket_fallback"] = int(frontier != "heap")
    serial = count()
    queue.push((initial_h, initial_h if tie_break == "low_h" else 0,
                next(serial), start, 0))
    best_g, parent, expanded_g = {start: 0}, {}, {}
    stats["generated_nodes"] = stats["max_frontier"] = 1
    while queue:
        f, _, _, node, g = queue.pop()
        stats["popped_nodes"] += 1
        if best_g[node] != g:
            stats["stale_pops"] += 1
            continue
        if record_trace:
            trace.append(dict(step=len(trace)+1, node=node, g=g, h=f-g, f=f))
        if node == end:
            path = [end]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            stats["unique_reached"] = len(best_g)
            return result(path, g)
        if node in expanded_g:
            stats["reopened_nodes"] += 1
        expanded_g[node] = g
        stats["expanded_nodes"] += 1
        if record_trace:
            expanded_order.append(node)
        for other in terrain.neighbors(node):
            nr, nc = other
            candidate = g+terrain.costs[nr][nc]
            if candidate < best_g.get(other, float("inf")):
                best_g[other], parent[other] = candidate, node
                h = estimates[other]
                queue.push((candidate+h, h if tie_break == "low_h" else 0,
                            next(serial), other, candidate))
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(queue))
    stats["unique_reached"] = len(best_g)
    return result([], -1)


def bucket_astar_search(start, end, grid, heuristic="row_column_tight", tie_break="low_h",
                        record_trace=False, max_bucket_slots=DEFAULT_MAX_BUCKET_SLOTS):
    return queue_astar_search(start, end, grid, heuristic, tie_break, "bucket",
                              record_trace, max_bucket_slots)
