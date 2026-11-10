"""真正的普通队列 BFS，以及仅替换 frontier 的堆版受控对照。"""

from collections import deque
from heapq import heappop, heappush
from itertools import count

from maze_astar import DIRECTIONS, SearchResult, _coordinate, _reconstruct, _shape


def uniform_search(start, end, grid, frontier="deque", record_trace=False, neighbor_order=None):
    """单位代价搜索；deque/heap 共用校验、发现、父指针及统计代码。

    两种容器都按步数层次、层内入队顺序处理。heap 仅用于区分容器
    成本，不是额外的优化算法。输入不变，结果格式同 astar_search。
    """
    if frontier not in ("deque", "heap"):
        raise ValueError("frontier 必须为 deque 或 heap")
    start, end = _coordinate(start, "起点"), _coordinate(end, "终点")
    height, width = _shape(grid)
    directions = DIRECTIONS if neighbor_order is None else tuple(
        _coordinate(item, "方向") for item in neighbor_order)
    if len(directions) != 4 or set(directions) != set(DIRECTIONS):
        raise ValueError("neighbor_order 必须是四个基本方向的排列")
    stats = dict(expanded_nodes=0, generated_nodes=0, popped_nodes=0,
                 max_frontier=0, stale_pops=0, reopened_nodes=0, unique_reached=0)
    trace, expanded_order = [], []
    if height == 0 or width == 0:
        return SearchResult([], -1, stats, trace, expanded_order)
    for name, node in (("起点", start), ("终点", end)):
        if not (0 <= node[0] < height and 0 <= node[1] < width):
            raise ValueError(f"{name}坐标超出地图范围")
    if grid[start[0]][start[1]] == "X" or grid[end[0]][end[1]] == "X":
        return SearchResult([], -1, stats, trace, expanded_order)

    if frontier == "deque":
        queue = deque([(start, 0)])

        def put(node, distance):
            queue.append((node, distance))

        def take():
            return queue.popleft()
    else:
        serial = count()
        queue = [(0, next(serial), start)]

        def put(node, distance):
            heappush(queue, (distance, next(serial), node))

        def take():
            distance, _, node = heappop(queue)
            return node, distance

    best_g, parent = {start: 0}, {}
    stats["generated_nodes"] = stats["max_frontier"] = 1
    while queue:
        current, current_g = take()
        stats["popped_nodes"] += 1
        if record_trace:
            trace.append(dict(step=len(trace)+1, node=current, g=current_g, h=0, f=current_g))
        if current == end:
            stats["unique_reached"] = len(best_g)
            return SearchResult(_reconstruct(parent, end), current_g, stats, trace, expanded_order)
        stats["expanded_nodes"] += 1
        if record_trace:
            expanded_order.append(current)
        for dr, dc in directions:
            neighbor = current[0]+dr, current[1]+dc
            nr, nc = neighbor
            if (0 <= nr < height and 0 <= nc < width and
                    grid[nr][nc] != "X" and neighbor not in best_g):
                best_g[neighbor] = current_g + 1
                parent[neighbor] = current
                put(neighbor, current_g + 1)
                stats["generated_nodes"] += 1
                stats["max_frontier"] = max(stats["max_frontier"], len(queue))
    stats["unique_reached"] = len(best_g)
    return SearchResult([], -1, stats, trace, expanded_order)


def bfs_search(start, end, grid, record_trace=False, neighbor_order=None):
    """普通 deque BFS，完整返回路径与统一口径的搜索统计。"""
    return uniform_search(start, end, grid, "deque", record_trace, neighbor_order)
