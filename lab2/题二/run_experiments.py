"""题二实验：进入地形代价、免费终点、可采纳性及行列下界。

纯标准库。默认正式求解配置都与独立 Dijkstra 核对；raw_M 和最少步数
BFS 是分析对照，允许在零权拓展或加权地形上给出非最优费用。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from weighted_astar import (SAMPLE_GOAL, SAMPLE_GRID, SAMPLE_START, astar_search,
                            heuristic_value, prepare_terrain)


SAFE = ("zero", "corrected", "row_column", "row_column_tight")
VARIANTS = (*SAFE, "raw_M")
LABELS = {"zero": "h=0", "corrected": "修正曼哈顿", "row_column": "行列组合", "row_column_tight": "行列组合+免费次数收紧", "raw_M": "原始曼哈顿（分析）"}
COLORS = {"zero": "#718da6", "corrected": "#419f96", "row_column": "#da9239", "row_column_tight": "#477fbd", "raw_M": "#ae759a"}
MOVES = ((0, 1), (1, 0), (0, -1), (-1, 0))
FONT = "'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif"


@dataclass
class Case:
    slug: str
    name: str
    grid: list
    start: tuple
    goal: tuple
    suite: str = "structure"
    distribution: str = ""
    seed: int = 0
    obstacle_probability: float = 0.0
    zero_extension: bool = False


def independent_costs(case):
    """测试侧独立转换，不复用正式求解器的规范化或邻居实现。"""
    costs = []
    for row in case.grid:
        converted = []
        for value in row:
            if value == "X":
                converted.append(None)
            elif value in ("S", "E"):
                converted.append(0)
            else:
                converted.append(int(value))
        costs.append(converted)
    costs[case.start[0]][case.start[1]] = 0
    costs[case.goal[0]][case.goal[1]] = 0
    return costs


def adjacent(costs, node):
    row, col = node
    for nr, nc in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
        if 0 <= nr < len(costs) and 0 <= nc < len(costs[0]) and costs[nr][nc] is not None:
            yield nr, nc


def reference_dijkstra(case, reverse=False):
    """独立的全距离 Dijkstra；反向边 v->u 的费用为原进入 v 的权重。"""
    costs = independent_costs(case)
    origin = case.goal if reverse else case.start
    distances, queue = {origin: 0}, [(0, origin)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        # 正式图删掉所有进入 start 的边；反图中的 start 无出边。
        if reverse and node == case.start:
            continue
        for other in adjacent(costs, node):
            if not reverse and other == case.start:
                continue
            weight = costs[node[0]][node[1]] if reverse else costs[other[0]][other[1]]
            candidate = distance + weight
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                heapq.heappush(queue, (candidate, other))
    return distances


def reference_bellman_ford(case):
    """小图的第二种参考算法：显式边松弛，与堆和 A* 都独立。"""
    costs = independent_costs(case)
    cells = [(r, c) for r, row in enumerate(costs) for c, weight in enumerate(row) if weight is not None]
    edges = [(u, v, costs[v[0]][v[1]]) for u in cells for v in adjacent(costs, u) if v != case.start]
    distances = {node: math.inf for node in cells}
    distances[case.start] = 0
    for _ in range(max(0, len(cells)-1)):
        changed = False
        for source, target, weight in edges:
            candidate = distances[source] + weight
            if candidate < distances[target]:
                distances[target] = candidate
                changed = True
        if not changed:
            break
    return distances[case.goal] if math.isfinite(distances[case.goal]) else -1


def bfs_path(case):
    """BFS 只最小化步数，随后独立累计权重，故不保证费用最优。"""
    costs = independent_costs(case)
    parent, queue = {case.start: None}, deque([case.start])
    while queue:
        node = queue.popleft()
        if node == case.goal:
            path = []
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))
        # 与正式算法同为右下左上，仅控制同深度路径选择。
        for dr, dc in MOVES:
            other = node[0]+dr, node[1]+dc
            if (0 <= other[0] < len(costs) and 0 <= other[1] < len(costs[0])
                    and costs[other[0]][other[1]] is not None and other not in parent):
                parent[other] = node
                queue.append(other)
    return []


def path_cost(case, path):
    costs = independent_costs(case)
    return sum(costs[r][c] for r, c in path[1:]) if path else -1


def validate_path(case, result, expected=None):
    if not result.path:
        if result.cost != -1 or expected is not None and expected != -1:
            raise AssertionError(f"{case.slug}: 无解结果与参考不符")
        return
    if result.path[0] != case.start or result.path[-1] != case.goal:
        raise AssertionError(f"{case.slug}: 端点错误")
    costs = independent_costs(case)
    for row, col in result.path:
        if not (0 <= row < len(costs) and 0 <= col < len(costs[0])) or costs[row][col] is None:
            raise AssertionError(f"{case.slug}: 路径越界或经过障碍")
    for first, second in zip(result.path, result.path[1:]):
        if abs(first[0]-second[0]) + abs(first[1]-second[1]) != 1:
            raise AssertionError(f"{case.slug}: 路径不是四方向连续移动")
    if case.start in result.path[1:]:
        raise AssertionError("路径不应再次进入起点")
    if result.cost != path_cost(case, result.path):
        raise AssertionError(f"{case.slug}: 返回费用和实际进入格累计不同")
    if expected is not None and result.cost != expected:
        raise AssertionError(f"{case.slug}: {result.cost} != 最小费用 {expected}")


def heuristic_checks(case, variant, reverse_distances):
    prepared = prepare_terrain(case.start, case.goal, case.grid)
    values = {}
    for row in range(prepared.rows):
        for col in range(prepared.cols):
            node = row, col
            if prepared.costs[row][col] is not None:
                values[node] = (abs(row-case.goal[0])+abs(col-case.goal[1])
                                if variant == "raw_M" else heuristic_value(prepared, node, variant))
    over = sum(values[node] > true + 1e-9 for node, true in reverse_distances.items())
    violations = []
    for node in values:
        for other in prepared.neighbors(node):
            weight = prepared.costs[other[0]][other[1]]
            if values[node] > weight + values[other] + 1e-9:
                violations.append((node, other, values[node], weight, values[other]))
    if variant in SAFE and (over or violations or values[case.goal] != 0):
        raise AssertionError(f"{case.slug}/{variant}: 安全启发的下界/一致性验证失败")
    optimal = reverse_distances.get(case.start, -1)
    return {"case": case.slug, "variant": variant, "safe_variant": variant in SAFE,
            "h_start": values[case.start], "true_cost": optimal,
            "finite_nodes_checked": len(reverse_distances), "overestimate_nodes": over,
            "consistency_violation_edges": len(violations),
            "mean_gap_on_finite_nodes": statistics.mean(true-values[node] for node, true in reverse_distances.items()),
            "first_consistency_violation": json.dumps(violations[0]) if violations else ""}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mark(grid, start, goal):
    grid[start[0]][start[1]] = "S"
    grid[goal[0]][goal[1]] = "E"
    return grid


def band_grid(n, zero=False):
    grid = [[1 for _ in range(n)] for _ in range(n)]
    for row in range(n):
        for col in range(n):
            if row in (n//3, 2*n//3):
                grid[row][col] = 8
            if col in (n//3, 2*n//3):
                grid[row][col] = max(grid[row][col], 6)
    if zero:
        grid[2][3] = 0
    return grid


def structure_cases():
    sample = Case("sample", "PDF 样例", [list(row) for row in SAMPLE_GRID], SAMPLE_START, SAMPLE_GOAL)
    trap = Case("bfs_trap", "短路昂贵、绕路便宜", [[1]*5, ["S", 9, 9, 9, "E"], [1]*5], (1, 0), (1, 4))
    bands = Case("bands", "整行整列高代价带", mark(band_grid(31), (0, 0), (30, 30)), (0, 0), (30, 30))
    zero_band = Case("zero_bands", "单个零权格与高代价带", mark(band_grid(31, True), (0, 0), (30, 30)),
                     (0, 0), (30, 30), zero_extension=True)
    zero_counter = Case("zero_counterexample", "零权拓展中的原始曼哈顿反例",
                        [["S", 1, "E"], [0, 0, 0]], (0, 0), (0, 2), zero_extension=True)
    blocked = Case("unreachable", "障碍隔断无解拓展", [["S", 1, "X", 1, "E"], [1, 1, "X", 1, 1]],
                   (0, 0), (0, 4))
    uniform = Case("uniform", "均匀正权开阔图", mark([[1]*31 for _ in range(31)], (0, 0), (30, 30)),
                   (0, 0), (30, 30))
    horizontal_grid = [[9 if row in (5, 10, 15) else 1 for _ in range(21)] for row in range(21)]
    horizontal = Case("horizontal_bands", "三条高权整行", mark(horizontal_grid, (0, 0), (20, 20)),
                      (0, 0), (20, 20), distribution="horizontal_bands")
    cells = [(row, col) for row in range(21) for col in range(21)
             if (row, col) not in ((0, 0), (20, 20))]
    values = [horizontal.grid[row][col] for row, col in cells]
    random.Random(20261028).shuffle(values)
    scattered_grid = [[1] * 21 for _ in range(21)]
    for (row, col), value in zip(cells, values):
        scattered_grid[row][col] = value
    scattered = Case("scattered_weights", "相同权重随机分散",
                     mark(scattered_grid, (0, 0), (20, 20)), (0, 0), (20, 20),
                     distribution="same_histogram_scattered", seed=20261028)
    if Counter(value for row in horizontal.grid for value in row) != Counter(value for row in scattered.grid for value in row):
        raise AssertionError("空间对照必须保持全部权重频数、尺寸和端点相同")
    return [sample, trap, bands, zero_band, zero_counter, blocked, uniform, horizontal, scattered]


def scaling_cases():
    result = []
    for factor in (1, 2, 4, 8):
        grid = [[value*factor if type(value) is int else value for value in row] for row in SAMPLE_GRID]
        result.append((Case(f"scaling_{factor}", f"所有地形权重 ×{factor}", grid,
                            SAMPLE_START, SAMPLE_GOAL, "scaling"), factor))
    return result


def gap_cases():
    cases = []
    for gap in (0, 5, 10, 15, 20):
        grid = [[1]*21 for _ in range(21)]
        for row in range(21):
            if row != gap:
                grid[row][10] = 20
        start, goal = (10, 0), (10, 20)
        cases.append((Case(f"high_cost_gap_{gap}", f"高代价列缺口行 {gap}", mark(grid, start, goal),
                            start, goal, "gap"), gap))
    return cases


def random_cases(seeds, seed):
    cases = []
    for n in (15, 30, 60):
        for distribution_index, distribution in enumerate(("low", "heterogeneous", "bands")):
            for probability_index, probability in enumerate((0.0, .15)):
                for instance in range(seeds):
                    case_seed = seed+n*100000+distribution_index*10000+probability_index*1000+instance
                    rng = random.Random(case_seed)
                    if distribution == "bands":
                        grid = band_grid(n)
                    else:
                        choices = (1, 1, 1, 2, 3) if distribution == "low" else (1, 1, 3, 8, 20)
                        grid = [[rng.choice(choices) for _ in range(n)] for _ in range(n)]
                    for row in range(n):
                        for col in range(n):
                            if rng.random() < probability:
                                grid[row][col] = "X"
                    start, goal = (0, 0), (n-1, n-1)
                    cases.append(Case(f"random_{n}_{distribution}_{int(probability*100)}_{instance}",
                                      "随机加权地图", mark(grid, start, goal), start, goal, "scale",
                                      distribution, case_seed, probability))
        for instance in range(seeds):
            case_seed = seed+n*100000+90000+instance
            rng = random.Random(case_seed)
            choices = (0, 1, 1, 3, 8)
            grid = [[rng.choice(choices) if rng.random() > .15 else "X" for _ in range(n)] for _ in range(n)]
            start, goal = (0, 0), (n-1, n-1)
            cases.append(Case(f"zero_random_{n}_{instance}", "零权地形拓展", mark(grid, start, goal),
                              start, goal, "zero_extension", "zero_mixture", case_seed, .15, True))
    return cases


def measure(case, variant, repeats, batch, expected, trace=False):
    heuristic = (lambda node, goal: abs(node[0]-goal[0])+abs(node[1]-goal[1])) if variant == "raw_M" else variant
    call = lambda record: astar_search(case.start, case.goal, case.grid, heuristic=heuristic,
                                       tie_break="low_h", record_trace=record)
    result = call(trace)
    validate_path(case, result, expected if variant in SAFE else None)
    call(False)
    samples = []
    for _ in range(repeats):
        before = perf_counter_ns()
        for _ in range(batch):
            timed = call(False)
        samples.append((perf_counter_ns()-before)/batch/1e6)
        if timed.path != result.path or timed.cost != result.cost:
            raise AssertionError("记录与计时模式不一致")
    prepared = prepare_terrain(case.start, case.goal, case.grid)
    row = {"suite": case.suite, "case": case.slug, "case_name": case.name,
           "rows": len(case.grid), "cols": len(case.grid[0]), "distribution": case.distribution,
           "seed": case.seed, "obstacle_probability": case.obstacle_probability,
           "zero_extension": case.zero_extension, "variant": variant, "safe_variant": variant in SAFE,
           "reachable": expected >= 0, "oracle_cost": expected, "returned_cost": result.cost,
           "optimal": result.cost == expected, "cost_gap": result.cost-expected if expected >= 0 else 0,
           "path_steps": len(result.path)-1 if result.path else -1, "min_step_cost": prepared.min_step_cost,
           **result.stats, "time_median_ms": statistics.median(samples), "time_min_ms": min(samples),
           "batch_average_ms": json.dumps(samples)}
    return result, row


def summarize(rows):
    grouped, summary = defaultdict(list), []
    keys = ("suite", "rows", "distribution", "obstacle_probability", "zero_extension", "reachable", "variant")
    for row in rows:
        if row["safe_variant"]:
            grouped[tuple(row[key] for key in keys)].append(row)
    for values, group in grouped.items():
        item = dict(zip(keys, values))
        item["n_inputs"] = len(group)
        for metric in ("expanded_nodes", "max_frontier", "time_median_ms", "path_steps", "oracle_cost"):
            data = [row[metric] for row in group]
            if metric in ("path_steps", "oracle_cost") and not item["reachable"]:
                continue
            q = statistics.quantiles(data, n=4, method="inclusive") if len(data)>1 else data*3
            item.update({metric+"_median": statistics.median(data), metric+"_q1": q[0], metric+"_q3": q[2]})
        summary.append(item)
    return summary


def svg_start(width, height, title, note):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f'<style>text{{font-family:{FONT};fill:#273b49}}.title{{font-size:22px;font-weight:700}}'
            '.label{font-size:14px}.note{font-size:12px;fill:#526777}</style>',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            f'<text class="title" x="24" y="34">{escape(title)}</text>',
            f'<text class="note" x="24" y="60">{escape(note)}</text>']


def plot_lines(path, title, panels, note):
    width, panel_h, left, right, top = 1180, 280, 100, 40, 110
    height = top+len(panels)*panel_h+30
    parts = svg_start(width, height, title, note)
    for pi, panel in enumerate(panels):
        xs, series = panel["x"], panel["series"]
        ymax = max(1, max(value for _, values, _ in series for value in values)*1.12)
        ytop, plot_h, plot_w = top+pi*panel_h, 170, width-left-right
        numeric = panel.get("numeric", False)
        def xpos(index):
            return (left+plot_w*(float(xs[index])-float(min(xs)))/max(1e-12,float(max(xs))-float(min(xs)))
                    if numeric else left+plot_w*(index+.5)/len(xs))
        parts.append(f'<text class="label" x="24" y="{ytop-20}">{escape(panel["title"])}</text>')
        for tick in range(5):
            value = ymax*tick/4
            y = ytop+plot_h*(1-tick/4)
            parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e0e7ec"/>')
            parts.append(f'<text class="note" x="{left-10}" y="{y+4}" text-anchor="end">{value:.2f}</text>')
        for j, label in enumerate(xs):
            parts.append(f'<text class="note" x="{xpos(j)}" y="{ytop+plot_h+22}" text-anchor="middle">{escape(str(label))}</text>')
        for si, (label, values, color) in enumerate(series):
            points = [(xpos(j),ytop+plot_h*(1-value/ymax)) for j,value in enumerate(values)]
            parts.append('<polyline fill="none" stroke="'+color+'" stroke-width="2" points="'+' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+'"/>')
            for x,y in points:
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            lx = 100+si*265
            parts.append(f'<line x1="{lx}" y1="{ytop+plot_h+47}" x2="{lx+18}" y2="{ytop+plot_h+47}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text class="note" x="{lx+25}" y="{ytop+plot_h+51}">{escape(label)}</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def path_svg(path, case, items, title, note):
    """每个面板保留原始权值与路径线，不用等步数冒充费用。"""
    cell = max(20,min(48,390//max(len(case.grid),len(case.grid[0]))))
    panel_width = max(360,len(case.grid[0])*cell+50)
    width, height = panel_width*len(items)+30, 140+len(case.grid)*cell+65
    parts = svg_start(width,height,title,note)
    for index,(label,result) in enumerate(items):
        left, ytop = 24+index*panel_width, 124
        parts.append(f'<text class="label" x="{left}" y="86">{escape(label)}：费用 {result.cost}，步数 {len(result.path)-1}</text>')
        pathset = set(result.path)
        for row,line in enumerate(case.grid):
            for col,value in enumerate(line):
                fill = '#273b49' if value=='X' else '#f8dba9' if (row,col) in pathset else '#f2f6f8'
                x,y = left+col*cell, ytop+row*cell
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d3dde4"/>')
                if value!='X':
                    text = 'S' if (row,col)==case.start else 'E=0' if (row,col)==case.goal else str(value)
                    parts.append(f'<text x="{x+cell/2}" y="{y+cell*.65}" text-anchor="middle" style="font-size:{max(9,min(14,cell*.4))}px">{escape(text)}</text>')
        if result.path:
            points=' '.join(f'{left+(col+.5)*cell},{ytop+(row+.5)*cell}' for row,col in result.path)
            parts.append(f'<polyline points="{points}" fill="none" stroke="#be6f1e" stroke-width="2.5" opacity=".65"/>')
    parts.append(f'<text class="note" x="24" y="{height-20}">S 首次费用为 0；只按进入下一格收费；E 为 0；橙线为返回路径。</text>')
    parts.append('</svg>')
    path.write_text('\n'.join(parts),encoding='utf-8')


def run(output, seeds=10, repeats=3, batch=3, seed=20261028):
    output.mkdir(parents=True,exist_ok=True)
    rows, checks, recorded, all_cases = [], [], {}, []

    def evaluate(case, check=False, trace=False):
        all_cases.append(case)
        exact = reference_dijkstra(case).get(case.goal,-1)
        reverse = reference_dijkstra(case,True) if check else None
        if check and len(case.grid)*len(case.grid[0])<=40 and reference_bellman_ford(case)!=exact:
            raise AssertionError("Dijkstra 与独立 Bellman-Ford 不一致")
        order=list(VARIANTS);random.Random(seed+len(all_cases)).shuffle(order)
        result_map={}
        for variant in order:
            result,row=measure(case,variant,repeats,batch,exact,trace)
            rows.append(row);result_map[variant]=result
            if check:checks.append(heuristic_checks(case,variant,reverse))
        recorded[case.slug]=result_map
        return result_map

    structures=structure_cases()
    for case in structures:evaluate(case,True,case.slug=='sample')
    scaling=[]
    for case,factor in scaling_cases():
        evaluate(case,True)
        scaling.extend({**row,"weight_multiplier":factor} for row in rows if row['case']==case.slug)
    for variant in SAFE:
        base_result=recorded['scaling_1'][variant]
        for factor in (2,4,8):
            scaled=recorded[f'scaling_{factor}'][variant]
            if (scaled.cost!=factor*base_result.cost or scaled.path!=base_result.path
                    or scaled.stats!=base_result.stats):
                raise AssertionError(f"{variant}: 等比例权重缩放应保持路径和搜索工作量，仅费用等比变化")
    gaps=[]
    for case,gap in gap_cases():
        evaluate(case,True)
        gaps.extend({**row,"gap_row":gap,"high_cost_cells":20} for row in rows if row['case']==case.slug)
    print('[1/3] 样例、免费终点、加权 BFS 反例、地形缩放和受控高代价格位置完成。',flush=True)
    generated=random_cases(seeds,seed)
    for index,case in enumerate(generated,1):
        evaluate(case)
        if index%30==0:print(f'随机加权地图 {index}/{len(generated)}',flush=True)
    print('[2/3] 正权与零权随机组完成；所有安全配置费用与独立 Dijkstra 相同。',flush=True)

    sample=structures[0]; sample_result=recorded['sample']['corrected']
    costs=independent_costs(sample)
    running=0;ledger=[]
    for index,node in enumerate(sample_result.path):
        fee=0 if index==0 else costs[node[0]][node[1]]
        running+=fee
        ledger.append({'step':index,'row':node[0],'col':node[1],
                       'enter_cost':fee,'cumulative_cost':running,'is_goal':node==sample.goal})
    write_csv(output/'sample_path_ledger.csv',ledger)
    traces=[]
    for variant,result in recorded['sample'].items():
        for item in result.trace:
            node=item['node'];traces.append({'variant':variant,'step':item['step'],'row':node[0],
                                           'col':node[1],'g':item['g'],'h':item['h'],'f':item['f']})
    write_csv(output/'sample_trace.csv',traces)
    comparisons=[];path_comparisons={}
    for case in (structures[1],structures[4]):
        path=bfs_path(case)
        bfs_result=type('BFSResult',(),{'path':path,'cost':path_cost(case,path)})()
        optimal=recorded[case.slug]['corrected']
        for method,result in [('bfs_min_steps',bfs_result),('astar_min_cost',optimal)]:
            comparisons.append({'case':case.slug,'method':method,'path_steps':len(result.path)-1,
                                'path_cost':result.cost,'optimal_cost':optimal.cost,
                                'path':json.dumps(result.path),'is_minimum_cost':result.cost==optimal.cost})
        path_comparisons[case.slug]=(bfs_result,optimal)
    write_csv(output/'bfs_vs_cost.csv',comparisons)
    write_csv(output/'comparison.csv',rows)
    write_csv(output/'safe_summary.csv',summarize(rows))
    write_csv(output/'heuristic_checks.csv',checks)
    write_csv(output/'weight_scaling.csv',scaling)
    write_csv(output/'high_cost_gap.csv',gaps)

    path_svg(output/'sample_path.svg',sample,[('修正曼哈顿',sample_result)],'题二 PDF 样例：进入费用累计',
             '返回最小费用 7；完整逐步账本见 sample_path_ledger.csv。')
    trap=structures[1];bfs_result,opt=path_comparisons[trap.slug]
    path_svg(output/'bfs_vs_cost.svg',trap,[('BFS 最少步数',bfs_result),('A* 最小费用',opt)],
             '步数最少不等于费用最小','同一地图和端点；BFS 路径合法，但权重不同使最短步数目标与费用目标不同。')
    counter=structures[4]
    path_svg(output/'zero_cost_counterexample.svg',counter,
             [('安全修正启发',recorded[counter.slug]['corrected']),('原始曼哈顿',recorded[counter.slug]['raw_M'])],
             '零权拓展：原始曼哈顿可以失去费用最优性','这不是 PDF 正权样例；低代价绕行被几何步数高估。')
    chosen=('sample','uniform','bands','zero_bands','horizontal_bands','scattered_weights','unreachable')
    plot_lines(output/'heuristic_nodes.svg','安全启发函数的搜索量与结构',[{
        'title':'实际扩展节点数（不含终点）；均为 low_h；各图尺寸见原始数据',
        'x':[next(c.name for c in structures if c.slug==slug) for slug in chosen],
        'series':[(LABELS[v],[recorded[slug][v].stats['expanded_nodes'] for slug in chosen],COLORS[v]) for v in SAFE]}],
        '图是不同结构案例比较，不把尺寸、权重分布变化解释成单变量因果。')
    plot_lines(output/'weight_and_position.svg','权重缩放与高代价位置的受控实验',[
        {'title':'所有数字权重等比缩放，E 仍免费：横轴倍数，纵轴最小费用','x':[1,2,4,8],'numeric':True,
         'series':[(LABELS[v],[next(r['returned_cost'] for r in scaling if r['variant']==v and r['weight_multiplier']==factor)
                                      for factor in (1,2,4,8)],COLORS[v]) for v in SAFE]},
        {'title':'保持 20 个高权格，只移动低费用缺口：横轴缺口行，纵轴最小费用','x':[0,5,10,15,20],'numeric':True,
         'series':[('独立验证的最小费用',[next(r['oracle_cost'] for r in gaps if r['gap_row']==gap) for gap in (0,5,10,15,20)],'#419f96')]}],
        '高费用格并非障碍，缺口太远时可能直接支付高费用；因此最优路线会切换。')

    by_variant=[]
    for suite in ('structure','scale','zero_extension'):
        for variant in VARIANTS:
            selected=[r for r in rows if r['suite']==suite and r['variant']==variant]
            by_variant.append({'suite':suite,'variant':variant,'configurations':len(selected),
                               'reachable_inputs':sum(r['reachable'] for r in selected),
                               'nonoptimal_results':sum(not r['optimal'] for r in selected)})
    metadata={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
              'platform':platform.platform(),'processor':platform.processor(),
              'parameters':{'seeds':seeds,'repeats':repeats,'batch':batch,'seed':seed,
                            'safe_variants':SAFE,'analysis_only':['raw_M','bfs_min_steps'],'tie_break':'low_h'},
              'rules':{'move':'four-direction','edge_cost':'enter destination terrain weight',
                       'start_cost':0,'goal_cost':0,'incoming_start_edges':'removed; nonnegative optimum has a simple path',
                       'zero_terrain':'explicit extension','X':'obstacle extension for no-path cases'},
              'timing':{'clock':'perf_counter_ns','warmups':1,'measure':'median of batch-average ms',
                        'includes':'complete astar_search including normalization/preprocessing and path reconstruction',
                        'excludes':'Dijkstra/Bellman-Ford validation, trace collection, graphs and file IO'},
              'source_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('weighted_astar.py','run_experiments.py')},
              'cases':[{'case':c.slug,'name':c.name,'suite':c.suite,'distribution':c.distribution,'grid':c.grid,
                        'start':c.start,'goal':c.goal,'seed':c.seed,'zero_extension':c.zero_extension,
                        'obstacle_probability':c.obstacle_probability} for c in all_cases],
              'paths':{slug:{variant:result.path for variant,result in variants.items()} for slug,variants in recorded.items()}}
    fingerprints=[hashlib.sha256(json.dumps({'grid':c.grid,'start':c.start,'goal':c.goal},sort_keys=True).encode()).hexdigest()
                  for c in all_cases]
    metadata['evidence_scope']={'input_instances':len(all_cases),'distinct_grid_endpoint_inputs':len(set(fingerprints)),
                                'statewise_checked_inputs':len(checks)//len(VARIANTS),
                                'statewise_heuristic_configurations':len(checks),
                                'duplicate_reason':'scaling factor 1 repeats PDF sample; deterministic bands with obstacle_probability=0 repeat across seeds',
                                'correctness':'all inputs checked for optimal cost for safe variants; statewise conditions checked on structure/scaling/gap suites',
                                'scaling_invariance':'all safe variants preserve path and search counters under positive integer scaling; cost scales'}
    (output/'reproducibility.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    outcomes={'counts':{'inputs':len(all_cases),'configurations':len(rows),'safe_configurations':sum(r['safe_variant'] for r in rows),
                        'distinct_inputs':len(set(fingerprints)),'statewise_checked_inputs':len(checks)//len(VARIANTS),
                        'statewise_heuristic_configurations':len(checks),
                        'positive_scale_maps':sum(c.suite=='scale' for c in all_cases),'zero_extension_maps':sum(c.zero_extension for c in all_cases)},
              'sample':{'cost':sample_result.cost,'path':sample_result.path,'ledger':ledger,
                        'variants':[r for r in rows if r['case']=='sample']},
              'structures':[r for r in rows if r['suite']=='structure'],'bfs_vs_cost':comparisons,
              'same_histogram_comparison':{'cases':['horizontal_bands','scattered_weights'],
                                            'same_dimensions_and_endpoints':True,'same_weight_frequencies':True},
              'weight_scaling':scaling,'high_cost_gap':gaps,'analysis_optimality':by_variant,
              'heuristic_checks':checks,'safe_summary':summarize(rows),
              'all_safe_optimal':all(r['optimal'] for r in rows if r['safe_variant']),
              'positive_raw_M_optimal':all(r['optimal'] for r in rows if not r['zero_extension'] and r['variant']=='raw_M'),
              'all_paths_match_reported_cost':True,'all_safe_heuristic_checks_passed':True,
              'safe_scaling_invariance_verified':True}
    (output/'outcomes.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"[3/3] 完成 {len(all_cases)} 个输入 × {len(VARIANTS)} 设置 = {len(rows)} 配置；安全版本均通过最小费用校验。",flush=True)
    print('PDF 样例费用：',sample_result.cost,'；路径步数：',len(sample_result.path)-1,flush=True)
    print('结果目录：',output.resolve(),flush=True)
    return outcomes


def positive(value):
    parsed=int(value)
    if parsed<1:raise argparse.ArgumentTypeError('必须是正整数')
    return parsed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=positive,default=10)
    parser.add_argument('--repeats',type=positive,default=3)
    parser.add_argument('--batch',type=positive,default=3)
    parser.add_argument('--seed',type=int,default=20261028)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    run(args.output_dir,args.seeds,args.repeats,args.batch,args.seed)


if __name__=='__main__':
    main()
