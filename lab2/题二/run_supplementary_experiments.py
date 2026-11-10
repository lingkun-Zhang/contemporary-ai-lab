"""题二补充实验：并列排序、费用比阈值、可达区域最小进入费。

所有配置均以独立 Dijkstra 核验；原始计时与完整输入随结果保存。
X 障碍、零权地形及可达区域预处理是明确标注的拓展。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from weighted_astar import SAFE_HEURISTICS, astar_search, heuristic_value, prepare_terrain
from run_experiments import (COLORS, LABELS, Case, adjacent, band_grid, independent_costs,
                             mark, path_svg, plot_lines, positive, random_cases,
                             reference_dijkstra, structure_cases, validate_path, write_csv)


TIES = ("fifo", "low_h")
HIGH_COSTS = (1, 2, 5, 10, 11, 12, 20, 40, 80)
COMPONENT_SIZES = (15, 31, 61, 101)


def reachable_nodes(case):
    """独立求从 S 可达的所有格；即使遇到 E 也不提前停止。"""
    costs = independent_costs(case)
    if costs[case.start[0]][case.start[1]] is None:
        return set()
    reached, queue = {case.start}, deque([case.start])
    while queue:
        node = queue.popleft()
        for other in adjacent(costs, node):
            if other != case.start and other not in reached:
                reached.add(other)
                queue.append(other)
    return reached


def platform_counts(result, goal, optimum):
    """只计算实际扩展，不把有效出队的终点算入最优 f 平台。"""
    expanded = [item for item in result.trace if tuple(item["node"]) != goal]
    if len(expanded) != result.stats["expanded_nodes"]:
        raise AssertionError("有效出队去掉终点后必须等于实际扩展数")
    if optimum < 0:
        return dict(expanded_f_below_optimum=None, expanded_f_equal_optimum=None,
                    expanded_f_above_optimum=None)
    counts = dict(expanded_f_below_optimum=sum(x["f"] < optimum for x in expanded),
                  expanded_f_equal_optimum=sum(x["f"] == optimum for x in expanded),
                  expanded_f_above_optimum=sum(x["f"] > optimum for x in expanded))
    if sum(counts.values()) != result.stats["expanded_nodes"] or counts["expanded_f_above_optimum"]:
        raise AssertionError("安全启发不应在最优终点之前扩展 f>C* 的格")
    return counts


def check_heuristic(case, variant, scope, reverse_distances, reached):
    """全图 m 检查全部状态；区域 m 只检查对后继封闭的 S 可达集合。"""
    prepared = prepare_terrain(case.start, case.goal, case.grid, min_scope=scope)
    domain = reached if scope == "reachable" else {
        (r, c) for r in range(prepared.rows) for c in range(prepared.cols)
        if prepared.costs[r][c] is not None}
    values = {node: heuristic_value(prepared, node, variant) for node in domain}
    over, violations, edges = 0, 0, 0
    for node, h in values.items():
        if node in reverse_distances and h > reverse_distances[node]:
            over += 1
        for other in prepared.neighbors(node):
            if other not in domain:
                raise AssertionError("区域检查集合必须对合法后继封闭")
            edges += 1
            if h > prepared.costs[other[0]][other[1]] + values[other]:
                violations += 1
    if over or violations:
        raise AssertionError(f"{case.slug}/{variant}/{scope}: 下界或一致性失败")
    if scope == "reachable":
        costs = independent_costs(case)
        expected = min((costs[r][c] for r, c in reached
                        if (r, c) not in (case.start, case.goal)), default=0)
        if prepared.component_size != len(reached) or prepared.min_step_cost != expected:
            raise AssertionError("区域大小或区域最小进入费与独立遍历不一致")
    return dict(case=case.slug, suite=case.suite, variant=variant, min_scope=scope,
                domain="S_reachable" if scope == "reachable" else "all_passable",
                checked_nodes=len(domain), checked_edges=edges,
                finite_distance_nodes=sum(n in reverse_distances for n in domain),
                overestimate_nodes=over, consistency_violation_edges=violations,
                min_step_cost=prepared.min_step_cost, component_size=prepared.component_size)


def ratio_cases():
    cases = []
    for high in HIGH_COSTS:
        grid = [[1] * 21 for _ in range(21)]
        for r in range(21):
            if r != 5:
                grid[r][10] = high
        cases.append((Case(f"ratio_{high}", f"高费 K={high}",
                           mark(grid, (10, 0), (10, 20)), (10, 0), (10, 20),
                           suite="cost_ratio", distribution="fixed_vertical_band"), high))
    return cases


def component_cases():
    cases = []
    for n in COMPONENT_SIZES:
        for isolated in (1, 0):
            grid = [row + ["X", "X"] for row in band_grid(n)]
            grid[n // 2][n + 1] = isolated
            case = Case(f"isolated_{n}_{isolated}", f"{n} 阶地形，隔离格={isolated}",
                        mark(grid, (0, 0), (n - 1, n - 1)), (0, 0), (n - 1, n - 1),
                        suite="component_minimum", distribution="isolated_pocket",
                        zero_extension=isolated == 0)
            cases.append((case, dict(base_size=n, pocket_weight=isolated,
                                      terrain_structure="bands",
                                      zero_location="isolated" if isolated == 0 else "none")))
    for n in (31, 101):
        for isolated in (1, 0):
            grid = [[1]*n+["X", "X"] for _ in range(n)]
            grid[n//2][n+1] = isolated
            case = Case(f"uniform_isolated_{n}_{isolated}", f"{n} 阶均匀地形，隔离格={isolated}",
                        mark(grid, (0, 0), (n-1, n-1)), (0, 0), (n-1, n-1),
                        suite="component_minimum", distribution="uniform_isolated_pocket",
                        zero_extension=isolated == 0)
            cases.append((case, dict(base_size=n, pocket_weight=isolated, terrain_structure="uniform",
                                      zero_location="isolated" if isolated == 0 else "none")))
    n = 31
    grid = [row + ["X", "X"] for row in band_grid(n)]
    grid[n // 2][n + 1] = 1
    grid[2][3] = 0
    cases.append((Case("reachable_zero_31", "31 阶可达零权负对照",
                       mark(grid, (0, 0), (30, 30)), (0, 0), (30, 30),
                       suite="component_minimum", distribution="reachable_zero", zero_extension=True),
                  dict(base_size=31, pocket_weight=1, terrain_structure="bands", zero_location="reachable")))
    cases.append((Case("component_unreachable", "区域最小费的无解对照",
                       [["S", 1, "X", 0, "E"], [1, 1, "X", 1, 1]],
                       (0, 0), (0, 4), suite="component_minimum",
                       distribution="unreachable", zero_extension=True),
                  dict(base_size=2, pocket_weight=None, terrain_structure="unreachable", zero_location="outside_S_component")))
    return cases


def timed_samples(calls, repeats, batch, rng):
    """每一批交错配置顺序，降低固定先后次序与系统漂移的关联。"""
    samples = {key: [] for key in calls}
    order_log = []
    warmup_order = list(calls)
    rng.shuffle(warmup_order)
    for key in warmup_order:
        calls[key]()
    for _ in range(repeats):
        order = list(calls)
        rng.shuffle(order)
        order_log.append(order)
        for key in order:
            before = perf_counter_ns()
            for _ in range(batch):
                calls[key]()
            samples[key].append((perf_counter_ns() - before) / batch / 1e6)
    return samples, dict(warmup=warmup_order, batches=order_log)


def evaluate(case, configs, repeats, batch, seed, prepare_timing=False):
    """参考求解、轨迹、条件检查不计时；完整 A* 调用包含所有预处理。"""
    oracle = reference_dijkstra(case).get(case.goal, -1)
    reverse = reference_dijkstra(case, reverse=True)
    reached = reachable_nodes(case)
    results, calls, checks = {}, {}, []
    for variant, tie, scope in configs:
        key = f"{variant}|{tie}|{scope}"
        result = astar_search(case.start, case.goal, case.grid, heuristic=variant,
                              tie_break=tie, record_trace=True, min_scope=scope)
        validate_path(case, result, oracle)
        if result.stats["reopened_nodes"]:
            raise AssertionError("一致启发不应实际重新扩展状态")
        results[key] = result
        def call(v=variant, t=tie, s=scope):
            return astar_search(case.start, case.goal, case.grid, heuristic=v,
                                tie_break=t, min_scope=s)
        calls[key] = call
    for variant, scope in dict.fromkeys((v, s) for v, _, s in configs):
        checks.append(check_heuristic(case, variant, scope, reverse, reached))
    rng = random.Random(seed)
    samples, order = timed_samples(calls, repeats, batch, rng)
    for key, call in calls.items():
        timed, recorded = call(), results[key]
        if timed.cost != recorded.cost or timed.path != recorded.path or timed.stats != recorded.stats:
            raise AssertionError("轨迹与关闭轨迹的计时结果不一致")
    prep_samples, prep_order = {}, None
    if prepare_timing:
        prep_calls = {}
        for _, _, scope in configs:
            if scope not in prep_calls:
                prep_calls[scope] = lambda s=scope: prepare_terrain(case.start, case.goal, case.grid, min_scope=s)
        prep_samples, prep_order = timed_samples(prep_calls, repeats, batch, rng)
    rows = []
    for variant, tie, scope in configs:
        key = f"{variant}|{tie}|{scope}"
        result, batches = results[key], samples[key]
        row = dict(suite=case.suite, case=case.slug, case_name=case.name,
                   rows=len(case.grid), cols=len(case.grid[0]), distribution=case.distribution,
                   seed=case.seed, obstacle_probability=case.obstacle_probability,
                   zero_extension=case.zero_extension, variant=variant, tie_break=tie, min_scope=scope,
                   reachable=oracle >= 0, oracle_cost=oracle, returned_cost=result.cost,
                   optimal=result.cost == oracle, path_steps=result.steps,
                   min_step_cost=result.min_step_cost, component_size=result.component_size,
                   independent_component_size=len(reached), **result.stats,
                   **platform_counts(result, case.goal, oracle),
                   total_time_median_ms=statistics.median(batches), total_time_min_ms=min(batches),
                   total_batch_average_ms=json.dumps(batches))
        if prepare_timing:
            prep = prep_samples[scope]
            row.update(prepare_time_median_ms=statistics.median(prep),
                       prepare_time_min_ms=min(prep), prepare_batch_average_ms=json.dumps(prep))
        rows.append(row)
    return rows, checks, results, dict(total=order, prepare=prep_order)


def paired_tie_summary(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault((row["case"], row["variant"], row["min_scope"]), {})[row["tie_break"]] = row
    summary = []
    for (_, _, _), pair in pairs.items():
        fifo, low = pair["fifo"], pair["low_h"]
        summary.append(dict(case=fifo["case"], case_name=fifo["case_name"], suite=fifo["suite"],
                            rows=fifo["rows"], cols=fifo["cols"], distribution=fifo["distribution"],
                            variant=fifo["variant"], min_scope=fifo["min_scope"], reachable=fifo["reachable"],
                            oracle_cost=fifo["oracle_cost"], fifo_expanded=fifo["expanded_nodes"],
                            low_h_expanded=low["expanded_nodes"],
                            low_h_minus_fifo_expanded=low["expanded_nodes"]-fifo["expanded_nodes"],
                            expanded_reduction_fraction=(1-low["expanded_nodes"]/fifo["expanded_nodes"])
                            if fifo["expanded_nodes"] else 0,
                            fifo_total_median_ms=fifo["total_time_median_ms"],
                            low_h_total_median_ms=low["total_time_median_ms"],
                            fifo_equal_f_expanded=fifo["expanded_f_equal_optimum"],
                            low_h_equal_f_expanded=low["expanded_f_equal_optimum"]))
    return summary


def make_figures(output, tie_rows, ratio_rows, comp_rows):
    selected = ("uniform", "bands", "horizontal_bands", "scattered_weights")
    names = [next(row["case_name"] for row in tie_rows if row["case"] == slug) for slug in selected]
    panels = []
    for tie in TIES:
        panels.append(dict(title=f"排序 {tie}：实际扩展数（不含终点），各场景同图配对", x=names,
                           series=[(LABELS[v], [next(r["expanded_nodes"] for r in tie_rows
                                   if r["case"] == c and r["variant"] == v and r["tie_break"] == tie)
                                   for c in selected], COLORS[v]) for v in SAFE_HEURISTICS]))
    plot_lines(output/"tie_break_nodes.svg", "启发函数与同分排序的交互", panels,
               "费用全部与独立最优值一致；不同场景的尺寸和结构不同，不把跨场景连线解释为单变量因果。")
    platform_series = []
    for v in ("corrected", "row_column", "row_column_tight"):
        platform_series.append((LABELS[v], [next(r[metric] for r in tie_rows
                                if r["case"] == "bands" and r["variant"] == v and r["tie_break"] == tie)
                                for tie, metric in (("fifo", "expanded_f_below_optimum"),
                                                    ("fifo", "expanded_f_equal_optimum"),
                                                    ("low_h", "expanded_f_below_optimum"),
                                                    ("low_h", "expanded_f_equal_optimum"))], COLORS[v]))
    plot_lines(output/"optimal_f_platform.svg", "同一高代价带的最优 f 平台", [dict(
        title="31×31 高代价带，C*=83；终点和过期堆项不计入扩展", x=["FIFO：f<C*", "FIFO：f=C*", "low_h：f<C*", "low_h：f=C*"],
        series=platform_series)], "更强下界减少低于最优费用的状态；同分排序决定最优 f 平台上还要搜索多少状态。")
    chosen = [next(r for r in ratio_rows if r["high_cost"] == k and r["variant"] == "row_column_tight"
                   and r["tie_break"] == "low_h") for k in HIGH_COSTS]
    plot_lines(output/"cost_ratio_threshold.svg", "固定地形位置：只改变高低费用比", [
        dict(title="离散 K 测试点等距排列；纵轴费用；固定缺口第 5 行，E 进入免费", x=HIGH_COSTS, numeric=False,
             series=[("直接穿越：18+K", [18+k for k in HIGH_COSTS], "#da9239"),
                     ("绕过缺口：29", [29]*len(HIGH_COSTS), "#718da6"),
                     ("独立验证最小费用", [r["returned_cost"] for r in chosen], "#419f96")]),
        dict(title="tight + low_h 所选路径步数；离散 K 等距排列；K=11 两类路线费用相同", x=HIGH_COSTS, numeric=False,
             series=[("返回路径步数", [r["path_steps"] for r in chosen], "#477fbd")])],
        "横轴为等距离散测试点，不代表连续比例；20 个指定高费格位置固定，K=1 时同价仍按位置统计穿越。")
    isolated = [r for r in comp_rows if r["zero_location"] == "isolated" and r["tie_break"] == "low_h"
                and r["terrain_structure"] == "bands"]
    plot_lines(output/"component_minimum.svg", "不可达零权格与区域最小进入费", [
        dict(title="横轴主体地形边长 n；纵轴扩展数；corrected + low_h", x=COMPONENT_SIZES, numeric=True,
             series=[(label, [next(r["expanded_nodes"] for r in isolated if r["base_size"] == n and r["min_scope"] == scope)
                             for n in COMPONENT_SIZES], color)
                     for scope, label, color in (("global", "全图 m（被隔离零拖成 0）", "#da9239"),
                                                 ("reachable", "起点可达区域 m", "#419f96"))]),
        dict(title="完整查询毫秒中位数，包含完整连通遍历；预处理另测且不与总时间相减", x=COMPONENT_SIZES, numeric=True,
             series=[(label, [next(r["total_time_median_ms"] for r in isolated if r["base_size"] == n and r["min_scope"] == scope)
                             for n in COMPONENT_SIZES], color)
                     for scope, label, color in (("global", "全图 m", "#da9239"), ("reachable", "可达区域 m", "#419f96"))])],
        "X 与零权是拓展；区域 m 不做无解提前返回，收益与开销均来自更强下界及其完整准备。")
    shape_series = (("bands", "global", "高费带：全图 m", "#da9239"),
                    ("bands", "reachable", "高费带：区域 m", "#419f96"),
                    ("uniform", "global", "均匀图：全图 m", "#718da6"),
                    ("uniform", "reachable", "均匀图：区域 m", "#477fbd"))
    shape_rows = [r for r in comp_rows if r["zero_location"]=="isolated" and r["tie_break"]=="low_h"]
    plot_lines(output/"component_structure_control.svg", "区域预处理的收益取决于地形结构", [
        dict(title=title, x=[31,101], numeric=True,
             series=[(label, [next(r[metric] for r in shape_rows if r["base_size"]==n
                                  and r["terrain_structure"]==shape and r["min_scope"]==scope)
                              for n in (31,101)], color)
                     for shape,scope,label,color in shape_series])
        for title,metric in (("隔离零权格固定存在；corrected + low_h 的实际 A* 扩展数", "expanded_nodes"),
                             ("完整查询毫秒中位数：含连通扫描、预处理、搜索和路径回溯", "total_time_median_ms"))],
        "同一地形的 scope 对照使用完全相同输入；均匀/带状是不同结构，节点收益须与扫描开销一起解释。")


def run(output, seeds=3, repeats=3, batch=3, seed=20261028):
    output.mkdir(parents=True, exist_ok=True)
    all_cases, checks, paths, orders = [], [], {}, {}
    tie_rows, ratio_rows, comp_rows = [], [], []
    configs = [(v, t, "global") for v in SAFE_HEURISTICS for t in TIES]
    generated = random_cases(seeds, seed)
    cases_a = structure_cases() + generated
    for index, case in enumerate(cases_a):
        case.suite = "tie_break"
        rows, checked, result_map, order = evaluate(case, configs, repeats, batch, seed+index)
        tie_rows.extend(rows); checks.extend(checked); all_cases.append(case)
        paths[case.slug] = {k: v.path for k, v in result_map.items()}; orders[case.slug] = order
        if (index+1) % 20 == 0:
            print(f"[1/3] 同分排序对照 {index+1}/{len(cases_a)}", flush=True)
    print(f"[1/3] 完成 {len(cases_a)} 输入 × 8 安全配置，最优费用和 f 平台核验通过。", flush=True)
    ratio_paths = {}
    for index, (case, high) in enumerate(ratio_cases()):
        rows, checked, result_map, order = evaluate(case, configs, repeats, batch, seed+10000+index)
        expected = min(18+high, 29)
        for row in rows:
            result = result_map[f"{row['variant']}|{row['tie_break']}|global"]
            crossings = sum(c == 10 and r != 5 for r, c in result.path)
            if row["oracle_cost"] != expected or high < 11 and result.steps != 20 or high > 11 and result.steps != 30:
                raise AssertionError("高低费用比实验不符合理论阈值")
            if high == 11 and result.steps not in (20, 30):
                raise AssertionError("阈值处最优路径应为20步直行或30步绕行")
            row.update(high_cost=high, low_cost=1, designated_band_cells=20,
                       designated_band_entries=crossings, theoretical_cost=expected,
                       direct_route_cost=18+high, detour_route_cost=29,
                       route_class="cross_band" if crossings else "through_gap")
        ratio_rows.extend(rows); checks.extend(checked); all_cases.append(case)
        paths[case.slug] = {k: v.path for k, v in result_map.items()}; orders[case.slug] = order
        if high in (10, 11, 12):
            ratio_paths[high] = (case, result_map["row_column_tight|low_h|global"])
    print("[2/3] 完成费用比参数实验，C*=min(18+K,29) 与路线阈值核验通过。", flush=True)
    component_configs = [("corrected", t, s) for s in ("global", "reachable") for t in TIES]
    for index, (case, extra) in enumerate(component_cases()):
        rows, checked, result_map, order = evaluate(case, component_configs, repeats, batch, seed+20000+index, True)
        for row in rows:
            row.update(extra)
            row.update(component_scan_nodes=row["component_size"] or 0,
                       scan_plus_search_node_visits=(row["component_size"] or 0)+row["expanded_nodes"])
            if row["min_scope"] == "reachable" and row["zero_location"] == "reachable" and row["min_step_cost"] != 0:
                raise AssertionError("实际可达零权不能被区域优化忽略")
            if case.slug == "component_unreachable" and row["expanded_nodes"] != len(reachable_nodes(case)):
                raise AssertionError("本实验不允许区域预处理提前返回无解")
        comp_rows.extend(rows); checks.extend(checked); all_cases.append(case)
        paths[case.slug] = {k: v.path for k, v in result_map.items()}; orders[case.slug] = order
    for shape, n in [("bands",n) for n in COMPONENT_SIZES]+[("uniform",n) for n in (31,101)]:
        pair = [r for r in comp_rows if r["base_size"] == n and r["terrain_structure"]==shape
                and r["zero_location"] in ("isolated","none")]
        if len({r["oracle_cost"] for r in pair}) != 1:
            raise AssertionError("改变不可达格不得改变主体路径最优费用")
        local = [r for r in pair if r["min_scope"] == "reachable"]
        for tie in TIES:
            a, b = [r for r in local if r["tie_break"] == tie]
            if a["expanded_nodes"] != b["expanded_nodes"] or a["min_step_cost"] != b["min_step_cost"]:
                raise AssertionError("不可达格权重不得影响区域下界和搜索量")
    print("[3/3] 完成区域最小费、可达零权负对照和无解对照；预处理开销已计入完整查询。", flush=True)
    all_rows = tie_rows + ratio_rows + comp_rows
    paired = paired_tie_summary(tie_rows)
    write_csv(output/"tie_break_comparison.csv", tie_rows)
    write_csv(output/"tie_break_paired.csv", paired)
    write_csv(output/"cost_ratio.csv", ratio_rows)
    write_csv(output/"component_minimum.csv", comp_rows)
    write_csv(output/"heuristic_checks.csv", checks)
    make_figures(output, tie_rows, ratio_rows, comp_rows)
    for high, (case, result) in ratio_paths.items():
        path_svg(output/f"ratio_path_{high}.svg", case, [("tight + low_h", result)],
                 f"高费 K={high}：费用 {result.cost}，步数 {result.steps}",
                 "固定地形，只改指定高费格的值；K=11 存在直行与绕行两类最优路线。")
    fingerprints = [hashlib.sha256(json.dumps(dict(grid=c.grid,start=c.start,goal=c.goal),sort_keys=True).encode()).hexdigest()
                    for c in all_cases]
    outcome = dict(counts=dict(tie_inputs=len(cases_a), tie_configurations=len(tie_rows),
                               ratio_inputs=len(HIGH_COSTS), ratio_configurations=len(ratio_rows),
                               component_inputs=len(component_cases()), component_configurations=len(comp_rows),
                               total_input_instances=len(all_cases), distinct_grid_endpoint_inputs=len(set(fingerprints)),
                               configurations=len(all_rows), heuristic_checks=len(checks)),
                   all_optimal=all(r["optimal"] for r in all_rows), all_heuristic_checks_passed=True,
                   all_platform_counts_passed=True, ratio_formula_verified=True,
                   isolated_weight_invariance_verified=True, reachable_zero_not_ignored=True,
                   no_early_unreachable_return_verified=True,
                   representative_tie_rows=[r for r in tie_rows if r["case"] in ("bands","uniform","horizontal_bands")],
                   ratio_tight_low_h=[r for r in ratio_rows if r["variant"]=="row_column_tight" and r["tie_break"]=="low_h"],
                   component_rows=comp_rows)
    (output/"outcomes.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                    platform=platform.platform(), processor=platform.processor(),
                    parameters=dict(seeds=seeds, repeats=repeats, batch=batch, seed=seed,
                                    safe_heuristics=SAFE_HEURISTICS, tie_breaks=TIES,
                                    high_costs=HIGH_COSTS, component_sizes=COMPONENT_SIZES),
                    timing=dict(clock="perf_counter_ns", warmups=1,
                                statistic="median of batch-average milliseconds",
                                total_includes="complete astar_search: validation, normalization, full connectivity traversal if requested, row/column preparation, heap search, path reconstruction",
                                total_excludes="reference algorithms, condition checks, trace collection, plotting, file IO",
                                preparation="independent prepare_terrain calls; not subtracted from total as exact search-only time",
                                order="seeded shuffle across configurations at warmup and each timing batch"),
                    rules=dict(goal_enter_cost=0, start_cost=0, incoming_start_edges="removed",
                               unreachable="component preparation never returns early; A* exhausts S component",
                               local_heuristic_domain="only states reachable from S; global rows/columns unchanged",
                               zero_and_X="explicit extensions"),
                    evidence_scope=outcome["counts"],
                    duplicates="deterministic no-obstacle band maps repeat across seeds; duplicates retained as input instances, never called independent maps",
                    source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("weighted_astar.py","run_experiments.py","run_supplementary_experiments.py")},
                    cases=[dict(case=c.slug,name=c.name,suite=c.suite,distribution=c.distribution,
                                grid=c.grid,start=c.start,goal=c.goal,seed=c.seed,
                                zero_extension=c.zero_extension,obstacle_probability=c.obstacle_probability)
                           for c in all_cases], paths=paths, timing_orders=orders)
    (output/"reproducibility.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"补充实验完成：{len(all_cases)} 输入记录，{len(all_rows)} 配置全部最优；结果 {output.resolve()}",flush=True)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=positive, default=3)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--batch", type=positive, default=3)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"results_supplementary")
    args = parser.parse_args()
    run(args.output_dir,args.seeds,args.repeats,args.batch,args.seed)


if __name__ == "__main__":
    main()
