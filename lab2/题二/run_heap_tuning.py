"""题二策略参数实验：堆清理阈值 rho 的时间、空间与维护成本。

rho=inf 是同内核不清理；original 是原始 A* 外部同批参照。
所有有效出队序列和最优路径在计时外与原 A* 逐项核对。
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
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from heap_compaction import compacted_astar_search, validate_rho
from run_experiments import (Case, band_grid, mark, plot_lines, positive,
                             reference_dijkstra, svg_start, validate_path, write_csv)
from weighted_astar import astar_search


DEFAULT_RHOS = (1.05, 1.2, 1.5, 2.0, 4.0, 8.0, 16.0, 32.0, math.inf)
STABLE_STATS = ("expanded_nodes", "generated_nodes", "reopened_nodes", "unique_reached")


def rho_label(rho):
    if rho == math.inf:
        return "inf"
    text = str(rho)
    return text[:-2] if text.endswith(".0") else text


def parse_rhos(text):
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
        for value in values:
            validate_rho(value)
        if len(set(values)) != len(values) or math.inf not in values:
            raise ValueError("rho 不可重复，且必须包含 inf 对照")
        return tuple(sorted(values))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def tuning_cases(seed=20261029, sizes=(31, 61, 101)):
    groups = []
    for n in sizes:
        case = Case(f"bands_{n}", f"{n} 阶高代价带", mark(band_grid(n), (0, 0), (n-1, n-1)),
                    (0, 0), (n-1, n-1), suite="heap_tuning", distribution="bands")
        groups.append((case, "corrected", "low_h", "global"))
    for case_seed in (seed, seed+1):
        rng, n = random.Random(case_seed), 61
        grid = [[rng.choice((1, 1, 3, 8, 20)) for _ in range(n)] for _ in range(n)]
        case = Case(f"heterogeneous_61_{case_seed}", f"61 阶随机权重，种子 {case_seed}",
                    mark(grid, (0, 0), (60, 60)), (0, 0), (60, 60), suite="heap_tuning",
                    distribution="heterogeneous", seed=case_seed)
        groups.append((case, "corrected", "low_h", "global"))
    n = max(sizes)
    uniform = Case(f"uniform_{n}", f"{n} 阶均匀图，无过期项对照",
                   mark([[1]*n for _ in range(n)], (0, 0), (n-1, n-1)), (0, 0), (n-1, n-1),
                   suite="heap_tuning", distribution="uniform")
    tight = Case(f"tight_bands_{n}", f"{n} 阶高代价带，强启发对照",
                 mark(band_grid(n), (0, 0), (n-1, n-1)), (0, 0), (n-1, n-1),
                 suite="heap_tuning", distribution="bands_tight_control")
    groups.extend(((uniform, "corrected", "low_h", "global"),
                   (tight, "row_column_tight", "low_h", "global")))
    return groups


def verify_effective_order(result, original):
    for name in ("path", "cost", "trace", "expanded_order", "path_costs", "cumulative_costs"):
        if getattr(result, name) != getattr(original, name):
            raise AssertionError(f"堆清理改变了有效搜索字段 {name}")
    if any(result.stats[key] != original.stats[key] for key in STABLE_STATS):
        raise AssertionError("堆清理改变了有效扩展/入堆/重开/到达统计")
    if result.stats["max_frontier"] > original.stats["max_frontier"]:
        raise AssertionError("保留相同有效顺序时，清理不应增加物理堆峰值")
    extra = result.compaction_stats
    if result.stats["generated_nodes"] != result.stats["popped_nodes"] + extra["removed_stale"] + extra["final_heap_size"]:
        raise AssertionError("物理记录不守恒")


def timing_summary(samples):
    quartiles = statistics.quantiles(samples, n=4, method="inclusive") if len(samples)>1 else samples*3
    return dict(time_median_ms=statistics.median(samples), time_q1_ms=quartiles[0],
                time_q3_ms=quartiles[2], time_min_ms=min(samples), time_max_ms=max(samples),
                batch_average_ms=json.dumps(samples))


def selection_rows(rows, rhos):
    output = []
    for slug in dict.fromkeys(row["case"] for row in rows):
        group = [row for row in rows if row["case"] == slug]
        compacted = {row["rho"]: row for row in group if row["implementation"] == "compacted"}
        original = next(row for row in group if row["implementation"] == "original")
        infinite = compacted["inf"]
        best = min(compacted.values(), key=lambda row: row["time_median_ms"])
        labels = [rho_label(rho) for rho in rhos]
        finite_labels = [label for label in labels if label != "inf"]
        finite_index = finite_labels.index(best["rho"]) if best["rho"] != "inf" else None
        overlap = [row["rho"] for row in compacted.values()
                   if max(row["time_q1_ms"], best["time_q1_ms"]) <= min(row["time_q3_ms"], best["time_q3_ms"])]
        output.append(dict(case=slug, case_name=best["case_name"], best_tested_rho=best["rho"],
                           best_is_finite=best["rho"] != "inf",
                           best_at_lower_boundary=finite_index == 0,
                           best_at_upper_boundary=finite_index is None or finite_index == len(finite_labels)-1,
                           best_is_internal=finite_index is not None and 0 < finite_index < len(finite_labels)-1,
                           infinite_control_is_upper_bracket=False,
                           best_median_ms=best["time_median_ms"], best_q1_ms=best["time_q1_ms"],
                           best_q3_ms=best["time_q3_ms"], same_kernel_inf_median_ms=infinite["time_median_ms"],
                           original_median_ms=original["time_median_ms"],
                           speedup_over_same_kernel_inf=infinite["time_median_ms"]/best["time_median_ms"],
                           speedup_over_original=original["time_median_ms"]/best["time_median_ms"],
                           iqr_overlap_with_best=json.dumps(overlap),
                           interpretation="tested workload only; IQR overlap is descriptive, not a significance test"))
    return output


def draw_normalized_ranges(path, rows, rhos, slugs, title):
    """同内核 inf 中位数归一化；误差棒是批耗时 IQR，非置信区间。"""
    width, height, left, right, top, bottom = 1180, 460, 100, 40, 110, 130
    parts = svg_start(width, height, title,
                      "横轴为等距离散候选；inf=同内核不清理；误差棒为批耗时 IQR，非置信区间；越低越好。")
    labels = [rho_label(rho) for rho in rhos]
    colors = ("#397f9b", "#ce8434", "#4b9a75", "#87649c")
    series = []
    for slug in slugs:
        group = [row for row in rows if row["case"] == slug and row["implementation"] == "compacted"]
        if not group:
            continue
        baseline = next(row for row in group if row["rho"] == "inf")["time_median_ms"]
        lookup = {row["rho"]: row for row in group}
        series.append((group[0]["case_name"], [(lookup[label]["time_median_ms"]/baseline,
                       lookup[label]["time_q1_ms"]/baseline, lookup[label]["time_q3_ms"]/baseline) for label in labels]))
    ymax = max(1.1, max(high for _, values in series for _, _, high in values)*1.12)
    plot_w, plot_h = width-left-right, height-top-bottom
    xpos = lambda i: left+plot_w*(i+.5)/len(labels)
    ypos = lambda value: top+plot_h*(1-value/ymax)
    for tick in range(5):
        value = ymax*tick/4
        y = ypos(value)
        parts.extend((f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e0e7ec"/>',
                      f'<text class="note" x="{left-10}" y="{y+4}" text-anchor="end">{value:.2f}</text>'))
    parts.append(f'<text class="label" x="24" y="90">完整查询耗时 / 同组 inf 中位数；包含规范化、open 管理、扫描及 heapify</text>')
    parts.append(f'<line x1="{left}" y1="{ypos(1)}" x2="{width-right}" y2="{ypos(1)}" stroke="#718da6" stroke-dasharray="5 4"/>')
    for i, label in enumerate(labels):
        parts.append(f'<text class="note" x="{xpos(i)}" y="{top+plot_h+24}" text-anchor="middle">{escape(label)}</text>')
    for index, (name, values) in enumerate(series):
        color = colors[index % len(colors)]
        points = ' '.join(f'{xpos(i):.2f},{ypos(value):.2f}' for i,(value,_,_) in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for i, (value, low, high) in enumerate(values):
            x = xpos(i)
            parts.extend((f'<line x1="{x}" x2="{x}" y1="{ypos(low)}" y2="{ypos(high)}" stroke="{color}"/>',
                          f'<line x1="{x-4}" x2="{x+4}" y1="{ypos(low)}" y2="{ypos(low)}" stroke="{color}"/>',
                          f'<line x1="{x-4}" x2="{x+4}" y1="{ypos(high)}" y2="{ypos(high)}" stroke="{color}"/>',
                          f'<circle cx="{x}" cy="{ypos(value)}" r="3.5" fill="{color}"/>'))
        lx = 100+(index % 2)*520
        ly = top+plot_h+56+(index//2)*25
        parts.extend((f'<line x1="{lx}" x2="{lx+20}" y1="{ly-4}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>',
                      f'<text class="note" x="{lx+28}" y="{ly}">{escape(name)}</text>'))
    parts.append('</svg>')
    path.write_text('\n'.join(parts), encoding="utf-8")


def make_figures(output, rows, rhos):
    bands = [slug for slug in dict.fromkeys(row["case"] for row in rows) if slug.startswith("bands_")]
    controls = [slug for slug in dict.fromkeys(row["case"] for row in rows) if not slug.startswith("bands_")]
    if bands:
        draw_normalized_ranges(output/"rho_time_bands.svg", rows, rhos, bands, "高代价带：堆清理阈值与完整查询成本")
    if controls:
        draw_normalized_ranges(output/"rho_time_controls.svg", rows, rhos, controls, "随机地形与无过期项对照：阈值收益是否成立")
    if bands:
        slug = bands[-1]
        group = [row for row in rows if row["case"] == slug and row["implementation"] == "compacted"]
        lookup = {row["rho"]: row for row in group}
        labels = [rho_label(rho) for rho in rhos]
        plot_lines(output/"rho_heap_work.svg", f'{group[0]["case_name"]}：记录数量与清理工作', [
            dict(title="同一有效搜索顺序：物理堆峰值与实际过期弹出次数", x=labels,
                 series=[(name,[lookup[label][key] for label in labels],color)
                         for name,key,color in (("物理堆峰值","max_frontier","#397f9b"),
                                                ("过期弹出","stale_pops","#ce8434"))]),
            dict(title="清理扫描的堆记录数与实际删除的旧项数（两者不是字节内存）", x=labels,
                 series=[(name,[lookup[label][key] for label in labels],color)
                         for name,key,color in (("扫描记录","scanned_entries","#87649c"),
                                                ("删除记录","removed_stale","#4b9a75"))])],
            "阈值小可能重复扫描；阈值大可能保留旧项；扩展顺序和费用均保持，峰值包含清理前瞬间。")


def run(output, rhos=DEFAULT_RHOS, repeats=9, batch=3, seed=20261029,
        sizes=(31, 61, 101), selected_cases=None):
    output.mkdir(parents=True, exist_ok=True)
    groups = tuning_cases(seed, sizes)
    if selected_cases:
        groups = [group for group in groups if group[0].slug in selected_cases]
        if set(selected_cases) != {group[0].slug for group in groups}:
            raise ValueError("存在未知 --cases 名称")
    rows, batch_rows, orders, cases_meta, paths = [], [], {}, [], {}
    for index, (case, heuristic, tie, scope) in enumerate(groups):
        expected = reference_dijkstra(case).get(case.goal, -1)
        original = astar_search(case.start, case.goal, case.grid, heuristic, tie, True, scope)
        validate_path(case, original, expected)
        calls = {"original": lambda record=False, c=case, h=heuristic, t=tie, s=scope:
                 astar_search(c.start,c.goal,c.grid,h,t,record,s)}
        recorded = {"original": original}
        for rho in rhos:
            key = rho_label(rho)
            call = lambda record=False, r=rho, c=case, h=heuristic, t=tie, s=scope: compacted_astar_search(
                c.start,c.goal,c.grid,h,t,record,s,r)
            answer = call(True)
            validate_path(case, answer, expected)
            verify_effective_order(answer, original)
            if rho == math.inf and answer.stats != original.stats:
                raise AssertionError("inf 对照的原有统计应与原求解器全部相同")
            calls[key], recorded[key] = call, answer
        rng = random.Random(seed+index)
        warmup = list(calls); rng.shuffle(warmup)
        for key in warmup:
            calls[key]()
        samples = {key: [] for key in calls}
        case_orders = []
        for repeat in range(repeats):
            order = list(calls); rng.shuffle(order); case_orders.append(order)
            for position,key in enumerate(order):
                before = perf_counter_ns()
                for _ in range(batch):
                    timed = calls[key]()
                elapsed = (perf_counter_ns()-before)/batch/1e6
                samples[key].append(elapsed)
                if timed.cost != recorded[key].cost or timed.path != recorded[key].path:
                    raise AssertionError("计时与轨迹配置不一致")
                batch_rows.append(dict(case=case.slug, implementation="original" if key=="original" else "compacted",
                                       rho=key, repeat=repeat+1, order_position=position+1,
                                       batch=batch, batch_average_ms=elapsed))
        for key, answer in recorded.items():
            extra = answer.compaction_stats if key != "original" else dict(
                compactions=0, removed_stale=0, scanned_entries=0, trigger_checks=0,
                peak_valid_open=None, final_heap_size=answer.stats["generated_nodes"]-answer.stats["popped_nodes"],
                final_valid_open=None)
            row = dict(case=case.slug, case_name=case.name, rows=len(case.grid), cols=len(case.grid[0]),
                       heuristic=heuristic, tie_break=tie, min_scope=scope, seed=case.seed,
                       distribution=case.distribution, implementation="original" if key=="original" else "compacted",
                       rho=key, optimal_cost=expected, returned_cost=answer.cost, path_steps=answer.steps,
                       same_effective_order=True, **answer.stats, **extra, **timing_summary(samples[key]))
            rows.append(row)
        paths[case.slug] = original.path
        cases_meta.append(dict(case=case.slug,name=case.name,grid=case.grid,start=case.start,goal=case.goal,
                               heuristic=heuristic,tie_break=tie,min_scope=scope,seed=case.seed,
                               distribution=case.distribution))
        orders[case.slug] = dict(warmup=warmup,batches=case_orders)
        print(f'堆阈值实验 {index+1}/{len(groups)}：{case.slug}，费用/路径/有效出队顺序全部一致。',flush=True)
    selections = selection_rows(rows, rhos)
    write_csv(output/"comparison.csv", rows)
    write_csv(output/"timing_batches.csv", batch_rows)
    write_csv(output/"selection.csv", selections)
    make_figures(output, rows, rhos)
    fingerprints={json.dumps(dict(grid=c[0].grid,start=c[0].start,goal=c[0].goal),sort_keys=True) for c in groups}
    outcome = dict(groups=len(groups),distinct_grid_endpoint_inputs=len(fingerprints),
                   compacted_configurations=len(groups)*len(rhos),original_reference_configurations=len(groups),
                   total_configurations=len(rows),timing_batches=len(batch_rows),
                   all_costs_optimal=True,all_paths_and_effective_orders_unchanged=True,
                   all_compacted_record_conservation_passed=True,infinite_original_stats_identical=True,
                   selections=selections)
    (output/"outcomes.json").write_text(json.dumps(outcome,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    names=["weighted_astar.py","run_experiments.py","heap_compaction.py","run_heap_tuning.py"]
    range_helper=Path(__file__).with_name("summarize_timing_ranges.py")
    if range_helper.is_file():
        from summarize_timing_ranges import summarize_existing_timings
        summarize_existing_timings(output)
        names.append("summarize_timing_ranges.py")
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                  platform=platform.platform(),processor=platform.processor(),
                  parameters=dict(rhos=[rho_label(r) for r in rhos],repeats=repeats,batch=batch,seed=seed,
                                  sizes=list(sizes),cases=selected_cases),
                  policy=dict(trigger="after all relaxations of each valid expansion",
                              ratio="physical heap size / distinct valid open states; zero valid states removes all remaining stale",
                              keep="entire original (f,secondary,serial,node,g) tuple if g == best_g[node]",
                              peak="actual physical heap peak immediately after each push, before any cleanup",
                              conservation="generated = popped + removed_stale + final_heap_size"),
                  timing=dict(clock="perf_counter_ns",warmups=1,
                              includes="full solver call: validation, normalization, heuristics, open maintenance, cleanup scanning, heapify, search, backtracking and result accounting",
                              excludes="oracle, trace collection, order/path verification, charts and file IO",
                              repeats="seeded random order across rho and original implementation in each batch",
                              references="same-kernel inf includes open bookkeeping; original astar is a separate same-batch reference",
                              error_bars="batch IQR, descriptive, not confidence intervals"),
                  interpretation="best in tested candidates/workload only; boundary minimum is not an internal optimum; do not force a U curve",
                  source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names},
                  counts={k:v for k,v in outcome.items() if isinstance(v,int) and not isinstance(v,bool)},
                  cases=cases_meta,paths=paths,timing_orders=orders)
    (output/"reproducibility.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(f'完成 {len(rows)} 配置；保留全部计时批值、同内核与原求解器参照。结果：{output.resolve()}',flush=True)
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhos",type=parse_rhos,default=DEFAULT_RHOS,help="逗号分隔，须包含 inf")
    parser.add_argument("--repeats",type=positive,default=9)
    parser.add_argument("--batch",type=positive,default=3)
    parser.add_argument("--seed",type=int,default=20261029)
    parser.add_argument("--sizes",default="31,61,101",help="带状地图边长，至少5")
    parser.add_argument("--cases",help="仅选指定组合，逗号分隔slug；用于预先声明的聚焦验证")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results_heap_tuning")
    args=parser.parse_args()
    try:
        sizes=tuple(int(n) for n in args.sizes.split(","))
        if not sizes or any(n<5 for n in sizes) or len(set(sizes)) != len(sizes):
            raise ValueError("sizes 必须为不重复且不小于5的整数")
        selected=args.cases.split(",") if args.cases else None
        run(args.output_dir,args.rhos,args.repeats,args.batch,args.seed,sizes,selected)
    except ValueError as exc:
        parser.error(str(exc))


if __name__=="__main__":
    main()
