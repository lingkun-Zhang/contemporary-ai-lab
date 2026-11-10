"""题一地标数调参：每批独立建表后完整执行同一组查询。

与旧 multi_query_cumulative 的逐查询中位数求和不同，本脚本直接计时整批。
只使用标准库；参考 BFS、轨迹核验、作图和文件写入都放在计时外。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from landmark_heuristic import build_landmark_index
from maze_astar import astar_search
from run_experiments import Case, bfs_distance, svg_start, validate_result, write_csv
from run_explorations import distance_field


DEFAULT_K = (0, 1, 2, 3, 4, 8)
CASE_ORDER = ("long_wall", "perfect_maze", "open")
COLORS = {"long_wall": "#327c99", "perfect_maze": "#c48136", "open": "#69915c"}


def load_legacy_reference(path=None):
    path = Path(path) if path else Path(__file__).resolve().parent/'results'/'exploration'/'reproducibility.json'
    return path, json.loads(path.read_text(encoding='utf-8'))


def selected_cases(legacy, names=CASE_ORDER):
    available = {item['case']: Case(item['case'],item['name'],[list(row) for row in item['grid']],
                                    tuple(item['start']),tuple(item['goal']))
                 for item in legacy['structure_maps']}
    return [available[name] for name in names]


def make_queries(case, max_landmarks, count, seed):
    """共同查询仅在计时外生成；选点索引不进入正式批次。"""
    index = build_landmark_index(case.grid, max_landmarks=max_landmarks)
    candidates = sorted(set(distance_field(case.grid, case.start)) - set(index.landmarks))
    if count < 1 or count > len(candidates) * (len(candidates) - 1):
        raise ValueError("不同有序查询数超过排除最高 k 地标后的候选容量")
    rng, pairs, used = random.Random(seed), [], set()
    while len(pairs) < count:
        pair = tuple(rng.sample(candidates, 2))
        if pair not in used:
            used.add(pair)
            pairs.append(pair)
    return pairs, tuple(index.landmarks)


def prepare_workload(case, max_landmarks, count, seed, legacy):
    """旧两图复用原查询；新k偶然命中端点只记录，不重新抽样。"""
    existing = next((item for item in legacy['multi_query_workloads'] if item['case']==case.slug),None)
    if existing is None:
        pairs,selected=make_queries(case,max_landmarks,count,seed)
        source='new_uniform_control_excluding_max_k_landmarks'
        original_landmarks=()
    else:
        if not 1<=count<=len(existing['queries']):
            raise ValueError('原工作负载复测的查询数必须在1到40之间；默认40完整复用')
        pairs=[tuple(tuple(point) for point in pair) for pair in existing['queries'][:count]]
        selected=tuple(build_landmark_index(case.grid,max_landmarks=max_landmarks).landmarks)
        original_landmarks=tuple(tuple(point) for point in existing['landmarks'])
        if selected[:min(4,max_landmarks)]!=original_landmarks[:min(4,max_landmarks)]:
            raise AssertionError('地标选择规则未保持原前缀，不可当作原参数低点的重复验证')
        if any(endpoint in original_landmarks for pair in pairs for endpoint in pair):
            raise AssertionError('原工作负载应已排除原4地标端点')
        source='exact_original_40_queries' if count==40 else 'prefix_of_original_40_queries'
    endpoint_hits={k:sum(endpoint in set(selected[:k]) for pair in pairs for endpoint in pair)
                   for k in range(1,max_landmarks+1)}
    new_hits=sum(endpoint in set(selected[4:]) for pair in pairs for endpoint in pair)
    return pairs,selected,dict(query_source=source,legacy_landmarks=original_landmarks,
                              max_k_landmark_endpoint_occurrences=endpoint_hits,
                              newly_added_after_four_endpoint_occurrences=new_hits)


def run_batch(grid, pairs, k):
    """独立建 k 张表，然后完整运行全部查询；返回后再校验结果。"""
    started = perf_counter_ns()
    index = build_landmark_index(grid, max_landmarks=k) if k else None
    after_prepare = perf_counter_ns() if index is not None else started
    heuristic = index.heuristic(k) if index is not None else "manhattan"
    results = [astar_search(start, goal, grid, heuristic=heuristic,
                            tie_break="low_h", record_trace=False) for start, goal in pairs]
    finished = perf_counter_ns()
    return dict(total_ms=(finished-started)/1e6,
                preprocess_ms=(after_prepare-started)/1e6,
                distance_table_entries=sum(len(t) for t in index.distance_tables) if index else 0,
                landmarks=tuple(index.landmarks) if index else (), results=results)


def validate_batch(case, pairs, oracle, batch, expected_landmarks):
    if batch["landmarks"] != expected_landmarks:
        raise AssertionError("各批次独立选择的地标必须等于最高 k 索引的相同前缀")
    if len(batch["results"]) != len(pairs):
        raise AssertionError("整批查询数量不一致")
    for (start, goal), expected, result in zip(pairs, oracle, batch["results"]):
        validate_result(Case(case.slug, case.name, case.grid, start, goal), result, expected)


def distribution(values):
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return dict(median=statistics.median(values), q1=quartiles[0], q3=quartiles[2],
                iqr=quartiles[2]-quartiles[0], min=min(values), max=max(values))


def summarize(raw, ks):
    summaries = []
    for slug in dict.fromkeys(row["case"] for row in raw):
        group = [row for row in raw if row["case"] == slug]
        baseline = {row["repeat"]: row["total_ms"] for row in group if row["k"] == 0}
        local = []
        for k in ks:
            rows = [row for row in group if row["k"] == k]
            row = dict(case=slug, case_name=rows[0]["case_name"], k=k,
                       queries=rows[0]["queries"], repeats=len(rows),
                       distance_table_entries=rows[0]["distance_table_entries"],
                       expanded_nodes_sum=rows[0]["expanded_nodes_sum"],
                       generated_nodes_sum=rows[0]["generated_nodes_sum"],
                       max_frontier_per_query_max=rows[0]["max_frontier_per_query_max"])
            for prefix, values in (("total_ms", [r["total_ms"] for r in rows]),
                                   ("preprocess_ms", [r["preprocess_ms"] for r in rows]),
                                   ("paired_speedup", [baseline[r["repeat"]]/r["total_ms"] for r in rows])):
                row.update({prefix+"_"+key: value for key, value in distribution(values).items()})
            local.append(row)
        best = min(local, key=lambda row: (row["total_ms_median"], row["k"]))["k"]
        for row in local:
            row.update(best_k_by_median=best, is_best_by_median=row["k"] == best,
                       best_is_interior=best not in (min(ks), max(ks)))
        summaries.extend(local)
    return summaries


def paired_comparisons(raw, ks):
    result = []
    for slug in dict.fromkeys(row["case"] for row in raw):
        group = [row for row in raw if row["case"] == slug]
        lookup = {(r["repeat"],r["k"]):r["total_ms"] for r in group}
        repeats = sorted({r["repeat"] for r in group})
        for left, right in zip(ks, ks[1:]):
            diffs = [lookup[(r,right)]-lookup[(r,left)] for r in repeats]
            result.append(dict(case=slug, left_k=left, right_k=right,
                               meaning="right total ms minus left total ms",
                               right_faster_batches=sum(x < 0 for x in diffs),
                               right_slower_batches=sum(x > 0 for x in diffs),
                               equal_batches=sum(x == 0 for x in diffs),
                               **{key+"_delta_ms":v for key,v in distribution(diffs).items()},
                               raw_delta_ms=json.dumps(diffs)))
    return result


def figures(output, summaries, ks):
    """一图给整批总成本与IQR，一图给同重复块配对加速比。"""
    width, panel_h, top, left, right = 1200, 250, 110, 95, 45
    cases = list(dict.fromkeys(r["case"] for r in summaries))
    for metric, name, title, label in (("total_ms", "batch_total.svg", "地标数量：真正整批端到端成本", "整批总耗时 ms（越低越好）"),
                                      ("paired_speedup", "batch_speedup.svg", "地标数量：相同重复块的配对加速比", "相对 k=0 的加速比（越高越好）")):
        height = top+panel_h*len(cases)+35
        parts = svg_start(width,height,title)
        parts.append('<text class="note" x="24" y="60">每批独立建表后执行同一组查询；点为中位数，竖线为IQR；重复批次不是独立地图。</text>')
        for pi,slug in enumerate(cases):
            group = [r for r in summaries if r["case"]==slug]
            ymax = max(1,max(r[metric+"_q3"] for r in group)*1.18)
            ytop, plot_h, plot_w = top+pi*panel_h,155,width-left-right
            xpos = lambda k: left+plot_w*(k-min(ks))/max(1,max(ks)-min(ks))
            ypos = lambda value: ytop+plot_h*(1-value/ymax)
            parts.append(f'<text class="label" x="24" y="{ytop-18}">{escape(group[0]["case_name"])}：{escape(label)}；横轴地标数 k</text>')
            for tick in range(5):
                value=ymax*tick/4;y=ypos(value)
                parts.append(f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#e0e7ec"/>')
                parts.append(f'<text class="note" x="{left-10}" y="{y+4}" text-anchor="end">{value:.2f}</text>')
            points=[]
            for row in group:
                x=xpos(row["k"]);lo=ypos(row[metric+"_q1"]);hi=ypos(row[metric+"_q3"]);y=ypos(row[metric+"_median"])
                points.append((x,y))
                parts.append(f'<line x1="{x}" y1="{lo}" x2="{x}" y2="{hi}" stroke="{COLORS[slug]}" stroke-width="2"/>')
                for cap in (lo,hi):parts.append(f'<line x1="{x-5}" y1="{cap}" x2="{x+5}" y2="{cap}" stroke="{COLORS[slug]}"/>')
                parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{COLORS[slug]}"/>')
                parts.append(f'<text class="note" x="{x}" y="{ytop+plot_h+22}" text-anchor="middle">{row["k"]}</text>')
            parts.append(f'<polyline fill="none" stroke="{COLORS[slug]}" stroke-width="2" points="'+" ".join(f"{x:.2f},{y:.2f}" for x,y in points)+'"/>')
            best=group[0]["best_k_by_median"]
            parts.append(f'<text class="note" x="{left}" y="{ytop+plot_h+48}">本批已测集合中最低总耗时 k={best}；须结合波动与其他输入，不能解释为通用最优。</text>')
        parts.append('</svg>')
        (output/name).write_text('\n'.join(parts),encoding='utf-8')


def run(output, ks=DEFAULT_K, queries=40, repeats=9, seed=20261029, case_names=CASE_ORDER,
        legacy_metadata=None):
    ks=tuple(sorted(set(ks)))
    if not ks or ks[0]!=0 or len(ks)<3 or any(type(k) is not int or k<0 for k in ks):
        raise ValueError("地标候选须含0及至少两个正整数")
    if repeats<1:raise ValueError("repeats 必须为正整数")
    output.mkdir(parents=True,exist_ok=True)
    raw, details, workloads, orders = [], [], [], {}
    legacy_path,legacy=load_legacy_reference(legacy_metadata)
    measured_queries=0
    for ci,case in enumerate(selected_cases(legacy,case_names)):
        pairs,selected,query_info = prepare_workload(case,max(ks),queries,seed+100*(ci+1),legacy)
        oracles=[bfs_distance(Case(case.slug,case.name,case.grid,start,goal)) for start,goal in pairs]
        if any(x<0 for x in oracles):raise AssertionError("查询须在同一S可达分量")
        workloads.append(dict(case=case.slug,case_name=case.name,grid=case.grid,queries=pairs,
                              expected_costs=oracles,max_k_landmarks=selected,**query_info))
        rng=random.Random(seed+10000*(ci+1))
        warmup=list(ks);rng.shuffle(warmup)
        for k in warmup:
            batch=run_batch(case.grid,pairs,k)
            validate_batch(case,pairs,oracles,batch,selected[:k])
        orders[case.slug]=dict(warmup=warmup,repeats=[])
        signatures={}
        for repeat in range(1,repeats+1):
            order=list(ks);rng.shuffle(order);orders[case.slug]["repeats"].append(order)
            for k in order:
                batch=run_batch(case.grid,pairs,k)
                validate_batch(case,pairs,oracles,batch,selected[:k])
                measured_queries+=len(pairs)
                signature=[(r.cost,r.path,r.stats) for r in batch["results"]]
                if k in signatures and signature!=signatures[k]:raise AssertionError("同k重复批次的路径或计数不稳定")
                signatures[k]=signature
                raw.append(dict(case=case.slug,case_name=case.name,repeat=repeat,k=k,queries=len(pairs),
                                total_ms=batch["total_ms"],preprocess_ms=batch["preprocess_ms"],
                                distance_table_entries=batch["distance_table_entries"],
                                expanded_nodes_sum=sum(r.stats["expanded_nodes"] for r in batch["results"]),
                                generated_nodes_sum=sum(r.stats["generated_nodes"] for r in batch["results"]),
                                max_frontier_per_query_max=max(r.stats["max_frontier"] for r in batch["results"])))
                if repeat==1:
                    for qid,((start,goal),expected,result) in enumerate(zip(pairs,oracles,batch["results"]),1):
                        details.append(dict(case=case.slug,k=k,query_id=qid,start=json.dumps(start),goal=json.dumps(goal),
                                            expected_cost=expected,returned_cost=result.cost,path=json.dumps(result.path),
                                            **result.stats))
        print(f"整批地标调参：{case.name}，{len(ks)}候选×{repeats}批×{queries}查询全部核验。",flush=True)
    summary=summarize(raw,ks);paired=paired_comparisons(raw,ks)
    write_csv(output/'batch_raw.csv',raw);write_csv(output/'batch_summary.csv',summary)
    write_csv(output/'paired_adjacent.csv',paired);write_csv(output/'query_details.csv',details)
    figures(output,summary,ks)
    outcome=dict(input_maps=len(workloads),queries_per_map=queries,candidates=ks,repeats=repeats,
                 timed_batches=len(raw),measured_query_executions=measured_queries,
                 unique_map_query_k_configurations=len(details),
                 all_shortest_costs_and_paths_verified=True,path_and_stats_stable_across_repeats=True,
                 fixed_query_sets_and_landmark_prefixes_verified=True,
                 best_by_case=[dict(case=slug,best_k=next(r['best_k_by_median'] for r in summary if r['case']==slug),
                                    best_is_interior=next(r['best_is_interior'] for r in summary if r['case']==slug))
                               for slug in case_names])
    (output/'outcomes.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
                  platform=platform.platform(),processor=platform.processor(),
                  parameters=dict(seed=seed,landmarks=ks,queries=queries,repeats=repeats,cases=case_names),
                  timing=dict(clock='perf_counter_ns',warmup_batches_per_configuration=1,
                              total='one wall-clock interval from before independent build(k) until all query results return',
                              includes='map validation, landmark selection/BFS, heuristic construction, every complete A* query and path reconstruction',
                              excludes='query selection, reference BFS, result validation, plotting and file IO',
                              gc='interpreter default',preprocess='nested wall-clock stage; total is directly measured, not a sum of medians'),
                  comparison='same queries and k-prefix landmark selection; each repetition shuffles configuration order',
                  sampling='long_wall/perfect_maze use exact saved original queries (or requested prefix), without rejecting new landmark endpoint hits; uniform is newly sampled excluding maximum-k landmarks; preparation index never reused in timed batches',
                  original_workload_source=dict(relative_path=str(legacy_path.resolve().relative_to(Path(__file__).resolve().parent))
                                                if legacy_path.resolve().is_relative_to(Path(__file__).resolve().parent) else str(legacy_path),
                                                sha256=hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                                                original_seed=legacy['parameters']['seed']),
                  evidence_scope='three fixed maps; timing repeats are not independent map samples; best is restricted to tested candidates and workload',
                  source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ('maze_astar.py','landmark_heuristic.py','run_experiments.py','run_explorations.py','run_landmark_tuning.py')},
                  workloads=workloads,timing_orders=orders)
    (output/'reproducibility.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"完成 {len(raw)} 个真正整批计时，{measured_queries} 次最短路核验；{output.resolve()}",flush=True)
    return outcome


def positive(value):
    value=int(value)
    if value<1:raise argparse.ArgumentTypeError('必须为正整数')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--landmarks',nargs='+',type=int,default=DEFAULT_K)
    parser.add_argument('--queries',type=positive,default=40)
    parser.add_argument('--repeats',type=positive,default=9)
    parser.add_argument('--seed',type=int,default=20261029)
    parser.add_argument('--cases',nargs='+',choices=CASE_ORDER,default=CASE_ORDER)
    parser.add_argument('--legacy-metadata',type=Path,default=None)
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results'/'landmark_tuning')
    args=parser.parse_args()
    try:run(args.output_dir,args.landmarks,args.queries,args.repeats,args.seed,args.cases,args.legacy_metadata)
    except ValueError as exc:parser.error(str(exc))


if __name__=='__main__':main()
