"""整数桶与费用单位归一化：同轨迹对照及完整调用成本（标准库）。

python -B run_bucket_study.py
所有配置预先固定，输出已存在时另建rerun目录，不覆盖历史记录。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics as st
import sys
from datetime import datetime, timezone
from heapq import heappop, heappush
from pathlib import Path
from time import perf_counter_ns

from bucket_astar import DEFAULT_MAX_BUCKET_SLOTS, queue_astar_search
from weighted_astar import astar_search

METHODS = ('original', 'heap', 'bucket', 'bucket_gcd')
TIES = ('fifo', 'low_h')
MULTIPLIERS = (1, 2, 4, 8, 16, 64, 256)
COMMON_STATS = ('expanded_nodes', 'generated_nodes', 'popped_nodes',
                'max_frontier', 'stale_pops', 'reopened_nodes', 'unique_reached')
EXTRA_STATS = ('h_evaluations', 'reduced_edges_checked', 'max_reduced_cost',
               'bucket_slots', 'empty_bucket_scans', 'max_active_buckets',
               'bucket_fallback', 'cost_quantum', 'priority_quantum',
               'bucket_priority_increment')
SOURCES = ('weighted_astar.py', 'bucket_astar.py', 'run_bucket_study.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_cases(seed):
    cases = []
    for n in (21, 31, 41):
        for family in ('uniform', 'bands', 'zero_random', 'sparse_high'):
            map_seed = seed+n*100+10000*('uniform', 'bands', 'zero_random', 'sparse_high').index(family)
            rng = random.Random(map_seed)
            grid = [[1]*n for _ in range(n)]
            if family == 'bands':
                for r in range(n):
                    for c in range(n):
                        if r == n//2 or c == n//2:
                            grid[r][c] = 12
            elif family == 'zero_random':
                grid = [['X' if rng.random() < .13 else rng.choice((0, 0, 1, 2, 7))
                         for _ in range(n)] for _ in range(n)]
            elif family == 'sparse_high':
                # 不在起终点；数值范围很大但不按实际搜索结果调整位置。
                grid[n//2][n//2] = 10000
                grid[n//3][n//2] = 10000
            start, end = (0, 0), (n-1, n-1)
            grid[0][0], grid[-1][-1] = 'S', 'E'
            cases.append(dict(case=f'{family}_{n}', family=family, size=n,
                map_seed=map_seed, start=start, end=end, grid=grid))
    assert len(cases) == 12
    assert len({json.dumps(c['grid']) for c in cases}) == len(cases)
    return cases


def scaled_grid(grid, multiplier):
    return [[value*multiplier if type(value) is int else value for value in row] for row in grid]


def oracle(grid, start, end):
    """独立Dijkstra，只依据原始网格解释进入费；不使用待测准备函数。"""
    distances, queue = {start: 0}, [(0, start)]
    while queue:
        cost, node = heappop(queue)
        if cost != distances[node]:
            continue
        if node == end:
            return cost
        r, c = node
        for other in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            nr, nc = other
            if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])) or grid[nr][nc] == 'X':
                continue
            weight = 0 if other == end or grid[nr][nc] == 'S' else grid[nr][nc]
            candidate = cost+weight
            if candidate < distances.get(other, float('inf')):
                distances[other] = candidate
                heappush(queue, (candidate, other))
    return -1


def call_solver(case, grid, method, tie, trace=False):
    start, end = tuple(case['start']), tuple(case['end'])
    if method == 'original':
        return astar_search(start, end, grid, 'row_column_tight', tie, trace)
    return queue_astar_search(start, end, grid, 'row_column_tight', tie, method, trace,
                              max_bucket_slots=DEFAULT_MAX_BUCKET_SLOTS)


def validate(grid, case, result, expected):
    assert result.cost == expected, (case['case'], result.cost, expected)
    if expected < 0:
        assert result.path == []
        return
    assert result.path[0] == tuple(case['start']) and result.path[-1] == tuple(case['end'])
    total, ledger = 0, [0]
    for rank, (r, c) in enumerate(result.path):
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != 'X'
        if rank:
            previous = result.path[rank-1]
            assert abs(previous[0]-r)+abs(previous[1]-c) == 1
            weight = 0 if (r, c) == tuple(case['end']) else grid[r][c]
            assert type(weight) is int
            ledger.append(weight)
            total += weight
    assert total == expected
    assert result.path_costs == ledger
    cumulative, value = [], 0
    for weight in ledger:
        value += weight
        cumulative.append(value)
    assert result.cumulative_costs == cumulative


def dist(values):
    quartiles = st.quantiles(values, n=4, method='inclusive')
    return dict(median=st.median(values), q1=quartiles[0], q3=quartiles[2])


def summarize(comparison, batches):
    groups, lookup = {}, {}
    for row in batches:
        key = row['case'], row['q'], row['tie'], row['method']
        groups.setdefault(key, []).append(row['total_ms'])
        lookup[key+(row['block'],)] = row['total_ms']
    summary = []
    for row in comparison:
        times = groups[(row['case'],row['q'],row['tie'],row['method'])]
        summary.append(dict(row, repeats=len(times), **{f'total_ms_{k}':v for k,v in dist(times).items()}))
    paired = []
    comparisons = (('bucket_gcd','bucket'), ('bucket_gcd','heap'), ('bucket_gcd','original'),
                   ('bucket','heap'), ('bucket','original'))
    tasks = list(dict.fromkeys((r['case'],r['family'],r['q'],r['tie']) for r in comparison))
    for case, family, q, tie in tasks:
        blocks = sorted({r['block'] for r in batches if r['case']==case and r['q']==q and r['tie']==tie})
        for candidate, reference in comparisons:
            values = [lookup[(case,q,tie,candidate,b)]-lookup[(case,q,tie,reference,b)] for b in blocks]
            paired.append(dict(case=case,family=family,q=q,tie=tie,candidate=candidate,
                reference=reference,faster_blocks=sum(v<0 for v in values),repeats=len(blocks),
                **{f'delta_{k}_ms':v for k,v in dist(values).items()}))
    return summary, paired


def aggregate(summary, paired):
    lookup = {(r['case'],r['q'],r['tie'],r['method']):r for r in summary}
    rows = []
    for family in ('uniform','bands','zero_random','sparse_high'):
        cases = list(dict.fromkeys(r['case'] for r in summary if r['family']==family))
        for q in MULTIPLIERS:
            for tie in TIES:
                for candidate, reference in (('bucket_gcd','bucket'),('bucket_gcd','heap'),
                        ('bucket_gcd','original'),('bucket','heap')):
                    ratios = [lookup[(case,q,tie,candidate)]['total_ms_median']/
                              lookup[(case,q,tie,reference)]['total_ms_median'] for case in cases]
                    differences = [r['delta_median_ms'] for r in paired if r['family']==family
                        and r['q']==q and r['tie']==tie and r['candidate']==candidate and r['reference']==reference]
                    rows.append(dict(family=family,q=q,tie=tie,candidate=candidate,reference=reference,
                        maps=len(cases),ratio_median=st.median(ratios),faster_maps=sum(v<0 for v in differences)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results_bucket')
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if args.repeats < 3 or args.batch < 1:
        parser.error('repeats>=3 and batch>=1 required')
    output = args.output
    if output.exists() and any(output.iterdir()):
        output = output/('rerun_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=True)
    cases = make_cases(args.seed)
    root = Path(__file__).resolve().parent
    protocol = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase='before oracle, solver warmup and timed calls',
        seed=args.seed,repeats=args.repeats,calls_per_batch=args.batch,cases=cases,
        multipliers=MULTIPLIERS,ties=TIES,methods=METHODS,
        source_sha256={name:sha(root/name) for name in SOURCES},
        design=dict(inputs='12 distinct base maps:4 structures x 3 sizes21/31/41; no reachability/performance filtering',
            scaled_inputs='all ordinary integer entry costs multiplied by q; S/E/X unchanged, E still free',
            candidate='raw circular f buckets and exact gcd cost-unit normalization; max32768 slots then exact heap fallback',
            matching='same heuristic row_column_tight, same tie, same insertion order; all valid pops/path/counters must agree',
            scaling_control='for each base input/tie, original heap route and valid pop nodes are q-invariant; g/h/f scale exactly by q',
            primary='paired complete-call time gcd bucket minus raw bucket and matched heap; original API is practical strong baseline',
            preparation='matched heap shares h table, gcd and full legal-edge D validation; original API does not pay new-module preparation',
            timing='serial, seeded shuffled tasks then methods within blocks; one untimed warmup per configuration; default GC',
            included='input validation, terrain preparation, all h/D/gcd checks, queue initialization, search and full path/fee ledger',
            excluded='scaled-map construction, independent Dijkstra, validation, trace, serialization, charts',
            interpretation='7 repeated blocks are not extra maps; q is input cost representation, not an algorithm optimum parameter',
            fallback='retain explicit bucket_fallback; fallback results are not pure-bucket speedups; no posthoc removal of losses'))
    with (output/'protocol.json').open('x',encoding='utf-8') as stream:
        json.dump(protocol,stream,ensure_ascii=False,indent=2)
    print('预注册协议已保存：12基础图、7费用倍率、2排序、4方法。',flush=True)
    if args.prepare_only:
        return
    problems, comparison, references = [], [], {}
    for case in cases:
        base_cost = oracle(case['grid'],tuple(case['start']),tuple(case['end']))
        base_traces = {}
        for tie in TIES:
            base = call_solver(case,case['grid'],'original',tie,True)
            validate(case['grid'],case,base,base_cost)
            base_traces[tie] = base
        for q in MULTIPLIERS:
            grid = scaled_grid(case['grid'],q)
            expected = oracle(grid,tuple(case['start']),tuple(case['end']))
            assert expected == (base_cost*q if base_cost>=0 else -1)
            for tie in TIES:
                base = base_traces[tie]
                for method in METHODS:
                    result = call_solver(case,grid,method,tie,True)
                    validate(grid,case,result,expected)
                    assert result.path == base.path
                    assert [r['node'] for r in result.trace] == [r['node'] for r in base.trace]
                    assert [(r['g'],r['h'],r['f']) for r in result.trace] == [
                        (r['g']*q,r['h']*q,r['f']*q) for r in base.trace]
                    assert all(result.stats[name]==base.stats[name] for name in COMMON_STATS)
                    key = case['case'],q,tie,method
                    references[key] = dict(cost=expected,path=result.path,stats=result.stats.copy())
                    comparison.append(dict(case=case['case'],family=case['family'],size=case['size'],q=q,
                        tie=tie,method=method,cost=expected,steps=result.steps,
                        trace_sha256=hashlib.sha256(json.dumps(result.trace,sort_keys=True).encode()).hexdigest(),
                        **{name:result.stats.get(name,0) for name in COMMON_STATS+EXTRA_STATS}))
                    warm = call_solver(case,grid,method,tie)
                    validate(grid,case,warm,expected)
                problems.append((case,q,tie,grid))
        print('已校验/预热 '+case['case'],flush=True)
    write_csv(output/'comparison.csv',comparison)
    calls, batches, rng = [], [], random.Random(args.seed+700000)
    for block in range(args.repeats):
        rng.shuffle(problems)
        for case,q,tie,grid in problems:
            methods = list(METHODS);rng.shuffle(methods)
            for method in methods:
                elapsed = []
                reference = references[(case['case'],q,tie,method)]
                for call in range(args.batch):
                    begin = perf_counter_ns()
                    result = call_solver(case,grid,method,tie)
                    end = perf_counter_ns()
                    ms = (end-begin)/1e6
                    validate(grid,case,result,reference['cost'])
                    assert result.path == reference['path'] and result.stats == reference['stats']
                    elapsed.append(ms)
                    calls.append(dict(case=case['case'],family=case['family'],q=q,tie=tie,
                        method=method,block=block,call=call,total_ms=ms))
                batches.append(dict(case=case['case'],family=case['family'],q=q,tie=tie,
                    method=method,block=block,total_ms=st.mean(elapsed)))
        write_csv(output/'timing_calls.csv',calls)
        write_csv(output/'timing_batches.csv',batches)
        print(f'正式计时块 {block+1}/{args.repeats} 完成',flush=True)
    summary, paired = summarize(comparison,batches)
    grouped = aggregate(summary,paired)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'paired_deltas.csv',paired)
    write_csv(output/'aggregate_summary.csv',grouped)
    save_json(output/'analysis.json',dict(aggregate=grouped,
        representatives=[r for r in summary if r['size']==31 and r['tie']=='low_h' and r['q'] in (1,256)]))
    save_json(output/'manifest.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,platform=platform.platform(),source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(output/'protocol.json'),base_maps=12,scaled_inputs=84,
        configurations=len(comparison),timing_batches=len(batches),timed_calls=len(calls),
        all_cost_paths_ledgers_and_trajectory_invariance_validated=True,
        files_sha256={path.name:sha(path) for path in output.iterdir() if path.is_file()}))
    print('桶实验完成：最短费用、账本、同序轨迹和倍率不变性全部核验。',flush=True)


if __name__ == '__main__':
    main()
