"""收益增量缓存：固定4/6/8宝藏输入，完整输出成本与同内核控制。

只使用标准库。--plan只打印设计，不调用求解器或开始计时。
已有结果目录会另建rerun子目录，保留全部原始记录。
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from incremental_cache_search import CacheStats, cache_search
from reference_tools import (dijkstra_matrix, history_ledger, root_values,
                             validate_path)
from treasure_game import (Problem, build_distance_matrix, game_search,
                           sample_problem)

VARIANTS = ('original_mm', 'original_ab', 'none_mm', 'none_ab',
            'fullscore_mm', 'fullscore_ab', 'incremental_mm', 'incremental_ab')
ORDERS = ('label', 'gain')
PHASES = ('game', 'end_to_end')
COUNTS = (4, 6, 8)
FAMILIES = ('open_weighted', 'detour', 'random_obstacles')
DEFAULT_SEED = 202611120
SOURCES = ('treasure_game.py', 'reference_tools.py', 'incremental_cache_search.py',
           'run_incremental_cache_study.py', 'test_incremental_cache.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def choose_output(path):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        output = output/('rerun_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=True)
    return output


def largest_component(grid):
    unseen = {(r, c) for r, row in enumerate(grid) for c, fee in enumerate(row) if fee is not None}
    largest = set()
    while unseen:
        first = min(unseen); unseen.remove(first)
        stack, component = [first], {first}
        while stack:
            r, c = stack.pop()
            for point in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                if point in unseen:
                    unseen.remove(point); component.add(point); stack.append(point)
        if len(component) > len(largest):
            largest = component
    return largest


def make_case(n, family, seed):
    """只按连通性/点互异接收输入，不读取任何搜索结果。"""
    rng = random.Random(seed)
    size = 9 if family == 'open_weighted' else 11
    for attempt in range(1, 1001):
        grid = [[rng.choice((1,1,2,4,8)) for _ in range(size)] for _ in range(size)]
        if family == 'detour':
            gap = rng.randrange(size)
            for r in range(size):
                if r != gap:
                    grid[r][size//2] = None
        elif family == 'random_obstacles':
            for row in grid:
                for c in range(size):
                    if rng.random() < .23:
                        row[c] = None
        component = largest_component(grid)
        if len(component) < max(20, n+2):
            continue
        points = tuple(rng.sample(sorted(component), n+2))
        for r, c in points:
            grid[r][c] = 1
        values = tuple(rng.randint(1, 16) for _ in range(n))
        problem = Problem(tuple(tuple(row) for row in grid),
            ('A', 'B', *(f'T{i+1}' for i in range(n))), points, values)
        return dict(case=f'{family}_n{n}_s{seed}', family=family, treasures=n,
                    seed=seed, attempt=attempt, problem=problem)
    raise RuntimeError('无法按固定结构规则生成连通输入')


def make_cases(seed):
    cases = []
    for n in COUNTS:
        for fi, family in enumerate(FAMILIES):
            for replica in range(2):
                cases.append(make_case(n, family, seed+n*1000+fi*100+replica))
    assert len({tuple(row['problem'].grid) for row in cases}) == 18
    return cases


def variant_search(problem, matrix, depth, variant, order):
    algorithm = 'minimax' if variant.endswith('_mm') else 'alphabeta'
    if variant.startswith('original_'):
        return game_search(problem, matrix, depth, algorithm, order)
    mode = variant.rsplit('_', 1)[0]
    return cache_search(problem, matrix, depth, algorithm, order, mode)


def validate_result(problem, matrix, depth, result, exact_roots):
    expected = max(exact_roots.values()) if exact_roots else 0
    assert result.best_value == expected
    best = {problem.labels[index+2] for index, value in exact_roots.items() if value == expected}
    assert result.best_target in best if best else result.best_target is None
    ledger = history_ledger(problem, matrix, result.principal_variation)
    assert len(ledger) == min(depth, len(problem.values))
    assert ledger[-1]['utility'] == expected if ledger else expected == 0
    assert result.best_target == (result.principal_variation[0] if ledger else None)
    return ledger


def check_distances(problem, matrix, exact):
    assert matrix.matrix == exact
    expected_calls = (len(problem.values)+2)*(len(problem.values)+1)
    assert len(matrix.queries) == expected_calls
    for (a, b), result in matrix.queries.items():
        validate_path(problem, a, b, result, exact[a][b])
        for r, c in result.path:
            assert 0 <= r < len(problem.grid) and 0 <= c < len(problem.grid[0])


def run_task(case, matrix, variant, order, phase):
    problem, depth = case['problem'], case['treasures']
    begin = perf_counter_ns()
    if phase == 'end_to_end':
        current = build_distance_matrix(problem)
    else:
        current = matrix
    built = perf_counter_ns()
    result = variant_search(problem, current, depth, variant, order)
    end = perf_counter_ns()
    return result, current, dict(build_ms=(built-begin)/1e6 if phase == 'end_to_end' else 0,
         game_ms=(end-built)/1e6, total_ms=(end-begin)/1e6,
         astar_calls=len(current.queries) if phase == 'end_to_end' else 0)


def stats_row(case, variant, order, result):
    fields = [field.name for field in dataclasses.fields(CacheStats)]
    stats = {field: getattr(result.stats, field, 0) for field in fields}
    return dict(case=case['case'], family=case['family'], treasures=case['treasures'],
                variant=variant, order=order, best_target=result.best_target,
                best_value=result.best_value, pv='>'.join(result.principal_variation), **stats)


def design(args, cases):
    base = Path(__file__).resolve().parent
    return dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase='before any solver/oracle warmup or timed call',
        source_sha256={name:sha(base/name) for name in SOURCES}, seed=args.seed,
        repeats=args.repeats, variants=VARIANTS, orders=ORDERS, phases=PHASES,
        independent_inputs=18, configurations=18*len(VARIANTS)*len(ORDERS)*len(PHASES),
        timed_top_calls=18*len(VARIANTS)*len(ORDERS)*len(PHASES)*args.repeats,
        rules={
            'state_model':'integer fees/values; formal utility U=scoreA-scoreB, exactly unchanged at cutoff; one action per depth',
            'generation':'18 new distinct maps, 3 families x 2 seeds x 4/6/8 treasures, structural connectivity only; depths equal treasure count',
            'four_treasures':'designated no-nonterminal-reuse negative control; official H2/H4 separately checked outside timing',
            'key':'fullscore stores absolute F and includes both scoreA and scoreB; incremental omits both and stores F-U; both contain exact depth, turn, both positions, mask',
            'terminal':'terminal U returned directly before cache lookup; no terminal entries stored in any cache mode',
            'bounds':'exact/lower/upper; relative value plus current U before bound comparison; unusable bounds continue with original window, no narrowing',
            'cache_lifetime':'new table inside every top-level call; no cross-map/cross-method/cross-call reuse',
            'primary':'incremental_ab vs fullscore_ab and none_ab, same kernel/order/map; original_ab is practical baseline',
            'game_time':'precomputed verified matrix outside timer; includes validation, table construction/probes, search, sorting, and PV',
            'end_to_end':'single full interval includes fresh (n+2)(n+1) ordered A* calls for complete matrix plus game result/PV; no symmetry shortcut',
            'oracle':'independent Dijkstra all distances, complete nonmemoized enumeration all root actions, independent PV ledger; outside timing',
            'warmup':'one game and end_to_end call per variant/order/input, discarded',
            'timing':'7 single-call blocks by default, seeded shuffle configurations and phase order, serial; default interpreter GC',
            'statistical_unit':'18 independent map/point/value inputs; blocks are timing repeats, not new inputs; retain every loss and count',
            'inference':'structural repeated states do not prove AB visits or elapsed speedup; full paths and full matrices have same output contract'},
        cases=[dict(case=c['case'], family=c['family'], treasures=c['treasures'],
            seed=c['seed'], attempt=c['attempt'], problem=dataclasses.asdict(c['problem'])) for c in cases])


def describe(values):
    q = statistics.quantiles(values, n=4, method='inclusive')
    return statistics.median(values), q[0], q[2]


def summarize(raw):
    groups, lookup = {}, {}
    for row in raw:
        key = row['case'], row['order'], row['phase'], row['variant']
        groups.setdefault(key, []).append(row)
        lookup[(*key, row['block'])] = row
    summary = []
    for (case, order, phase, variant), rows in groups.items():
        first = rows[0]
        row = dict(case=case, family=first['family'], treasures=first['treasures'],
                   order=order, phase=phase, variant=variant, repeats=len(rows))
        for metric in ('build_ms', 'game_ms', 'total_ms'):
            for name, value in zip(('median','q1','q3'), describe([r[metric] for r in rows])):
                row[metric+'_'+name] = value
        row['astar_calls'] = first['astar_calls']
        summary.append(row)
    comparisons = (('incremental_ab','fullscore_ab'), ('incremental_ab','none_ab'),
                   ('incremental_ab','original_ab'), ('fullscore_ab','none_ab'),
                   ('incremental_mm','fullscore_mm'), ('incremental_mm','none_mm'),
                   ('incremental_mm','original_mm'), ('incremental_ab','incremental_mm'))
    pairs = []
    for case, order, phase in dict.fromkeys(key[:3] for key in groups):
        sample = groups[(case,order,phase,VARIANTS[0])][0]
        blocks = sorted({r['block'] for r in groups[(case,order,phase,VARIANTS[0])]})
        for candidate, reference in comparisons:
            delta = [lookup[(case,order,phase,candidate,b)]['total_ms']-
                     lookup[(case,order,phase,reference,b)]['total_ms'] for b in blocks]
            med,q1,q3 = describe(delta)
            pairs.append(dict(case=case, family=sample['family'], treasures=sample['treasures'],
                order=order,phase=phase,candidate=candidate,reference=reference,
                delta_median_ms=med,delta_q1_ms=q1,delta_q3_ms=q3,
                faster_blocks=sum(v<0 for v in delta),repeats=len(delta)))
    return summary,pairs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results_incremental_cache')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--plan', action='store_true')
    args=parser.parse_args()
    if args.repeats<3:
        parser.error('at least 3 timing blocks')
    cases=make_cases(args.seed)
    protocol=design(args,cases)
    if args.plan:
        print(json.dumps({k:protocol[k] for k in ('independent_inputs','configurations','timed_top_calls','repeats')},ensure_ascii=False))
        return
    output=choose_output(args.output)
    with (output/'protocol.json').open('x',encoding='utf-8') as stream:
        json.dump(protocol,stream,ensure_ascii=False,indent=2)
    print('协议已写入，开始独立距离/全部根分支/PV校验及预热。',flush=True)
    official=sample_problem();official_matrix=build_distance_matrix(official)
    official_rows=[]
    for h,target,value in ((2,'T2',-1),(4,'T3',3)):
        roots=root_values(official_matrix.matrix,official.values,h)
        for order in ORDERS:
            for variant in VARIANTS:
                result=variant_search(official,official_matrix,h,variant,order)
                validate_result(official,official_matrix.matrix,h,result,roots)
                assert (result.best_target,result.best_value)==(target,value)
                official_rows.append(dict(depth=h,order=order,variant=variant,
                    best_target=result.best_target,best_value=result.best_value,
                    pv=result.principal_variation))
    save_json(output/'official_validation.json',official_rows)
    matrices,exact_distances,exact_roots,stats,oracle_rows={},{},{},[],[]
    for case in cases:
        name,problem=case['case'],case['problem']
        exact=dijkstra_matrix(problem)
        matrix=build_distance_matrix(problem);check_distances(problem,matrix,exact)
        roots=root_values(exact,problem.values,case['treasures'])
        matrices[name],exact_distances[name],exact_roots[name]=matrix,exact,roots
        oracle_rows.append(dict(case=name, distance_matrix=exact,
            root_values={problem.labels[a+2]:v for a,v in roots.items()}))
        for order in ORDERS:
            for variant in VARIANTS:
                for phase in PHASES:
                    result,current,_=run_task(case,matrix,variant,order,phase)
                    validate_result(problem,current.matrix,case['treasures'],result,roots)
                    if phase=='end_to_end':check_distances(problem,current,exact)
                    else:stats.append(stats_row(case,variant,order,result))
        print('已校验/预热 '+name,flush=True)
    save_json(output/'oracle.json',oracle_rows)
    write_csv(output/'search_stats.csv',stats)
    raw=[];rng=random.Random(args.seed+900001)
    tasks=[(case,variant,order) for case in cases for variant in VARIANTS for order in ORDERS]
    for block in range(args.repeats):
        rng.shuffle(tasks)
        for case,variant,order in tasks:
            phases=list(PHASES);rng.shuffle(phases)
            for phase in phases:
                name=case['case']
                result,current,times=run_task(case,matrices[name],variant,order,phase)
                validate_result(case['problem'],current.matrix,case['treasures'],result,exact_roots[name])
                if phase=='end_to_end':check_distances(case['problem'],current,exact_distances[name])
                raw.append(dict(case=name,family=case['family'],treasures=case['treasures'],
                    variant=variant,order=order,phase=phase,block=block,**times))
        write_csv(output/'timings_raw.csv',raw)
        print(f'计时块 {block+1}/{args.repeats} 完成',flush=True)
    summary,pairs=summarize(raw)
    write_csv(output/'timing_summary.csv',summary)
    write_csv(output/'paired_deltas.csv',pairs)
    base=Path(__file__).resolve().parent
    save_json(output/'manifest.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={name:sha(base/name) for name in SOURCES},
        protocol_sha256=sha(output/'protocol.json'),python=sys.version,platform=platform.platform(),
        independent_inputs=len(cases),game_configurations=len(stats),timed_top_calls=len(raw),
        warmup_top_calls=len(tasks)*2,official_validated_calls=len(official_rows),
        all_distance_root_value_pv_checks_passed=True,
        full_matrix_astar_calls=sum(r['astar_calls'] for r in raw),
        sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('实验完成，全部原始计时和损失保留。',flush=True)


if __name__=='__main__':
    main()
