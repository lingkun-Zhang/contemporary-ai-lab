"""Q4 matched generation experiment, with two separate output contracts.

The thirty generated maps deliberately reuse the robustness study's maps.
Only lazy versus matched_eager isolates generation timing in the same kernel;
the original solver is a separate practical baseline, including its different
instrumentation overhead. All engines use identical rules and visit order.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from treasure_game import build_distance_matrix, game_search, sample_problem
from lazy_game_search import lazy_game_search
from reference_tools import dijkstra_matrix, history_ledger, root_values, validate_path
from run_experiments import RoutingProvider
from run_robustness_study import (
    FAMILY_SEEDS, build_cases, quantiles, register_protocol, sha, write_csv, write_json,
)


BASE = Path(__file__).resolve().parent
ENGINES = ('original', 'matched_eager', 'lazy')
TIMING_SEED = 202611150
TRAVERSAL_FIELDS = ('visited_nodes', 'internal_nodes', 'leaf_evaluations',
                    'cutoff_events', 'skipped_child_roots', 'bound_crossings',
                    'sorting_calls', 'sorted_actions')


def build_inputs():
    """Generate maps without invoking either formal solver; fix route subset."""
    cases, specs = build_cases()
    problem = sample_problem()
    spec = dict(case='pdf_sample', family='official', seed=0, attempts=1,
                **dataclasses.asdict(problem))
    spec['input_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    official = {key: spec[key] for key in ('case', 'family', 'seed', 'attempts', 'input_sha256')}
    official['problem'] = problem
    cases.insert(0, official); specs.insert(0, spec)
    # Selection is structural and fixed before any solution or time is known.
    route_names = {'pdf_sample'}
    route_names.update(f'{family}_s{seed}' for family, seeds in FAMILY_SEEDS.items() for seed in seeds[:2])
    assert len(cases) == 31 and len(route_names) == 7
    return cases, specs, route_names


def build_configurations(cases, route_names):
    configs = []
    for case in cases:
        for depth in (2, 4):
            for algorithm in ('minimax', 'alphabeta'):
                for order in ('label', 'reverse', 'gain'):
                    for engine in ENGINES:
                        configs.append((case, dict(phase='search_only', case=case['case'],
                            family=case['family'], depth=depth, algorithm=algorithm,
                            order=order, engine=engine, strategy='prepared', output_mode='decision_only')))
    for case in cases:
        if case['case'] not in route_names:
            continue
        for depth in (2, 4):
            for order in ('label', 'gain'):
                for strategy in ('precompute', 'on_demand', 'repeated'):
                    for output_mode in ('decision_only', 'complete_task'):
                        for engine in ENGINES:
                            configs.append((case, dict(phase='routing', case=case['case'],
                                family=case['family'], depth=depth, algorithm='alphabeta',
                                order=order, engine=engine, strategy=strategy, output_mode=output_mode)))
    assert sum(c['phase']=='search_only' for _, c in configs) == 1116
    assert sum(c['phase']=='routing' for _, c in configs) == 504
    return configs


def solve(problem, distances, config):
    arguments = (problem, distances, config['depth'], config['algorithm'], config['order'])
    if config['engine'] == 'original':
        return game_search(*arguments)
    return lazy_game_search(*arguments, generation='eager' if config['engine']=='matched_eager' else 'lazy')


def normalized_stats(result, engine):
    """Old-engine additional diagnostics are derived AFTER the timed interval.

    All formal named endpoints are connected. For the original kernel every
    candidate slot is prepared, and visited-1 actions become child states.
    These derived fields do not add instrumentation to the original timing.
    """
    stats = dataclasses.asdict(result.stats)
    if engine == 'original':
        stats.update(scheduled_candidates=stats['distance_lookups'],
                     considered_candidates=stats['distance_lookups'],
                     unreachable_candidates=stats['distance_lookups']-stats['generated_actions'],
                     materialized_actions=stats['visited_nodes']-1, unqueried_candidates=0,
                     skipped_candidates=stats['skipped_child_roots'])
    return stats


def invoke(case, config):
    started = perf_counter_ns()
    if config['phase'] == 'search_only':
        result = solve(case['problem'], case['prepared'], config)
        finished = perf_counter_ns()
        times = dict(total_ms=(finished-started)/1e6, prepare_ms=0.0,
                     search_ms=(finished-started)/1e6, completion_ms=0.0)
        routing = dict(astar_calls=0, astar_expanded_nodes=0, astar_generated_nodes=0,
                       astar_max_frontier=0, distance_requests=0, distance_cache_hits=0,
                       cached_distance_entries=0, preparation_distance_requests=0,
                       decision_distance_requests=0, completion_distance_requests=0,
                       decision_astar_calls=0, completion_astar_calls=0)
        seen, emitted = {}, None
    else:
        provider = RoutingProvider(case['problem'], config['strategy'])
        after_prepare = perf_counter_ns()
        prep_requests, prep_calls = provider.requests, provider.calls
        result = solve(case['problem'], provider, config)
        after_search = perf_counter_ns()
        decision_requests = provider.requests-prep_requests
        decision_calls = provider.calls-prep_calls
        emitted = provider.complete_matrix() if config['output_mode']=='complete_task' else None
        finished = perf_counter_ns()
        times = dict(total_ms=(finished-started)/1e6, prepare_ms=(after_prepare-started)/1e6,
                     search_ms=(after_search-after_prepare)/1e6, completion_ms=(finished-after_search)/1e6)
        routing = dict(astar_calls=provider.calls, astar_expanded_nodes=provider.expanded,
                       astar_generated_nodes=provider.generated, astar_max_frontier=provider.max_frontier,
                       distance_requests=provider.requests, distance_cache_hits=provider.hits,
                       cached_distance_entries=len(provider.cache), preparation_distance_requests=prep_requests,
                       decision_distance_requests=decision_requests,
                       completion_distance_requests=provider.requests-prep_requests-decision_requests,
                       decision_astar_calls=decision_calls,
                       completion_astar_calls=provider.calls-prep_calls-decision_calls)
        seen = provider.seen_costs
    # All conversion/validation/oracle operations happen after finished.
    return result, dict(**normalized_stats(result, config['engine']), **routing), times, seen, emitted


def signature(result):
    return (result.best_target, result.best_value, result.principal_variation,
            tuple(getattr(result.stats, name) for name in TRAVERSAL_FIELDS))


def validate(case, config, result, counters, seen, emitted, expected_signature=None, stable=None,
             check_ledger=False):
    truth = case['reference_roots'][config['depth']]
    assert result.best_value == max(truth.values())
    index = case['problem'].labels[2:].index(result.best_target)
    assert truth[index] == result.best_value
    if expected_signature is not None:
        assert signature(result) == expected_signature
    if stable is not None:
        assert counters == stable
    assert counters['visited_nodes'] == counters['internal_nodes']+counters['leaf_evaluations']
    assert counters['materialized_actions'] == counters['visited_nodes']-1
    assert counters['scheduled_candidates'] == counters['considered_candidates']+counters['unqueried_candidates']
    assert counters['considered_candidates'] == counters['distance_lookups']
    assert counters['unreachable_candidates'] == 0
    assert counters['generated_actions'] == counters['distance_lookups']
    if config['engine']=='lazy' and config['order']!='gain':
        assert counters['distance_lookups'] == counters['visited_nodes']-1
    else:
        assert counters['distance_lookups'] == counters['visited_nodes']-1+counters['skipped_child_roots']
        assert counters['unqueried_candidates'] == 0
    if config['phase']=='routing':
        assert counters['decision_distance_requests'] == counters['distance_lookups']
        assert all(cost==case['reference_matrix'][s][t] for (s,t), cost in seen.items())
        assert counters['preparation_distance_requests'] == (30 if config['strategy']=='precompute' else 0)
        assert counters['completion_distance_requests'] == (30 if config['output_mode']=='complete_task' else 0)
        if config['output_mode']=='complete_task':
            assert emitted == case['reference_matrix']
            assert counters['astar_calls'] == (counters['distance_lookups']+30 if config['strategy']=='repeated' else 30)
        else:
            assert emitted is None
            if config['strategy']=='precompute': assert counters['astar_calls']==30
            if config['strategy']=='repeated': assert counters['astar_calls']==counters['distance_lookups']
    if check_ledger:
        ledger = history_ledger(case['problem'], case['reference_matrix'], result.principal_variation)
        assert len(ledger)==config['depth'] and ledger[-1]['utility']==result.best_value


def make_pairs(rows, raw, repeats):
    """Keep one paired input as the cross-map observational unit."""
    keys = ('phase','case','depth','algorithm','order','strategy','output_mode')
    lookup = {(tuple(r[key] for key in keys), r['engine']):r for r in rows}
    timing = {(r['config'],r['repeat']):r['total_ms'] for r in raw}
    pairs=[]; paired_blocks=[]
    for target in rows:
        if target['engine']!='lazy': continue
        for reference_engine in ('matched_eager','original'):
            reference = lookup[tuple(target[key] for key in keys), reference_engine]
            differences=[]
            for repeat in range(repeats):
                delta = timing[target['config'],repeat]-timing[reference['config'],repeat]
                differences.append(delta)
                paired_blocks.append(dict(target_config=target['config'],reference_config=reference['config'],
                    reference_engine=reference_engine,repeat=repeat,total_delta_ms=delta))
            med,q1,q3=quantiles(differences)
            pairs.append(dict(phase=target['phase'],case=target['case'],family=target['family'],depth=target['depth'],
                algorithm=target['algorithm'],order=target['order'],strategy=target['strategy'],output_mode=target['output_mode'],
                reference_engine=reference_engine,target_config=target['config'],reference_config=reference['config'],
                target_lookups=target['distance_lookups'],reference_lookups=reference['distance_lookups'],
                lookup_delta=target['distance_lookups']-reference['distance_lookups'],
                lookup_ratio=target['distance_lookups']/reference['distance_lookups'],
                target_astar_calls=target['astar_calls'],reference_astar_calls=reference['astar_calls'],
                astar_delta=target['astar_calls']-reference['astar_calls'],
                visited_delta=target['visited_nodes']-reference['visited_nodes'],
                leaf_delta=target['leaf_evaluations']-reference['leaf_evaluations'],
                median_delta_ms=med,q1_delta_ms=q1,q3_delta_ms=q3,
                median_time_ratio=target['median_total_ms']/reference['median_total_ms'],
                faster_blocks=sum(v<0 for v in differences),slower_blocks=sum(v>0 for v in differences)))
    return pairs,paired_blocks


def summarize(pairs):
    grouping=('phase','depth','algorithm','order','strategy','output_mode','reference_engine')
    grouped=defaultdict(list)
    for row in pairs:
        for family in (row['family'],'all'):
            grouped[(family,*(row[key] for key in grouping))].append(row)
    rows=[]
    for (family,*key), group in sorted(grouped.items()):
        ratio,q1,q3=quantiles([r['lookup_ratio'] for r in group])
        delta,dq1,dq3=quantiles([r['median_delta_ms'] for r in group])
        time_ratio,tq1,tq3=quantiles([r['median_time_ratio'] for r in group])
        rows.append(dict(family=family,**dict(zip(grouping,key)),map_count=len(group),
            median_lookup_ratio=ratio,q1_lookup_ratio=q1,q3_lookup_ratio=q3,
            lookup_fewer_maps=sum(r['lookup_delta']<0 for r in group),lookup_equal_maps=sum(r['lookup_delta']==0 for r in group),
            astar_fewer_maps=sum(r['astar_delta']<0 for r in group),astar_equal_maps=sum(r['astar_delta']==0 for r in group),
            median_paired_delta_ms=delta,q1_paired_delta_ms=dq1,q3_paired_delta_ms=dq3,
            median_time_ratio=time_ratio,q1_time_ratio=tq1,q3_time_ratio=tq3,
            time_faster_maps=sum(r['median_delta_ms']<0 for r in group),time_slower_maps=sum(r['median_delta_ms']>0 for r in group),
            worst_paired_delta_ms=max(r['median_delta_ms'] for r in group)))
    return rows


def draw_figures(out, rows, pairs):
    def start(title,subtitle):
        return ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">',
            '<rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#293e4d">',
            f'<text x="32" y="40" font-size="25" font-weight="bold">{escape(title)}</text>',
            f'<text x="32" y="75" font-size="14" fill="#566d82">{escape(subtitle)}</text>']
    svg=start('延迟准备被剪枝的候选：相同访问树，较少距离请求','搜索阶段使用已验证矩阵；比值为lazy / matched_eager，31输入内配对后取中位数。')
    selected=[r for r in rows if r['family']=='all' and r['phase']=='search_only' and r['reference_engine']=='matched_eager']
    left,top,width,height=100,140,1000,400
    for i in range(6):
        y=top+height-height*i/5
        svg.extend([f'<line x1="100" y1="{y}" x2="1110" y2="{y}" stroke="#dce5ed"/>',
                    f'<text x="82" y="{y+5}" text-anchor="end" font-size="14">{i/5:.1f}</text>'])
    groups=[('AB H2','alphabeta',2),('AB H4','alphabeta',4),('MM H2','minimax',2),('MM H4','minimax',4)]
    colors=['#347f9b','#c68435','#64945e']
    for i,(label,algorithm,depth) in enumerate(groups):
        center=left+width*(i+.5)/4
        svg.append(f'<text x="{center}" y="575" text-anchor="middle" font-size="17">{label}</text>')
        for j,order in enumerate(('label','reverse','gain')):
            row=next(r for r in selected if r['algorithm']==algorithm and r['depth']==depth and r['order']==order)
            x=center+(j-1)*42-17; value=row['median_lookup_ratio']; y=top+height-value*height
            svg.extend([f'<rect x="{x}" y="{y}" width="34" height="{value*height}" fill="{colors[j]}"/>',
                        f'<text x="{x+17}" y="{y-10}" text-anchor="middle" font-size="13">{value:.2f}</text>'])
    for j,order in enumerate(('label','reverse','gain')):
        x=130+j*330
        svg.extend([f'<rect x="{x}" y="615" width="22" height="15" fill="{colors[j]}"/>',
                    f'<text x="{x+31}" y="628" font-size="15">{order}</text>'])
    svg.append('<text x="32" y="680" font-size="14">MM无剪枝、gain必须预先计算排序依据：两者为请求量不变的边界对照。</text>')
    (out/'lazy_lookups.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')
    svg=start('总成本边界：查询减少是否变成A*调用减少','官方样例、H4、AB、label；lazy / matched_eager。完整任务要求所有方法输出完整6×6矩阵。')
    chosen=[r for r in pairs if r['phase']=='routing' and r['case']=='pdf_sample' and r['depth']==4 and r['order']=='label' and r['reference_engine']=='matched_eager']
    columns=[('策略',55),('输出合同',245),('距离请求',455),('A*调用',660),('配对耗时差 ms',870)]
    for text,x in columns:svg.append(f'<text x="{x}" y="135" font-size="17" font-weight="bold">{text}</text>')
    for i,row in enumerate(chosen):
        y=190+i*68
        if i%2==0:svg.append(f'<rect x="35" y="{y-27}" width="1130" height="58" fill="#f0f5f8"/>')
        entries=[row['strategy'],row['output_mode'],f"{row['target_lookups']} / {row['reference_lookups']}",
                 f"{row['target_astar_calls']} / {row['reference_astar_calls']}",f"{row['median_delta_ms']:+.4f}"]
        for value,(_,x) in zip(entries,columns):svg.append(f'<text x="{x}" y="{y}" font-size="16">{escape(str(value))}</text>')
    svg.extend(['<text x="38" y="645" font-size="14">耗时差 = lazy − matched_eager；负值较快。单图微小差异只作描述，跨图分布见summary.csv。</text>',
                '<text x="38" y="680" font-size="14">precompute / on_demand在完整任务中始终调用30次A*；不能把决策输出与完整输出混比。</text>'])
    (out/'lazy_routing_boundary.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results_lazy')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--batch',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<3 or args.batch<1: parser.error('repeats>=3 and batch>=1 required')
    out=args.output_dir
    if out.exists() and any(out.iterdir()):
        out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=True)
    cases,specs,route_names=build_inputs()
    for case,spec in zip(cases,specs):
        case['reference_matrix']=dijkstra_matrix(case['problem'])
        case['reference_roots']={h:root_values(case['reference_matrix'],case['problem'].values,h) for h in (2,4)}
        spec['reference_matrix']=case['reference_matrix']
    write_json(out/'inputs.json',specs)
    paths=[Path(__file__),BASE/'treasure_game.py',BASE/'lazy_game_search.py',BASE/'reference_tools.py',
           BASE/'run_robustness_study.py',BASE/'run_experiments.py',BASE/'samples/pdf_sample.json']
    source_hashes={str(p.relative_to(BASE)):sha(p) for p in paths}
    configs=build_configurations(cases,route_names)
    protocol=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase='Before first formal A*/game solver warmup.',
        inputs=dict(relative_path='inputs.json',sha256=sha(out/'inputs.json'),official_map=1,new_generated_maps=30,
                    generated_maps_reused_from_robustness=True),
        route_selection='Official sample plus first two fixed seeds of each family, chosen before solving any instance.',
        route_cases=sorted(route_names),source_sha256=source_hashes,
        design=dict(depths=[2,4],search_algorithms=['minimax','alphabeta'],search_orders=['label','reverse','gain'],
                    route_algorithms=['alphabeta'],route_orders=['label','gain'],engines=list(ENGINES),
                    strategies=['precompute','on_demand','repeated'],output_modes=['decision_only','complete_task'],
                    repeats=args.repeats,batch=args.batch,timing_seed=TIMING_SEED),
        counts=dict(map_inputs=31,route_map_inputs=7,search_configurations=1116,route_configurations=504,
                    total_configurations=1620,timing_batches=1620*args.repeats,timed_top_level_calls=1620*args.repeats*args.batch),
        primary_comparison='lazy minus matched_eager: same instrumented kernel, only generation timing differs. Original baseline is secondary and has different instrumentation overhead.',
        counters='All formal endpoints connected. Old-engine generation diagnostics are derived after the clock stops. generated_actions means prepared legal actions, so eager prepares skipped candidates but lazy label/reverse prepares visited candidates only. materialized_actions=visited-1 in every engine.',
        timing_boundary='Search-only excludes prepared matrix. Routing includes fresh RoutingProvider, preparation A*, game search, and if complete_task all30 ordered matrix output requests. Validation, oracle, trace, stats conversion and file output excluded. Each batch value is mean of separately timed calls.',
        contracts='decision_only emits selected first target and value; it is not a replacement for the PDF full matrix requirement. complete_task emits the same full6x6 matrix for all methods. Never pool contracts or add independently measured component medians.',
        controls='Minimax (no pruning) and gain order (requires all distances before sorting) must show unchanged lookup counts; no speed benefit is presumed.',
        interpretation='Paired block deltas use the same randomized repetition. Maps, not repetitions, are cross-input units. Descriptive IQR, no confidence/significance claims. The same30 maps appear in robustness study and are not another30 independent inputs.')
    protocol_hash=register_protocol(out/'protocol.json',protocol)
    path_checks=0
    for case in cases:
        case['prepared']=build_distance_matrix(case['problem'])
        assert case['prepared'].matrix==case['reference_matrix']
        for (s,t),result in case['prepared'].queries.items():
            validate_path(case['problem'],s,t,result,case['reference_matrix'][s][t]);path_checks+=1
    stable=[];first=[];expected={}
    for case,config in configs:
        result,counters,_,seen,emitted=invoke(case,config)
        key=(case['case'],config['depth'],config['algorithm'],config['order'])
        if key not in expected:
            assert config['engine']=='original'
            expected[key]=signature(result)
        validate(case,config,result,counters,seen,emitted,expected[key],check_ledger=True)
        stable.append(counters);first.append(result)
    raw=[];raw_calls=[];rng=random.Random(TIMING_SEED)
    for repeat in range(args.repeats):
        indices=list(range(len(configs)));rng.shuffle(indices)
        for index in indices:
            case,config=configs[index];times=[]
            key=(case['case'],config['depth'],config['algorithm'],config['order'])
            for call in range(args.batch):
                result,counters,elapsed,seen,emitted=invoke(case,config)
                times.append(elapsed)
                validate(case,config,result,counters,seen,emitted,expected[key],stable[index])
                raw_calls.append(dict(config=index,case=case['case'],family=case['family'],phase=config['phase'],
                    repeat=repeat,call=call,**elapsed))
            raw.append(dict(config=index,case=case['case'],family=case['family'],phase=config['phase'],
                repeat=repeat,batch=args.batch,**{metric:statistics.mean(t[metric] for t in times) for metric in times[0]}))
        print(f'Lazy generation block {repeat+1}/{args.repeats} verified',flush=True)
    assert path_checks==930 and sha(out/'protocol.json')==protocol_hash
    assert all(sha(p)==source_hashes[str(p.relative_to(BASE))] for p in paths)
    timing_groups=defaultdict(list)
    for r in raw:timing_groups[r['config']].append(r)
    rows=[]
    for index,(case,config) in enumerate(configs):
        distribution={}
        for metric in ('total_ms','prepare_ms','search_ms','completion_ms'):
            median,q1,q3=quantiles([r[metric] for r in timing_groups[index]])
            distribution.update({f'median_{metric}':median,f'q1_{metric}':q1,f'q3_{metric}':q3})
        rows.append(dict(config=index,**config,seed=case['seed'],input_sha256=case['input_sha256'],
            best_target=first[index].best_target,best_value=first[index].best_value,
            principal_variation='|'.join(first[index].principal_variation),**stable[index],**distribution))
    pairs,paired_blocks=make_pairs(rows,raw,args.repeats);summary=summarize(pairs)
    # Deterministic negative controls must hold across all maps, not only sample.
    assert all(p['visited_delta']==p['leaf_delta']==0 for p in pairs)
    assert all(p['lookup_delta']==0 for p in pairs if p['algorithm']=='minimax' or p['order']=='gain')
    assert all(p['astar_delta']==0 for p in pairs if p['phase']=='routing' and p['output_mode']=='complete_task' and p['strategy']!='repeated')
    write_csv(out/'comparison.csv',rows);write_csv(out/'timing_batches.csv',raw)
    write_csv(out/'timing_calls.csv',raw_calls)
    write_csv(out/'paired_differences.csv',pairs);write_csv(out/'paired_blocks.csv',paired_blocks)
    write_csv(out/'summary.csv',summary);draw_figures(out,summary,pairs)
    outcomes=dict(all930_prepared_paths_valid=True,all_selected_actions_and_values_correct=True,
        all_same_order_target_pv_and_visit_counters_identical=True,all_counters_stable=True,
        all_complete_contract_matrices_identical=True,negative_controls_unchanged_lookups=True,
        complete_cached_contract_astar_always30=True,protocol_unchanged=True,sources_unchanged=True,
        total_configurations=len(rows),timing_batches=len(raw),timed_top_level_calls=len(raw_calls))
    write_json(out/'outcomes.json',outcomes)
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
        protocol=dict(relative_path='protocol.json',sha256=protocol_hash),source_sha256=source_hashes,
        input_sha256=sha(out/'inputs.json'),counts=protocol['counts'],design=protocol['design'],
        primary_comparison=protocol['primary_comparison'],timing_boundary=protocol['timing_boundary'],
        contracts=protocol['contracts'],interpretation=protocol['interpretation'],
        result_files_sha256={name:sha(out/name) for name in ['comparison.csv','timing_batches.csv','timing_calls.csv',
            'paired_differences.csv','paired_blocks.csv','summary.csv','outcomes.json']})
    write_json(out/'reproducibility.json',metadata)
    print(f'Saved1620 configurations / {len(raw)} batches / {len(raw)*args.batch} calls to {out}')
    return out


if __name__=='__main__':main()
