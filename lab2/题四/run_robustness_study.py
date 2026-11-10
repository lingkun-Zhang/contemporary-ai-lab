"""Q4: paired search and depth-decision evidence on thirty fixed new maps.

Generation conditions concern connectivity only.  The same thirty maps are also
reused by the lazy-successor study; that reuse is not another independent sample.
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
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from treasure_game import Problem, build_distance_matrix, game_search
from reference_tools import dijkstra_matrix, history_ledger, root_values, validate_path


BASE=Path(__file__).resolve().parent
FAMILY_SEEDS={
    'open_weighted':tuple(range(202611101,202611111)),
    'detour':tuple(range(202611201,202611211)),
    'random_obstacles':tuple(range(202611301,202611311)),
}
LABELS=('A','B','T1','T2','T3','T4')
VALUES=(9,6,7,7)
ORDERS=('label','reverse','gain')
TIMING_SEED=202611140


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(path,records):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
def quantiles(values):
    q=statistics.quantiles(values,n=4,method='inclusive') if len(values)>1 else values*3
    return statistics.median(values),q[0],q[2]


def connected_components(grid):
    """Deterministic structural traversal; costs never affect acceptance."""
    unseen={(r,c) for r,row in enumerate(grid) for c,fee in enumerate(row) if fee is not None}
    components=[]
    while unseen:
        start=min(unseen);unseen.remove(start);stack=[start];component={start}
        while stack:
            r,c=stack.pop()
            for point in ((r-1,c),(r,c-1),(r,c+1),(r+1,c)):
                if point in unseen:
                    unseen.remove(point);component.add(point);stack.append(point)
        components.append(component)
    return components


def generate_case(family,seed):
    if family not in FAMILY_SEEDS:raise ValueError('unknown map family')
    rng=random.Random(seed)
    for attempt in range(1,1001):
        size=7 if family=='open_weighted' else 9
        if family=='open_weighted':
            grid=[[rng.randint(1,9) for _ in range(size)] for _ in range(size)]
            gap=None
        elif family=='detour':
            grid=[[rng.choice((1,1,1,2,4,8)) for _ in range(size)] for _ in range(size)]
            middle=size//2;gap=rng.randrange(size)
            for r in range(size):
                for c in range(size):
                    if c==middle:
                        if r!=gap:grid[r][c]=None
                    elif rng.random()<.07:grid[r][c]=None
        else:
            grid=[[None if rng.random()<.23 else rng.choice((1,1,1,2,4,8)) for _ in range(size)] for _ in range(size)]
            gap=None
        components=connected_components(grid)
        largest=max(components,key=len) if components else set()
        if len(largest)<20:continue
        if family=='detour':
            left=sorted(p for p in largest if p[1]<size//2)
            right=sorted(p for p in largest if p[1]>size//2)
            if len(left)<3 or len(right)<3:continue
            a,b=rng.choice(left),rng.choice(right)
            points=(a,b,*rng.sample(sorted(largest-{a,b}),4))
        else:
            points=tuple(rng.sample(sorted(largest),6))
        before=[row[:] for row in grid]
        for r,c in points:grid[r][c]=1
        problem=Problem(grid,LABELS,points,VALUES)
        specification=dict(case=f'{family}_s{seed}',family=family,seed=seed,attempts=attempt,
                           grid=problem.grid,points=problem.points,labels=problem.labels,values=problem.values,
                           ordinary_grid_before_named_normalization=before,
                           passable_cells=sum(fee is not None for row in grid for fee in row),
                           largest_component_size=len(largest),component_count=len(components),
                           detour_gap_row=gap,endpoint_condition='All six endpoints in the largest component (size>=20). Detour also puts A left and B right.',
                           seed_note='Integer RNG identifier, not a calendar date.')
        fingerprint=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
        specification['input_sha256']=fingerprint
        case={key:specification[key] for key in ('case','family','seed','attempts','input_sha256')}
        case['problem']=problem
        return case,specification
    raise RuntimeError('structural generation failed after1000 attempts')


def build_cases():
    """Return reusable deterministic cases and JSON-ready numeric specifications."""
    cases,specifications=[],[]
    for family,seeds in FAMILY_SEEDS.items():
        for seed in seeds:
            case,specification=generate_case(family,seed)
            cases.append(case);specifications.append(specification)
    return cases,specifications


def full_horizon_regret(problem,full_values,selected_target):
    """Evaluate the actual selected shallow action, retaining order-dependent ties."""
    action=problem.labels[2:].index(selected_target)
    optimum=max(full_values.values())
    return optimum-full_values[action],full_values[action],optimum


def register_protocol(path,protocol):
    with Path(path).open('x',encoding='utf-8') as stream:json.dump(protocol,stream,ensure_ascii=False,indent=2)
    return sha(path)


def invoke(case,config):
    started=perf_counter_ns()
    if config['phase']=='matrix':
        result=build_distance_matrix(case['problem'],config['heuristic'])
    else:
        result=game_search(case['problem'],case['prepared'],config['depth'],config['algorithm'],config['order'])
    finished=perf_counter_ns()
    if config['phase']=='matrix':
        counters=dict(astar_calls=len(result.queries),astar_expanded_nodes=sum(r.stats.expanded_nodes for r in result.queries.values()),
                      astar_generated_nodes=sum(r.stats.generated_nodes for r in result.queries.values()),
                      astar_max_frontier=max(r.stats.max_frontier for r in result.queries.values()))
    else:
        counters=dataclasses.asdict(result.stats)
    return result,counters,(finished-started)/1e6


def check_result(case,config,result,counters,stable=None):
    if config['phase']=='matrix':
        assert result.matrix==case['reference_matrix'] and counters['astar_calls']==30
    else:
        truth=case['reference_roots'][config['depth']]
        assert result.best_value==max(truth.values())
        target=case['problem'].labels[2:].index(result.best_target)
        assert truth[target]==max(truth.values())
        assert len(result.principal_variation)==min(config['depth'],len(case['problem'].values))
        assert result.principal_variation[0]==result.best_target
        ledger=history_ledger(case['problem'],case['reference_matrix'],result.principal_variation)
        assert ledger[-1]['utility']==result.best_value
        assert counters['visited_nodes']==counters['internal_nodes']+counters['leaf_evaluations']
        assert counters['generated_actions']==counters['visited_nodes']-1+counters['skipped_child_roots']
        assert counters['distance_lookups']==counters['generated_actions']
    if stable is not None:assert counters==stable


def summaries(game_rows,pairs,matrix_rows):
    """One input contributes one observation to each cross-map summary."""
    regret=[];order_summary=[];matrix_summary=[]
    families=(*FAMILY_SEEDS,'all')
    for family in families:
        chosen=[r for r in game_rows if family=='all' or r['family']==family]
        for depth in range(1,5):
            for algorithm in ('minimax','alphabeta'):
                for order in ORDERS:
                    group=[r for r in chosen if r['depth']==depth and r['algorithm']==algorithm and r['order']==order]
                    med,q1,q3=quantiles([r['full_horizon_regret'] for r in group])
                    regret.append(dict(family=family,depth=depth,algorithm=algorithm,order=order,map_count=len(group),
                                       full_optimal_first_count=sum(r['full_horizon_regret']==0 for r in group),
                                       positive_regret_count=sum(r['full_horizon_regret']>0 for r in group),
                                       median_regret=med,q1_regret=q1,q3_regret=q3,max_regret=max(r['full_horizon_regret'] for r in group)))
            for comparison_kind in ('alphabeta_vs_minimax','gain_vs_label'):
                matching=[p for p in pairs if p['comparison_kind']==comparison_kind and p['depth']==depth and (family=='all' or p['family']==family)]
                if comparison_kind=='alphabeta_vs_minimax':
                    order_options=ORDERS
                else:order_options=('gain',)
                for order in order_options:
                    group=[p for p in matching if p['target_order']==order]
                    ratios=[p['visit_ratio'] for p in group]
                    med,q1,q3=quantiles(ratios)
                    order_summary.append(dict(family=family,depth=depth,comparison_kind=comparison_kind,target_order=order,map_count=len(group),
                                              median_visit_ratio=med,q1_visit_ratio=q1,q3_visit_ratio=q3,
                                              node_fewer_maps=sum(p['visit_delta']<0 for p in group),node_equal_maps=sum(p['visit_delta']==0 for p in group),node_more_maps=sum(p['visit_delta']>0 for p in group),
                                              time_faster_maps=sum(p['median_delta_ms']<0 for p in group),time_equal_maps=sum(p['median_delta_ms']==0 for p in group),time_slower_maps=sum(p['median_delta_ms']>0 for p in group),
                                              worst_time_delta_ms=max(p['median_delta_ms'] for p in group)))
        for heuristic in ('zero','manhattan'):
            group=[r for r in matrix_rows if r['heuristic']==heuristic and (family=='all' or r['family']==family)]
            med,q1,q3=quantiles([r['median_total_ms'] for r in group])
            matrix_summary.append(dict(family=family,heuristic=heuristic,map_count=len(group),
                                       median_matrix_ms=med,q1_matrix_ms=q1,q3_matrix_ms=q3,
                                       median_expanded=statistics.median(r['astar_expanded_nodes'] for r in group),
                                       min_expanded=min(r['astar_expanded_nodes'] for r in group),max_expanded=max(r['astar_expanded_nodes'] for r in group)))
    return regret,order_summary,matrix_summary


def draw_figures(out,order_rows,regret_rows):
    def start(title,subtitle):
        return ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">',
                '<rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#293e4d">',
                f'<text x="28" y="38" font-size="25" font-weight="bold">{escape(title)}</text>',
                f'<text x="28" y="69" font-size="14" fill="#566d82">{escape(subtitle)}</text>']
    def panel(svg,y,title,series,maximum=None):
        svg.append(f'<text x="28" y="{y}" font-size="17">{escape(title)}</text>')
        left,top,w,h=105,y+28,1030,185
        high=maximum if maximum is not None else max([1]+[value for _,values in series for value in values])*1.12
        colors=['#347f9b','#c68435','#64945e','#82629a']
        for tick in range(5):
            py=top+h-tick*h/4
            svg.extend([f'<line x1="{left}" y1="{py}" x2="1145" y2="{py}" stroke="#dce5ed"/>',f'<text x="90" y="{py+5}" text-anchor="end" font-size="13">{high*tick/4:.2f}</text>'])
        for i in range(4):svg.append(f'<text x="{left+w*(i+.5)/4}" y="{top+h+26}" text-anchor="middle" font-size="14">H={i+1}</text>')
        for j,(name,values) in enumerate(series):
            points=[]
            for i,value in enumerate(values):
                px,py=left+w*(i+.5)/4,top+h-value/high*h
                points.append(f'{px},{py}');svg.append(f'<circle cx="{px}" cy="{py}" r="3.7" fill="{colors[j]}"/>')
            svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[j]}" stroke-width="2"/>')
            lx=105+j*280
            svg.extend([f'<path d="M{lx},{top+h+53}h22" stroke="{colors[j]}" stroke-width="3"/>',f'<text x="{lx+30}" y="{top+h+58}" font-size="13">{escape(name)}</text>'])
    svg=start('三十张新地图：剪枝和排序是否稳健','每地图内配对；地图才是跨输入单位，重复计时块不是独立地图。')
    pruning=[r for r in order_rows if r['family']=='all' and r['comparison_kind']=='alphabeta_vs_minimax']
    panel(svg,115,'AB访问量 / 同序Minimax访问量（30地图比值的中位数）',[(order,[next(r['median_visit_ratio'] for r in pruning if r['depth']==h and r['target_order']==order) for h in range(1,5)]) for order in ORDERS],maximum=1.05)
    gain=[r for r in order_rows if r['family']=='all' and r['comparison_kind']=='gain_vs_label']
    panel(svg,455,'Gain排序较标签排序：访问更少 / 相同 / 更多的地图数',[('访问更少',[next(r['node_fewer_maps'] for r in gain if r['depth']==h) for h in range(1,5)]),('相同',[next(r['node_equal_maps'] for r in gain if r['depth']==h) for h in range(1,5)]),('更多',[next(r['node_more_maps'] for r in gain if r['depth']==h) for h in range(1,5)])],maximum=30)
    (out/'robustness_pruning.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')
    svg=start('浅层首步的完整博弈后悔值','对真实返回首步统一用同图H4精确值再评价；浅层并列首步可以有不同后悔值。')
    selected=[r for r in regret_rows if r['family']=='all' and r['algorithm']=='alphabeta']
    panel(svg,115,'后悔值中位数（完整最优值 − 所选首步的完整后续值）',[(order,[next(r['median_regret'] for r in selected if r['depth']==h and r['order']==order) for h in range(1,5)]) for order in ORDERS])
    panel(svg,455,'完整最优首步命中地图数 / 30',[(order,[next(r['full_optimal_first_count'] for r in selected if r['depth']==h and r['order']==order) for h in range(1,5)]) for order in ORDERS],maximum=30)
    (out/'depth_regret.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results_robustness')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--batch',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<3 or args.batch<1:parser.error('repeats>=3 and batch>=1 required')
    out=args.output_dir
    if out.exists() and any(out.iterdir()):out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=True)
    cases,specs=build_cases()
    for case,spec in zip(cases,specs):
        case['reference_matrix']=dijkstra_matrix(case['problem'])
        case['reference_roots']={h:root_values(case['reference_matrix'],case['problem'].values,h) for h in range(1,5)}
        spec['reference_matrix']=case['reference_matrix']
        spec['reference_roots']={str(h):{case['problem'].labels[i+2]:v for i,v in vector.items()} for h,vector in case['reference_roots'].items()}
    write_json(out/'inputs.json',specs)
    source_paths=[Path(__file__),BASE/'treasure_game.py',BASE/'reference_tools.py']
    source_hashes={p.name:sha(p) for p in source_paths}
    protocol=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),registration_phase='Before first formal A*/game solver warmup.',
                  families={family:list(seeds) for family,seeds in FAMILY_SEEDS.items()},values=list(VALUES),
                  generation='Connectivity-conditioned maps; largest passable component>=20. Named cells cost1. Detour A/B on opposite sides. No rejection based on search results.',
                  rejection_audit=[dict(case=c['case'],attempts=c['attempts']) for c in cases],
                  inputs=dict(relative_path='inputs.json',sha256=sha(out/'inputs.json'),independent_maps=30),
                  design=dict(depths=[1,2,3,4],algorithms=['minimax','alphabeta'],orders=list(ORDERS),heuristics=['zero','manhattan'],repeats=args.repeats,batch=args.batch,timing_seed=TIMING_SEED),
                  counts=dict(map_inputs=30,game_configurations=720,matrix_configurations=60,total_configurations=780,timing_batches=780*args.repeats,timed_top_level_calls=780*args.repeats*args.batch),
                  regret_rule='Use the actual returned first target at H; regret=max(H4 root action values)-H4 value of that action. Do not merge shallow ties or evaluate with shallow U.',
                  timing_boundary='Game phase uses a previously computed verified A* matrix, excludes matrix preparation, includes eager candidates/sort/full recursion. Matrix phase includes fresh complete30 A* queries. Validation/oracle/path checks/trace/output excluded. Matrix and game medians are separate and must not be added as an end-to-end measurement.',
                  timing_data='timing_calls.csv retains every timed top-level call. timing_batches.csv is the mean per-call time of each batch; per-configuration medians/IQR and paired differences use those repeat-block batch means, not individual calls as independent maps.',
                  reuse_boundary='These same30 maps are deliberately reused by the lazy-successor study. They are not two independent samples.',
                  interpretation='Three intentionally different map families, paired methods within every instance; family differences are not a single-factor causal effect. Repeated timing IQR is descriptive, not confidence or independent input count.',
                  source_sha256=source_hashes)
    protocol_hash=register_protocol(out/'protocol.json',protocol)
    configs=[];stable=[];first=[];path_checks=0
    for case in cases:
        for heuristic in ('zero','manhattan'):
            config=dict(phase='matrix',case=case['case'],family=case['family'],heuristic=heuristic,depth=0,algorithm='',order='')
            result,counters,_=invoke(case,config);check_result(case,config,result,counters)
            for (s,t),path in result.queries.items():validate_path(case['problem'],s,t,path,case['reference_matrix'][s][t]);path_checks+=1
            if heuristic=='manhattan':case['prepared']=result
            configs.append((case,config));stable.append(counters);first.append(result)
    for case in cases:
        for depth in range(1,5):
            for algorithm in ('minimax','alphabeta'):
                for order in ORDERS:
                    config=dict(phase='game',case=case['case'],family=case['family'],heuristic='manhattan',depth=depth,algorithm=algorithm,order=order)
                    result,counters,_=invoke(case,config);check_result(case,config,result,counters)
                    configs.append((case,config));stable.append(counters);first.append(result)
    assert len(configs)==780 and path_checks==1800
    raw=[];raw_calls=[];rng=random.Random(TIMING_SEED)
    for repeat in range(args.repeats):
        order=list(enumerate(configs));rng.shuffle(order)
        for index,(case,config) in order:
            elapsed=[]
            for call in range(args.batch):
                result,counters,ms=invoke(case,config);elapsed.append(ms)
                raw_calls.append(dict(config=index,case=case['case'],family=case['family'],phase=config['phase'],repeat=repeat,call=call,total_ms=ms))
                check_result(case,config,result,counters,stable[index])
                if config['phase']=='game':
                    assert result.best_target==first[index].best_target
                    assert result.principal_variation==first[index].principal_variation
            raw.append(dict(config=index,case=case['case'],family=case['family'],phase=config['phase'],repeat=repeat,batch=args.batch,total_ms=statistics.mean(elapsed)))
        print(f'Robustness block {repeat+1}/{args.repeats} verified',flush=True)
    assert sha(out/'protocol.json')==protocol_hash and all(sha(p)==source_hashes[p.name] for p in source_paths)
    rows=[];all_fields=sorted({field for counters in stable for field in counters})
    for index,(case,config) in enumerate(configs):
        result=first[index];median,q1,q3=quantiles([r['total_ms'] for r in raw if r['config']==index])
        if config['phase']=='game':
            regret,selected_full,full_optimum=full_horizon_regret(case['problem'],case['reference_roots'][4],result.best_target)
            truth=case['reference_roots'][config['depth']]
            best_value=result.best_value;target=result.best_target
            optimal='|'.join(case['problem'].labels[i+2] for i,v in truth.items() if v==best_value)
        else:regret=selected_full=full_optimum=best_value='';target=optimal=''
        rows.append(dict(config=index,**config,seed=case['seed'],attempts=case['attempts'],input_sha256=case['input_sha256'],
                         best_target=target,best_value=best_value,optimal_targets=optimal,
                         principal_variation='|'.join(result.principal_variation) if config['phase']=='game' else '',
                         full_horizon_regret=regret,full_horizon_value_of_selected=selected_full,full_optimal_value=full_optimum,
                         **{field:stable[index].get(field,0) for field in all_fields},
                         median_total_ms=median,q1_total_ms=q1,q3_total_ms=q3))
    lookup={(r['case'],r['phase'],r['depth'],r['algorithm'],r['order'],r['heuristic']):r for r in rows}
    time_lookup={(r['config'],r['repeat']):r['total_ms'] for r in raw};pairs=[]
    for target in rows:
        kind=None
        if target['phase']=='matrix' and target['heuristic']=='manhattan':
            reference=lookup[target['case'],'matrix',0,'','','zero'];kind='manhattan_vs_zero';work_field='astar_expanded_nodes'
        elif target['phase']=='game' and target['algorithm']=='alphabeta':
            reference=lookup[target['case'],'game',target['depth'],'minimax',target['order'],'manhattan'];kind='alphabeta_vs_minimax';work_field='visited_nodes'
        else:continue
        candidates=[(kind,reference,work_field)]
        if target['phase']=='game' and target['order']=='gain':
            candidates.append(('gain_vs_label',lookup[target['case'],'game',target['depth'],'alphabeta','label','manhattan'],'visited_nodes'))
        for kind,reference,work_field in candidates:
            differences=[time_lookup[target['config'],repeat]-time_lookup[reference['config'],repeat] for repeat in range(args.repeats)]
            median,q1,q3=quantiles(differences)
            pairs.append(dict(case=target['case'],family=target['family'],depth=target['depth'],comparison_kind=kind,
                              target_order=target['order'],target_config=target['config'],reference_config=reference['config'],
                              work_metric=work_field,visit_delta=target[work_field]-reference[work_field],visit_ratio=target[work_field]/reference[work_field],
                              median_delta_ms=median,q1_delta_ms=q1,q3_delta_ms=q3,
                              faster_blocks=sum(d<0 for d in differences),slower_blocks=sum(d>0 for d in differences)))
    game_rows=[r for r in rows if r['phase']=='game'];matrix_rows=[r for r in rows if r['phase']=='matrix']
    regrets,order_summary,matrix_summary=summaries(game_rows,pairs,matrix_rows)
    write_csv(out/'comparison.csv',rows);write_csv(out/'timing_batches.csv',raw);write_csv(out/'timing_calls.csv',raw_calls);write_csv(out/'paired_differences.csv',pairs)
    write_csv(out/'depth_regret_summary.csv',regrets);write_csv(out/'order_summary.csv',order_summary);write_csv(out/'matrix_summary.csv',matrix_summary)
    draw_figures(out,order_summary,regrets)
    outcomes=dict(all_distances_and1800_paths_valid=True,all_game_values_and_selected_actions_correct=True,all_repeat_counters_stable=True,
                  all_principal_variations_valid=True,all_repeat_principal_variations_stable=True,
                  all_h4_selected_actions_have_zero_regret=all(r['full_horizon_regret']==0 for r in game_rows if r['depth']==4),
                  protocol_unchanged=True,source_unchanged=True,map_inputs=30,game_configurations=720,matrix_configurations=60,
                  total_configurations=len(rows),timing_batches=len(raw),timed_top_level_calls=len(raw)*args.batch,
                  no_new_champion_selected=True)
    write_json(out/'outcomes.json',outcomes)
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                  protocol=dict(relative_path='protocol.json',sha256=protocol_hash),source_sha256=source_hashes,
                  input_sha256=sha(out/'inputs.json'),counts=protocol['counts'],design=protocol['design'],
                  timing_boundary=protocol['timing_boundary'],timing_data=protocol['timing_data'],reuse_boundary=protocol['reuse_boundary'],interpretation=protocol['interpretation'],
                  result_files_sha256={name:sha(out/name) for name in ['comparison.csv','timing_batches.csv','timing_calls.csv','paired_differences.csv','depth_regret_summary.csv','order_summary.csv','matrix_summary.csv','outcomes.json','robustness_pruning.svg','depth_regret.svg']})
    write_json(out/'reproducibility.json',metadata)
    print(f'Saved780 configurations / {len(raw)} batches / {len(raw)*args.batch} calls to {out}')
    return out


if __name__=='__main__':main()
