"""Q4 controlled studies: depth, ordering, routing cost, and T3 value boundaries."""
from __future__ import annotations

import argparse
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
import csv

from treasure_game import astar_search, build_distance_matrix, game_search, sample_problem
from reference_tools import dijkstra_matrix, enumerate_game, history_ledger, root_values, validate_path

BASE=Path(__file__).resolve().parent
SEED=20261101
ORDERS=('label','reverse','gain')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [safe(v) for v in value]
    return value
def write_json(path,value):path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2),encoding='utf-8')
def distribution(values):
    q=statistics.quantiles(values,n=4,method='inclusive')
    return statistics.median(values),q[0],q[2]


class _Row:
    def __init__(self,provider,source):self.provider,self.source=provider,source
    def __len__(self):return len(self.provider.problem.points)
    def __getitem__(self,target):
        if not 0<=target<len(self):raise IndexError(target)
        return self.provider.get(self.source,target)


class RoutingProvider:
    """Identical game interface; only A* reuse/preparation changes."""
    def __init__(self,problem,strategy):
        if strategy not in ('precompute','on_demand','repeated'):raise ValueError(strategy)
        self.problem,self.strategy,self.labels=problem,strategy,problem.labels
        self.matrix=tuple(_Row(self,i) for i in range(len(problem.points)))
        self.cache={};self.seen_costs={}
        self.requests=self.hits=self.calls=self.expanded=self.generated=self.max_frontier=0
        if strategy=='precompute':self.complete_matrix()

    def get(self,source,target):
        self.requests+=1
        if source==target:return 0
        key=source,target
        if key in self.cache:
            self.hits+=1;return self.cache[key]
        result=astar_search(self.problem.grid,self.problem.points[source],self.problem.points[target])
        self.calls+=1;self.expanded+=result.stats.expanded_nodes
        self.generated+=result.stats.generated_nodes
        self.max_frontier=max(self.max_frontier,result.stats.max_frontier)
        self.seen_costs[key]=result.cost
        if self.strategy!='repeated':self.cache[key]=result.cost
        return result.cost

    def complete_matrix(self):
        return tuple(tuple(0 if i==j else self.get(i,j) for j in range(len(self.problem.points)))
                     for i in range(len(self.problem.points)))


def invoke_routing(problem,depth,algorithm,order,strategy,output_mode):
    started=perf_counter_ns()
    provider=RoutingProvider(problem,strategy)
    after_prepare=perf_counter_ns()
    prep_requests=provider.requests
    result=game_search(problem,provider,depth,algorithm,order)
    after_search=perf_counter_ns()
    decision_requests=provider.requests-prep_requests
    emitted=provider.complete_matrix() if output_mode=='complete_task' else None
    finished=perf_counter_ns()
    counts=dict(astar_calls=provider.calls,astar_expanded_nodes=provider.expanded,
                astar_generated_nodes=provider.generated,astar_max_frontier=provider.max_frontier,
                distance_requests=provider.requests,distance_cache_hits=provider.hits,
                cached_distance_entries=len(provider.cache),decision_distance_requests=decision_requests,
                completion_distance_requests=provider.requests-prep_requests-decision_requests)
    return result,counts,dict(total_ms=(finished-started)/1e6,prepare_ms=(after_prepare-started)/1e6,
                             search_ms=(after_search-after_prepare)/1e6,completion_ms=(finished-after_search)/1e6),provider.seen_costs,emitted


def validate_game(problem,matrix,depth,result,oracle_values=None):
    expected,line=enumerate_game(matrix,problem.values,depth)
    assert result.best_value==expected
    if depth:
        values=root_values(matrix,problem.values,depth) if oracle_values is None else oracle_values
        best=max(values.values())
        optimal={problem.labels[i+2] for i,v in values.items() if v==best}
        assert result.best_target in optimal
    else:
        assert result.best_target is None
    stats=dataclasses.asdict(result.stats)
    assert stats['visited_nodes']==stats['internal_nodes']+stats['leaf_evaluations']
    assert stats['generated_actions']==stats['visited_nodes']-1+stats['skipped_child_roots']
    ledger=history_ledger(problem,matrix,result.principal_variation)
    assert len(ledger)==min(depth,len(problem.values))
    assert (ledger[-1]['utility'] if ledger else 0)==expected
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results')
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--batch',type=int,default=3)
    args=parser.parse_args()
    if args.repeats<3 or args.batch<1:parser.error('repeats至少3，batch至少1')
    out=args.output_dir
    if out.exists() and any(out.iterdir()):out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=True)
    problem=sample_problem()
    reference=dijkstra_matrix(problem)
    assert problem.labels==('A','B','T1','T2','T3','T4') and problem.values==(9,6,7,7)
    assert root_values(reference,problem.values,2)=={0:-4,1:-1,2:-2,3:-4}
    assert root_values(reference,problem.values,4)=={0:-5,1:1,2:3,3:-7}
    cases={v:dataclasses.replace(problem,values=(9,6,v,7)) for v in range(15)}
    expected_actions={(v,h):root_values(reference,c.values,h) for v,c in cases.items() for h in range(1,5)}
    sources=[Path(__file__),BASE/'treasure_game.py',BASE/'reference_tools.py',BASE/'samples/pdf_sample.json']
    source_hashes={p.name:sha(p) for p in sources}
    protocol=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),registration_phase='Before first formal solver warmup.',
                  sample_input=json.loads((BASE/'samples/pdf_sample.json').read_text(encoding='utf-8')),
                  independent_reference_matrix=reference,
                  phases=dict(path_matrix=dict(heuristics=['zero','manhattan'],configurations=2),
                              sample_depth=dict(depths=[0,1,2,3,4],algorithms=['minimax','alphabeta'],orders=list(ORDERS),configurations=30),
                              value_sweep=dict(t3_values=list(range(15)),depths=[2,4],algorithms=['minimax','alphabeta'],orders=list(ORDERS),configurations=180),
                              routing=dict(depths=[1,2,3,4],algorithms=['minimax','alphabeta'],orders=['label','gain'],
                                           strategies=['precompute','on_demand','repeated'],output_modes=['decision_only','complete_task'],configurations=96)),
                  repeats=args.repeats,batch=args.batch,timing_seed=SEED,
                  counts=dict(total_configurations=308,timing_batches=308*args.repeats,timed_top_level_calls=308*args.repeats*args.batch,
                              terrain_maps=1,reward_vectors=15),source_sha256=source_hashes,
                  timing_boundary='Static game phase excludes one prepared matrix, and includes candidate generation/sort/search. Matrix phase includes all30 ordered A* calls. Routing phase includes fresh provider, A*, cache, sort/search and required output completion. Validation/oracle/trace/file output excluded.',
                  comparison_scope='Decision-only routing emits the same optimal first move/value, but not all strategies emit a complete matrix. Complete-task phase separately includes full matrix output for every strategy. These output contracts must never be pooled.',
                  scientific_limits='One terrain map and15 deterministic reward vectors, not308 independent inputs. Exploration of value boundaries, not a prespecified interior timing optimum. IQR descriptive, no confidence/significance claim.')
    with (out/'protocol.json').open('x',encoding='utf-8') as stream:json.dump(safe(protocol),stream,ensure_ascii=False,indent=2)
    protocol_hash=sha(out/'protocol.json')

    matrices={};path_rows=[];path_data=[]
    for heuristic in ('zero','manhattan'):
        distances=build_distance_matrix(problem,heuristic,record_trace=True)
        assert distances.matrix==reference
        matrices[heuristic]=distances
        for (source,target),result in distances.queries.items():
            validate_path(problem,source,target,result,reference[source][target])
            path_rows.append(dict(heuristic=heuristic,source=problem.labels[source],target=problem.labels[target],cost=result.cost,
                                  steps=len(result.path)-1,**dataclasses.asdict(result.stats)))
            path_data.append(dict(heuristic=heuristic,source=problem.labels[source],target=problem.labels[target],**result.to_dict()))
    write_csv(out/'distance_matrix.csv',[dict(source=label,**dict(zip(problem.labels,reference[i]))) for i,label in enumerate(problem.labels)])
    write_csv(out/'path_queries.csv',path_rows);write_json(out/'paths.json',path_data)
    prepared=matrices['manhattan']
    analysis=dict(grid=problem.grid,labels=problem.labels,points=problem.points,values=problem.values,matrix=reference,depths=[],ledgers={})
    for depth in range(1,5):
        vector=expected_actions[(7,depth)];best=max(vector.values());optimal=[problem.labels[i+2] for i,v in vector.items() if v==best]
        _,line=enumerate_game(reference,problem.values,depth)
        full_root=expected_actions[(7,4)]
        analysis['depths'].append(dict(depth=depth,root_action_values={problem.labels[i+2]:v for i,v in vector.items()},
                                       best_value=best,best_targets=optimal,canonical_line=[problem.labels[i+2] for i in line],
                                       full_horizon_value_of_selected_first=full_root[line[0]],full_horizon_first_action_regret=max(full_root.values())-full_root[line[0]]))
    for depth in (2,4):
        line=analysis['depths'][depth-1]['canonical_line']
        analysis['ledgers'][f'H{depth}']=history_ledger(problem,reference,line)
    histories=[('T1','T2','T3','T4'),('T2','T1','T3','T4')]
    alias=[dict(history=line,ledger=history_ledger(problem,reference,line),final_positions=['T3','T4'],remaining=[],turn='A') for line in histories]
    assert [item['ledger'][-1]['utility'] for item in alias]==[1,7]
    analysis['state_alias_counterexample']=alias
    write_json(out/'sample_analysis.json',analysis)
    traces=[]
    for depth in (2,4):
        result=game_search(problem,prepared,depth,record_trace=True)
        traces.append(dict(depth=depth,**result.to_dict()))
    write_json(out/'sample_traces.json',traces)

    configs=[]
    for heuristic in ('zero','manhattan'):
        configs.append(dict(phase='path_matrix',heuristic=heuristic))
    for depth in range(5):
        for algorithm in ('minimax','alphabeta'):
            for order in ORDERS:configs.append(dict(phase='sample_depth',depth=depth,algorithm=algorithm,order=order,t3_value=7))
    for v in range(15):
        for depth in (2,4):
            for algorithm in ('minimax','alphabeta'):
                for order in ORDERS:configs.append(dict(phase='value_sweep',depth=depth,algorithm=algorithm,order=order,t3_value=v))
    for output_mode in ('decision_only','complete_task'):
        for depth in range(1,5):
            for algorithm in ('minimax','alphabeta'):
                for order in ('label','gain'):
                    for strategy in ('precompute','on_demand','repeated'):
                        configs.append(dict(phase='routing',depth=depth,algorithm=algorithm,order=order,t3_value=7,strategy=strategy,output_mode=output_mode))
    assert len(configs)==308

    def invoke(config):
        if config['phase']=='path_matrix':
            started=perf_counter_ns();distance=build_distance_matrix(problem,config['heuristic']);finished=perf_counter_ns()
            stats=dict(astar_calls=len(distance.queries),astar_expanded_nodes=sum(r.stats.expanded_nodes for r in distance.queries.values()),
                       astar_generated_nodes=sum(r.stats.generated_nodes for r in distance.queries.values()))
            return distance,stats,dict(total_ms=(finished-started)/1e6,prepare_ms=0,search_ms=0,completion_ms=0),None
        case=cases[config['t3_value']]
        if config['phase']=='routing':
            result,counters,times,seen,emitted=invoke_routing(case,config['depth'],config['algorithm'],config['order'],config['strategy'],config['output_mode'])
            return result,{**dataclasses.asdict(result.stats),**counters},times,(seen,emitted)
        started=perf_counter_ns();result=game_search(case,prepared,config['depth'],config['algorithm'],config['order']);finished=perf_counter_ns()
        return result,dataclasses.asdict(result.stats),dict(total_ms=(finished-started)/1e6,prepare_ms=0,search_ms=(finished-started)/1e6,completion_ms=0),None

    def check(config,result,counts,extra):
        if config['phase']=='path_matrix':
            assert result.matrix==reference and counts['astar_calls']==30
            return
        case=cases[config['t3_value']]
        validate_game(case,reference,config['depth'],result,expected_actions.get((config['t3_value'],config['depth'])))
        if config['phase']=='routing':
            seen,emitted=extra
            assert all(value==reference[s][t] for (s,t),value in seen.items())
            if config['output_mode']=='complete_task':assert emitted==reference
            if config['strategy']=='precompute':assert counts['astar_calls']==30
            if config['strategy']=='on_demand':
                assert counts['astar_calls']==counts['cached_distance_entries']<=30
                if config['output_mode']=='complete_task':assert counts['astar_calls']==30
            if config['strategy']=='repeated':
                assert counts['cached_distance_entries']==0
                assert counts['astar_calls']==counts['decision_distance_requests']+(30 if config['output_mode']=='complete_task' else 0)

    stable=[];initial_results=[]
    for config in configs:
        result,counts,_,extra=invoke(config);check(config,result,counts,extra)
        stable.append(counts);initial_results.append(result)
    raw=[];rng=random.Random(SEED)
    for repeat in range(args.repeats):
        order=list(enumerate(configs));rng.shuffle(order)
        for index,config in order:
            recorded=[]
            for _ in range(args.batch):
                result,counts,times,extra=invoke(config)
                check(config,result,counts,extra)
                assert counts==stable[index]
                recorded.append(times)
            raw.append(dict(config=index,phase=config['phase'],repeat=repeat,batch=args.batch,
                            **{name:statistics.mean(t[name] for t in recorded) for name in ('total_ms','prepare_ms','search_ms','completion_ms')}))
        print(f'Q4 timing block {repeat+1}/{args.repeats} verified',flush=True)
    assert sha(out/'protocol.json')==protocol_hash and all(sha(p)==source_hashes[p.name] for p in sources)
    summaries=[]
    for index,config in enumerate(configs):
        row=dict(config=index,phase=config['phase'],depth=config.get('depth',''),algorithm=config.get('algorithm',''),
                 order=config.get('order',''),t3_value=config.get('t3_value',''),heuristic=config.get('heuristic',''),
                 strategy=config.get('strategy',''),output_mode=config.get('output_mode',''))
        result=initial_results[index]
        row.update(best_target='' if config['phase']=='path_matrix' else result.best_target,
                   best_value='' if config['phase']=='path_matrix' else result.best_value)
        all_count_names=sorted(set(key for counts in stable for key in counts))
        row.update({key:stable[index].get(key,0) for key in all_count_names})
        for metric in ('total_ms','prepare_ms','search_ms','completion_ms'):
            median,q1,q3=distribution([r[metric] for r in raw if r['config']==index])
            row.update({f'median_{metric}':median,f'q1_{metric}':q1,f'q3_{metric}':q3})
        summaries.append(row)
    write_csv(out/'comparison.csv',summaries);write_csv(out/'timing_batches.csv',raw)
    for phase,filename in [('sample_depth','sample_order_comparison.csv'),('value_sweep','value_sweep.csv'),('routing','routing_comparison.csv'),('path_matrix','path_matrix_comparison.csv')]:
        write_csv(out/filename,[r for r in summaries if r['phase']==phase])
    action_rows=[]
    for v in range(15):
        for depth in range(1,5):
            vector=expected_actions[(v,depth)];best=max(vector.values())
            for action,value in vector.items():
                action_rows.append(dict(t3_value=v,depth=depth,target=problem.labels[action+2],exact_value=value,is_optimal=value==best))
    write_csv(out/'root_action_values.csv',action_rows)
    pairs=[]
    raw_lookup={(r['config'],r['repeat']):r['total_ms'] for r in raw}
    for target in summaries:
        reference_row=None
        if target['phase'] in ('sample_depth','value_sweep') and target['algorithm']=='alphabeta':
            reference_row=next(r for r in summaries if r['phase']==target['phase'] and r['depth']==target['depth'] and r['t3_value']==target['t3_value'] and r['order']==target['order'] and r['algorithm']=='minimax')
        elif target['phase']=='routing' and target['strategy']!='precompute':
            reference_row=next(r for r in summaries if r['phase']=='routing' and all(r[key]==target[key] for key in ('depth','algorithm','order','output_mode')) and r['strategy']=='precompute')
        if reference_row is None:continue
        values=[raw_lookup[(target['config'],repeat)]-raw_lookup[(reference_row['config'],repeat)] for repeat in range(args.repeats)]
        median,q1,q3=distribution(values)
        pairs.append(dict(target_config=target['config'],reference_config=reference_row['config'],phase=target['phase'],
                          output_mode=target['output_mode'],median_delta_ms=median,q1_delta_ms=q1,q3_delta_ms=q3,
                          faster_blocks=sum(v<0 for v in values),slower_blocks=sum(v>0 for v in values)))
    write_csv(out/'paired_differences.csv',pairs)
    outcomes=dict(all_distances_match_independent_dijkstra=True,all_paths_and_enter_ledgers_valid=True,
                  all_game_values_and_first_actions_correct=True,all_counters_stable=True,
                  both_output_contracts_preserved=True,protocol_unchanged=True,source_unchanged=True,
                  configurations=len(configs),timing_batches=len(raw),timed_top_level_calls=len(raw)*args.batch,
                  official_results={'H2':{'best_target':'T2','best_value':-1},'H4':{'best_target':'T3','best_value':3}})
    write_json(out/'outcomes.json',outcomes)
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                  protocol=dict(relative_path='protocol.json',sha256=protocol_hash),source_sha256=source_hashes,
                  parameters=dict(repeats=args.repeats,batch=args.batch,timing_seed=SEED),counts=protocol['counts'],
                  timing_boundary=protocol['timing_boundary'],comparison_scope=protocol['comparison_scope'],scientific_limits=protocol['scientific_limits'],
                  result_files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('protocol.json','reproducibility.json')})
    write_json(out/'reproducibility.json',metadata)
    print(f'Saved {len(configs)} configurations / {len(raw)} batches / {len(raw)*args.batch} top-level calls to {out}')


if __name__=='__main__':main()
