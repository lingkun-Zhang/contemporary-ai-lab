"""Q3: bounded LRU cache capacity, repeat work, storage and full query cost."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from alphabeta_tree import search, validate_tree
from bounded_transposition import BoundedStats, bounded_search
from run_transposition_study import csv_write, encode_graph, make_graph, panel, quantiles, semantic_oracle, svg_start
from transposition_search import cached_search


BASE=Path(__file__).resolve().parent
SEEDS=(20261301,20261302,20261303)
PROBABILITIES=(0,.5,1)
CAPACITIES=(0,4,8,16,32,64,128,None)
TIMING_SEED=20261301


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_label(capacity): return 'all' if capacity is None else capacity


def invoke_capacity(case,method,capacity=None):
    started=perf_counter_ns()
    if method=='original': result=search(case['root'],validate=False)
    elif method=='old_bounds': result=cached_search(case['root'],validate=False)
    elif method=='lru': result=bounded_search(case['root'],capacity=capacity,validate=False)
    else: raise ValueError(method)
    elapsed=(perf_counter_ns()-started)/1e6
    stats=dataclasses.asdict(result.stats)
    for field in dataclasses.fields(BoundedStats): stats.setdefault(field.name,0)
    return result.value,stats,elapsed


def check_stats(case,method,capacity,expected_stats=None):
    value,stats,elapsed=invoke_capacity(case,method,capacity)
    assert value==case['expected'],(case['case'],method,capacity,value,case['expected'])
    assert stats['visited_nodes']==stats['internal_nodes']+stats['leaf_evaluations']+stats['tt_returns']
    assert stats['tt_hits']==stats['exact_hits']+stats['bound_cutoffs']+stats['noncutting_hits']
    assert stats['tt_returns']==stats['exact_hits']+stats['bound_cutoffs']
    if method=='lru':
        assert stats['stores']==stats['new_insertions']+stats['entry_updates']
        assert stats['table_final_entries']==stats['new_insertions']-stats['evictions']
        if capacity is not None: assert stats['table_peak_entries']<=capacity
    if expected_stats is not None: assert stats==expected_stats
    return stats,elapsed


def draw_figures(out,rows):
    svg=svg_start('缓存容量：淘汰与重复搜索的代价','固定LRU；三个新种子，所有查询完整搜索终局，容量只约束表项数。',850)
    for k,(field,title) in enumerate((('visited_nodes','三个种子节点访问中位数；横轴为最大条目数'),('median_total_ms','逐图计时中位数的中位数（ms，含全部LRU维护）'))):
        series=[]
        for p,label in ((0,'无复用纯树'),(.5,'部分共享'),(1,'完整共享')):
            vals=[statistics.median(r[field] for r in rows if r['method']=='lru' and r['sharing_probability']==p and r['capacity']==capacity_label(c)) for c in CAPACITIES]
            series.append((label,vals))
        panel(svg,115+k*340,title,[capacity_label(c) for c in CAPACITIES],series)
    (out/'capacity_cost.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')
    svg=svg_start('固定共享图：容量、缓存命中与淘汰','seed=20261301，p=1；条目不是字节内存，命中数不等于实际避免的搜索量。',850)
    chosen=[next(r for r in rows if r['case']=='dag_s20261301_p1' and r['method']=='lru' and r['capacity']==capacity_label(c)) for c in CAPACITIES]
    panel(svg,115,'表峰值与淘汰次数',[capacity_label(c) for c in CAPACITIES],[('峰值条目',[r['table_peak_entries'] for r in chosen]),('淘汰次数',[r['evictions'] for r in chosen])])
    panel(svg,455,'真正缓存返回与非闭窗命中',[capacity_label(c) for c in CAPACITIES],[('缓存直接返回',[r['tt_returns'] for r in chosen]),('命中仍需搜索',[r['noncutting_hits'] for r in chosen])])
    (out/'capacity_evictions.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results_capacity_study')
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--batch',type=int,default=3)
    args=parser.parse_args(argv)
    if args.repeats<3 or args.batch<1: parser.error('repeats至少3，batch至少1')
    out=args.output_dir
    if out.exists() and any(out.iterdir()): out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),BASE/'bounded_transposition.py',BASE/'transposition_search.py',BASE/'alphabeta_tree.py',BASE/'run_transposition_study.py']
    hashes={p.name:sha(p) for p in source_paths}
    cases,specs=[],[]
    for seed in SEEDS:
        for p in PROBABILITIES:
            root=make_graph(seed,p)
            validate_tree(root)
            value,nodes,leaves,digest=semantic_oracle(root)
            graph=encode_graph(root)
            case=dict(case=f'dag_s{seed}_p{p:g}',seed=seed,sharing_probability=p,root=root,expected=value,
                      expanded_nodes=nodes,expanded_leaves=leaves,unique_objects=len(graph['nodes']),ordered_expansion_sha256=digest)
            cases.append(case)
            specs.append({**{k:v for k,v in case.items() if k!='root'},'graph':graph})
    for seed in SEEDS:
        assert len({c['ordered_expansion_sha256'] for c in cases if c['seed']==seed})==1
    assert len({c['ordered_expansion_sha256'] for c in cases})==3
    protocol=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),registration_phase='Before first solver/warmup call.',
                  graph_parameters=dict(branching=3,depth=8,width=12,seeds=list(SEEDS),sharing_probabilities=list(PROBABILITIES),seed_note='Integers are RNG identifiers, not calendar dates.'),
                  methods=['original','old_bounds','lru'],capacities=[capacity_label(c) for c in CAPACITIES],
                  fixed_policy='LRU: every probe hit refreshes recency including nonclosing bounds; accepted writes refresh; new insertion evicts oldest when full.',
                  capacity_zero='No table probes/stores in bounded kernel; full terminal search.',
                  references=['original','lru_capacity_0','old_bounds_unlimited','lru_unlimited'],
                  repeats=args.repeats,batch=args.batch,timing_seed=TIMING_SEED,
                  counts=dict(object_graphs=9,distinct_ordered_expanded_inputs=3,configurations=90,timing_batches=90*args.repeats,timed_solver_calls=90*args.repeats*args.batch),
                  cases=specs,source_sha256=hashes,
                  timing_boundary='Preconstructed and validated inputs; includes fresh OrderedDict, probes, LRU refreshes, stores, eviction and full search. Excludes generation, independent oracle, validation, trace and output.',
                  interpretation='Exploratory capacity sweep; no prespecified interior optimum. Per-block IQR is descriptive, not confidence or independent inputs. Table entries are not total bytes of memory.')
    protocol_path=out/'protocol.json'
    with protocol_path.open('x',encoding='utf-8') as stream: json.dump(protocol,stream,ensure_ascii=False,indent=2)
    protocol_hash=sha(protocol_path)
    configs=[]
    for case in cases:
        for method,capacity in [('original',None),('old_bounds',None)]+[('lru',cap) for cap in CAPACITIES]:
            stats,_=check_stats(case,method,capacity)
            configs.append((case,method,capacity,stats))
        original=next(s for c,m,_,s in configs if c is case and m=='original')
        off=next(s for c,m,cap,s in configs if c is case and m=='lru' and cap==0)
        for field in ('visited_nodes','internal_nodes','leaf_evaluations','cutoff_events','skipped_child_roots','bound_crossings'):
            assert original[field]==off[field]
        old=next(s for c,m,_,s in configs if c is case and m=='old_bounds')
        unlimited=next(s for c,m,cap,s in configs if c is case and m=='lru' and cap is None)
        for field in ('visited_nodes','internal_nodes','leaf_evaluations','tt_returns','tt_hits','bound_cutoffs','noncutting_hits','stores'):
            assert old[field]==unlimited[field]
        if case['sharing_probability']==0:
            assert all(s['tt_hits']==0 for c,_,_,s in configs if c is case)
    raw=[];rng=random.Random(TIMING_SEED)
    for repeat in range(args.repeats):
        shuffled=list(enumerate(configs));rng.shuffle(shuffled)
        for index,(case,method,capacity,stats) in shuffled:
            elapsed=[check_stats(case,method,capacity,stats)[1] for _ in range(args.batch)]
            raw.append(dict(config=index,case=case['case'],method=method,capacity=capacity_label(capacity),repeat=repeat,batch=args.batch,total_ms=statistics.mean(elapsed)))
        print(f'Capacity block {repeat+1}/{args.repeats} verified',flush=True)
    assert sha(protocol_path)==protocol_hash
    assert all(sha(p)==hashes[p.name] for p in source_paths)
    rows=[]
    for index,(case,method,capacity,stats) in enumerate(configs):
        median,q1,q3=quantiles([b['total_ms'] for b in raw if b['config']==index])
        rows.append(dict(config=index,**{k:v for k,v in case.items() if k!='root'},method=method,capacity=capacity_label(capacity),value=case['expected'],**stats,median_total_ms=median,q1_total_ms=q1,q3_total_ms=q3))
    csv_write(out/'comparison.csv',rows);csv_write(out/'timing_batches.csv',raw)
    pairs=[]
    for row in rows:
        if row['method']!='lru': continue
        references={
            'original':next(r for r in rows if r['case']==row['case'] and r['method']=='original'),
            'lru_capacity_0':next(r for r in rows if r['case']==row['case'] and r['method']=='lru' and r['capacity']==0),
            'old_bounds_unlimited':next(r for r in rows if r['case']==row['case'] and r['method']=='old_bounds'),
            'lru_unlimited':next(r for r in rows if r['case']==row['case'] and r['method']=='lru' and r['capacity']=='all')}
        for label,reference in references.items():
            if row['config']==reference['config']: continue
            deltas=[]
            for repeat in range(args.repeats):
                t=next(b['total_ms'] for b in raw if b['config']==row['config'] and b['repeat']==repeat)
                ref=next(b['total_ms'] for b in raw if b['config']==reference['config'] and b['repeat']==repeat)
                deltas.append(t-ref)
            median,q1,q3=quantiles(deltas)
            pairs.append(dict(case=row['case'],capacity=row['capacity'],target_config=row['config'],reference_config=reference['config'],reference=label,median_delta_ms=median,q1_delta_ms=q1,q3_delta_ms=q3,faster_blocks=sum(v<0 for v in deltas),slower_blocks=sum(v>0 for v in deltas)))
    csv_write(out/'paired_differences.csv',pairs)
    draw_figures(out,rows)
    outcomes=dict(all_safe_values_correct=True,all_counters_stable=True,capacity_never_exceeded=True,store_conservation=True,
                  all_sharing_preserves_expansion=True,unlimited_lru_matches_old_search_counts=True,zero_capacity_matches_original_search_counts=True,pure_tree_has_no_hits=True,
                  object_graphs=9,distinct_ordered_expanded_inputs=3,configurations=len(rows),timing_batches=len(raw),timed_solver_calls=len(raw)*args.batch,
                  interior_optimum_presupposed=False)
    (out/'outcomes.json').write_text(json.dumps(outcomes,indent=2),encoding='utf-8')
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                  protocol=dict(relative_path='protocol.json',sha256=protocol_hash,declared_at_utc=protocol['declared_at_utc']),
                  source_sha256=hashes,counts=protocol['counts'],parameters=dict(capacities=protocol['capacities'],seeds=list(SEEDS),sharing_probabilities=list(PROBABILITIES),repeats=args.repeats,batch=args.batch),
                  result_files_sha256={name:sha(out/name) for name in ('comparison.csv','timing_batches.csv','paired_differences.csv','outcomes.json')},
                  timing_boundary=protocol['timing_boundary'],interpretation=protocol['interpretation'])
    (out/'reproducibility.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(rows)} configurations / {len(raw)} batches / {len(raw)*args.batch} timed calls to {out}')
    return out


if __name__=='__main__': main()
