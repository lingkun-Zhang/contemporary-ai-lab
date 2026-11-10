"""题三：固定展开语义的共享对照、缓存界消融与准入深度扫描。"""
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
from html import escape
from pathlib import Path
from time import perf_counter_ns

from alphabeta_tree import TreeNode, parse_tree, search, validate_tree
from transposition_search import cached_search

BASE=Path(__file__).resolve().parent
SEED=20261030
LIMITS=(0,1,2,3,4,5,6,7,8,None)


def encode_graph(root):
    objects,indices={},{}
    def visit(node):
        if id(node) in indices:
            return indices[id(node)]
        key=f'n{len(indices)}'; indices[id(node)]=key
        objects[key]={}
        objects[key].update(dict(children=[visit(c) for c in node.children]) if node.children else dict(value=node.value))
        return key
    return dict(root=visit(root),nodes=objects)


def make_graph(seed,share_probability,branching=3,depth=8,width=12):
    """同一基础有向无环图按引用发生位置选择共享或复制，展开语义不变。"""
    rng=random.Random(seed)
    layer=[TreeNode(rng.randint(-100,100)) for _ in range(width)]
    for level in range(depth-1,0,-1):
        layer=[TreeNode(children=rng.sample(layer,branching)) for _ in range(width)]
    original=TreeNode(children=rng.sample(layer,branching))
    aliases={}
    def materialize(node,path):
        digest=hashlib.sha256(f'{seed}|{path}'.encode()).digest()
        use_shared=int.from_bytes(digest[:8],'big')/2**64 < share_probability
        if use_shared and id(node) in aliases:
            return aliases[id(node)]
        copy=TreeNode(node.value)
        if use_shared:
            aliases[id(node)]=copy
        copy.children=[materialize(c,path+'/'+str(i)) for i,c in enumerate(node.children)]
        return copy
    return materialize(original,'root')


def semantic_oracle(root):
    """独立按(对象,玩家)后序计算全终局值、展开次数和有序语义指纹。"""
    memo={}
    def visit(node,maximizing):
        key=(id(node),maximizing)
        if key in memo: return memo[key]
        if not node.children:
            value,nodes,leaves=node.value,1,1
            serial=json.dumps({'value':value},sort_keys=True)
        else:
            child=[visit(c,not maximizing) for c in node.children]
            value=(max if maximizing else min)(v[0] for v in child)
            nodes,leaves=1+sum(v[1] for v in child),sum(v[2] for v in child)
            serial=json.dumps({'children':[v[3] for v in child]},sort_keys=True)
        answer=(value,nodes,leaves,hashlib.sha256(serial.encode()).hexdigest())
        memo[key]=answer
        return answer
    return visit(root,True)


def counterexample():
    shared=TreeNode(children=[TreeNode(6),TreeNode(9)])
    return TreeNode(children=[TreeNode(children=[TreeNode(5),shared]),TreeNode(children=[shared,TreeNode(8)])])


def incorrect_cache_demo(root):
    """故意错误消融：将所有截断返回都视为精确项，绝不用于正式求解。"""
    table,events={},[]
    def visit(node,maximizing,alpha,beta,path):
        key=(id(node),maximizing)
        if key in table:
            events.append(dict(event='wrong_exact_hit',path=path,value=table[key]))
            return table[key]
        if not node.children:
            value=node.value
        else:
            value=-math.inf if maximizing else math.inf
            for i,child in enumerate(node.children):
                candidate=visit(child,not maximizing,alpha,beta,f'{path}/{i}')
                value=(max if maximizing else min)(value,candidate)
                if maximizing: alpha=max(alpha,value)
                else: beta=min(beta,value)
                if alpha>=beta: break
        table[key]=value
        events.append(dict(event='wrong_store_as_exact',path=path,value=value))
        return value
    return dict(value=visit(root,True,-math.inf,math.inf,'root'),trace=events,
                warning='Intentionally incorrect ablation only; a cutoff bound is not an exact value.')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def quantiles(values):
    q=statistics.quantiles(values,n=4,method='inclusive') if len(values)>1 else values*3
    return statistics.median(values),q[0],q[2]


def invoke(case,method,limit):
    started=perf_counter_ns()
    if method=='original':
        result=search(case['root'],validate=False)
    else:
        result=cached_search(case['root'],mode=method,max_cache_depth=limit,validate=False)
    elapsed=(perf_counter_ns()-started)/1e6
    stats=dataclasses.asdict(result.stats)
    for field in ('tt_returns','tt_probes','tt_hits','exact_hits','bound_cutoffs','noncutting_hits','stores','table_peak_entries','table_final_entries'):
        stats.setdefault(field,0)
    return result.value,stats,elapsed


def svg_start(title,subtitle,height):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
            '<rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#293e4d">',
            f'<text x="28" y="38" font-size="25" font-weight="bold">{escape(title)}</text>',
            f'<text x="28" y="69" font-size="14" fill="#566c80">{escape(subtitle)}</text>']


def panel(svg,y,title,xs,series,height=250):
    svg.append(f'<text x="28" y="{y}" font-size="17">{escape(title)}</text>')
    left,top,width,h=100,y+25,1050,height-75
    maximum=max(v for _,vals,*_ in series for v in vals)*1.12 or 1
    colors=['#64748b','#bb7d34','#3586a0','#66965f']
    for tick in range(5):
        py=top+h-tick*h/4
        svg.extend([f'<line x1="{left}" y1="{py}" x2="1150" y2="{py}" stroke="#dce5ed"/>',
                    f'<text x="88" y="{py+5}" text-anchor="end" font-size="13">{maximum*tick/4:.2f}</text>'])
    for j,x in enumerate(xs):
        svg.append(f'<text x="{left+width*(j+.5)/len(xs)}" y="{top+h+25}" text-anchor="middle" font-size="14">{escape(str(x))}</text>')
    for i,(name,vals,*bands) in enumerate(series):
        points=[]
        for j,v in enumerate(vals):
            px,py=left+width*(j+.5)/len(xs),top+h-v/maximum*h
            points.append(f'{px},{py}')
            if bands:
                low,high=bands[0][j]
                y1,y2=top+h-low/maximum*h,top+h-high/maximum*h
                svg.append(f'<path d="M{px},{y1}V{y2} M{px-4},{y1}h8 M{px-4},{y2}h8" stroke="{colors[i]}"/>')
            svg.append(f'<circle cx="{px}" cy="{py}" r="3.4" fill="{colors[i]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        x=105+i*265
        svg.extend([f'<path d="M{x},{top+h+53}h22" stroke="{colors[i]}" stroke-width="3"/>',f'<text x="{x+30}" y="{top+h+58}" font-size="13">{escape(name)}</text>'])


def figures(out,rows):
    names={'original':'原始AB','off':'同内核关闭缓存','exact':'仅精确缓存','bounds':'界感知缓存'}
    primary=[r for r in rows if r['experiment']=='sharing']
    svg=svg_start('同一展开树：共享对象能否带来缓存收益','三个固定种子；各共享程度保持有序展开指纹、根值和原始AB计数不变。',850)
    for k,field in enumerate(('visited_nodes','median_total_ms')):
        series=[]
        for method in ('original','off','exact','bounds'):
            series.append((names[method],[statistics.median(r[field] for r in primary if r['sharing_probability']==p and r['method']==method) for p in (0,.5,1)]))
        panel(svg,115+k*340,'访问发生中位数' if k==0 else '逐树计时中位数的中位数（ms，包含新表和表操作）',(0,.5,1),series)
    (out/'sharing_cost.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')
    svg=svg_start('缓存准入深度：只调哪些层查表，不截短搜索','选定展示 seed=20261030、完整共享；点为9批中位数，竖线IQR，非置信区间。',850)
    selected=[next(r for r in rows if r['experiment']=='admission' and r['case']=='dag_s20261030_p1' and r['max_cache_depth']==('all' if q is None else q)) for q in LIMITS]
    panel(svg,115,'包含缓存全部工作的新查询时间（ms）',[*range(9),'all'],[('界感知缓存',[r['median_total_ms'] for r in selected],[(r['q1_total_ms'],r['q3_total_ms']) for r in selected])])
    panel(svg,455,'搜索访问与表峰值条目（条目不是总字节内存）',[*range(9),'all'],[('搜索访问',[r['visited_nodes'] for r in selected]),('表峰值条目',[r['table_peak_entries'] for r in selected])])
    (out/'admission_depth.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results_transposition')
    parser.add_argument('--repeats',type=int,default=9)
    parser.add_argument('--batch',type=int,default=5)
    args=parser.parse_args()
    if args.repeats<3 or args.batch<1: parser.error('repeats至少3，batch至少1')
    out=args.output_dir
    if out.exists() and any(out.iterdir()): out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True)
    cases=[]
    for seed in range(SEED,SEED+3):
        for p in (0,.5,1):
            cases.append(dict(case=f'dag_s{seed}_p{p:g}',seed=seed,sharing_probability=p,root=make_graph(seed,p)))
    payload=json.loads((BASE/'samples/pdf_sample.json').read_text(encoding='utf-8'))
    cases.append(dict(case='pdf_json',seed=SEED,sharing_probability=-1,root=parse_tree(payload)))
    leaf=lambda a,b:TreeNode(children=[TreeNode(a),TreeNode(b)])
    shared=leaf(6,9)
    cases.append(dict(case='pdf_shared',seed=SEED,sharing_probability=-1,root=TreeNode(children=[TreeNode(children=[leaf(3,5),shared]),TreeNode(children=[shared,leaf(1,2)])])))
    cases.append(dict(case='cache_counterexample',seed=SEED,sharing_probability=-1,root=counterexample()))
    specs=[]
    for case in cases:
        validate_tree(case['root'])
        value,nodes,leaves,digest=semantic_oracle(case['root'])
        graph=encode_graph(case['root'])
        case.update(expected=value,expanded_nodes=nodes,expanded_leaves=leaves,unique_objects=len(graph['nodes']),ordered_expansion_sha256=digest)
        specs.append({**{k:v for k,v in case.items() if k!='root'},'graph':graph})
    for seed in range(SEED,SEED+3):
        group=[c for c in cases if c['case'].startswith(f'dag_s{seed}_')]
        assert len({c['ordered_expansion_sha256'] for c in group})==1
    (out/'inputs.json').write_text(json.dumps(specs,ensure_ascii=False),encoding='utf-8')
    configs=[]
    def add(case,method,limit,experiment):
        value,stats,_=invoke(case,method,limit)
        assert value==case['expected'],(case['case'],method,limit,value,case['expected'])
        assert stats['visited_nodes']==stats['internal_nodes']+stats['leaf_evaluations']+stats['tt_returns']
        assert stats['tt_hits']==stats['exact_hits']+stats['bound_cutoffs']+stats['noncutting_hits']
        configs.append((case,method,limit,experiment,stats))
    for case in cases:
        for method in ('original','off','exact','bounds'): add(case,method,None,'sharing' if case['sharing_probability']>=0 else 'examples')
    for case in cases:
        if case['sharing_probability'] in (.5,1) or case['case']=='dag_s20261030_p0':
            for limit in LIMITS: add(case,'bounds',limit,'admission')
    for seed in range(SEED,SEED+3):
        original=[stats['visited_nodes'] for case,method,_,experiment,stats in configs if case['seed']==seed and experiment=='sharing' and method=='original']
        assert len(set(original))==1,'sharing altered original search work'
    raw=[];rng=random.Random(SEED)
    for repeat in range(args.repeats):
        order=list(enumerate(configs));rng.shuffle(order)
        for index,(case,method,limit,experiment,stats) in order:
            elapsed=[]
            for _ in range(args.batch):
                value,actual,ms=invoke(case,method,limit);elapsed.append(ms)
                assert value==case['expected'] and actual==stats
            raw.append(dict(config=index,case=case['case'],experiment=experiment,method=method,max_cache_depth='all' if limit is None else limit,repeat=repeat,batch=args.batch,total_ms=statistics.mean(elapsed)))
        print(f'Block {repeat+1}/{args.repeats} verified',flush=True)
    rows=[]
    for index,(case,method,limit,experiment,stats) in enumerate(configs):
        median,q1,q3=quantiles([r['total_ms'] for r in raw if r['config']==index])
        rows.append(dict(config=index,**{k:v for k,v in case.items() if k!='root'},experiment=experiment,method=method,max_cache_depth='all' if limit is None else limit,value=case['expected'],**stats,median_total_ms=median,q1_total_ms=q1,q3_total_ms=q3))
    csv_write(out/'comparison.csv',rows);csv_write(out/'timing_batches.csv',raw)
    pairs=[]
    for row in rows:
        if row['method']!='bounds': continue
        for ref in ('original','off','exact'):
            baseline=next(r for r in rows if r['case']==row['case'] and r['method']==ref and r['experiment'] in ('sharing','examples'))
            diffs=[]
            for repeat in range(args.repeats):
                target=next(r['total_ms'] for r in raw if r['config']==row['config'] and r['repeat']==repeat)
                control=next(r['total_ms'] for r in raw if r['config']==baseline['config'] and r['repeat']==repeat)
                diffs.append(target-control)
            med,q1,q3=quantiles(diffs)
            pairs.append(dict(case=row['case'],experiment=row['experiment'],max_cache_depth=row['max_cache_depth'],target_config=row['config'],reference_config=baseline['config'],reference=ref,median_delta_ms=med,q1_delta_ms=q1,q3_delta_ms=q3,faster_blocks=sum(x<0 for x in diffs),slower_blocks=sum(x>0 for x in diffs)))
    csv_write(out/'paired_differences.csv',pairs)
    demo=next(c for c in cases if c['case']=='cache_counterexample')
    wrong=incorrect_cache_demo(demo['root'])
    safe=cached_search(demo['root'],trace=True)
    assert demo['expected']==safe.value==8 and wrong['value']==6
    def json_safe(obj):
        if isinstance(obj,float) and not math.isfinite(obj): return str(obj)
        if isinstance(obj,dict): return {k:json_safe(v) for k,v in obj.items()}
        if isinstance(obj,(tuple,list)): return [json_safe(v) for v in obj]
        return obj
    (out/'cache_counterexample.json').write_text(json.dumps(json_safe(dict(expected=8,incorrect=wrong,safe=dataclasses.asdict(safe))),ensure_ascii=False,indent=2),encoding='utf-8')
    figures(out,rows)
    metadata=dict(declared_design='Same ordered expanded game tree across sharing probabilities; fresh table each call; left-to-right; full terminal search.',generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),seed=SEED,repeats=args.repeats,batch=args.batch,input_records=len(cases),distinct_ordered_expanded_inputs=len({c['ordered_expansion_sha256'] for c in cases}),configurations=len(configs),timing_batches=len(raw),timed_solver_calls=len(raw)*args.batch,max_cache_depth_candidates=['all' if q is None else q for q in LIMITS],source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),BASE/'transposition_search.py',BASE/'alphabeta_tree.py',BASE/'samples/pdf_sample.json']},input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),timing_boundary='Already constructed and validated inputs. Includes fresh table creation, lookup, bound checks, stores and all search. Excludes generation, independent oracle, validation, trace and file output. No concurrent benchmark worker.',interpretation='Exploratory timing on fixed workloads; IQR is descriptive, not confidence. Do not select and independently confirm a parameter in the same data. Object identity is not structural equality; cache lookup never merges equal independent copies.')
    (out/'reproducibility.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'outcomes.json').write_text(json.dumps(dict(all_safe_values_correct=True,all_counters_stable=True,sharing_preserves_ordered_expansion=True,sharing_preserves_original_ab_counts=True,incorrect_cache_expected_failure=True,safe_counterexample_value=8,incorrect_counterexample_value=6),indent=2),encoding='utf-8')
    print(f'Saved {len(configs)} configurations / {len(raw)} batches / {len(raw)*args.batch} calls to {out}')


if __name__=='__main__': main()
