"""题三：层级叶值相关性、采样预测质量与包含探查成本的排序实验。"""
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
from html import escape
from pathlib import Path

from alphabeta_tree import TreeNode, validate_tree
from run_experiments import encode, execute, oracle, quantiles, write_csv


BASE = Path(__file__).resolve().parent
SEEDS = tuple(range(20261201, 20261207))
LAMBDAS = (0, .25, .5, .75, 1)
BUDGETS = (0, 1, 4, 8)
BRANCHING, DEPTH, DECAY = 3, 8, .65
TIMING_SEED = 20261210


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_basis(seed, branching=BRANCHING, depth=DEPTH, decay=DECAY):
    """Every lambda reuses these independent leaf and hierarchical edge draws."""
    if branching < 2 or depth < 1 or not 0 < decay <= 1:
        raise ValueError("branching>=2, depth>=1, 0<decay<=1 required")
    rng = random.Random(seed)
    leaves = branching ** depth
    edges = sum(branching ** level for level in range(1, depth+1))
    return dict(seed=seed, branching=branching, depth=depth, decay=decay,
                leaf_noise=[rng.randint(-100,100) for _ in range(leaves)],
                edge_noise=[rng.randint(-100,100) for _ in range(edges)],
                edge_order="preorder; consume one draw immediately before descending each child")


def make_tree(basis, mixing):
    """sqrt(1-lambda)*z + sqrt(lambda)*H; H is a normalized path signal."""
    if isinstance(mixing, bool) or not isinstance(mixing,(int,float)) or not math.isfinite(mixing) or not 0 <= mixing <= 1:
        raise ValueError("mixing must lie in [0,1]")
    branching,depth,decay = basis['branching'],basis['depth'],basis['decay']
    if len(basis['leaf_noise']) != branching**depth or len(basis['edge_noise']) != sum(branching**k for k in range(1,depth+1)):
        raise ValueError("basis draw counts do not match shape")
    weights = [decay**k for k in range(depth)]
    norm = math.sqrt(sum(w*w for w in weights))
    leaf_weight, signal_weight = math.sqrt(1-mixing), math.sqrt(mixing)
    edge_index,leaf_index,values = 0,0,[]
    def build(level,path_sum):
        nonlocal edge_index,leaf_index
        if level == depth:
            value = leaf_weight*basis['leaf_noise'][leaf_index] + signal_weight*path_sum/norm
            leaf_index += 1
            values.append(value)
            return TreeNode(value)
        children=[]
        for _ in range(branching):
            draw = basis['edge_noise'][edge_index]
            edge_index += 1
            children.append(build(level+1,path_sum+weights[level]*draw))
        return TreeNode(children=children)
    root=build(0,0)
    assert edge_index==len(basis['edge_noise']) and leaf_index==len(basis['leaf_noise'])
    return root,values


def pearson(left,right):
    if len(left)!=len(right) or len(left)<2:
        return None
    ml,mr=statistics.mean(left),statistics.mean(right)
    dl,dr=[v-ml for v in left],[v-mr for v in right]
    denominator=math.sqrt(sum(v*v for v in dl)*sum(v*v for v in dr))
    return sum(a*b for a,b in zip(dl,dr))/denominator if denominator else None


def distribution_metrics(values,branching):
    # Each pair is leaf0/leaf1 from a different final-layer parent.
    left=[values[i] for i in range(0,len(values),branching)]
    right=[values[i+1] for i in range(0,len(values),branching)]
    return dict(leaf_mean=statistics.mean(values), leaf_variance=statistics.pvariance(values),
                sibling_pair_count=len(left), sibling_leaf_correlation=pearson(left,right),
                minimum_leaf=min(values), maximum_leaf=max(values))


def first_leaf_mean(node,budget):
    if budget<1:
        raise ValueError("budget must be positive")
    stack,samples=[node],[]
    while stack and len(samples)<budget:
        item=stack.pop()
        if item.children:
            stack.extend(reversed(item.children))
        else:
            samples.append(item.value)
    return sum(samples)/len(samples)


def pair_counts(estimates,truths):
    if len(estimates)!=len(truths):
        raise ValueError("score lengths differ")
    counts=dict(eligible_pairs=0,true_tied_pairs=0,predicted_tied_pairs=0,concordant_pairs=0,discordant_pairs=0)
    for i in range(len(truths)):
        for j in range(i+1,len(truths)):
            actual,predicted=truths[i]-truths[j],estimates[i]-estimates[j]
            if actual==0:
                counts['true_tied_pairs']+=1
                continue
            counts['eligible_pairs']+=1
            if predicted==0:
                counts['predicted_tied_pairs']+=1
            elif (actual>0)==(predicted>0):
                counts['concordant_pairs']+=1
            else:
                counts['discordant_pairs']+=1
    return counts


def quality_metrics(root,budget,truth_values=None):
    """All-parent diagnostics, separately stratifying nonterminal children."""
    truth_values=oracle(root)[1] if truth_values is None else truth_values
    buckets={}
    def add(label,counts,error,children):
        row=buckets.setdefault(label,dict(parent_depth=label,parents=0,child_scores=0,
                                          absolute_error_sum=0,eligible_pairs=0,true_tied_pairs=0,
                                          predicted_tied_pairs=0,concordant_pairs=0,discordant_pairs=0))
        row['parents']+=1;row['child_scores']+=children;row['absolute_error_sum']+=error
        for key,value in counts.items(): row[key]+=value
    stack=[(root,True,0)]
    while stack:
        node,maximizing,depth=stack.pop()
        if not node.children: continue
        estimates=[first_leaf_mean(child,budget) for child in node.children]
        truth=[truth_values[(id(child),not maximizing)] for child in node.children]
        counts=pair_counts(estimates,truth)
        error=sum(abs(a-b) for a,b in zip(estimates,truth))
        add(str(depth),counts,error,len(node.children));add('all',counts,error,len(node.children))
        if any(child.children for child in node.children):
            add('nonterminal_children',counts,error,len(node.children))
        stack.extend((child,not maximizing,depth+1) for child in reversed(node.children))
    for row in buckets.values():
        row['pairwise_agreement']=(row['concordant_pairs']+.5*row['predicted_tied_pairs'])/row['eligible_pairs'] if row['eligible_pairs'] else None
        row['mean_absolute_error']=row['absolute_error_sum']/row['child_scores']
    return buckets


def output_directory(requested):
    out=Path(requested)
    if out.exists() and any(out.iterdir()):
        out=out/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    out.mkdir(parents=True)
    return out


def write_protocol(path,payload):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2)
    return sha(path)


def svg_start(title,subtitle):
    return ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">',
            '<rect width="100%" height="100%" fill="white"/><g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#293e4d">',
            f'<text x="28" y="38" font-size="24" font-weight="bold">{escape(title)}</text>',
            f'<text x="28" y="68" font-size="14" fill="#566d82">{escape(subtitle)}</text>']


def panel(svg,y,title,series,ymin=0,ymax=None):
    svg.append(f'<text x="28" y="{y}" font-size="17">{escape(title)}</text>')
    left,top,width,height=105,y+30,1040,185
    maximum=ymax if ymax is not None else max([1]+[v for _,vals in series for v in vals])*1.12
    colors=['#61768c','#31839e','#c78337','#64945a']
    for tick in range(5):
        value=ymin+(maximum-ymin)*tick/4;py=top+height-tick*height/4
        svg.extend([f'<line x1="{left}" y1="{py}" x2="1145" y2="{py}" stroke="#dce5ed"/>',
                    f'<text x="90" y="{py+5}" text-anchor="end" font-size="13">{value:.2f}</text>'])
    for j,x in enumerate(LAMBDAS):
        svg.append(f'<text x="{left+width*(j+.5)/len(LAMBDAS)}" y="{top+height+26}" text-anchor="middle" font-size="14">{x:g}</text>')
    for i,(name,values) in enumerate(series):
        points=[]
        for j,value in enumerate(values):
            px=left+width*(j+.5)/len(values);py=top+height-(value-ymin)/(maximum-ymin)*height
            points.append(f'{px},{py}')
            svg.append(f'<circle cx="{px}" cy="{py}" r="3.4" fill="{colors[i]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        lx=105+i*258
        svg.extend([f'<path d="M{lx},{top+height+54}h21" stroke="{colors[i]}" stroke-width="3"/>',
                    f'<text x="{lx+28}" y="{top+height+59}" font-size="13">{escape(name)}</text>'])


def draw_figures(out,aggregates):
    def values(field,q):
        return [next(r[field] for r in aggregates if r['mixing']==lam and r['budget']==q) for lam in LAMBDAS]
    svg=svg_start('层级相关叶值：采样估计是否更有排序信息','六个共同基础输入；横轴为混合参数λ；末层叶对相关性实测，完整数据保留每树差异。')
    panel(svg,115,'同父叶0/1的跨父节点Pearson相关系数（每树计算后取中位数）',
          [('实测叶对相关',values('median_sibling_correlation',0))],ymin=-.1,ymax=1)
    panel(svg,455,'非终局孩子的成对排序一致率（汇总所有可比较对子，预测同分计半分）',
          [(f'q={q}',values('pooled_nonterminal_agreement',q)) for q in (1,4,8)],ymax=1)
    (out/'correlation_quality.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')
    svg=svg_start('相关性与排序收益：完整成本保留额外探查','横轴λ；逐树重复计时中位数再做跨六树汇总；合成模型不等于真实游戏。')
    panel(svg,115,'完整时间相对同树q=0的比值（各树比值中位数，低于1才更快）',
          [(f'q={q}',values('median_time_ratio_to_q0',q)) for q in BUDGETS])
    panel(svg,455,'正式搜索与排序额外读取的总叶值次数（各树中位数，非时间等价量）',
          [(f'q={q}',values('median_total_leaf_reads',q)) for q in BUDGETS])
    (out/'correlation_cost.svg').write_text('\n'.join(svg+['</g></svg>']),encoding='utf-8')


def run(output_dir=None,repeats=5,batch=3):
    if repeats<3 or batch<1: raise ValueError('repeats>=3 and batch>=1 required')
    out=output_directory(output_dir or BASE/'results_correlation_study')
    bases=[make_basis(seed) for seed in SEEDS]
    (out/'inputs.json').write_text(json.dumps(dict(model='hierarchical_path_signal_v1',bases=bases),ensure_ascii=False),encoding='utf-8')
    cases,quality_rows,specs=[],[],[]
    for basis in bases:
        basis_sha=hashlib.sha256(json.dumps(basis,sort_keys=True).encode()).hexdigest()
        for mixing in LAMBDAS:
            root,leaves=make_tree(basis,mixing);validate_tree(root)
            expected,truth,nodes,leaf_count=oracle(root)
            case=dict(case=f'correlation_s{basis["seed"]}_l{mixing:g}',seed=basis['seed'],mixing=mixing,
                      branching=BRANCHING,depth=DEPTH,basis_sha256=basis_sha,expected=expected,
                      full_nodes=nodes,full_leaves=leaf_count,tree_sha256=hashlib.sha256(json.dumps(encode(root),sort_keys=True).encode()).hexdigest(),
                      **distribution_metrics(leaves,BRANCHING),root=root,quality={})
            for q in BUDGETS[1:]:
                diagnostics=quality_metrics(root,q,truth)
                case['quality'][q]=diagnostics
                quality_rows.extend(dict(case=case['case'],seed=case['seed'],mixing=mixing,budget=q,**row) for row in diagnostics.values())
            cases.append(case)
            specs.append({k:v for k,v in case.items() if k not in ('root','quality')})
    sources=[Path(__file__),BASE/'run_experiments.py',BASE/'alphabeta_tree.py']
    source_hashes={p.name:sha(p) for p in sources}
    protocol=dict(declared_at_utc=datetime.now(timezone.utc).isoformat(),registration_phase='before any solver warmup',
                  design=dict(seeds=list(SEEDS),mixing=list(LAMBDAS),budgets=list(BUDGETS),branching=BRANCHING,depth=DEPTH,decay=DECAY,
                              repeats=repeats,batch=batch,timing_seed=TIMING_SEED),
                  model='Leaf=sqrt(1-lambda)*independent_noise + sqrt(lambda)*normalized hierarchical path signal. Independent edge draws use weights 0.65**(level-1).',
                  model_limits='Theoretical marginal variance is constant across lambda over the generator distribution. Finite-sample variance and marginal shape can change; lambda is not a pure empirical-correlation intervention. lambda=1 is pure hierarchical signal, not constant leaves.',
                  inputs=dict(relative_path='inputs.json',sha256=sha(out/'inputs.json'),basis_records=6,case_records=30),cases=specs,
                  counts=dict(configurations=120,timing_batches=120*repeats,timed_solver_calls=120*repeats*batch),source_sha256=source_hashes,
                  diagnostics='All-parent pairwise agreement excludes true ties and gives predicted ties half-credit. Nonterminal-child stratum excludes trivial direct-leaf predictions. Oracle and diagnostics remain outside solver/timing.',
                  selection_rule='All q=0,1,4,8 retained; no champion selected from these exploratory results.',
                  timing_boundary='Preconstructed validated tree. Includes order preparation, extra sampling, sorting and search; excludes generation, diagnostics, independent oracle, validation, file output and trace.')
    protocol_hash=write_protocol(out/'protocol.json',protocol)
    configs=[]
    for case in cases:
        for q in BUDGETS:
            result,counters,_,_=execute(case,'alphabeta',f'sample_{q}',True)
            if result.value!=case['expected']: raise AssertionError('incorrect warmup value')
            stats=dataclasses.asdict(result.stats)
            if stats['visited_nodes']!=stats['internal_nodes']+stats['leaf_evaluations']: raise AssertionError('node conservation')
            if q==0 and counters['ordering_probe_leaves']!=0: raise AssertionError('q0 must not probe')
            configs.append((case,q,stats,counters))
    raw=[];rng=random.Random(TIMING_SEED)
    for repeat in range(repeats):
        order=list(enumerate(configs));rng.shuffle(order)
        for index,(case,q,stats,counters) in order:
            totals=[]
            for _ in range(batch):
                result,actual,total_ms,_=execute(case,'alphabeta',f'sample_{q}',True)
                totals.append(total_ms)
                if result.value!=case['expected'] or dataclasses.asdict(result.stats)!=stats or actual!=counters:
                    raise AssertionError('timed correctness/counters changed')
            raw.append(dict(config=index,case=case['case'],seed=case['seed'],mixing=case['mixing'],budget=q,repeat=repeat,batch=batch,total_ms=statistics.mean(totals)))
        print(f'Correlation timing block {repeat+1}/{repeats} verified',flush=True)
    if sha(out/'protocol.json')!=protocol_hash or any(sha(p)!=source_hashes[p.name] for p in sources):
        raise AssertionError('protocol/source changed')
    rows=[]
    for index,(case,q,stats,counters) in enumerate(configs):
        median,q1,q3=quantiles([r['total_ms'] for r in raw if r['config']==index])
        diag=case['quality'].get(q,{})
        informative=diag.get('nonterminal_children',{})
        all_parents=diag.get('all',{})
        rows.append(dict(config=index,**{k:v for k,v in case.items() if k not in ('root','quality')},budget=q,value=case['expected'],
                         **stats,**counters,total_leaf_reads=stats['leaf_evaluations']+counters['ordering_probe_leaves'],
                         pairwise_agreement_all=all_parents.get('pairwise_agreement'),
                         pairwise_agreement_nonterminal=informative.get('pairwise_agreement'),
                         eligible_pairs_nonterminal=informative.get('eligible_pairs',0),true_tied_pairs_nonterminal=informative.get('true_tied_pairs',0),
                         predicted_tied_pairs_nonterminal=informative.get('predicted_tied_pairs',0),concordant_pairs_nonterminal=informative.get('concordant_pairs',0),
                         median_total_ms=median,q1_total_ms=q1,q3_total_ms=q3))
    pairs=[]
    for row in rows:
        if row['budget']==0: continue
        control=next(r for r in rows if r['case']==row['case'] and r['budget']==0)
        differences=[]
        for repeat in range(repeats):
            target=next(r['total_ms'] for r in raw if r['config']==row['config'] and r['repeat']==repeat)
            reference=next(r['total_ms'] for r in raw if r['config']==control['config'] and r['repeat']==repeat)
            differences.append(target-reference)
        median,q1,q3=quantiles(differences)
        pairs.append(dict(case=row['case'],seed=row['seed'],mixing=row['mixing'],budget=row['budget'],target_config=row['config'],reference_config=control['config'],
                          median_delta_ms=median,q1_delta_ms=q1,q3_delta_ms=q3,faster_blocks=sum(d<0 for d in differences),slower_blocks=sum(d>0 for d in differences)))
    aggregates=[]
    for mixing in LAMBDAS:
        for q in BUDGETS:
            group=[r for r in rows if r['mixing']==mixing and r['budget']==q]
            eligible=sum(r['eligible_pairs_nonterminal'] for r in group)
            score=sum(r['concordant_pairs_nonterminal']+.5*r['predicted_tied_pairs_nonterminal'] for r in group)
            ratios=[r['median_total_ms']/next(c['median_total_ms'] for c in rows if c['case']==r['case'] and c['budget']==0) for r in group]
            aggregates.append(dict(mixing=mixing,budget=q,input_trees=len(group),median_leaf_variance=statistics.median(r['leaf_variance'] for r in group),
                                   median_sibling_correlation=statistics.median(r['sibling_leaf_correlation'] for r in group),
                                   median_visited_nodes=statistics.median(r['visited_nodes'] for r in group),median_search_leaf_reads=statistics.median(r['leaf_evaluations'] for r in group),
                                   median_extra_probe_leaves=statistics.median(r['ordering_probe_leaves'] for r in group),median_total_leaf_reads=statistics.median(r['total_leaf_reads'] for r in group),
                                   median_total_ms=statistics.median(r['median_total_ms'] for r in group),median_time_ratio_to_q0=statistics.median(ratios),
                                   pooled_eligible_nonterminal_pairs=eligible,pooled_true_tied_nonterminal_pairs=sum(r['true_tied_pairs_nonterminal'] for r in group),
                                   pooled_nonterminal_agreement=score/eligible if eligible else None))
    write_csv(out/'comparison.csv',rows);write_csv(out/'timing_batches.csv',raw);write_csv(out/'paired_differences.csv',pairs)
    write_csv(out/'prediction_quality.csv',quality_rows);write_csv(out/'aggregates.csv',aggregates)
    draw_figures(out,aggregates)
    outcomes=dict(all_values_correct=True,all_counters_stable=True,protocol_unchanged=True,source_unchanged=True,
                  base_draw_records=6,mixing_case_records=30,configurations=len(rows),timing_batches=len(raw),timed_solver_calls=len(raw)*batch,
                  no_new_winner_selected=True,diagnostics_excluded_from_search=True,
                  scientific_support_not_assumed='All ratios, variance, correlations and negative results retained. No assertion that a larger lambda or budget improves speed.')
    (out/'outcomes.json').write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding='utf-8')
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
                  protocol=dict(relative_path='protocol.json',sha256=protocol_hash,declared_at_utc=protocol['declared_at_utc']),
                  source_sha256=source_hashes,input_sha256=sha(out/'inputs.json'),counts=protocol['counts'],design=protocol['design'],
                  result_files_sha256={name:sha(out/name) for name in ('comparison.csv','timing_batches.csv','paired_differences.csv','prediction_quality.csv','aggregates.csv','outcomes.json')},
                  timing_boundary=protocol['timing_boundary'],model_limits=protocol['model_limits'],diagnostics=protocol['diagnostics'],
                  statistics='Timing IQR describes repeated batches on fixed inputs. Six paired basis seeds, not thirty independent draws. Cross-lambda dependencies retained; no confidence/significance claim.')
    (out/'reproducibility.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(rows)} configs / {len(raw)} batches / {len(raw)*batch} calls to {out}')
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BASE/'results_correlation_study')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--batch',type=int,default=3)
    args=parser.parse_args()
    try: run(args.output_dir,args.repeats,args.batch)
    except ValueError as error: parser.error(str(error))


if __name__=='__main__': main()
