"""Register then compare Boolean Scout with matched/original Alpha-Beta.

Standard library only, Python3.9+. Default: 30 ordered inputs, 3 orders,
3 implementations, 7 shuffled serial blocks with 3 calls per batch.
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
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from time import perf_counter_ns

from alphabeta_tree import TreeNode, sample_tree, search, validate_tree
from scout_search import ScoutStats, scout_search

METHODS = ('original_ab', 'matched_ab', 'scout')
ORDERS = ('left', 'right', 'fixed_random')
SOURCES = ('alphabeta_tree.py', 'scout_search.py', 'test_scout_search.py', 'run_scout_study.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def encode(node):
    return {'value':node.value} if not node.children else {'children':[encode(c) for c in node.children]}


def oracle(node, maximizing=True):
    """Independent unpruned full recursion; no tested search is called."""
    if not node.children:
        return node.value, 1, 1
    results = [oracle(child, not maximizing) for child in node.children]
    value = (max if maximizing else min)(r[0] for r in results)
    return value, 1+sum(r[1] for r in results), sum(r[2] for r in results)


def random_tree(b, d, seed):
    rng = random.Random(seed)
    def make(depth):
        if depth == 0:
            return TreeNode(rng.randint(-1000, 1000))
        return TreeNode(children=[make(depth-1) for _ in range(b)])
    return make(d)


def structured_pair(b, d):
    """Position weights construct an ordered input, without oracle sorting.

    A high digit at MAX / low digit at MIN dominates all following digits.
    Mirroring every child list preserves unordered game semantics and changes
    the first child from best to worst. This is a boundary input, not a policy.
    """
    def make(level, value):
        if level == d:
            return TreeNode(value)
        maximizing = level % 2 == 0
        weight = b ** (d-level-1)
        return TreeNode(children=[make(level+1, value +
            (b-1-i if maximizing else i)*weight) for i in range(b)])
    def mirror(node):
        if not node.children:
            return TreeNode(node.value)
        return TreeNode(children=[mirror(c) for c in reversed(node.children)])
    first = make(0, 0)
    return first, mirror(first)


def build_cases(seed):
    cases = []
    for b in (2, 3, 4):
        for d in (4, 6):
            for replica in range(3):
                item_seed = seed*100 + b*1000 + d*10 + replica
                cases.append(dict(case=f'random_b{b}_d{d}_{replica}',group='random',
                    branching=b,depth=d,seed=item_seed,root=random_tree(b,d,item_seed)))
    for b in (3,4):
        for d in (4,6):
            for group,root in zip(('best_first','late_improvement'),structured_pair(b,d)):
                cases.append(dict(case=f'{group}_b{b}_d{d}',group=group,
                    branching=b,depth=d,seed=None,root=root))
    def n(*args):
        return TreeNode(children=[x if isinstance(x,TreeNode) else TreeNode(x) for x in args])
    equal = random_tree(3,6,seed)
    def set_equal(node):
        if not node.children:node.value=7
        for child in node.children:set_equal(child)
    set_equal(equal)
    special = [('sample',sample_tree()),('equal',equal),
        ('mixed_float',n(.1,n(-.4,.2,n(.7,.9)),n(n(-.8,.3),.5))),
        ('negative',n(n(-10,-3),n(-9,n(-2,-7)),n(-6,-5,-8)))]
    for group,root in special:
        cases.append(dict(case=group,group=group,branching=None,depth=None,seed=None,root=root))
    assert len(cases)==30
    return cases


def fixed_order(root, seed):
    """Prepare a reproducible per-object random permutation, inside timing."""
    rng, pending, table = random.Random(seed), [root], {}
    while pending:
        node = pending.pop()
        if id(node) in table:
            continue
        indices = list(range(len(node.children)))
        rng.shuffle(indices)
        table[id(node)] = indices
        pending.extend(reversed(node.children))
    return lambda node, maximizing: table[id(node)]


def execute(case, method, order, order_seed):
    started = perf_counter_ns()
    actual_order = fixed_order(case['root'],order_seed) if order=='fixed_random' else order
    prepared = perf_counter_ns()
    if method == 'original_ab':
        result = search(case['root'],order=actual_order,validate=False)
    else:
        result = scout_search(case['root'],mode='scout' if method=='scout' else 'alphabeta',
                              order=actual_order,validate=False)
    ended = perf_counter_ns()
    if method == 'original_ab':
        counts = asdict(ScoutStats(exact_visits=result.stats.visited_nodes,
            exact_internal_visits=result.stats.internal_nodes,
            exact_leaf_reads=result.stats.leaf_evaluations,
            exact_cutoffs=result.stats.cutoff_events,
            total_visits=result.stats.visited_nodes,
            total_leaf_reads=result.stats.leaf_evaluations))
    else:
        counts = asdict(result.stats)
    return result.value, counts, dict(preparation_ms=(prepared-started)/1e6,
            search_ms=(ended-prepared)/1e6,total_ms=(ended-started)/1e6)


def check(case, method, value, counts, expected_counts=None):
    assert value==case['expected'],(case['case'],method,value,case['expected'])
    assert counts['total_visits']==counts['exact_visits']+counts['test_visits']
    assert counts['total_leaf_reads']==counts['exact_leaf_reads']+counts['test_leaf_reads']
    assert counts['research_calls']==counts['test_true']
    assert counts['probe_calls']==counts['test_true']+counts['test_false']
    if expected_counts is not None:
        assert counts==expected_counts,(case['case'],method,'nondeterministic statistics')


def distribution(values):
    quartiles = st.quantiles(values,n=4,method='inclusive')
    return dict(median=st.median(values),q1=quartiles[0],q3=quartiles[2])


def summarize(comparison,batches):
    groups,lookup = {},{}
    for row in batches:
        key = row['case'],row['order'],row['method']
        groups.setdefault(key,[]).append(row)
        lookup[(row['case'],row['order'],row['block'],row['method'])]=row
    results=[]
    for row in comparison:
        group=groups[(row['case'],row['order'],row['method'])]
        result=dict(row)
        for metric in ('preparation_ms','search_ms','total_ms'):
            for name,value in distribution([r[metric] for r in group]).items():
                result[metric+'_'+name]=value
        results.append(result)
    paired=[]
    for case,order in dict.fromkeys((r['case'],r['order']) for r in comparison):
        blocks=sorted({r['block'] for r in batches if r['case']==case and r['order']==order})
        group=next(r['group'] for r in comparison if r['case']==case)
        for reference in ('matched_ab','original_ab'):
            deltas=[lookup[(case,order,b,'scout')]['total_ms']-
                    lookup[(case,order,b,reference)]['total_ms'] for b in blocks]
            paired.append(dict(case=case,group=group,order=order,candidate='scout',reference=reference,
                delta_median_ms=st.median(deltas),delta_q1_ms=distribution(deltas)['q1'],
                delta_q3_ms=distribution(deltas)['q3'],faster_blocks=sum(v<0 for v in deltas),blocks=len(blocks)))
    return results,paired


def aggregate(results,paired):
    lookup={(r['case'],r['order'],r['method']):r for r in results}
    rows=[]
    for group,order in dict.fromkeys((r['group'],r['order']) for r in results):
        cases=list(dict.fromkeys(r['case'] for r in results if r['group']==group and r['order']==order))
        for reference in ('matched_ab','original_ab'):
            ratios=[lookup[(case,order,'scout')]['total_ms_median']/lookup[(case,order,reference)]['total_ms_median'] for case in cases]
            leaf_ratios=[lookup[(case,order,'scout')]['total_leaf_reads']/lookup[(case,order,reference)]['total_leaf_reads'] for case in cases]
            diffs=[next(r['delta_median_ms'] for r in paired if r['case']==case and r['order']==order and r['reference']==reference) for case in cases]
            rows.append(dict(group=group,order=order,reference=reference,ordered_inputs=len(cases),
                time_ratio_median=st.median(ratios),leaf_ratio_median=st.median(leaf_ratios),
                faster_inputs=sum(d<0 for d in diffs),delta_median_ms=st.median(diffs)))
    return rows


def draw_svg(output,results):
    """Compare actual leaf reads, including all repeated TEST work."""
    wanted=('best_first_b4_d6','late_improvement_b4_d6','random_b4_d6_0','equal')
    labels=('首孩子最佳','后置孩子逐次改善','独立随机树','全同分树')
    selected=[r for case in wanted for r in results if r['case']==case and r['order']=='left' and r['method'] in ('matched_ab','scout')]
    maximum=max(r['total_leaf_reads'] for r in selected)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="660" viewBox="0 0 1200 660">',
         '<rect width="1200" height="660" fill="white"/><g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">',
         '<text x="65" y="48" font-size="29" fill="#203947">Scout省下的求值是否抵消阈值探测与重搜</text>',
         '<text x="65" y="84" font-size="18" fill="#52616a">左序；代表输入固定为b4d6编号0及全同分树；所有重复叶读取均计入</text>']
    x0,y0,w,h=100,120,1040,390
    for i in range(6):
        value=maximum*i/5;y=y0+h-h*i/5
        svg.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#e3e8eb"/>')
        svg.append(f'<text x="{x0-14}" y="{y+6}" text-anchor="end" font-size="16">{value:.0f}</text>')
    for gi,(case,label) in enumerate(zip(wanted,labels)):
        center=x0+130+260*gi
        for mi,method in enumerate(('matched_ab','scout')):
            r=next(v for v in selected if v['case']==case and v['method']==method)
            x=center-80+mi*90
            heights=(r['exact_leaf_reads']/maximum*h,r['test_leaf_reads']/maximum*h)
            y=y0+h
            for part,color in zip(heights,('#438795','#d19048')):
                y-=part
                svg.append(f'<rect x="{x}" y="{y}" width="62" height="{part}" fill="{color}"/>')
            svg.append(f'<text x="{x+31}" y="{y-8}" font-size="16" text-anchor="middle">{r["total_leaf_reads"]}</text>')
            svg.append(f'<text x="{x+31}" y="{y0+h+27}" font-size="16" text-anchor="middle">{"AB" if mi==0 else "Scout"}</text>')
        svg.append(f'<text x="{center-4}" y="{y0+h+62}" font-size="18" text-anchor="middle">{escape(label)}</text>')
    svg.extend(['<rect x="320" y="615" width="20" height="16" fill="#438795"/><text x="352" y="630" font-size="17">EVAL/AB阶段读叶</text>',
        '<rect x="660" y="615" width="20" height="16" fill="#d19048"/><text x="692" y="630" font-size="17">TEST阶段读叶</text></g></svg>'])
    (output/'scout_leaf_work.svg').write_text('\n'.join(svg),encoding='utf-8')


def prepare_protocol(output,seed,repeats,batch):
    cases=build_cases(seed)
    specifications=[]
    for index,case in enumerate(cases):
        case['order_seed']=seed+770000+index
        specifications.append({**{k:v for k,v in case.items() if k!='root'},'tree':encode(case['root'])})
    base=Path(__file__).resolve().parent
    protocol=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        registration_phase='before any oracle, solver warmup or timed call',
        seed=seed,repeats=repeats,calls_per_batch=batch,methods=METHODS,orders=ORDERS,
        cases=specifications,source_sha256={name:sha(base/name) for name in SOURCES},
        design=dict(inputs='30 ordered inputs:18 independent random trees;4 constructed unordered semantics each with best-first/late-improvement representations;4 special inputs',
            construction='positional base-b leaf digits construct first-best ordering; no oracle-based runtime reorder or free true-value lookahead',
            scope='full finite tree, no truncation/iterative deepening/transposition table; finite int/float leaves',
            primary='Scout minus same-module AB complete time and all TEST+EVAL leaf reads, paired on same tree/order/block',
            secondary='original AB practical baseline; instrumentation and implementation differ',
            random_order='fixed per-node permutation rebuilt within every timed call and included in total preparation cost',
            timing='serial; one discarded warmup per configuration; shuffled case/order tasks and shuffled methods per block; default GC',
            included='order preparation, all TEST/EVAL visits, counters and all re-searches; output format conversion outside timer for all methods',
            excluded='tree construction, structural validation, independent minimax oracle, diagnostics, trace, serialization and charts',
            units='7 repeated blocks do not add independent trees; paired IQR is descriptive, not a confidence interval',
            honest_results='no performance-based input filtering; retain every loss and non-beneficial case'))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'protocol.json').open('x',encoding='utf-8') as stream:
        json.dump(protocol,stream,ensure_ascii=False,indent=2)
    return cases,protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).parent/'results_scout')
    parser.add_argument('--seed',type=int,default=20261110)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--batch',type=int,default=3)
    parser.add_argument('--prepare-only',action='store_true',help='write fixed protocol only; no solver/oracle/timing')
    args=parser.parse_args()
    if args.repeats<3 or args.batch<1:parser.error('repeats>=3 and batch>=1 required')
    output=args.output
    if output.exists() and any(output.iterdir()):
        output=output/('rerun_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    cases,protocol=prepare_protocol(output,args.seed,args.repeats,args.batch)
    print('固定协议已保存：'+str(output/'protocol.json'),flush=True)
    if args.prepare_only:return
    counts_by_key,comparison={},[]
    for case in cases:
        validate_tree(case['root'])
        case['expected'],case['full_nodes'],case['full_leaves']=oracle(case['root'])
        for order in ORDERS:
            for method in METHODS:
                value,counts,_=execute(case,method,order,case['order_seed'])
                check(case,method,value,counts)
                counts_by_key[(case['case'],order,method)]=counts
                comparison.append(dict(case=case['case'],group=case['group'],order=order,method=method,
                    value=value,full_nodes=case['full_nodes'],full_leaves=case['full_leaves'],**counts))
        print('已核验/预热 '+case['case'],flush=True)
    for b in (3,4):
        for d in (4,6):
            pair=[c for c in cases if c['case'] in (f'best_first_b{b}_d{d}',f'late_improvement_b{b}_d{d}')]
            assert pair[0]['expected']==pair[1]['expected']
    write_csv(output/'comparison.csv',comparison)
    calls,batches=[],[]
    rng=random.Random(args.seed+990000)
    tasks=[(case,order) for case in cases for order in ORDERS]
    for block in range(args.repeats):
        rng.shuffle(tasks)
        for case,order in tasks:
            methods=list(METHODS);rng.shuffle(methods)
            for method in methods:
                times=[]
                for call in range(args.batch):
                    value,counts,timing=execute(case,method,order,case['order_seed'])
                    check(case,method,value,counts,counts_by_key[(case['case'],order,method)])
                    times.append(timing)
                    calls.append(dict(case=case['case'],group=case['group'],order=order,method=method,
                        block=block,call=call,**timing))
                batches.append(dict(case=case['case'],group=case['group'],order=order,method=method,
                    block=block,**{metric:st.mean(t[metric] for t in times)
                        for metric in ('preparation_ms','search_ms','total_ms')}))
        write_csv(output/'timing_calls.csv',calls)
        write_csv(output/'timing_batches.csv',batches)
        print(f'正式计时块 {block+1}/{args.repeats} 完成',flush=True)
    results,paired=summarize(comparison,batches)
    groups=aggregate(results,paired)
    write_csv(output/'summary.csv',results)
    write_csv(output/'paired_deltas.csv',paired)
    write_csv(output/'aggregate_summary.csv',groups)
    draw_svg(output,results)
    save_json(output/'analysis.json',dict(aggregate=groups,
        representative_left=[r for r in results if r['order']=='left' and r['case'] in
            ('best_first_b4_d6','late_improvement_b4_d6','random_b4_d6_0','equal','sample','mixed_float')]))
    save_json(output/'manifest.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,platform=platform.platform(),source_sha256=protocol['source_sha256'],
        protocol_sha256=sha(output/'protocol.json'),ordered_input_records=30,
        independent_random_trees=18,structured_underlying_semantics=4,
        configurations=len(comparison),timing_batches=len(batches),timed_calls=len(calls),
        all_values_and_count_identities_validated=True,
        files_sha256={path.name:sha(path) for path in output.iterdir() if path.is_file()}))
    print('Scout实验完成；所有根值和重复工作统计均核验。',flush=True)


if __name__=='__main__':
    main()
