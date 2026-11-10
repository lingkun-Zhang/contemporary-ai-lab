"""从原始记录独立复算增量缓存结果并画SVG；仅使用标准库。

不运行搜索、不修改协议、原始计时或manifest。已有派生文件可重复生成。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict
from html import escape
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    q = st.quantiles(values, n=4, method='inclusive')
    return st.median(values), q[0], q[2]


def close(actual, expected):
    assert math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-9), (actual, expected)


def summarize(data, source):
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    protocol = json.loads((data/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        assert sha(data/name) == digest, name
    for name, digest in manifest['source_sha256'].items():
        assert sha(source/name) == digest, name
    assert sha(data/'protocol.json') == manifest['protocol_sha256']
    assert protocol['source_sha256'] == manifest['source_sha256']
    raw = read_csv(data/'timings_raw.csv')
    summary = read_csv(data/'timing_summary.csv')
    paired = read_csv(data/'paired_deltas.csv')
    stats = read_csv(data/'search_stats.csv')
    case_specs = {case['case']: case for case in protocol['cases']}
    oracle = {case['case']: case for case in json.loads((data/'oracle.json').read_text(encoding='utf-8'))}
    assert len(case_specs) == len(oracle) == 18
    assert len(raw) == protocol['timed_top_calls'] == manifest['timed_top_calls']
    assert len(stats) == manifest['game_configurations'] == 288
    assert len(summary) == protocol['configurations'] == 576
    lookup, groups = {}, defaultdict(list)
    for row in raw:
        key = (row['case'], row['order'], row['phase'], row['variant'])
        full = (*key, int(row['block']))
        assert full not in lookup, full
        lookup[full] = row
        groups[key].append(row)
        if row['phase'] == 'end_to_end':
            close(row['total_ms'], float(row['build_ms']) + float(row['game_ms']))
        else:
            # game模式总时间还含极小的阶段分派；build_ms=0表示没有建矩阵。
            assert float(row['build_ms']) == 0
            assert float(row['total_ms']) >= float(row['game_ms'])
        n = int(row['treasures'])
        assert int(row['astar_calls']) == ((n+2)*(n+1) if row['phase'] == 'end_to_end' else 0)
    summary_lookup = {}
    for row in summary:
        key = (row['case'], row['order'], row['phase'], row['variant'])
        assert key not in summary_lookup
        summary_lookup[key] = row
        group = groups[key]
        assert sorted(int(r['block']) for r in group) == list(range(protocol['repeats']))
        for metric in ('build_ms', 'game_ms', 'total_ms'):
            expected = describe([float(r[metric]) for r in group])
            for field, value in zip(('median', 'q1', 'q3'), expected):
                close(row[metric+'_'+field], value)
    pair_lookup = {}
    for row in paired:
        key = (row['case'], row['order'], row['phase'], row['candidate'], row['reference'])
        assert key not in pair_lookup
        pair_lookup[key] = row
        case, order, phase, candidate, reference = key
        deltas = [float(lookup[(case,order,phase,candidate,b)]['total_ms']) -
                  float(lookup[(case,order,phase,reference,b)]['total_ms'])
                  for b in range(protocol['repeats'])]
        for field, value in zip(('median','q1','q3'), describe(deltas)):
            close(row['delta_'+field+'_ms'], value)
        assert int(row['faster_blocks']) == sum(v < 0 for v in deltas)

    stat_lookup = {}
    for row in stats:
        case, variant, order = row['case'], row['variant'], row['order']
        key = case, order, variant
        assert key not in stat_lookup
        stat_lookup[key] = row
        visits, internal, leaves, hits = (int(row[k]) for k in
            ('visited_nodes','internal_nodes','leaf_evaluations','cache_hits'))
        assert visits == internal + leaves + hits
        assert hits == sum(int(row[k]) for k in ('cache_exact_hits','cache_lower_hits','cache_upper_hits'))
        if int(row['treasures']) == 4:
            assert hits == 0
        roots = oracle[case]['root_values']
        best = max(roots.values())
        assert int(row['best_value']) == best
        assert roots[row['best_target']] == best
        # 独立重算保存PV的行动/费用账本，不调用被测模块。
        n = int(row['treasures'])
        matrix = oracle[case]['distance_matrix']
        values = case_specs[case]['problem']['values']
        pa, pb, utility, remaining = 0, 1, 0, set(range(n))
        pv = row['pv'].split('>')
        assert len(pv) == n
        for step, label in enumerate(pv):
            action = int(label[1:])-1
            assert action in remaining
            if step % 2 == 0:
                utility += values[action] - matrix[pa][action+2]
                pa = action+2
            else:
                utility -= values[action] - matrix[pb][action+2]
                pb = action+2
            remaining.remove(action)
        assert utility == best

    comparisons = tuple(dict.fromkeys((r['candidate'],r['reference']) for r in paired))
    aggregates = []
    for n in (4,6,8):
        cases = [case for case, spec in case_specs.items() if spec['treasures'] == n]
        assert len(cases) == 6
        for order in protocol['orders']:
            for phase in protocol['phases']:
                for candidate, reference in comparisons:
                    ratios = [float(summary_lookup[(c,order,phase,candidate)]['total_ms_median']) /
                              float(summary_lookup[(c,order,phase,reference)]['total_ms_median']) for c in cases]
                    pairs = [pair_lookup[(c,order,phase,candidate,reference)] for c in cases]
                    med, q1, q3 = describe(ratios)
                    row = dict(treasures=n,order=order,phase=phase,candidate=candidate,
                        reference=reference,inputs=len(cases),time_ratio_median=med,
                        time_ratio_q1=q1,time_ratio_q3=q3,
                        faster_inputs=sum(float(p['delta_median_ms'])<0 for p in pairs),
                        faster_blocks=sum(int(p['faster_blocks']) for p in pairs),
                        total_paired_blocks=len(cases)*protocol['repeats'],
                        paired_delta_median_ms=st.median(float(p['delta_median_ms']) for p in pairs))
                    for metric in ('visited_nodes','internal_nodes','leaf_evaluations'):
                        row[metric+'_ratio_median'] = st.median(
                            int(stat_lookup[(c,order,candidate)][metric]) /
                            int(stat_lookup[(c,order,reference)][metric]) for c in cases)
                    for metric in ('cache_hits','cache_peak_entries'):
                        row['candidate_'+metric+'_median'] = st.median(
                            int(stat_lookup[(c,order,candidate)][metric]) for c in cases)
                    aggregates.append(row)

    theoretical = []
    for n in (4,6,8):
        histories = sum(math.factorial(n)//math.factorial(n-k) for k in range(n))
        states = 1+n+sum(math.comb(n,k)*k*(k-1) for k in range(2,n))
        theoretical.append(dict(treasures=n,nonterminal_histories=histories,
                                structural_nonterminal_keys=states))
        for case, spec in case_specs.items():
            if spec['treasures'] == n:
                for order in protocol['orders']:
                    assert int(stat_lookup[(case,order,'none_mm')]['internal_nodes']) == histories
                    assert int(stat_lookup[(case,order,'incremental_mm')]['internal_nodes']) == states
    representatives = []
    for n in (4,6,8):
        spec = min((s for s in case_specs.values() if s['treasures']==n and s['family']=='open_weighted'),
                   key=lambda s:s['seed'])
        case = spec['case']
        for variant in protocol['variants']:
            representatives.append(dict(case=case,treasures=n,variant=variant,
                **{phase+'_ms':float(summary_lookup[(case,'label',phase,variant)]['total_ms_median'])
                   for phase in protocol['phases']},
                **{metric:int(stat_lookup[(case,'label',variant)][metric]) for metric in
                   ('visited_nodes','internal_nodes','leaf_evaluations','cache_hits','cache_peak_entries')}))
    audit = dict(raw_source_hashes_match=True,raw_timing_rows=len(raw),
        summary_groups=len(summary),paired_groups=len(paired),
        independent_inputs=len(case_specs),timing_blocks=protocol['repeats'],
        stats_rows=len(stats),all_saved_pv_ledgers_recomputed=True,
        all_summaries_and_pairs_recomputed=True,
        full_matrix_astar_calls=sum(int(r['astar_calls']) for r in raw),
        theoretical_counts_are_not_ab_hit_counts=True)
    assert audit['full_matrix_astar_calls'] == manifest['full_matrix_astar_calls']
    return dict(aggregate=aggregates,representatives=representatives,
                theoretical=theoretical,audit=audit)


def draw(data, rows):
    colors={'label':'#337f91','gain':'#b47135'}
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="720" viewBox="0 0 1500 720">',
         '<rect width="1500" height="720" fill="white"/>',
         '<g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">']
    def text(x,y,value,size=22,anchor='start',color='#233e4d'):
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{color}">{escape(str(value))}</text>')
    def line(x1,y1,x2,y2,color='#dce4e9',width=2,extra=''):
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>')
    text(60,60,'收益增量缓存的实际成本边界',36)
    text(60,100,'相对原版 Alpha-Beta；每图先取7块中位数，再取6图中位数与跨图IQR',21)
    selected=[r for r in rows if r['candidate']=='incremental_ab' and r['reference']=='original_ab']
    values=[v for r in selected for v in (r['time_ratio_q1'],r['time_ratio_q3'])]
    lo,hi=.25,2.0
    # 本次数据采用共同的0.25至2范围；复跑若超界则扩展，避免隐藏误差棒。
    while min(values)<lo:lo/=2
    while max(values)>hi:hi*=2
    for pi,phase in enumerate(('game','end_to_end')):
        x0,y0,w,h=140+pi*720,190,530,360
        xp=lambda n:x0+(n-4)/4*w
        yp=lambda v:y0+h-(math.log10(v)-math.log10(lo))/(math.log10(hi)-math.log10(lo))*h
        text(x0+w/2,152,'仅博弈搜索' if phase=='game' else '完整矩阵与博弈',27,'middle')
        ticks=[2.0**p for p in range(-20,21) if lo<=2.0**p<=hi]
        for value in ticks:
            y=yp(value);line(x0,y,x0+w,y);text(x0-15,y+8,f'{value:g}',20,'end')
        line(x0,yp(1),x0+w,yp(1),'#40505b',2,'stroke-dasharray="10 7"')
        line(x0,y0,x0,y0+h,'#788995');line(x0,y0+h,x0+w,y0+h,'#788995')
        for n in (4,6,8):text(xp(n),y0+h+32,n,22,'middle')
        for order in ('label','gain'):
            group=[r for r in selected if r['phase']==phase and r['order']==order]
            coords=' '.join(f"{xp(r['treasures'])},{yp(r['time_ratio_median'])}" for r in group)
            svg.append(f'<polyline points="{coords}" stroke="{colors[order]}" fill="none" stroke-width="4"/>')
            for row in group:
                x=xp(row['treasures']);y=yp(row['time_ratio_median'])
                a,b=yp(row['time_ratio_q1']),yp(row['time_ratio_q3'])
                line(x,a,x,b,colors[order],2)
                for value in (a,b):line(x-7,value,x+7,value,colors[order],2)
                svg.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colors[order]}"/>')
        text(x0+w/2,610,'宝藏数',23,'middle')
    text(60,690,'纵轴为耗时比（对数刻度），虚线1为原版AB；小于1更快，4宝藏为负对照。',21)
    for i,order in enumerate(('label','gain')):
        x=950+i*230;line(x,655,x+40,655,colors[order],4)
        text(x+52,663,'标签顺序' if order=='label' else 'Gain顺序',21)
    svg.append('</g></svg>')
    (data/'incremental_cost_boundary.svg').write_text('\n'.join(svg),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).parent/'results_incremental_cache')
    args=parser.parse_args()
    result=summarize(args.data,Path(__file__).resolve().parent)
    write_csv(args.data/'aggregate_summary.csv',result['aggregate'])
    (args.data/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    draw(args.data,result['aggregate'])
    print(json.dumps(result['audit'],ensure_ascii=False))


if __name__=='__main__':
    main()
