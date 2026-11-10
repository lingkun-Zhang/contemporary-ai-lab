"""只从保存的原始记录产生汇总与图，不修改或过滤计时记录。"""
from pathlib import Path
import csv
import json
import math
import statistics as st
import argparse
from html import escape

parser=argparse.ArgumentParser(description="从走廊实验原始汇总生成跨图结果和SVG；只使用标准库")
parser.add_argument('--data', type=Path, default=Path(__file__).parent/'results/corridor_study')
args=parser.parse_args()
DATA=args.data

def read(name):
    with (DATA/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def median(values):
    return st.median(values)

def quartile(values):
    return st.quantiles(values, n=4, method='inclusive')[::2]

summary = read('timing_summary.csv')
paired = read('paired_deltas.csv')
structure = read('structures.csv')
stats = read('query_stats.csv')
lookup = {(r['case'],int(r['queries']),r['variant']):r for r in summary}
plook = {(r['case'],int(r['queries']),r['candidate'],r['reference']):r for r in paired}
families = ('open','random','maze','snake')
labels = {'open':'开阔图','random':'随机障碍','maze':'完美迷宫','snake':'蛇形长廊'}
colors = {'open':'#6b788a','random':'#ad6846','maze':'#287f87','snake':'#4f6c9c'}
family_rows = []
for family in families:
    maps = [r for r in structure if r['family']==family]
    for q in (1,4,16,64):
        for candidate, reference in (('corridor_low_h','grid_low_h'),
                                    ('corridor_low_h','identity_low_h'),
                                    ('corridor_landmark','landmark_low_h'),
                                    ('corridor_landmark','identity_landmark'),
                                    ('corridor_landmark','corridor_low_h')):
            ratios = [float(lookup[(r['case'],q,candidate)]['total_ms_median']) /
                      float(lookup[(r['case'],q,reference)]['total_ms_median']) for r in maps]
            deltas = [float(plook[(r['case'],q,candidate,reference)]['delta_median_ms']) for r in maps]
            family_rows.append(dict(family=family,queries=q,candidate=candidate,
                reference=reference,maps=len(maps),ratio_median=median(ratios),
                ratio_q1=quartile(ratios)[0],ratio_q3=quartile(ratios)[1],
                faster_maps=sum(x<0 for x in deltas),
                paired_delta_median_ms=median(deltas)))
with (DATA/'family_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(family_rows[0]));w.writeheader();w.writerows(family_rows)

per_map=[]
for r in structure:
    case=r['case']
    row=dict(case=case,family=r['family'],size=int(r['size']),
        walkable=int(r['walkable_cells']),retained=int(r['retained_nodes']),
        degree2_fraction=int(r['degree_two_cells'])/int(r['walkable_cells']),
        solved=int(r['solved_queries']),failed=64-int(r['solved_queries']))
    for variant in ('grid_low_h','landmark_low_h','identity_low_h','identity_landmark',
                    'corridor_low_h','corridor_landmark'):
        s=lookup[(case,64,variant)]
        row[variant+'_ms']=float(s['total_ms_median'])
        row[variant+'_build_ms']=float(s['build_ms_median'])
        row[variant+'_query_ms']=float(s['query_ms_median'])
        row[variant+'_expanded']=sum(int(x['expanded_nodes']) for x in stats
                                 if x['case']==case and x['variant']==variant)
    first=[q for q in (1,4,16,64) if float(plook[(case,q,'corridor_low_h','grid_low_h')]['delta_median_ms'])<0]
    row['first_tested_faster_count']=first[0] if first else '>64'
    per_map.append(row)
(DATA/'analysis.json').write_text(json.dumps(dict(family_summary=family_rows,per_map=per_map),
    ensure_ascii=False,indent=2),encoding='utf-8')


# 以下图形只展示保存记录的派生统计，不删改原始耗时。
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1900" height="900" viewBox="0 0 1900 900">',
     '<rect width="1900" height="900" fill="white"/>',
     '<g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">']
def text(x,y,value,size=25,color='#253a46',anchor='start'):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')
def line(x1,y1,x2,y2,color='#e5e9ec',width=2,extra=''):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>')
text(72,75,'走廊压缩何时值得预处理',44,'#172d3b')
text(72,122,'总成本包括建图、端点接入、搜索和逐格路径还原；虚线=原版 low_h')
text(72,168,'压缩/原版总耗时比（对数刻度，越低越快）',24)
x0,y0,w,h=175,185,1100,560
def xp(q):return x0+math.log(q,4)/3*w
all_q=[v for r in family_rows if r['candidate']=='corridor_low_h' and r['reference']=='grid_low_h'
       for v in (r['ratio_q1'],r['ratio_q3'])]
lo=min(.05,10**math.floor(math.log10(min(all_q))))
hi=max(10,10**math.ceil(math.log10(max(all_q))))
def yp(r):return y0+h-(math.log10(r)-math.log10(lo))/(math.log10(hi)-math.log10(lo))*h
ticks=[v for power in range(-4,4) for v in (10**power,2*10**power,5*10**power) if lo<=v<=hi]
for value in ticks:
    y=yp(value);line(x0,y,x0+w,y);text(x0-20,y+8,f'{value:g}',25,'#455660','end')
for q in (1,4,16,64):
    x=xp(q);line(x,y0,x,y0+h,'#eef1f3',1);text(x,y0+h+36,q,27,'#455660','middle')
line(x0,yp(1),x0+w,yp(1),'#2b3943',3,'stroke-dasharray="12 8"')
line(x0,y0,x0,y0+h,'#697782');line(x0,y0+h,x0+w,y0+h,'#697782')
for fi,family in enumerate(families):
    rows=[r for r in family_rows if r['family']==family and r['candidate']=='corridor_low_h' and r['reference']=='grid_low_h']
    points=[(xp(r['queries']),yp(r['ratio_median'])) for r in rows]
    coords=' '.join(f'{x},{y}' for x,y in points)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[family]}" stroke-width="5"/>')
    for r,(x,y) in zip(rows,points):
        a,b=yp(r['ratio_q1']),yp(r['ratio_q3']);line(x,b,x,a,colors[family],3)
        for z in (a,b):line(x-8,z,x+8,z,colors[family],3)
        svg.append(f'<circle cx="{x}" cy="{y}" r="7" fill="{colors[family]}"/>')
    ly=245+fi*95;line(1370,ly,1420,ly,colors[family],5)
    text(1440,ly+10,f"{labels[family]}（{rows[0]['maps']}图）",27)
text(x0+w/2,834,'同图查询次数（对数刻度）',28,anchor='middle')
text(1350,687,'点：跨图中位数',23);text(1350,725,'竖线：跨图四分位区间',23)
text(1350,763,'每图：7个重复计时块',23)
svg.append('</g></svg>')
(DATA/'corridor_cost_curve.svg').write_text('\n'.join(svg),encoding='utf-8')
print('已生成family_summary.csv、analysis.json和corridor_cost_curve.svg；原始记录保持原样。')
