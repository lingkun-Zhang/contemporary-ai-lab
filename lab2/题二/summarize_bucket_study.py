"""从保存的CSV生成费用倍率图；仅使用标准库，不修改原始记录。"""
import argparse
import csv
from html import escape
import json
import math
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,default=Path(__file__).parent/'results_bucket')
args=parser.parse_args()
with (args.data/'summary.csv').open(encoding='utf-8-sig',newline='') as stream:
    summary=list(csv.DictReader(stream))
lookup={(r['case'],int(r['q']),r['tie'],r['method']):r for r in summary}
cases=sorted({r['case'] for r in summary if r['family']=='zero_random'})
assert len(cases)==3
qs=(1,2,4,8,16,64,256)
def percentile(xs,p):
    xs=sorted(xs);position=(len(xs)-1)*p;a=int(position);b=min(a+1,len(xs)-1)
    return xs[a]+(xs[b]-xs[a])*(position-a)
curves={}
for method in ('heap','bucket','bucket_gcd'):
    curves[method]=[]
    for q in qs:
        values=[float(lookup[(case,q,'low_h',method)]['total_ms_median'])/
                float(lookup[(case,q,'low_h','original')]['total_ms_median']) for case in cases]
        curves[method].append(dict(q=q,median=percentile(values,.5),
                                   q1=percentile(values,.25),q3=percentile(values,.75)))
svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="800" viewBox="0 0 1400 800">',
     '<rect width="1400" height="800" fill="white"/>',
     '<g font-family="Microsoft YaHei,Noto Sans CJK SC,sans-serif">']
def text(x,y,value,size=24,fill='#253a46',anchor='start'):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}">{escape(str(value))}</text>')
def line(x1,y1,x2,y2,fill='#dfe6ea',width=2,extra=''):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{fill}" stroke-width="{width}" {extra}/>')
text(60,65,'费用单位改变时，队列节省能否抵消准备成本',36)
text(60,110,'零权随机图、low_h、完整调用成本；均相对同倍率原求解器',24)
text(60,148,'点为3张图的中位比，竖线为跨图四分位范围；每图内部先汇总7个计时块',20,'#536774')
x0,y0,w,h=135,210,870,450
lo=.75
hi=math.ceil(max(row['q3'] for rows in curves.values() for row in rows)*2)/2
def xp(q):return x0+math.log2(q)/8*w
def yp(value):return y0+h-(value-lo)/(hi-lo)*h
tick=.5
value=1.
while value<=hi:
    y=yp(value);line(x0,y,x0+w,y);text(x0-20,y+8,f'{value:g}',22,anchor='end');value+=tick
for q in qs:
    x=xp(q);line(x,y0,x,y0+h,'#eef1f3',1);text(x,y0+h+36,q,23,anchor='middle')
line(x0,yp(1),x0+w,yp(1),'#263e4a',3,'stroke-dasharray="10 7"')
line(x0,y0,x0,y0+h,'#657985');line(x0,y0+h,x0+w,y0+h,'#657985')
colors={'heap':'#6c7a8f','bucket':'#ba7046','bucket_gcd':'#2a8187'}
labels={'heap':'同内核堆','bucket':'原始桶','bucket_gcd':'公因数归一化桶'}
for rank,(method,rows) in enumerate(curves.items()):
    coords=' '.join(f"{xp(r['q'])},{yp(r['median'])}" for r in rows)
    svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[method]}" stroke-width="4"/>')
    for r in rows:
        x,y=xp(r['q']),yp(r['median']);a,b=yp(r['q1']),yp(r['q3'])
        line(x,a,x,b,colors[method],2)
        for z in (a,b):line(x-6,z,x+6,z,colors[method],2)
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{colors[method]}"/>')
    ly=280+rank*90;line(1060,ly,1110,ly,colors[method],4)
    text(1130,ly+8,labels[method],20)
text(1060,585,'虚线=原求解器',20,'#536774')
text(1060,625,'所有曲线仍高于1',20,'#536774')
text(x0+w/2,745,'普通格费用倍率q（对数刻度）',25,anchor='middle')
text(135,190,'完整耗时比（越低越快）',22)
svg.append('</g></svg>')
(args.data/'bucket_cost_scale.svg').write_text('\n'.join(svg),encoding='utf-8')
(args.data/'bucket_plot_data.json').write_text(json.dumps(curves,ensure_ascii=False,indent=2),encoding='utf-8')
print('已生成bucket_cost_scale.svg与其派生数据；原始记录未改写。')
