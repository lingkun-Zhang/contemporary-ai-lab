"""题三配对实验：正确性、访问顺序、规模、等号剪枝与采样排序成本。"""
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

from alphabeta_tree import TreeNode, search, validate_tree


BASE = Path(__file__).resolve().parent
SEED = 20261030
ORDERS = ("left", "right", "random", "oracle_best", "oracle_worst")
COLORS = ("#64748b", "#287e9c", "#db8a32", "#8159a1", "#5d965e", "#bc5264")


def oracle(root):
    """独立后序全树计算；不用正式Minimax/Alpha-Beta的返回值。"""
    values, nodes, leaves = {}, 0, 0
    stack = [(root, True, False)]
    while stack:
        node, maximizing, done = stack.pop()
        if not done:
            nodes += 1
            if not node.children:
                values[(id(node), maximizing)] = node.value
                leaves += 1
            else:
                stack.append((node, maximizing, True))
                stack.extend((child, not maximizing, False) for child in reversed(node.children))
        else:
            child_values = [values[(id(child), not maximizing)] for child in node.children]
            values[(id(node), maximizing)] = (max if maximizing else min)(child_values)
    return values[(id(root), True)], values, nodes, leaves


def encode(node):
    return {"children": [encode(child) for child in node.children]} if node.children else {"value": node.value}


def decode(obj):
    return TreeNode(children=[decode(child) for child in obj["children"]]) if obj.get("children") else TreeNode(value=obj["value"])


def full_tree(branching, depth, seed, constant=False):
    rng = random.Random(seed)
    def build(level):
        return TreeNode(value=7 if constant else rng.randint(-100, 100)) if level == depth else TreeNode(children=[build(level+1) for _ in range(branching)])
    return build(0)


def prepare_order(root, mode, seed):
    counters = dict(ordering_probe_leaves=0, ordering_internal_nodes=0, oracle_leaf_reads=0)
    if mode in ("left", "right"):
        return mode, counters
    if mode.startswith("oracle_"):
        _, values, _, leaves = oracle(root)
        counters["oracle_leaf_reads"] = leaves
        def exact_order(node, maximizing):
            counters["ordering_internal_nodes"] += 1
            reverse = maximizing if mode == "oracle_best" else not maximizing
            return sorted(range(len(node.children)), key=lambda i: values[(id(node.children[i]), not maximizing)], reverse=reverse)
        return exact_order, counters
    if mode == "random":
        rng, permutations, stack = random.Random(seed), {}, [root]
        while stack:
            node = stack.pop()
            if node.children:
                indices = list(range(len(node.children)))
                rng.shuffle(indices)
                permutations[id(node)] = indices
                stack.extend(reversed(node.children))
        def random_order(node, maximizing):
            counters["ordering_internal_nodes"] += 1
            return permutations[id(node)]
        return random_order, counters
    if mode.startswith("sample_"):
        budget = int(mode.split("_")[1])
        if budget == 0:
            return "left", counters
        def estimate(child):
            stack, samples = [child], []
            while stack and len(samples) < budget:
                node = stack.pop()
                if node.children:
                    stack.extend(reversed(node.children))
                else:
                    counters["ordering_probe_leaves"] += 1
                    samples.append(node.value)
            return sum(samples) / len(samples)
        def sampled_order(node, maximizing):
            counters["ordering_internal_nodes"] += 1
            scores = [estimate(child) for child in node.children]
            return sorted(range(len(scores)), key=lambda i: scores[i], reverse=maximizing)
        return sampled_order, counters
    raise ValueError(mode)


def execute(case, algorithm, mode, equal):
    started = perf_counter_ns()
    order, counters = prepare_order(case["root"], mode, case["seed"])
    after_prepare = perf_counter_ns()
    result = search(case["root"], algorithm=algorithm, order=order, trace=False,
                    cutoff_on_equal=equal, validate=False)
    finished = perf_counter_ns()
    return result, counters, (finished-started)/1e6, (finished-after_prepare)/1e6


def quantiles(values):
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values)>1 else values*3
    return statistics.median(values), q[0], q[2]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_start(width, height, title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" fill="#273c4c">',
            f'<text x="28" y="38" font-size="25" font-weight="bold">{escape(title)}</text>',
            f'<text x="28" y="68" font-size="14" fill="#536a80">{escape(subtitle)}</text>']


def line_panel(svg, x, y, width, height, xs, series, title, labels=None):
    svg.append(f'<text x="{x}" y="{y}" font-size="17">{escape(title)}</text>')
    left, top, w, h = x+70, y+25, width-100, height-95
    maximum = max([v for _, values, *_ in series for v in values] + [1])*1.10
    for tick in range(5):
        py, v = top+h-tick*h/4, maximum*tick/4
        svg.extend([f'<line x1="{left}" y1="{py}" x2="{left+w}" y2="{py}" stroke="#dce5ed"/>',
                    f'<text x="{left-12}" y="{py+5}" text-anchor="end" font-size="13">{v:.1f}</text>'])
    for i, value in enumerate(xs):
        px = left+w*(i+.5)/len(xs)
        svg.append(f'<text x="{px}" y="{top+h+25}" text-anchor="middle" font-size="14">{escape(str(value))}</text>')
    for k, (name, values, *intervals) in enumerate(series):
        color = COLORS[k % len(COLORS)]
        points=[]
        for i, value in enumerate(values):
            px, py = left+w*(i+.5)/len(xs), top+h-value/maximum*h
            points.append(f'{px},{py}')
            if intervals:
                low, high = intervals[0][i]
                y1,y2=top+h-low/maximum*h,top+h-high/maximum*h
                svg.extend([f'<line x1="{px}" y1="{y1}" x2="{px}" y2="{y2}" stroke="{color}"/>',f'<path d="M{px-4},{y1}h8 M{px-4},{y2}h8" stroke="{color}"/>'])
            svg.append(f'<circle cx="{px}" cy="{py}" r="3.6" fill="{color}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" stroke="{color}" fill="none" stroke-width="2"/>')
        lx,ly=x+70+(k%3)*330,top+h+53+(k//3)*24
        svg.extend([f'<line x1="{lx}" y1="{ly-5}" x2="{lx+22}" y2="{ly-5}" stroke="{color}" stroke-width="3"/>',f'<text x="{lx+30}" y="{ly}" font-size="13">{escape(name)}</text>'])


def figures(output, rows):
    labels={"left":"左到右", "right":"右到左", "random":"固定随机", "oracle_best":"理想好序（oracle）", "oracle_worst":"理想差序（oracle）"}
    svg=svg_start(1200,1160,"分支因子、深度与访问顺序", "每点是3棵固定随机树的平均访问次数；oracle只作排序参照，其完整准备成本另计。")
    for k,b in enumerate((2,3,4)):
        series=[]
        for alg,mode in [("minimax","left")]+[("alphabeta",mode) for mode in ORDERS]:
            vals=[statistics.mean(r["visited_nodes"] for r in rows if r["experiment"]=="scale" and r["branching"]==b and r["depth"]==d and r["algorithm"]==alg and r["order"]==mode) for d in (2,4,6)]
            series.append(("Minimax" if alg=="minimax" else "AB "+labels[mode],vals))
        line_panel(svg,28,105+k*340,1130,280,(2,4,6),series,f"分支因子 b={b}；横轴为边深度，纵轴为节点访问发生次数")
    (output/"ordering_scale.svg").write_text("\n".join(svg+["</g></svg>"]),encoding="utf-8")
    budgets=(0,1,2,4,8,16)
    subset=[r for r in rows if r["experiment"]=="sampling"]
    svg=svg_start(1200,820,"采样排序：减少搜索是否抵得过取样成本", "六棵3分支6层固定树；每个孩子最多探查q个叶值，仅改变顺序；取样不缓存为真实Minimax值。")
    work=[]
    for field,name in [("leaf_evaluations","正式搜索叶评估"),("ordering_probe_leaves","额外取样叶读取")]:
        work.append((name,[statistics.mean(r[field] for r in subset if r["order"]==f"sample_{q}") for q in budgets]))
    line_panel(svg,28,110,1130,290,budgets,work,"横轴 q；纵轴为叶值读取次数（不同操作不等价于同样时间）")
    vals=[statistics.median(r["median_total_ms"] for r in subset if r["order"]==f"sample_{q}") for q in budgets]
    line_panel(svg,28,465,1130,270,budgets,[("各树计时中位数的中位数",vals)],"横轴 q；纵轴为完整查询毫秒（包含取样与排序）；不代表独立确认")
    (output/"sampling_cost.svg").write_text("\n".join(svg+["</g></svg>"]),encoding="utf-8")
    svg=svg_start(1200,450,"相同叶值负载：等号剪枝影响工作量", "相同3分支6层、所有叶值7；三种算法根值均为7；没有剩余孩子的界交叉不计有效剪枝。")
    ties=[next(r for r in rows if r["experiment"]=="equal" and r["algorithm"]==alg and r["equality"]==eq and r["order"]=="left") for alg,eq in [("minimax",True),("alphabeta",True),("alphabeta",False)]]
    names=["Minimax","AB α≥β","AB α>β"]
    maxv=max(r["visited_nodes"] for r in ties)
    for i,(name,r) in enumerate(zip(names,ties)):
        y=140+i*85
        svg.extend([f'<text x="28" y="{y+22}" font-size="16">{escape(name)}</text>',f'<rect x="200" y="{y}" width="{r["visited_nodes"]/maxv*800}" height="38" fill="{COLORS[i]}"/>',f'<text x="{210+r["visited_nodes"]/maxv*800}" y="{y+24}" font-size="15">{r["visited_nodes"]}次 / {r["leaf_evaluations"]}叶</text>'])
    (output/"equality.svg").write_text("\n".join(svg+["</g></svg>"]),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE/"results")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=3)
    args=parser.parse_args()
    if args.repeats<3 or args.batch<1:
        parser.error("repeats至少3，batch至少1")
    output=args.output_dir
    if output.exists() and any(output.iterdir()):
        output=output/("run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(parents=True)
    cases=[]
    sample_data=json.loads((BASE/"samples/pdf_sample.json").read_text(encoding="utf-8"))
    sample=dict(case="pdf_sample",experiment="sample",branching=2,depth=3,seed=SEED,root=decode(sample_data.get("tree_structure",sample_data)))
    cases.append(sample)
    for b in (2,3,4):
        for d in (2,4,6):
            for seed in range(SEED,SEED+3):
                cases.append(dict(case=f"b{b}_d{d}_s{seed}",experiment="scale",branching=b,depth=d,seed=seed,root=full_tree(b,d,seed)))
    cases.append(dict(case="equal_b3_d6",experiment="equal",branching=3,depth=6,seed=SEED,root=full_tree(3,6,SEED,True)))
    for seed in range(SEED,SEED+6):
        cases.append(dict(case=f"sample_b3_d6_s{seed}",experiment="sampling",branching=3,depth=6,seed=seed,root=full_tree(3,6,seed)))
    specs=[]
    for case in cases:
        validate_tree(case["root"])
        expected,_,nodes,leaves=oracle(case["root"])
        case.update(expected=expected,full_nodes=nodes,full_leaves=leaves)
        specs.append({**{k:v for k,v in case.items() if k!="root"},"tree":encode(case["root"])})
    (output/"inputs.json").write_text(json.dumps(specs,ensure_ascii=False),encoding="utf-8")
    configs=[]
    for case in cases:
        if case["experiment"]=="scale":
            settings=[(alg,mode,True) for alg in ("minimax","alphabeta") for mode in ORDERS]
        elif case["experiment"]=="sampling":
            settings=[("alphabeta",f"sample_{q}",True) for q in (0,1,2,4,8,16)]
        else:
            settings=[(alg,mode,eq) for mode in ("left","right") for alg,eq in [("minimax",True),("alphabeta",True),("alphabeta",False)]]
        for alg,mode,eq in settings:
            result,counters,_,_=execute(case,alg,mode,eq)
            if result.value!=case["expected"]:
                raise AssertionError((case["case"],alg,mode,"wrong root value"))
            stats=dataclasses.asdict(result.stats)
            if stats["visited_nodes"]!=stats["internal_nodes"]+stats["leaf_evaluations"]:
                raise AssertionError("node conservation failed")
            if alg=="minimax" and stats["visited_nodes"]!=case["full_nodes"]:
                raise AssertionError("Minimax must traverse every occurrence")
            if not 1<=stats["visited_nodes"]<=case["full_nodes"]:
                raise AssertionError("invalid visited count")
            configs.append((case,alg,mode,eq,stats,counters))
    raw=[]
    rng=random.Random(SEED)
    for repeat in range(args.repeats):
        shuffled=list(enumerate(configs)); rng.shuffle(shuffled)
        for index,(case,alg,mode,eq,stats,counters) in shuffled:
            totals,search_times=[],[]
            for _ in range(args.batch):
                result,actual,total_ms,search_ms=execute(case,alg,mode,eq)
                totals.append(total_ms); search_times.append(search_ms)
                if result.value!=case["expected"] or dataclasses.asdict(result.stats)!=stats or actual!=counters:
                    raise AssertionError("timed result/counters changed")
            raw.append(dict(config=index,case=case["case"],algorithm=alg,order=mode,equality=eq,repeat=repeat,batch=args.batch,total_ms=statistics.mean(totals),search_ms=statistics.mean(search_times)))
        print(f"Timing block {repeat+1}/{args.repeats} verified",flush=True)
    rows=[]
    for index,(case,alg,mode,eq,stats,counters) in enumerate(configs):
        batches=[r for r in raw if r["config"]==index]
        mt,q1,q3=quantiles([r["total_ms"] for r in batches])
        st,s1,s3=quantiles([r["search_ms"] for r in batches])
        rows.append({"config":index,**{k:v for k,v in case.items() if k!="root"},"algorithm":alg,"order":mode,"equality":eq,"value":case["expected"],**stats,**counters,"median_total_ms":mt,"q1_total_ms":q1,"q3_total_ms":q3,"median_search_ms":st,"q1_search_ms":s1,"q3_search_ms":s3})
    write_csv(output/"comparison.csv",rows)
    write_csv(output/"timing_batches.csv",raw)
    trace=search(sample["root"],trace=True)
    trace_obj=dataclasses.asdict(trace)
    def finite(obj):
        if isinstance(obj,float) and not math.isfinite(obj): return str(obj)
        if isinstance(obj,dict): return {k:finite(v) for k,v in obj.items()}
        if isinstance(obj,(tuple,list)): return [finite(v) for v in obj]
        return obj
    (output/"sample_trace.json").write_text(json.dumps(finite(trace_obj),ensure_ascii=False,indent=2),encoding="utf-8")
    figures(output,rows)
    source_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/"alphabeta_tree.py",Path(__file__),BASE/"samples/pdf_sample.json")}
    metadata=dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),seed=SEED,repeats=args.repeats,batch=args.batch,cases=len(cases),distinct_inputs=len({json.dumps(encode(c["root"]),sort_keys=True) for c in cases}),configurations=len(configs),timing_batches=len(raw),timed_solver_calls=len(raw)*args.batch,source_sha256=source_hash,input_sha256=hashlib.sha256((output/"inputs.json").read_bytes()).hexdigest(),timing_boundary="Prepared TreeNode input; validation, output and trace excluded. Total includes ordering preparation/probes/sort. Search-only excludes preparation, includes callback probes/sort. Seven randomized blocks; no simultaneous timed workers.",oracle_warning="Exact full-tree values require prior whole-tree evaluation, counted in total; oracle order is an explanatory reference, not a deployable speed claim.",sampling_warning="Same trees, approximate leaf mean affects order only; no admissibility or exact-value claim; budget is exploratory, not independently confirmed.")
    (output/"reproducibility.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"outcomes.json").write_text(json.dumps(dict(all_root_values_correct=True,all_timing_counters_stable=True,node_conservation=True,configuration_count=len(rows),input_count=len(cases),distinct_tree_inputs=metadata["distinct_inputs"]),indent=2),encoding="utf-8")
    print(f"Saved {len(rows)} configurations, {len(raw)} timing batches to {output}")


if __name__=="__main__":
    main()
