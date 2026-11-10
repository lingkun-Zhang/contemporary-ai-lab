"""题一节点重开探究：反例族、随机地图、正确性消融与可复现证据。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import statistics
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from maze_astar import astar_search, manhattan
from reopen_probe import analyze_heuristic, corridor_case, probe_astar, table_heuristic
from run_experiments import svg_start, write_csv


def bfs_distances(grid, goal):
    """独立的四方向 BFS，既不调用 A* 也不调用实验内核。"""
    distances, queue = {goal: 0}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and
                    grid[nr][nc] != "X" and (nr,nc) not in distances):
                distances[(nr,nc)] = distances[(r,c)] + 1
                queue.append((nr,nc))
    return distances


def check_path(grid, start, goal, result):
    if result.cost < 0:
        assert result.path == []
        return
    assert result.path[0] == start and result.path[-1] == goal
    assert len(result.path)-1 == result.cost
    for r,c in result.path:
        assert 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "X"
    assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(result.path,result.path[1:]))


def heuristic_table(grid, goal, kind, lowered):
    return {(r,c): 0 if kind == "zero" or kind == "lowered_manhattan" and (r,c) in lowered
            else manhattan((r,c),goal)
            for r,row in enumerate(grid) for c,cell in enumerate(row) if cell != "X"}


def run_case(slug, cohort, grid, start, goal, lowered, extra, trace=False):
    exact = bfs_distances(grid, goal)
    optimum = exact.get(start, -1)
    rows, details, traces = [], [], []
    for kind in ("zero", "manhattan", "lowered_manhattan"):
        table = heuristic_table(grid, goal, kind, lowered)
        h = table_heuristic(table)
        analysis = analyze_heuristic(grid, goal, h, exact)
        assert analysis["admissibility_violations"] == analysis["invalid_values"] == analysis["goal_value"] == 0
        core = astar_search(start, goal, grid, heuristic=h)
        assert core.cost == optimum, f"Existing core failed: {slug}/{kind}"
        for allow in (True, False):
            result = probe_astar(start, goal, grid, h, allow, trace)
            check_path(grid, start, goal, result)
            if allow:
                assert result.cost == optimum
                assert result.path == core.path
                for metric in core.stats:
                    assert result.stats[metric] == core.stats[metric], (slug, kind, metric)
            row = {"case": slug, "cohort": cohort, "rows": len(grid), "cols": len(grid[0]),
                   **extra, "heuristic": kind, "allow_reopen": allow,
                   "reachable": optimum >= 0, "optimal_cost": optimum, "returned_cost": result.cost,
                   "path_node_count": len(result.path), "path_matches_reported_cost": True,
                   "optimality_failure": result.cost != optimum,
                   "extra_steps": result.cost-optimum if optimum >= 0 else "",
                   "cost_over_optimum": result.cost/optimum if optimum > 0 else "",
                   "admissibility_violations": analysis["admissibility_violations"],
                   "consistency_violation_edges": analysis["consistency_violation_edges"],
                   "heuristic_is_consistent": analysis["consistency_violation_edges"] == 0,
                   "core_cost": core.cost, "core_reopened_nodes": core.stats["reopened_nodes"],
                   **result.stats}
            rows.append(row)
            if trace and kind == "lowered_manhattan":
                traces.append((allow, result))
        details.append({"heuristic": kind, "h_table": [[table.get((r,c), "X")
                        for c in range(len(grid[0]))] for r in range(len(grid))],
                        "checks": analysis})
    return rows, {"case": slug, "cohort": cohort, "grid": ["".join(row) for row in grid],
                  "start": start, "goal": goal, "parameters": extra, "heuristics": details}, traces


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["cohort"], row["heuristic"], row["allow_reopen"],
               row["reachable"], row["heuristic_is_consistent"])
        groups[key].append(row)
    summaries = []
    for (cohort, kind, allow, reachable, consistent), group in groups.items():
        gaps = [row["extra_steps"] for row in group] if reachable else []
        failures = sum(row["optimality_failure"] for row in group)
        summaries.append({"cohort": cohort, "heuristic": kind, "allow_reopen": allow,
                          "reachable": reachable, "heuristic_is_consistent": consistent,
                          "n_inputs": len(group), "optimality_failures": failures,
                          "failure_fraction": failures/len(group),
                          "extra_steps_mean": statistics.mean(gaps) if gaps else "",
                          "extra_steps_max": max(gaps) if gaps else "",
                          "reopened_nodes_mean": statistics.mean(row["reopened_nodes"] for row in group),
                          "ignored_closed_improvements_sum": sum(row["ignored_closed_improvements"] for row in group),
                          "expanded_nodes_median": statistics.median(row["expanded_nodes"] for row in group)})
    return summaries


def counterexample_svg(path, grid, h_table, traces):
    parts = svg_start(1140, 680, "可采纳但不一致：关闭节点后还要不要重新扩展？")
    parts.append('<text class="note" x="24" y="60">同一内核、同一 FIFO 顺序，只改变重开策略；红色边框表示被再次扩展的格子。</text>')
    cell, top = 49, 112
    for index, (allow, result) in enumerate(traces):
        left = 28 + index*565
        title = f'{"允许重开" if allow else "冻结闭集（错误消融）"}：返回 {result.cost} 步；重开 {result.stats["reopened_nodes"]} 次'
        parts.append(f'<text class="label" x="{left}" y="90">{escape(title)}</text>')
        visits = defaultdict(int)
        for node in result.expanded_order:
            visits[node] += 1
        for r,row in enumerate(grid):
            for c,letter in enumerate(row):
                node=(r,c); x=left+c*cell; y=top+r*cell
                color = "#283c4b" if letter == "X" else "#f6d5a2" if node in result.path else "#e7f1f6"
                stroke = "#c3424e" if visits[node] > 1 else "#aabac6"
                parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="{stroke}" stroke-width="{3 if visits[node]>1 else 1}"/>')
                if letter != "X":
                    label = letter if letter in "SE" else f'{r},{c}'
                    parts.append(f'<text class="note" x="{x+cell/2}" y="{y+17}" text-anchor="middle">{label}</text>')
                    parts.append(f'<text class="label" x="{x+cell/2}" y="{y+37}" text-anchor="middle">h={h_table[(r,c)]}</text>')
        parts.append(f'<text class="note" x="{left}" y="233">路径含 {len(result.path)} 个坐标；长度 = 坐标数 − 1；所有移动仍是单位代价。</text>')
    parts.append('<text class="label" x="24" y="280">有效出队的 f 值；横轴为实际出队次序（含目标），红点标出重开事件</text>')
    max_step = max(len(result.trace) for _,result in traces)
    max_f = max(item["f"] for _,result in traces for item in result.trace)
    x0,y0,w,h=75,315,1010,250
    for value in range(0,int(max_f)+1,2):
        y=y0+h*(1-value/max_f)
        parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0+w}" y2="{y}" stroke="#e2e8ee"/><text class="note" x="{x0-10}" y="{y+4}" text-anchor="end">{value}</text>')
    for step in range(1,max_step+1):
        x=x0+(step-1)/(max_step-1)*w
        if step%2 or step==max_step:
            parts.append(f'<text class="note" x="{x}" y="{y0+h+20}" text-anchor="middle">{step}</text>')
    for allow,result in traces:
        color="#338f91" if allow else "#d68d36"
        points=[(x0+(item["step"]-1)/(max_step-1)*w,y0+h*(1-item["f"]/max_f)) for item in result.trace]
        parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for item,(x,y) in zip(result.trace,points):
            if item["reopened"]:
                parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#c3424e"/>')
    parts.append('<text class="label" x="75" y="620" fill="#338f91">青色：允许重开</text><text class="label" x="340" y="620" fill="#d68d36">橙色：冻结闭集</text>')
    parts.append('<text class="note" x="24" y="657">h 在 (0,3)→(0,4) 从 5 降到 0，违反 5≤1+0；但 h 从未超过真实距离。f 下降不等于 h 高估。</text></svg>')
    path.write_text("\n".join(parts),encoding="utf-8")


def failure_svg(path, rows):
    parts=svg_start(1140,500,"重开消融：不一致并不意味着每个实例都会失去最优性")
    parts.append('<text class="note" x="24" y="60">只统计有解输入；每个数值旁列出失败数/地图数。故意错误的冻结闭集版本不能替代正式求解器。</text>')
    labels={"zero":"h=0", "manhattan":"曼哈顿", "lowered_manhattan":"降低部分曼哈顿值"}
    for panel,cohort in enumerate(("corridor_family","random")):
        ybase=230+panel*215
        parts.append(f'<text class="label" x="24" y="{ybase-125}">{"固定参数走廊族" if panel==0 else "随机地图"}：最短性失败比例</text>')
        for j,kind in enumerate(labels):
            selected=[r for r in rows if r["cohort"]==cohort and r["heuristic"]==kind and r["reachable"]]
            x=100+j*330
            parts.append(f'<text class="label" x="{x}" y="{ybase+25}">{labels[kind]}</text>')
            for k,allow in enumerate((True,False)):
                group=[r for r in selected if r["allow_reopen"]==allow]
                failures=sum(r["optimality_failure"] for r in group)
                fraction=failures/len(group) if group else 0
                yy=ybase-90+k*43
                color="#338f91" if allow else "#d68d36"
                parts.append(f'<rect x="{x}" y="{yy}" width="{fraction*225:.2f}" height="24" fill="{color}"/>')
                parts.append(f'<text class="note" x="{x+fraction*225+7}" y="{yy+17}">{failures}/{len(group)} ({fraction:.1%})</text>')
    parts.append('<text class="note" x="24" y="486">每组第一条：允许重开；第二条：冻结闭集。无解结果和一致/不一致分层均完整保留在 CSV。</text></svg>')
    path.write_text("\n".join(parts),encoding="utf-8")


def family_svg(path, rows):
    widths=(9,11,13,15); params=list(itertools.product((2,4,6),(-1,0,1)))
    parts=svg_start(1140,430,"反例族：固定规则枚举，保留消融未失败的参数组合")
    parts.append('<text class="note" x="24" y="60">颜色表示冻结闭集的最短性；格内数值是返回路径比 BFS 多出的步数。下层终止列 = cutoff + delta。</text>')
    for j,(cutoff,delta) in enumerate(params):
        x=150+j*106
        parts.append(f'<text class="note" x="{x+48}" y="98" text-anchor="middle">c={cutoff}, Δ={delta}</text>')
    for i,width in enumerate(widths):
        y=120+i*58
        parts.append(f'<text class="label" x="24" y="{y+28}">宽度 {width}</text>')
        for j,(cutoff,delta) in enumerate(params):
            row=next(r for r in rows if r["cohort"]=="corridor_family" and r["heuristic"]=="lowered_manhattan" and not r["allow_reopen"] and r["width"]==width and r["cutoff"]==cutoff and r["branch_delta"]==delta)
            x=150+j*106; fill="#efc4bf" if row["optimality_failure"] else "#c8e3d7"
            parts.append(f'<rect x="{x}" y="{y}" width="96" height="43" fill="{fill}"/><text class="label" x="{x+48}" y="{y+27}" text-anchor="middle">+{row["extra_steps"]}</text>')
    parts.append('<text class="note" x="24" y="405">绿色：本实例仍最优；红色：次优。正确允许重开的版本在全部参数组合中均返回最短路径。</text></svg>')
    path.write_text("\n".join(parts),encoding="utf-8")


def run(output, seeds=20, seed=20261028):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    all_rows, metadata_cases=[],[]
    grid,start,goal,table=corridor_case()
    lowered={node for node,value in table.items() if value==0}
    canonical_rows,canonical_meta,traces=run_case("canonical_2x9","canonical",grid,start,goal,lowered,{},True)
    all_rows.extend(canonical_rows);metadata_cases.append(canonical_meta)
    trace_rows=[]
    for allow,result in traces:
        for item in result.trace:
            trace_rows.append({"allow_reopen":allow,"step":item["step"],"row":item["node"][0],"col":item["node"][1],
                               **{key:item[key] for key in ("g","h","f","reopened","is_goal")}})
    write_csv(output/"canonical_trace.csv",trace_rows)
    assert traces[0][1].cost==8 and traces[1][1].cost==10
    assert traces[0][1].stats["reopened_nodes"]==3
    counterexample_svg(output/"counterexample.svg",grid,table,traces)

    for width,cutoff,delta in itertools.product((9,11,13,15),(2,4,6),(-1,0,1)):
        branch_end=cutoff+delta
        grid,start,goal,table=corridor_case(width,branch_end,cutoff)
        lowered={node for node,value in table.items() if value==0}
        rows,meta,_=run_case(f"corridor_w{width}_c{cutoff}_d{delta}","corridor_family",grid,start,goal,lowered,
                             {"width":width,"cutoff":cutoff,"branch_delta":delta,"branch_end":branch_end})
        all_rows.extend(rows);metadata_cases.append(meta)
    print("[1/2] 2×9反例与36组固定参数走廊完成；正确版全部等于BFS。",flush=True)

    for n,p in itertools.product((10,20,30),(.15,.30)):
        for instance in range(seeds):
            case_seed=seed+n*100000+round(p*100)*1000+instance
            rng=random.Random(case_seed)
            grid=[["X" if rng.random()<p else "." for _ in range(n)] for _ in range(n)]
            start,goal=(0,0),(n-1,n-1);grid[0][0],grid[-1][-1]="S","E"
            lowered={(r,c) for r,row in enumerate(grid) for c,cell in enumerate(row) if cell!="X" and rng.random()<.5}
            rows,meta,_=run_case(f"random_{n}_{int(p*100)}_{instance}","random",grid,start,goal,lowered,
                                 {"size":n,"density":p,"instance":instance,"case_seed":case_seed})
            all_rows.extend(rows);metadata_cases.append(meta)
    # 不同分组参数的列补空，保留统一、可直接筛选的CSV。
    fields=list(dict.fromkeys(key for row in all_rows for key in row))
    normalized=[{key:row.get(key,"") for key in fields} for row in all_rows]
    summaries=summarize(all_rows)
    write_csv(output/"raw.csv",normalized)
    write_csv(output/"summary.csv",summaries)
    failure_svg(output/"failure_rates.svg",all_rows)
    family_svg(output/"corridor_family.svg",all_rows)
    random_lowered=[r for r in all_rows if r["cohort"]=="random" and r["heuristic"]=="lowered_manhattan" and not r["allow_reopen"]]
    family_lowered=[r for r in all_rows if r["cohort"]=="corridor_family" and r["heuristic"]=="lowered_manhattan" and not r["allow_reopen"]]
    outcomes={"canonical":canonical_rows,"summary":summaries,
              "counts":{"map_instances":len(metadata_cases),"corridor_maps":36,"random_maps":6*seeds,
                        "probe_configurations":len(all_rows),"core_crosschecks":len(all_rows)//2},
              "random_lowered":{"total":len(random_lowered),"reachable":sum(r["reachable"] for r in random_lowered),
                                "failures":sum(r["optimality_failure"] for r in random_lowered),
                                "inconsistent":sum(not r["heuristic_is_consistent"] for r in random_lowered),
                                "reopened_inputs_correct":sum(r["core_reopened_nodes"]>0 for r in random_lowered)},
              "family_lowered":{"total":len(family_lowered),"failures":sum(r["optimality_failure"] for r in family_lowered)},
              "correct_configurations_all_optimal":True,"all_returned_paths_match_reported_cost":True,
              "all_heuristics_admissible":True}
    metadata={"generated_at_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
              "platform":platform.platform(),"parameters":{"seeds_per_random_group":seeds,"seed":seed,
              "sizes":[10,20,30],"densities":[.15,.30],"lower_to_zero_probability":.5,
              "random_inputs_filtered":False,"corridor_widths":[9,11,13,15],"cutoffs":[2,4,6],"branch_deltas":[-1,0,1]},
              "methodology":{"grid":"four-direction unit-cost; immutable inputs",
              "control":"same probe kernel and FIFO order, only allow_reopen changes",
              "false_variant":"deliberately incorrect ablation; never use as submitted solver",
              "path_validation":"every returned coordinate path legal and len(path)-1 equals returned cost",
              "oracle":"independent BFS; original core crosschecked for every case and heuristic",
              "admissibility":"zero or pointwise-lowered Manhattan; checked on all finite true distances",
              "consistency":"all directed open-grid edges checked, including nodes disconnected from goal",
              "f_decrease_events":"successive effective pop f decreases; includes goal pop; not a heuristic overestimate",
              "reopened_nodes":"actual repeated expansion; target not expanded",
              "no_performance_ranking":"no speed-up claim; ablation studies correctness and expansion events"},
              "source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ("maze_astar.py","reopen_probe.py","run_reopen_study.py")},
              "cases":metadata_cases}
    (output/"outcomes.json").write_text(json.dumps(outcomes,ensure_ascii=False,indent=2),encoding="utf-8")
    (output/"reproducibility.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[2/2] {6*seeds}张未筛选随机图完成；{len(all_rows)}个消融配置、{len(all_rows)//2}次原core交叉核对。",flush=True)
    print(json.dumps({key:outcomes[key] for key in ("counts","random_lowered","family_lowered")},ensure_ascii=False),flush=True)
    return outcomes


def positive(text):
    value=int(text)
    if value<1:raise argparse.ArgumentTypeError("必须为正整数")
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=positive,default=20)
    parser.add_argument("--seed",type=int,default=20261028)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"results"/"bonus_reopen")
    args=parser.parse_args();run(args.output_dir,args.seeds,args.seed)


if __name__=="__main__":
    main()
