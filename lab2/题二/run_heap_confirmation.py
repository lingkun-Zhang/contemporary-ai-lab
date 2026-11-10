"""题二预先固定参数的独立计时确认；不利用新结果重新选择阈值。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from run_experiments import positive, svg_start, write_csv
from run_heap_tuning import rho_label, run as run_tuning, tuning_cases


ROOT = Path(__file__).resolve().parent
FIXED_RHOS = (1.05, 1.2, 1.5, 2.0, math.inf)
FIXED_CASES = ("bands_31", "bands_101", "uniform_101", "tight_bands_101")
TARGET = "1.5"
REFERENCES = ("original", "inf", "1.2", "2")
PRIMARY_CASE = "bands_101"
CASE_ROLES = {"bands_101":"primary_confirmation", "bands_31":"cost_transfer",
              "uniform_101":"no_cleanup_action_control", "tight_bands_101":"no_cleanup_action_control"}
SOURCE_NAMES = ("weighted_astar.py", "heap_compaction.py", "run_experiments.py",
                "run_heap_tuning.py", "run_heap_confirmation.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def fingerprint(value):
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def case_payload(case, heuristic, tie, scope):
    return dict(case=case.slug, name=case.name, grid=case.grid, start=list(case.start),
                goal=list(case.goal), heuristic=heuristic, tie_break=tie, min_scope=scope,
                distribution=case.distribution)


def comparable_payload(item):
    return {key: item[key] for key in ("case", "grid", "start", "goal", "heuristic", "tie_break", "min_scope", "distribution")}


def fresh_output_dir(requested):
    """保留已交付结果，默认重跑安全写入新的子目录。"""
    target = Path(requested)
    if target.exists() and any(target.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        target = target / f"run_{stamp}"
        target.mkdir(parents=True, exist_ok=False)
    else:
        target.mkdir(parents=True, exist_ok=True)
    return target


def write_protocol(output, protocol):
    path = Path(output) / "protocol.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return sha(path)


def prepare_protocol(requested, repeats=15, batch=3, seed=20261030):
    if type(repeats) is not int or repeats < 1 or type(batch) is not int or batch < 1:
        raise ValueError("repeats 与 batch 必须为正整数")
    previous = ROOT / "results_heap_tuning"
    old = json.loads((previous / "reproducibility.json").read_text(encoding="utf-8"))
    if seed == old["parameters"]["seed"]:
        raise ValueError("确认轮使用不同于原调参轮的新计时顺序种子")
    for name, expected in old["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"原调参来源已改变，不能直接声称同实现确认：{name}")
    generated = [case_payload(*group) for group in tuning_cases(seed, sizes=(31, 101))
                 if group[0].slug in FIXED_CASES]
    if tuple(item["case"] for item in generated) != FIXED_CASES:
        raise AssertionError("固定确认组合缺失或顺序变化")
    prior = {item["case"]: item for item in old["cases"]}
    for item in generated:
        if comparable_payload(item) != comparable_payload(prior[item["case"]]):
            raise AssertionError(f"确认轮不能改变原图/端点/启发/排序：{item['case']}")
    maps = {fingerprint({key:item[key] for key in ("grid", "start", "goal")}) for item in generated}
    protocol = dict(
        declared_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="new timing blocks with frozen prior workloads and a predetermined rho; no new winner selection",
        parameters=dict(rhos=[rho_label(rho) for rho in FIXED_RHOS], target_rho=TARGET,
                        references=list(REFERENCES), cases=list(FIXED_CASES), repeats=repeats,
                        batch=batch, timing_seed=seed),
        primary=dict(case=PRIMARY_CASE, target_rho=TARGET, comparison="original",
                     rationale="earlier tuning suggested rho=1.5 on bands_101; assess this fixed choice in new timing blocks"),
        predefined_comparisons="rho=1.5 minus original/inf/left-neighbor1.2/right-neighbor2 on every declared workload",
        descriptive_rule=dict(
            difference="same-repeat batch average ms at rho=1.5 minus reference; negative is faster",
            improvement="paired Q3 < 0 gives directionally consistent improvement in these recorded timing blocks",
            slowdown="paired Q1 > 0 gives directionally consistent slowdown in these recorded timing blocks",
            otherwise="mixed or unresolved direction; do not change target rho after seeing results",
            local_lowpoint="both finite-neighbor paired Q3 values must be below zero for descriptive support of a strict local lowpoint",
            limitations="IQR is descriptive machine timing variability, not a confidence interval, significance test or generalization across maps"),
        controls="uniform_101 and tight_bands_101 have no cleanup actions; nominal timing rankings cannot be attributed to cleanup",
        counts=dict(groups=len(generated), distinct_grid_endpoint_inputs=len(maps),
                    configurations=len(generated)*(len(FIXED_RHOS)+1),
                    timing_batches=len(generated)*(len(FIXED_RHOS)+1)*repeats,
                    timed_solver_calls=len(generated)*(len(FIXED_RHOS)+1)*repeats*batch),
        input_cases=[dict(**item, role=CASE_ROLES[item["case"]],
                          configuration_sha256=fingerprint(comparable_payload(item))) for item in generated],
        prior_generated_at_utc=old.get("generated_at_utc"),
        prior_source_files={name:sha(previous/name) for name in ("reproducibility.json", "comparison.csv", "timing_batches.csv", "selection.csv")},
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},
        timing="complete solver calls including normalization, open maintenance, filtering, heapify and backtracking; oracle/trace/verification/plotting outside",
        output_policy="if requested directory is nonempty, create a fresh timestamped run_* subdirectory; never overwrite an existing protocol",
        auxiliary_selection="selection.csv produced by reused runner is descriptive only; confirmation statements evaluate fixed rho=1.5",
    )
    output = fresh_output_dir(requested)
    digest = write_protocol(output, protocol)
    return output, protocol, digest


def delta_summary(values):
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values*3
    median = statistics.median(values)
    if q[2] < 0:
        direction = "directionally_consistent_improvement"
    elif q[0] > 0:
        direction = "directionally_consistent_slowdown"
    else:
        direction = "mixed_or_unresolved"
    return dict(delta_ms_median=median, delta_ms_q1=q[0], delta_ms_q3=q[2],
                delta_ms_min=min(values), delta_ms_max=max(values),
                faster_blocks=sum(value < 0 for value in values),
                slower_blocks=sum(value > 0 for value in values),
                equal_blocks=sum(value == 0 for value in values), direction=direction,
                raw_delta_ms=json.dumps(values))


def fixed_pair_rows(batch_rows, comparison_rows):
    cases = list(dict.fromkeys(row["case"] for row in batch_rows))
    result = []
    for slug in cases:
        group = [row for row in batch_rows if row["case"] == slug]
        lookup = {(int(row["repeat"]), row["rho"]): float(row["batch_average_ms"]) for row in group}
        repeats = sorted({int(row["repeat"]) for row in group})
        measured = next(row for row in comparison_rows if row["case"] == slug and row["rho"] == TARGET)
        for reference in REFERENCES:
            values = [lookup[(repeat,TARGET)]-lookup[(repeat,reference)] for repeat in repeats]
            action = int(measured["compactions"]) > 0
            result.append(dict(case=slug, case_name=measured["case_name"], case_role=CASE_ROLES.get(slug,"unspecified"), target_rho=TARGET,
                               reference=reference, repeats=len(repeats),
                               compactions_at_target=int(measured["compactions"]),
                               removed_stale_at_target=int(measured.get("removed_stale",0)),
                               cleanup_action_present=action,
                               causal_interpretation="timing difference cannot be attributed to cleanup because no cleanup action occurred" if not action else "interpret with measured cleanup work and complete timing cost",
                               **delta_summary(values)))
    return result


def draw_fixed_pairs(output, paired):
    labels = {"bands_31":"31阶带状", "bands_101":"101阶带状", "uniform_101":"均匀101", "tight_bands_101":"tight101"}
    width, height, left, plot_width, panel_h = 1180, 1095, 95, 1045, 245
    parts = svg_start(width,height,"预定ρ=1.5：新计时块的固定配对确认",
                      "负值表示预定参数更快；竖线为配对差IQR，非置信区间；参数不根据本轮结果重选。")
    for index,reference in enumerate(REFERENCES):
        top, plot_h = 110+panel_h*index, 150
        values = [next(row for row in paired if row["case"]==slug and row["reference"]==reference) for slug in FIXED_CASES]
        low = min(0,min(row["delta_ms_q1"] for row in values))*1.15
        high = max(0,max(row["delta_ms_q3"] for row in values))*1.15
        if high-low < .05:
            low-=.025;high+=.025
        xpos=lambda i:left+plot_width*(i+.5)/len(values)
        ypos=lambda y:top+plot_h*(1-(y-low)/(high-low))
        title = f"固定ρ=1.5 − {reference}：批平均单次毫秒差（越低越好）"
        parts.append(f'<text class="label" x="24" y="{top-18}">{escape(title)}</text>')
        for tick in range(5):
            value=low+(high-low)*tick/4;y=ypos(value)
            parts.append(f'<line x1="{left}" x2="{left+plot_width}" y1="{y}" y2="{y}" stroke="#e0e7ec"/>')
            parts.append(f'<text class="note" x="{left-10}" y="{y+4}" text-anchor="end">{value:.3f}</text>')
        parts.append(f'<line x1="{left}" x2="{left+plot_width}" y1="{ypos(0)}" y2="{ypos(0)}" stroke="#718da6" stroke-dasharray="5 4"/>')
        for i,row in enumerate(values):
            x=xpos(i);lo=ypos(row["delta_ms_q1"]);hi=ypos(row["delta_ms_q3"]);y=ypos(row["delta_ms_median"])
            color="#337c98" if row["cleanup_action_present"] else "#ad7842"
            parts.append(f'<line x1="{x}" x2="{x}" y1="{lo}" y2="{hi}" stroke="{color}" stroke-width="2"/>')
            for cap in (lo,hi):parts.append(f'<line x1="{x-5}" x2="{x+5}" y1="{cap}" y2="{cap}" stroke="{color}"/>')
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            parts.append(f'<text class="note" x="{x}" y="{top+plot_h+22}" text-anchor="middle">{labels[row["case"]]}</text>')
        parts.append(f'<text class="note" x="{left}" y="{top+plot_h+49}">蓝色有清理动作，棕色无清理动作；后一类的时间差不能解释成清理收益。</text>')
    parts.append('</svg>')
    (Path(output)/"confirmation_fixed_pairs.svg").write_text("\n".join(parts),encoding="utf-8")


def run_confirmation(output=None, repeats=15, batch=3, seed=20261030):
    requested = Path(output) if output is not None else ROOT/"results_heap_confirmation"
    actual, protocol, protocol_sha = prepare_protocol(requested,repeats,batch,seed)
    print(f"确认协议已在首个预热前保存：{actual/'protocol.json'}",flush=True)
    # The reused runner's sampling seed affects order only for these deterministic groups.
    run_tuning(actual,rhos=FIXED_RHOS,repeats=repeats,batch=batch,seed=seed,
               sizes=(31,101),selected_cases=list(FIXED_CASES))
    if sha(actual/"protocol.json") != protocol_sha:
        raise AssertionError("预先保存的确认协议被改变")
    meta=json.loads((actual/"reproducibility.json").read_text(encoding="utf-8"))
    observed={item["case"]:item for item in meta["cases"]}
    for item in protocol["input_cases"]:
        if comparable_payload(observed[item["case"]]) != comparable_payload(item):
            raise AssertionError("实际确认输入与预声明不一致")
    if meta["parameters"]["rhos"] != protocol["parameters"]["rhos"]:
        raise AssertionError("实际候选与预声明不一致")
    comparison=read_csv(actual/"comparison.csv")
    batches=read_csv(actual/"timing_batches.csv")
    if len(comparison)!=protocol["counts"]["configurations"] or len(batches)!=protocol["counts"]["timing_batches"]:
        raise AssertionError("确认配置或批次数与预声明不一致")
    pairs=fixed_pair_rows(batches,comparison)
    write_csv(actual/"confirmation_pairs.csv",pairs)
    draw_fixed_pairs(actual,pairs)
    primary=next(row for row in pairs if row["case"]==PRIMARY_CASE and row["reference"]=="original")
    neighbors=[row for row in pairs if row["case"]==PRIMARY_CASE and row["reference"] in ("1.2","2")]
    outcome=dict(
        counts=protocol["counts"], target_rho=TARGET, primary_case=PRIMARY_CASE,
        primary_reference="original", primary_direction=primary["direction"],
        fixed_local_lowpoint_directionally_supported=all(row["direction"]=="directionally_consistent_improvement" for row in neighbors),
        all_costs_optimal=True,all_paths_and_effective_orders_unchanged=True,
        actual_inputs_match_frozen_prior_workloads=True,protocol_unchanged=True,
        new_winner_selected=False,auxiliary_selection_only=True,
        interpretation="new timing blocks on the same three maps; descriptive support only, no significance or across-map generalization claim",
        primary_pair=primary,finite_neighbor_pairs=neighbors)
    (actual/"confirmation_outcomes.json").write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding="utf-8")
    confirmation_meta=dict(
        protocol_sha256=protocol_sha,protocol_written_before_warmup=True,
        source_sha256={name:sha(ROOT/name) for name in SOURCE_NAMES},
        frozen_prior_sources=protocol["prior_source_files"],
        result_files_sha256={name:sha(actual/name) for name in ("comparison.csv","timing_batches.csv","confirmation_pairs.csv","confirmation_outcomes.json")},
        selection_policy="selection.csv is auxiliary; all confirmation decisions evaluate only predeclared rho=1.5",
        counts=protocol["counts"])
    (actual/"confirmation_reproducibility.json").write_text(json.dumps(confirmation_meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"独立计时确认完成：{len(comparison)}配置、{len(batches)}计时批；预定ρ保持1.5；{actual.resolve()}",flush=True)
    return outcome


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats",type=positive,default=15)
    parser.add_argument("--batch",type=positive,default=3)
    parser.add_argument("--seed",type=int,default=20261030)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"results_heap_confirmation")
    args=parser.parse_args()
    try:run_confirmation(args.output_dir,args.repeats,args.batch,args.seed)
    except (ValueError,FileExistsError) as exc:parser.error(str(exc))


if __name__=="__main__":main()
