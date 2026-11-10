"""按地图配对审计已有规模实验；使用可复现的百分位 bootstrap。

读取已有原始数据，不重新计时，不修改原始结果，也不依赖 SciPy。
区间仅用于探索性稳健性说明，不能视为多重比较后的显著性结论。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from run_experiments import svg_start, write_csv


def percentile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError("需要非空样本且概率位于 [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_median_interval(pairs, resamples=2000, seed=20261028):
    """配对以地图为单位；先逐图算收益，再重采样地图索引。"""
    if len(pairs) < 5 or resamples < 1:
        raise ValueError("至少需要 5 个地图样本和一次重采样")
    rng = random.Random(seed)
    node_samples, time_samples = [], []
    for _ in range(resamples):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        node_samples.append(statistics.median(pair[0] for pair in selected))
        time_samples.append(statistics.median(pair[1] for pair in selected))
    return (
        percentile(node_samples, .025), percentile(node_samples, .975),
        percentile(time_samples, .025), percentile(time_samples, .975),
    )


def map_fingerprint(item):
    relevant = {key: item[key] for key in ("grid", "start", "goal")}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def save_intervals(path, rows):
    selected = [row for row in rows if row["rows"] == 40 and row["reachable"]
                and row["variant"] == "manhattan"]
    width, height, left, top = 1100, 440, 100, 110
    parts = svg_start(width, height, "按地图配对的节点收益：探索性 95% bootstrap 区间")
    parts.append('<text class="note" x="24" y="60">40×40；曼哈顿 low_h 相对 h=0；少于 5 张不同地图时不计算区间。</text>')
    plot_w, plot_h = 920, 230
    for tick in range(5):
        value = tick * .25
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#e1e7ec"/>')
        parts.append(f'<text class="note" x="{left-12}" y="{y+4}" text-anchor="end">{value:.0%}</text>')
    for index, row in enumerate(selected):
        x = left + plot_w * (index + .5) / len(selected)
        parts.append(f'<text class="label" x="{x}" y="{top+plot_h+26}" text-anchor="middle">p={row["density_requested"]:g}</text>')
        parts.append(f'<text class="note" x="{x}" y="{top+plot_h+48}" text-anchor="middle">不同地图 n={row["n_unique_maps"]}</text>')
        if row["interval_status"] != "computed":
            parts.append(f'<text class="note" x="{x}" y="{top+plot_h/2}" text-anchor="middle">区间不计算</text>')
            continue
        center = row["paired_node_reduction_median"]
        low, high = row["node_reduction_ci_low"], row["node_reduction_ci_high"]
        y0, y1 = top + plot_h * (1-high), top + plot_h * (1-low)
        parts.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1}" stroke="#2b857c" stroke-width="3"/>')
        for y in (y0, y1):
            parts.append(f'<line x1="{x-8}" y1="{y}" x2="{x+8}" y2="{y}" stroke="#2b857c" stroke-width="2"/>')
        parts.append(f'<circle cx="{x}" cy="{top+plot_h*(1-center)}" r="5" fill="#d78a32"/>')
    parts.append('<text class="note" x="24" y="420">点为逐图减少率的中位数；区间不代表个体地图取值范围，也不保证其他地图或机器上的收益。</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def run(output, resamples=2000, seed=20261028):
    base = Path(__file__).resolve().parent
    source = base / "results" / "exploration"
    raw_path = source / "density_raw.csv"
    meta_path = source / "reproducibility.json"
    rows = list(csv.DictReader(raw_path.open(encoding="utf-8-sig")))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fingerprints = {item["case"]: map_fingerprint(item) for item in meta["random_maps"]}
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["case"]][row["variant"]] = row
    groups = defaultdict(list)
    for case, variants in sorted(by_case.items()):
        reference = variants["zero"]
        for name in ("chebyshev", "euclidean", "manhattan"):
            current = variants[name]
            assert current["optimal_steps"] == reference["optimal_steps"]
            key = (int(current["rows"]), float(current["density_requested"]),
                   current["reachable"] == "True", name)
            reduction = 1 - int(current["expanded_nodes"]) / int(reference["expanded_nodes"])
            speedup = float(reference["time_median_ms"]) / float(current["time_median_ms"])
            groups[key].append((case, fingerprints[case], reduction, speedup))

    output.mkdir(parents=True, exist_ok=True)
    results = []
    for number, (key, observations) in enumerate(sorted(groups.items())):
        size, obstacle_probability, reachable, variant = key
        # 同一地图/端点的多个种子只保留一份，防止 p=0 产生伪独立样本。
        distinct = defaultdict(list)
        for case, fingerprint, reduction, speedup in observations:
            distinct[fingerprint].append((reduction, speedup))
        pairs = [(statistics.median(item[0] for item in repeated),
                  statistics.median(item[1] for item in repeated))
                 for repeated in distinct.values()]
        row = {
            "rows": size, "density_requested": obstacle_probability,
            "reachable": reachable, "variant": variant,
            "n_records": len(observations), "n_unique_maps": len(pairs),
            "paired_node_reduction_median": statistics.median(pair[0] for pair in pairs),
            "paired_time_speedup_median": statistics.median(pair[1] for pair in pairs),
            "node_reduction_ci_low": "", "node_reduction_ci_high": "",
            "time_speedup_ci_low": "", "time_speedup_ci_high": "",
            "interval_status": "computed" if len(pairs) >= 5 else "too_few_unique_maps",
        }
        if len(pairs) >= 5:
            lo_n, hi_n, lo_t, hi_t = paired_median_interval(pairs, resamples, seed + number)
            row.update(node_reduction_ci_low=lo_n, node_reduction_ci_high=hi_n,
                       time_speedup_ci_low=lo_t, time_speedup_ci_high=hi_t)
        results.append(row)
    write_csv(output / "paired_bootstrap.csv", results)
    save_intervals(output / "paired_node_intervals.svg", results)
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "parameters": {"seed": seed, "resamples": resamples, "confidence_level": .95,
                       "method": "paired_map_percentile_bootstrap_of_medians",
                       "minimum_unique_maps_for_interval": 5},
        "source_sha256": {Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "input_sha256": {str(path.relative_to(base)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (raw_path, meta_path)},
        "limitations": [
            "Intervals describe this generated-map population conditional on reachability.",
            "Percentile intervals are approximate; small/degenerate samples can have poor coverage.",
            "No multiple-comparison correction or universal speed ranking is claimed.",
            "Timing data are from the original measurement run; this audit does not retime it.",
            "Within-map timing batches are not independent maps; duplicate maps are collapsed.",
        ],
    }
    (output / "reproducibility.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "outcomes.json").write_text(json.dumps({"groups": results,
        "n_groups": len(results), "computed_groups": sum(r["interval_status"] == "computed" for r in results)},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {len(results)} 个配对分组的证据审计；未重新计时旧实验。")
    return results


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=positive, default=2000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "bonus_audit")
    args = parser.parse_args()
    run(args.output_dir, args.resamples, args.seed)


if __name__ == "__main__":
    main()
