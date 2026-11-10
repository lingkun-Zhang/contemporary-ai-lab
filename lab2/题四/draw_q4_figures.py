"""SVG figures for Q4's recorded map, game decisions and paired studies.

Only Python's standard library is required.  This module reads saved results;
it does not solve a path/game, generate samples or perform benchmark timing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from html import escape
from pathlib import Path


COLORS = ("#2f809e", "#ca8432", "#5b9565", "#9867a7")
FONT = "Microsoft YaHei, Noto Sans CJK SC, sans-serif"


def number(value, digits=2):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Figure data must contain finite numbers")
    return str(int(value)) if value.is_integer() else f"{value:.{digits}f}"


def start(title, subtitle, height=850):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<g font-family="{FONT}" fill="#2c4050">',
        f'<text x="28" y="39" font-size="25" font-weight="bold">{escape(title)}</text>',
        f'<text x="28" y="70" font-size="14" fill="#536b80">{escape(subtitle)}</text>',
    ]


def text(svg, x, y, label, size=15, color="#2c4050", anchor="start", bold=False):
    weight = ' font-weight="bold"' if bold else ""
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{weight}>{escape(str(label))}</text>')


def save(path, svg):
    Path(path).write_text("\n".join(svg + ["</g></svg>"]), encoding="utf-8")


def line_panel(svg, xs, series, title, y, height=245, signed=False, xlabels=None, legend=True):
    """Return geometry so annotations can be positioned without guessing."""
    text(svg, 28, y, title, 17)
    left, top, width, plot_h = 100, y + 26, 1050, height - 75
    numbers = [float(v) for _, values in series for v in values]
    if not numbers or not all(math.isfinite(v) for v in numbers):
        raise ValueError("Missing or non-finite series values")
    if signed:
        low, high = min(numbers + [0.0]), max(numbers + [0.0])
        padding = max((high - low) * .12, .5)
        low, high = low - padding, high + padding
    else:
        low, high = 0, max(numbers + [1e-6]) * 1.15
    def point(index, value):
        return (left + width * (index + .5) / len(xs),
                top + plot_h - (float(value) - low) / (high - low) * plot_h)
    for tick in range(5):
        value = low + (high - low) * tick / 4
        yy = top + plot_h - tick * plot_h / 4
        svg.append(f'<line x1="{left}" y1="{yy}" x2="1150" y2="{yy}" stroke="#dce5ed"/>')
        text(svg, left - 12, yy + 5, number(value), 13, "#536b80", "end")
    if signed and low < 0 < high:
        yy = point(0, 0)[1]
        svg.append(f'<line x1="{left}" y1="{yy}" x2="1150" y2="{yy}" stroke="#748ca0" stroke-dasharray="5 5"/>')
    for index, label in enumerate(xlabels or xs):
        xx, _ = point(index, low)
        text(svg, xx, top + plot_h + 26, label, 14, anchor="middle")
    for k, (name, values) in enumerate(series):
        if len(values) != len(xs):
            raise ValueError("Each series must match the x positions")
        color, points = COLORS[k % len(COLORS)], []
        for i, value in enumerate(values):
            xx, yy = point(i, value)
            points.append(f"{xx},{yy}")
            svg.append(f'<circle cx="{xx}" cy="{yy}" r="3.6" fill="{color}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        if legend:
            xx, yy = 105 + k * 258, top + plot_h + 57
            svg.append(f'<path d="M{xx},{yy - 5}h22" stroke="{color}" stroke-width="3"/>')
            text(svg, xx + 30, yy, name, 14)
    return point


def target_label(value):
    if isinstance(value, list):
        return "/".join(str(item) for item in value)
    return str(value)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def draw_map(results, sample):
    grid, labels, points = sample["grid"], sample["labels"], sample["points"]
    if len(labels) != len(points) or not grid or not grid[0]:
        raise ValueError("Invalid saved map schema")
    markers = {tuple(point): label for label, point in zip(labels, points)}
    ledgers = sample.get("ledgers", {})
    # The parent may keep ledgers as a separate top-level sibling; read only
    # values actually present in the saved data rather than hardcoding values.
    values = {row["target"]: row["value"] for group in ledgers.values() for row in group}
    svg = start("题四：地图进入费用与两种深度的收益账本", "A、B、T格进入费均为1；路径经过其他宝藏不自动获取，收益为价值减移动费。")
    rows, columns = len(grid), len(grid[0])
    cell = min(72, 525 / max(rows, columns))
    x0, y0 = 76, 150
    for col in range(columns):
        text(svg, x0 + cell * (col + .5), y0 - 17, col, 14, anchor="middle")
    for row in range(rows):
        text(svg, x0 - 23, y0 + cell * (row + .5) + 5, row, 14, anchor="middle")
        for col in range(columns):
            xx, yy = x0 + col * cell, y0 + row * cell
            value, label = grid[row][col], markers.get((row, col))
            if value is None:
                fill, border = "#e2e7ec", "#c8d2db"
            elif label in ("A", "B"):
                fill, border = ("#dceef4", "#75a6b9") if label == "A" else ("#fbe9d4", "#c7a36e")
            elif label:
                fill, border = "#fff4d7", "#d4b56c"
            else:
                fill, border = "#ffffff", "#c8d2db"
            svg.append(f'<rect x="{xx}" y="{yy}" width="{cell}" height="{cell}" fill="{fill}" stroke="{border}"/>')
            text(svg, xx + cell / 2, yy + cell / 2 + (0 if label else 7), "X" if value is None else (label or number(value)), 21, anchor="middle", bold=bool(label))
            if label:
                suffix = f"V={number(values[label])}" if label in values else "进入费1"
                text(svg, xx + cell / 2, yy + cell - 11, suffix, 12, "#62798c", "middle")
    text(svg, 76, y0 + rows * cell + 35, "起点不收费；每次移动按进入格收费。", 15)
    text(svg, 76, y0 + rows * cell + 62, "获得宝藏后只更新行动玩家的位置。", 15)
    def ledger_panel(depth, top):
        data = ledgers.get(f"H{depth}", ledgers.get(str(depth), []))
        svg.append(f'<rect x="655" y="{top - 27}" width="515" height="{72 + 46 * len(data)}" rx="8" fill="#f3f7fa"/>')
        text(svg, 675, top, f"H={depth}：按最优延续的一条路径", 18, bold=True)
        if not data:
            text(svg, 675, top + 31, "账本数据未记录", 14, "#62798c")
            return
        for i, row in enumerate(data):
            yy = top + 34 + 46 * i
            text(svg, 675, yy, f"{row['step']}. {row['player']}：{row['source']} → {row['target']}", 15)
            text(svg, 695, yy + 21,
                 f"费用{number(row['travel_cost'])}，收益{number(row['gain'])}；A={number(row['score_a'])}，B={number(row['score_b'])}，U={number(row['utility'])}", 14, "#536b80")
        last = data[-1]
        text(svg, 675, top + 39 + 46 * len(data), f"最终收益差 U={number(last['utility'])}", 17, bold=True)
    ledger_panel(2, 147)
    ledger_panel(4, 365)
    text(svg, 675, 680, "H按单次行动计：A → B → A → B。", 15)
    text(svg, 675, 709, "收益为负仍须选择剩余宝藏，不能跳过。", 15)
    text(svg, 28, 809, "图示来自已保存样例分析；同分路径可以有多条，此处仅展示记录的一条。", 13, "#62798c")
    save(results / "sample_map.svg", svg)


def draw_depths(results, sample):
    records = sorted(sample["depths"], key=lambda item: int(item["depth"]))
    depths = [int(item["depth"]) for item in records]
    targets = [label for label in sample["labels"] if label not in ("A", "B")]
    svg = start("搜索深度改变首步：每个根动作的精确回传值", "每个根动作均独立核验；H按单次行动计，截断时评价当前Score_A−Score_B。", 820)
    series = [(target, [item["root_action_values"][target] for item in records]) for target in targets]
    point = line_panel(svg, depths, series, "纵轴：强制该首步后的最优收益差；横轴：H", 130, height=470, signed=True)
    for depth in (2, 4):
        item = next((record for record in records if int(record["depth"]) == depth), None)
        if item is None:
            continue
        index = depths.index(depth)
        px, py = point(index, item["best_value"])
        xx = min(max(px - 105, 110), 925)
        yy = min(max(py + 17, 160), 550)
        svg.append(f'<rect x="{xx}" y="{yy}" width="225" height="39" rx="6" fill="#fff9e9" stroke="#d2b66b"/>')
        label = f"H={depth}：{target_label(item['best_targets'])}，U={number(item['best_value'])}"
        text(svg, xx + 12, yy + 25, label, 16, bold=True)
    text(svg, 100, 720, "H2只评估双方各一次行动；H4将第二次移动及其可能的负收益纳入回传。", 15)
    text(svg, 100, 752, "不同深度使用不同截断位置，根值无需单调；同一深度的Minimax与Alpha-Beta应一致。", 14, "#536b80")
    save(results / "depth_decisions.svg", svg)


def draw_routing(results, source):
    records = [row for row in read_csv(source)
               if normalized(row.get("algorithm")) in ("alphabeta", "ab")
               and normalized(row.get("order")) == "label"
               and row.get("output_mode", "decision_only") == "decision_only"]
    if not records:
        raise ValueError("routing_comparison.csv has no Alpha-Beta/label rows")
    depths = sorted({int(row["depth"]) for row in records})
    strategies = (("precompute", "预计算矩阵"), ("on_demand", "按需缓存"), ("repeated", "逐次重新寻路"))
    selected = {}
    for strategy, _ in strategies:
        for depth in depths:
            group = [row for row in records if row["strategy"] == strategy and int(row["depth"]) == depth]
            if len(group) != 1:
                raise ValueError(f"Expected exactly one routing row: {strategy}, H={depth}")
            selected[strategy, depth] = group[0]
    svg = start("首步决策子任务：预计算与重复寻路的成本", "固定Alpha-Beta、标签顺序、decision_only；时间包含该策略的准备、寻路与博弈搜索。")
    for y, field, title in ((115, "astar_calls", "A*调用次数；横轴为搜索深度H"),
                            (465, "median_total_ms", "首步子任务总时间中位数（ms）；横轴为H")):
        series = [(label, [float(selected[strategy, depth][field]) for depth in depths]) for strategy, label in strategies]
        line_panel(svg, depths, series, title, y)
    text(svg, 100, 808, "本图只比较首步子任务；正式完整任务还须输出距离矩阵，其补齐成本另记于complete_task。", 13, "#62798c")
    save(results / "routing_cost.svg", svg)


def draw_sweep(results, source):
    records = [row for row in read_csv(source)
               if normalized(row.get("algorithm")) in ("minimax", "mm")
               and normalized(row.get("order")) == "label" and int(row["depth"]) in (2, 4)]
    if not records:
        raise ValueError("value_sweep.csv has no Minimax/label H2/H4 rows")
    xs = sorted({float(row["t3_value"]) for row in records})
    selected = {}
    for depth in (2, 4):
        for x in xs:
            group = [row for row in records if int(row["depth"]) == depth and float(row["t3_value"]) == x]
            if len(group) != 1:
                raise ValueError(f"Expected one value-sweep row: V3={x}, H={depth}")
            selected[depth, x] = group[0]
    svg = start("固定地图与其他价值：只改变T3的价值", "固定Minimax、标签顺序；上方为根收益差，下方逐值列出最优首步；正式样例V3=7。")
    line_panel(svg, xs, [(f"H={depth}", [float(selected[depth, x]["best_value"]) for x in xs]) for depth in (2, 4)],
               "横轴：T3价值；纵轴：最优收益差U", 120, height=360, signed=True, xlabels=[number(x) for x in xs])
    text(svg, 28, 558, "同一输入分别核验各深度；参数改变价值与决策，不改变移动费用。", 16)
    left, width, label_width, top, row_h = 32, 1135, 100, 590, 51
    cell_w = (width - label_width) / len(xs)
    for row_index, label in enumerate(("T3价值", "H=2首步", "H=4首步")):
        yy = top + row_index * row_h
        svg.append(f'<rect x="{left}" y="{yy}" width="{label_width}" height="{row_h}" fill="#edf3f7" stroke="#cbd7e0"/>')
        text(svg, left + label_width / 2, yy + 31, label, 13, anchor="middle", bold=True)
        for i, x in enumerate(xs):
            xx = left + label_width + i * cell_w
            value = number(x) if row_index == 0 else selected[(2 if row_index == 1 else 4), x]["best_target"]
            svg.append(f'<rect x="{xx}" y="{yy}" width="{cell_w}" height="{row_h}" fill="white" stroke="#cbd7e0"/>')
            text(svg, xx + cell_w / 2, yy + 31, value, 13, anchor="middle")
    text(svg, 32, 796, "首步相同的连续区间可用于解释决策边界；并列解的处理以原始数据和求解器规则为准。", 13, "#62798c")
    save(results / "value_sweep.svg", svg)


def draw_all(results_dir):
    """Read available saved evidence and return the generated figure paths."""
    results = Path(results_dir)
    sample_path = results / "sample_analysis.json"
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    ledger_path = results / "sample_ledgers.json"
    if "ledgers" not in sample and ledger_path.is_file():
        sample["ledgers"] = json.loads(ledger_path.read_text(encoding="utf-8"))
    draw_map(results, sample)
    draw_depths(results, sample)
    generated = [results / "sample_map.svg", results / "depth_decisions.svg"]
    if (results / "routing_comparison.csv").is_file():
        draw_routing(results, results / "routing_comparison.csv")
        generated.append(results / "routing_cost.svg")
    if (results / "value_sweep.csv").is_file():
        draw_sweep(results, results / "value_sweep.csv")
        generated.append(results / "value_sweep.svg")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    for path in draw_all(args.results_dir):
        print(path)


if __name__ == "__main__":
    main()
