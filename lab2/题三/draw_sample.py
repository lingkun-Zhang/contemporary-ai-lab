"""Draw the teacher's sample tree and its left-to-right pruning as SVG."""

import argparse
from html import escape
from pathlib import Path


def draw_sample(path):
    """Write a self-contained diagram; full subtree truth and bounds stay distinct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="640" '
        'viewBox="0 0 1200 640" role="img" aria-labelledby="title desc">',
        '<title id="title">题三样例：精确Minimax回传与左到右Alpha-Beta剪枝</title>',
        '<desc id="desc">根MAX的真值5；左右MIN真值5和2；四个MAX真值5、9、9、2。'
        '左侧第二个MAX在beta为5时看到6即截断，返回下界6，叶9未访问。'
        'Alpha-Beta访问14次、评估7叶，完整Minimax访问15次、评估8叶。</desc>',
        '<rect width="1200" height="640" fill="#ffffff"/>',
        '<g font-family="Microsoft YaHei, Noto Sans CJK SC, sans-serif" '
        'fill="#24384b">',
    ]

    def text(x, y, content, size=18, fill="#24384b", anchor="middle", weight="normal"):
        parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escape(content)}</text>'
        )

    def edge(x1, y1, x2, y2, skipped=False):
        style = ' stroke-dasharray="7 5"' if skipped else ""
        color = "#c64141" if skipped else "#9aa9b8"
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="2.2"{style}/>'
        )

    def node(x, y, label, value, color="#eaf3f8", width=142):
        parts.append(
            f'<rect x="{x-width/2:g}" y="{y-30}" width="{width}" height="60" '
            f'rx="12" fill="{color}" stroke="#557a94" stroke-width="1.7"/>'
        )
        text(x, y-5, label, 16)
        text(x, y+19, value, 21, weight="bold")

    text(30, 37, "题三样例：回传真值与实际剪枝", 27, anchor="start", weight="bold")
    text(30, 67, "正式访问顺序：从左到右；层次为 MAX → MIN → MAX → 叶值。", 17, anchor="start")

    # Edges are painted first so labels and nodes remain legible.
    for x in (310, 890):
        edge(600, 140, x, 186)
    for parent_x, child_xs in ((310, (165, 455)), (890, (745, 1035))):
        for x in child_xs:
            edge(parent_x, 246, x, 300)
    leaf_pairs = ((165, (100, 230)), (455, (390, 520)),
                  (745, (680, 810)), (1035, (970, 1100)))
    for parent_x, child_xs in leaf_pairs:
        for x in child_xs:
            edge(parent_x, 360, x, 424, skipped=(x == 520))

    node(600, 110, "MAX · 根", "真值 5", "#d9edf5")
    node(310, 216, "MIN", "真值 5", "#edf0fb")
    node(890, 216, "MIN", "真值 2", "#edf0fb")
    node(165, 330, "MAX", "真值 5")
    node(455, 330, "MAX · β=5 时截断", "真值 9", "#fff1df", width=222)
    node(745, 330, "MAX", "真值 9")
    node(1035, 330, "MAX", "真值 2")
    parts.append('<rect x="333" y="367" width="244" height="26" rx="4" fill="#ffffff"/>')
    text(455, 387, "截断返回 6，是下界（≥6）", 16, "#9a5f17")

    for x, value in ((100, 3), (230, 5), (390, 6), (520, 9),
                     (680, 6), (810, 9), (970, 1), (1100, 2)):
        skipped = x == 520
        color = "#c64141" if skipped else "#557a94"
        fill = "#fff5f5" if skipped else "#f4f8fb"
        dash = ' stroke-dasharray="6 4"' if skipped else ""
        parts.append(
            f'<circle cx="{x}" cy="447" r="24" fill="{fill}" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        text(x, 455, str(value), 23, color, weight="bold")
        if skipped:
            text(x, 499, "未访问", 16, color)

    parts.append('<rect x="30" y="529" width="1140" height="90" rx="10" fill="#f0f5f8"/>')
    text(50, 558, "完整 Minimax：15 次节点访问 / 8 次叶评估；Alpha-Beta：14 / 7；根值均为 5。", 18, anchor="start")
    text(50, 586, "真正节省工作的剪枝 1 次；右 MIN 最后一个孩子回传后虽 α≥β，但已无剩余孩子可省。", 17, anchor="start")
    text(50, 608, "图中“真值”由完整 Minimax 得到；截断返回的界不能当作该子树的精确值。", 15, "#607487", anchor="start")
    parts.extend(("</g>", "</svg>"))
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "sample_tree.svg")
    args = parser.parse_args()
    print(draw_sample(args.output))


if __name__ == "__main__":
    main()
