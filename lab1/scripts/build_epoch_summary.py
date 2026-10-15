"""从已保存的 MLP 每轮原始记录生成每 20 轮的阅读摘要，不训练模型。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path


RUN_NAMES = {
    "mlp_lr001_e30": "学习率 0.001，30 轮",
    "mlp_lr0001_e30": "学习率 0.0001，30 轮",
    "mlp_alpha001_e60": "延长训练：alpha=0.001，60 轮",
    "mlp_alpha005_e50": "训练过程：alpha=0.005，50 轮",
    "mlp_alpha005_e60": "训练过程：alpha=0.005，60 轮",
    "mlp_alpha05_e100": "较强正则化：alpha=0.05，100 轮",
    "mlp_class2_w1.5": "类别 2 加权：相对权重 1.5",
    "mlp_width25_s42": "隐藏层 25 个神经元，种子 42",
    "mlp_width50_s42": "隐藏层 50 个神经元，种子 42",
    "mlp_width100_s42": "隐藏层 100 个神经元，种子 42",
    "mlp_width50_s43": "隐藏层 50 个神经元，种子 43",
    "mlp_width100_s43": "隐藏层 100 个神经元，种子 43",
    "mlp_width50_s44": "隐藏层 50 个神经元，种子 44",
    "mlp_width100_s44": "隐藏层 100 个神经元，种子 44",
    "mlp_bigram": "单词和双词短语，词表上限 5000",
    "mlp_logtf": "对数词频，词表上限 5000",
    "mlp_vocab10000": "词表上限 10000",
}

PREFERRED_ORDER = list(RUN_NAMES)
REQUIRED_COLUMNS = ("epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy")


def load_run(run_dir: Path, records_dir: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("model") != "MLP":
        raise ValueError(f"不是 MLP 记录：{run_dir}")
    history_path = run_dir / "history.csv"
    with history_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or ())
        rows = list(reader)
    missing = set(REQUIRED_COLUMNS).difference(columns)
    if missing:
        raise ValueError(f"{history_path} 缺少列：{sorted(missing)}")
    epochs = int(config["epochs"])
    actual_epochs = [int(row["epoch"]) for row in rows]
    if actual_epochs != list(range(1, epochs + 1)):
        raise ValueError(f"{history_path} 未完整覆盖 1..{epochs}，不生成不完整摘要")
    for row in rows:
        for key in REQUIRED_COLUMNS[1:]:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"{history_path}: epoch={row['epoch']} 的 {key} 非有限数")
            if key.endswith("accuracy") and not 0 <= value <= 1:
                raise ValueError(f"{history_path}: {key} 不在 0..1")
            if key.endswith("loss") and value < 0:
                raise ValueError(f"{history_path}: {key} 为负数")
    curves_path = run_dir / "curves.png"
    if not curves_path.is_file():
        raise FileNotFoundError(f"缺少对应曲线：{curves_path}")
    selected_epochs = sorted({1, epochs, *range(20, epochs + 1, 20)})
    selected = [row for row in rows if int(row["epoch"]) in selected_epochs]
    if [int(row["epoch"]) for row in selected] != selected_epochs:
        raise AssertionError(f"{history_path} 的抽取轮次错误")
    return {
        "run_id": run_dir.name,
        "config": config,
        "columns": columns,
        "rows": rows,
        "selected": selected,
        "history_link": Path(os.path.relpath(history_path, records_dir)).as_posix(),
        "curves_link": Path(os.path.relpath(curves_path, records_dir)).as_posix(),
        "config_link": Path(os.path.relpath(run_dir / "config.json", records_dir)).as_posix(),
    }


def describe_config(config: dict) -> str:
    ngram = tuple(config["ngram_range"])
    features = "单词" if ngram == (1, 1) else f"ngram={ngram}"
    frequency = "对数词频" if config["sublinear_tf"] else "普通词频"
    weighting = (
        "不加权"
        if config["weight_class"] is None
        else f"类别 {config['weight_class']} 相对权重 {config['relative_weight']}"
    )
    return (
        f"词表上限 {config['max_features']}；{features}；{frequency}；"
        f"隐藏层 {config['hidden_units']}；学习率 {config['learning_rate_init']}；"
        f"alpha={config['alpha']}；batch_size={config['batch_size']}；"
        f"训练种子 {config['seed']}；{weighting}；共 {config['epochs']} 轮。"
    )


def build_summary(records_dir: Path, output: Path) -> dict:
    records_dir = records_dir.resolve()
    runs_dir = records_dir / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"找不到记录目录：{runs_dir}")
    runs = [load_run(path.parent, output.resolve()) for path in runs_dir.glob("*/history.csv")]
    if not runs:
        raise ValueError("没有找到 MLP 每轮记录")
    ranks = {run_id: rank for rank, run_id in enumerate(PREFERRED_ORDER)}
    runs.sort(key=lambda run: (ranks.get(run["run_id"], len(ranks)), run["run_id"]))
    # 保留原始 CSV 字符串，不在生成 CSV 时舍入损失和准确率。
    history_columns = list(dict.fromkeys(key for run in runs for key in run["columns"]))
    metadata_columns = [
        "run_id", "model", "max_features", "ngram_range", "sublinear_tf", "seed",
        "weight_class", "relative_weight", "hidden_units", "learning_rate_init",
        "alpha", "batch_size", "epochs", "history_source", "curve_source",
    ]
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "MLP每20轮摘要.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata_columns + history_columns)
        writer.writeheader()
        for run in runs:
            metadata = {key: run["config"].get(key) for key in metadata_columns}
            metadata.update(
                run_id=run["run_id"],
                ngram_range=json.dumps(run["config"]["ngram_range"]),
                history_source=run["history_link"],
                curve_source=run["curves_link"],
            )
            for row in run["selected"]:
                writer.writerow({**metadata, **row})

    original_count = sum(len(run["rows"]) for run in runs)
    summary_count = sum(len(run["selected"]) for run in runs)
    lines = [
        "# MLP 每 20 轮训练摘要",
        "",
        f"覆盖全部 {len(runs)} 个 MLP 配置，从完整的 {original_count} 条逐轮记录中抽取 {summary_count} 条阅读摘要。",
        "",
        "抽取规则：第 1 轮、每个 20 的倍数，以及实际最后一轮；重复轮次只保留一次。"
        "因此 30 轮实验显示 1/20/30，50 轮实验显示 1/20/40/50，60 轮实验显示 1/20/40/60。",
        "",
        "这里只是方便阅读和截图的摘要。原来的 history.csv 保留每一轮，curves.png 也用全部逐轮数据绘制，"
        "没有用这份摘要替代原始记录或曲线。CSV 保留原始数值；下列表格的交叉熵显示 6 位小数、准确率显示 2 位百分数。",
        "",
        "训练/验证损失指在各自完整数据上计算的、不加权的预测交叉熵，不包含 L2 惩罚。"
        "这不代表训练时没有正则化：模型仍按配置中的 alpha 使用 L2。加权组的训练优化目标与此处统一的评估交叉熵也不完全相同。",
        "",
        "这里的一轮是 partial_fit 对完整训练集的一次遍历。当前逻辑回归使用 lbfgs，SVM 使用 SVC 求解器，"
        "它们没有像 MLP 这样记录完整数据遍历的逐轮指标，不能把求解器迭代数强行解释为训练轮次，也不补造每 20 轮损失表。",
        "",
    ]
    for index, run in enumerate(runs, start=1):
        name = RUN_NAMES.get(run["run_id"], run["run_id"])
        lines.extend([
            f"## {index}. {name}",
            "",
            f"配置编号：`{run['run_id']}`。{describe_config(run['config'])}",
            "",
            f"[完整逐轮数据]({run['history_link']}) · [对应训练/验证曲线]({run['curves_link']}) · [参数配置]({run['config_link']})",
            "",
            "| 轮次 | 训练交叉熵 | 验证交叉熵 | 训练准确率 | 验证准确率 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in run["selected"]:
            lines.append(
                f"| {row['epoch']} | {float(row['train_loss']):.6f} | {float(row['val_loss']):.6f} | "
                f"{float(row['train_accuracy']):.2%} | {float(row['val_accuracy']):.2%} |"
            )
        lines.append("")
    md_path = output / "MLP每20轮摘要.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 反读生成文件，核对每个原始值、抽取轮次及总行数。
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    expected = [(run, row) for run in runs for row in run["selected"]]
    if len(exported) != summary_count:
        raise AssertionError("摘要总行数不正确")
    for saved, (run, original) in zip(exported, expected):
        if saved["run_id"] != run["run_id"]:
            raise AssertionError("摘要配置编号不正确")
        for key in run["columns"]:
            if saved[key] != original[key]:
                raise AssertionError(f"{run['run_id']} 的 {key} 未保留原始值")
    result = {
        "mlp_config_count": len(runs),
        "raw_epoch_rows": original_count,
        "summary_rows": summary_count,
        "selection_rule": "epoch 1, every multiple of 20, final epoch; deduplicated",
        "raw_records_modified": False,
        "training_performed": False,
        "csv": csv_path.name,
        "markdown": md_path.name,
        "per_run_epochs": {
            run["run_id"]: [int(row["epoch"]) for row in run["selected"]] for run in runs
        },
    }
    (output / "每20轮摘要核验.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True, help="含 session.json 和 runs/ 的公开记录目录")
    parser.add_argument("--output", type=Path, required=True, help="新建的输出目录，不覆盖归档摘要")
    args = parser.parse_args()
    print(json.dumps(build_summary(args.records_dir, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
