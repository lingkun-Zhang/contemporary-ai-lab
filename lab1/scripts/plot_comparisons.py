"""从公开保存的逐轮数据与末次指标重画17组对照图，不训练模型。"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

LAB = Path(__file__).resolve().parents[1]
PUBLIC = LAB / "results/reproduction_20261014"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TITLES = {
    "LR_C": "逻辑回归：C 对照", "LR_weight": "逻辑回归：类别7训练权重",
    "LR_vocabulary": "逻辑回归：词表容量", "SVM_C": "线性SVM：C 对照",
    "SVM_vocabulary": "线性SVM：词表容量", "MLP_learning_rate": "MLP：学习率对照（30轮）",
    "MLP_epochs_alpha001": "MLP：轮数对照（alpha=0.001）",
    "MLP_epochs_alpha005": "MLP：轮数对照（alpha=0.005）",
    "MLP_regularization": "MLP：正则化对照", "MLP_width": "MLP：隐藏层宽度",
    "MLP_seed42": "MLP：宽度对照（训练种子42）", "MLP_seed43": "MLP：宽度对照（训练种子43）",
    "MLP_seed44": "MLP：宽度对照（训练种子44）", "MLP_weight": "MLP：类别2训练权重",
    "MLP_ngram": "MLP：单词与单词+短语", "MLP_tf_scaling": "MLP：普通与对数词频",
    "MLP_vocabulary": "MLP：词表容量",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def labels_for(name, summaries):
    if name.endswith("vocabulary"):
        return [f"词表 {row['max_features']}" for row in summaries]
    if name.endswith("_C"):
        return [f"C={row['C']:g}" for row in summaries]
    if name == "LR_weight" or name == "MLP_weight":
        return [f"训练相对权重 {row['relative_weight']:g}" for row in summaries]
    if name == "MLP_learning_rate":
        return [f"学习率 {row['learning_rate_init']:g}" for row in summaries]
    if name.startswith("MLP_epochs"):
        return [f"训练 {row['epochs']} 轮" for row in summaries]
    if name == "MLP_regularization":
        return [f"alpha={row['alpha']:g}" for row in summaries]
    if name == "MLP_ngram":
        return ["仅单词", "单词+短语"]
    if name == "MLP_tf_scaling":
        return ["普通词频", "对数词频"]
    return [f"隐藏层 {row['hidden_units']}" for row in summaries]


def make_comparison(name, ids, plot_dir):
    summaries = [json.loads((PUBLIC / "runs" / rid / "summary.json").read_text(encoding="utf-8")) for rid in ids]
    labels = labels_for(name, summaries)
    source_records = []
    for rid in ids:
        path = PUBLIC / "runs" / rid / "summary.json"
        source_records.append({"file": Path(os.path.relpath(path, plot_dir)).as_posix(), "sha256": sha(path)})
    if summaries[0]["model"] == "MLP":
        histories = [pd.read_csv(PUBLIC / "runs" / rid / "history.csv") for rid in ids]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), layout="constrained")
        for history, label, color, rid in zip(histories, labels, ("#2166AC", "#D6604D", "#27804F"), ids):
            epochs = history.epoch
            for ax, field in zip(axes[0], ("loss", "accuracy")):
                ax.plot(epochs, history[f"train_{field}"], color=color, marker=".", markersize=3,
                        linewidth=1, label=label + "：训练")
                ax.plot(epochs, history[f"val_{field}"], color=color, linestyle="--", marker=".",
                        markersize=3, linewidth=1, label=label + "：验证")
            for ax, field in zip(axes[1], ("loss", "accuracy")):
                ax.plot(epochs, history[f"val_{field}"], color=color, marker=".", markersize=3,
                        linewidth=1, label=label + "：验证")
            path = PUBLIC / "runs" / rid / "history.csv"
            source_records.append({"file": Path(os.path.relpath(path, plot_dir)).as_posix(), "sha256": sha(path),
                                   "points": len(history), "epochs": [int(epochs.min()), int(epochs.max())]})
        end = max(history.epoch.max() for history in histories)
        zoom_start = max(1, min(history.epoch.max() for history in histories) - 20)
        for ax in axes[0]:
            ax.set_xlim(1, end)
        for ax in axes[1]:
            ax.set_xlim(zoom_start, end)
        for column, field in enumerate(("loss", "accuracy")):
            for ax in axes[:, column]:
                ax.set(xlabel="训练轮次", ylabel="平均交叉熵（不含L2惩罚）" if field == "loss" else "准确率")
                if field == "accuracy":
                    ax.yaxis.set_major_formatter(PercentFormatter(1))
            late_values = np.concatenate([history.loc[history.epoch >= zoom_start, f"val_{field}"].to_numpy()
                                          for history in histories])
            low, high = late_values.min(), late_values.max()
            margin = max((high - low) * .12, .0005 if field == "loss" else .001)
            axes[1, column].set_ylim(low - margin, high + margin)
            axes[1, column].set_title("后段验证曲线（局部放大）")
        axes[0, 0].set_title("全部逐轮损失：训练与验证")
        axes[0, 1].set_title("全部逐轮准确率：训练与验证")
        for ax in axes.ravel():
            ax.grid(alpha=.25)
            ax.legend(fontsize=8, ncol=1 if len(ids) == 2 else 2)
        note = "实线：训练，虚线：验证；下排为后段验证曲线。"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
        field = "C" if name.endswith("_C") else "max_features" if name.endswith("vocabulary") else "relative_weight"
        x = [row[field] for row in summaries]
        xlabel = {"C": "C（越小，正则化越强）", "max_features": "词表上限", "relative_weight": "目标类别训练相对权重"}[field]
        axes[0].plot(x, [row["train_accuracy"] for row in summaries], marker="o", label="训练")
        axes[0].plot(x, [row["val_accuracy"] for row in summaries], marker="o", label="验证")
        axes[0].set(ylabel="准确率", title="每个配置的最终准确率")
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        if summaries[0]["model"] == "LR":
            axes[1].plot(x, [row["train_loss"] for row in summaries], marker="o", label="训练")
            axes[1].plot(x, [row["val_loss"] for row in summaries], marker="o", label="验证")
            axes[1].set(ylabel="平均交叉熵（不含L2惩罚）", title="每个配置的最终交叉熵")
        else:
            axes[1].plot(x, [row["support_vectors"] for row in summaries], marker="o", label="支持向量数")
            axes[1].set(ylabel="支持向量数量", title="支持向量数量")
        for ax in axes:
            ax.set_xlabel(xlabel)
            if field == "C":
                ax.set_xscale("log")
            ax.set_xticks(x, [str(value) for value in x])
            ax.grid(alpha=.25)
            ax.legend()
        note = "横轴是参数，不是训练轮次；该求解器没有保存逐轮曲线。"
    fig.suptitle(TITLES[name] + "\n" + note, fontsize=13)
    path = plot_dir / f"{name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    save_json(plot_dir / f"{name}_图源.json", {
        "comparison": name, "runs": ids, "input_files": source_records,
        "smooth": False, "interpolation": False, "points_removed": False,
        "y_axis_note": "cross-entropy excludes L2; accuracy is fraction", "plot_sha256": sha(path),
    })


def main():
    global PUBLIC
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PUBLIC)
    parser.add_argument("--output", type=Path, default=LAB / "local_plots/comparisons")
    args = parser.parse_args()
    PUBLIC, output = args.results.resolve(), args.output.resolve()
    session = json.loads((PUBLIC / "session.json").read_text(encoding="utf-8"))
    groups = session["comparison_groups"]
    # 输出使用新目录，保留归档图片和图源校验值。
    if output.exists():
        raise FileExistsError("Output already exists; choose a new directory.")
    for ids in groups.values():
        for run_id in ids:
            directory = PUBLIC / "runs" / run_id
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            if summary["model"] == "MLP":
                history = pd.read_csv(directory / "history.csv")
                if history.epoch.tolist() != list(range(1, int(summary["epochs"]) + 1)):
                    raise ValueError("Incomplete epoch record: " + run_id)
    output.mkdir(parents=True)
    for name, ids in groups.items():
        make_comparison(name, ids, output)
    print(json.dumps({"comparisons": len(groups), "training_performed": False,
                      "stored_records_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
