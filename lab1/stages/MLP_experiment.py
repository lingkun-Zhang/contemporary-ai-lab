"""MLP 训练：每轮记录训练、验证损失和准确率，并保存曲线。"""

import argparse
import json
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits


DATA_DIR = Path(__file__).resolve().parent

# 训练参数。
SEED = 42
MAX_FEATURES = 5000
HIDDEN_UNITS = 50
LEARNING_RATE = 0.0001
ALPHA = 0.05
BATCH_SIZE = 64
EPOCHS = 100  # 每轮遍历一次训练集，包含多次小批次更新。


def main():
    # 按脚本位置找数据，避免受运行目录影响。
    train_df = pd.read_csv(DATA_DIR / "train_data.csv")
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].to_numpy()

    # 沿用逻辑回归的数据划分，保留原索引方便回查错例。
    X_fit, X_val, y_fit, y_val, _fit_indices, val_indices = train_test_split(
        X_train, y_train, np.arange(len(train_df)),
        test_size=0.2, random_state=SEED, stratify=y_train
    )

    # 验证集不能参与拟合词表和 IDF。
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_fit_tfidf = vectorizer.fit_transform(X_fit)
    X_val_tfidf = vectorizer.transform(X_val)
    classes = np.unique(y_fit)

    # 不在每轮重建模型，否则权重和 Adam 的状态会被重置。
    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,),
        solver="adam",
        learning_rate_init=LEARNING_RATE,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        early_stopping=False,
        random_state=np.random.RandomState(SEED),
    )

    print(f"实际训练集：{len(X_fit)}，验证集：{len(X_val)}")
    print(f"输入特征数：{X_fit_tfidf.shape[1]}，类别数：{len(classes)}")
    print(f"隐藏层：{HIDDEN_UNITS}，学习率：{LEARNING_RATE}，轮数：{EPOCHS}")
    print(f"训练使用 L2（alpha={ALPHA:g}）；下方损失只统计轮末交叉熵，不加 L2 项。")

    history = []
    training_seconds = 0.0
    # 限制并行线程，避免小批次矩阵计算产生过多线程开销。
    with threadpool_limits(limits=2):
        for epoch in range(1, EPOCHS + 1):
            start = perf_counter()
            # partial_fit 接着上一轮训练；验证数据只用来评估。
            model.partial_fit(X_fit_tfidf, y_fit, classes=classes)
            training_seconds += perf_counter() - start

            train_proba = model.predict_proba(X_fit_tfidf)
            val_proba = model.predict_proba(X_val_tfidf)
            # 概率列按 classes_ 排列，不能直接把列号当标签。
            train_pred = model.classes_[train_proba.argmax(axis=1)]
            val_pred = model.classes_[val_proba.argmax(axis=1)]

            row = {
                "epoch": epoch,
                "train_loss": log_loss(y_fit, train_proba, labels=model.classes_),
                "val_loss": log_loss(y_val, val_proba, labels=model.classes_),
                "train_accuracy": accuracy_score(y_fit, train_pred),
                "val_accuracy": accuracy_score(y_val, val_pred),
                "training_seconds": training_seconds,
            }
            history.append(row)
            print(
                f"第 {epoch:02d} 轮 | "
                f"训练损失={row['train_loss']:.4f} | "
                f"验证损失={row['val_loss']:.4f} | "
                f"训练准确率={row['train_accuracy']:.2%} | "
                f"验证准确率={row['val_accuracy']:.2%}",
                flush=True,
            )

    # 每次用新目录保存，方便比较，也不会覆盖旧结果。
    run_time = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"lr{LEARNING_RATE:g}_e{EPOCHS}_{run_time}"
    output_dir.mkdir(parents=True, exist_ok=False)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False, encoding="utf-8-sig")
    config = {
        "seed": SEED,
        "validation_fraction": 0.2,
        "max_features": MAX_FEATURES,
        "hidden_units": HIDDEN_UNITS,
        "solver": "adam",
        "learning_rate_init": LEARNING_RATE,
        "alpha": ALPHA,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "training_threads": 2,
        "loss_definition": "epoch-end mean cross-entropy without regularization",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "matplotlib_version": version("matplotlib"),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n第 {EPOCHS} 轮模型的验证准确率：{history[-1]['val_accuracy']:.2%}")
    print(f"第 {EPOCHS} 轮模型的验证 Macro-F1：{f1_score(y_val, val_pred, average='macro'):.4f}")
    print(f"模型训练耗时（不含评估）：{training_seconds:.2f} 秒")
    print(f"原始日志和参数已保存到：{output_dir}")
    # 用最后一轮的预测保存错例。
    save_validation_analysis(
        output_dir, X_val, y_val, val_pred, val_proba,
        model.classes_, val_indices, epoch=EPOCHS,
    )
    plot_history(output_dir)


def save_validation_analysis(
    output_dir, X_val, y_val, val_pred, val_proba, classes, source_indices, *, epoch
):
    """保存最后一轮的验证预测和错例；验证错例不加入训练。"""
    title = f"第 {epoch} 轮模型：验证集分类报告"
    report = classification_report(y_val, val_pred, labels=classes, digits=4, zero_division=0)
    print("\n" + title)
    print(report)
    (output_dir / "classification_report.txt").write_text(
        title + "\n\n" + report, encoding="utf-8"
    )

    # 概率列顺序由 classes_ 决定，类别编号不能直接当列号。
    label_to_column = {label: column for column, label in enumerate(classes)}
    true_columns = [label_to_column[label] for label in y_val]
    predicted_columns = [label_to_column[label] for label in val_pred]
    positions = np.arange(len(y_val))
    probabilities = np.asarray(val_proba)
    results = pd.DataFrame({
        "epoch": epoch,
        "validation_index": positions,
        "source_index": source_indices,  # 原 CSV 的记录索引，从 0 开始，不是文件行号。
        "true_label": y_val,
        "predicted_label": val_pred,
        "predicted_probability": probabilities[positions, predicted_columns],
        "true_label_probability": probabilities[positions, true_columns],
        "is_correct": np.asarray(y_val) == np.asarray(val_pred),
        "text": X_val,  # CSV 保留全文，终端只预览一部分。
    })
    results.to_csv(output_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    errors = results.loc[~results["is_correct"]].sort_values(
        ["predicted_probability", "validation_index"], ascending=[False, True]
    )
    errors.to_csv(output_dir / "wrong_predictions.csv", index=False, encoding="utf-8-sig")

    print(f"验证集错分：{len(errors)} / {len(results)} 篇")
    if errors.empty:
        print("没有错分样本，错误文件只包含表头。")
    else:
        print("预览 3 条预测概率最高的错例：")
        for number, row in enumerate(errors.head(3).itertuples(index=False), start=1):
            print(f"\n错例 {number} | 验证集索引={row.validation_index} | 原数据索引={row.source_index}")
            print(f"真实类别={row.true_label}，预测类别={row.predicted_label}")
            print(
                f"预测类别概率={row.predicted_probability:.2%}，"
                f"真实类别概率={row.true_label_probability:.2%}"
            )
            print(row.text[:800])
            if len(row.text) > 800:
                print("……（全文在 wrong_predictions.csv）")
    print("分类报告和完整预测、错例已保存。")


def plot_history(run_dir):
    """读取已保存的逐轮日志画图，不重新训练。"""
    import matplotlib

    matplotlib.use("Agg")  # 直接保存图片，不依赖弹出绘图窗口。
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    history = pd.read_csv(run_dir / "history.csv")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    columns = ["epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    if not set(columns).issubset(history.columns):
        raise ValueError("日志缺少绘图所需的指标列。")
    if not np.isfinite(history[columns].to_numpy()).all():
        raise ValueError("日志中含有非有限数值，请先检查原始记录。")
    if history["epoch"].tolist() != list(range(1, config["epochs"] + 1)):
        raise ValueError("日志轮数或顺序与运行配置不一致。")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    train_color, val_color = "#2166AC", "#D6604D"
    best_loss = history.loc[history["val_loss"].idxmin()]
    best_accuracy = history.loc[history["val_accuracy"].idxmax()]
    caption = (
        f"学习率={config['learning_rate_init']:g}，隐藏层={config['hidden_units']}，"
        f"批次={config['batch_size']}，alpha={config['alpha']:g}，随机种子={config['seed']}"
    )

    # 左图看全程，右图放大后段验证损失，便于观察细小变化。
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for index, ax in enumerate(axes):
        if index == 0:
            ax.plot(history["epoch"], history["train_loss"], color=train_color, label="训练集")
        ax.plot(history["epoch"], history["val_loss"], color=val_color, label="验证集")
        ax.set_xlabel("训练轮数（epoch）")
        ax.set_ylabel("轮末平均交叉熵（不含正则化惩罚）")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_title("完整训练过程")
    axes[0].set_xlim(1, config["epochs"])
    axes[0].set_ylim(bottom=0)
    zoom_start = max(1, config["epochs"] // 3)
    axes[1].set_title(f"第 {zoom_start} 轮起的验证损失放大")
    axes[1].set_ylabel("验证集平均交叉熵（纵轴局部放大）")
    axes[1].set_xlim(zoom_start, config["epochs"])
    zoom_values = history.loc[history["epoch"] >= zoom_start, "val_loss"]
    zoom_min, zoom_max = float(zoom_values.min()), float(zoom_values.max())
    zoom_padding = max((zoom_max - zoom_min) * 0.15, 0.002)
    axes[1].set_ylim(max(0, zoom_min - zoom_padding), zoom_max + zoom_padding)
    axes[1].axvline(best_loss["epoch"], color=val_color, linestyle="--", alpha=0.5)
    axes[1].scatter(best_loss["epoch"], best_loss["val_loss"], color=val_color, zorder=3)
    axes[1].annotate(
        f"验证损失最低：第 {int(best_loss['epoch'])} 轮\n{best_loss['val_loss']:.4f}",
        xy=(best_loss["epoch"], best_loss["val_loss"]),
        xytext=(0.40, 0.87), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": val_color}, fontsize=9,
    )
    fig.suptitle("MLP 训练集与验证集损失\n" + caption, fontsize=12)
    fig.savefig(run_dir / "loss_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    ax.plot(history["epoch"], history["train_accuracy"], color=train_color, label="训练集")
    ax.plot(history["epoch"], history["val_accuracy"], color=val_color, label="验证集")
    ax.scatter(best_accuracy["epoch"], best_accuracy["val_accuracy"], color=val_color, zorder=3)
    ax.annotate(
        f"验证准确率最高：第 {int(best_accuracy['epoch'])} 轮\n"
        f"{best_accuracy['val_accuracy']:.2%}",
        xy=(best_accuracy["epoch"], best_accuracy["val_accuracy"]),
        xytext=(0.37, 0.40), textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": val_color}, fontsize=10,
    )
    ax.set(title="MLP 训练集与验证集准确率\n" + caption,
           xlabel="训练轮数（epoch）", ylabel="准确率", xlim=(1, config["epochs"]))
    ax.set_ylim(max(0, float(history[["train_accuracy", "val_accuracy"]].min().min()) - 0.05), 1.01)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.savefig(run_dir / "accuracy_curve.png", dpi=220)
    plt.close(fig)

    print(f"验证损失最低的记录：第 {int(best_loss['epoch'])} 轮，{best_loss['val_loss']:.4f}")
    print(f"验证准确率最高的记录：第 {int(best_accuracy['epoch'])} 轮，{best_accuracy['val_accuracy']:.2%}")
    print("未保存各轮模型；指标峰值仅作标记，评估使用最后一轮模型。")
    print(f"损失和准确率曲线已保存到：{run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 MLP 基线，或只绘制已有日志。")
    parser.add_argument("--plot-only", action="store_true", help="只绘制最近一次运行的日志，不重新训练。")
    args = parser.parse_args()
    if args.plot_only:
        histories = list((DATA_DIR / "mlp_results").glob("*/history.csv"))
        if not histories:
            parser.error("未找到 history.csv，请先完成一次训练。")
        latest_history = max(histories, key=lambda path: path.stat().st_mtime)
        plot_history(latest_history.parent)
    else:
        main()
