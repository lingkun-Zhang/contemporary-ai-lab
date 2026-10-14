"""比较单隐藏层 25/50/100 个神经元的验证效果。

三组只改变宽度，共用数据和训练配置。
直接运行训练三组；--check-only 只检查数据划分和配置。
"""

import argparse
import hashlib
import io
import json
import unicodedata
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
SEED = 42
MAX_FEATURES = 5000
HIDDEN_SIZES = (25, 50, 100)
LEARNING_RATE = 0.0001
ALPHA = 0.005  # 固定基线，不跟随原实验脚本的参数改动。
BATCH_SIZE = 64
EPOCHS = 100


def load_split():
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    texts, labels = frame["text"].astype(str).tolist(), frame["target"].to_numpy()
    # 固定分层划分，让三组使用相同样本。
    return train_test_split(
        texts, labels, np.arange(len(frame)),
        test_size=0.2, random_state=SEED, stratify=labels,
    )


def train_one(hidden_units, X_fit, X_val, y_fit, y_val, *, seed=SEED):
    # 每组从头训练；同一组保留模型及 Adam 状态。
    # 不同宽度消耗的随机数不同，同种子不代表相同初始化或打乱顺序。
    model = MLPClassifier(
        hidden_layer_sizes=(hidden_units,), activation="relu", solver="adam",
        learning_rate_init=LEARNING_RATE, alpha=ALPHA, batch_size=BATCH_SIZE,
        early_stopping=False, random_state=np.random.RandomState(seed),
    )
    classes = np.unique(y_fit)
    history, training_seconds = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        # partial_fit 遍历一次训练集，作为一轮；验证集不参与更新。
        model.partial_fit(X_fit, y_fit, classes=classes)
        training_seconds += perf_counter() - start
        train_proba, val_proba = model.predict_proba(X_fit), model.predict_proba(X_val)
        train_pred = model.classes_[train_proba.argmax(axis=1)]
        val_pred = model.classes_[val_proba.argmax(axis=1)]
        # 轮末重新评估两组数据；这里的交叉熵不含训练中的 L2 惩罚。
        row = {
            "epoch": epoch,
            "train_loss": log_loss(y_fit, train_proba, labels=model.classes_),
            "val_loss": log_loss(y_val, val_proba, labels=model.classes_),
            "train_accuracy": accuracy_score(y_fit, train_pred),
            "val_accuracy": accuracy_score(y_val, val_pred),
            "training_seconds": training_seconds,
        }
        history.append(row)
        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(
                f"第 {epoch:3d} 轮 | 训练损失={row['train_loss']:.4f} | 验证损失={row['val_loss']:.4f}\n"
                f"          训练准确率={row['train_accuracy']:.2%} | 验证准确率={row['val_accuracy']:.2%}",
                flush=True,
            )
    # 包括权重和偏置，便于比较模型规模。
    parameter_count = sum(array.size for array in model.coefs_ + model.intercepts_)
    summary = {
        "hidden_units": hidden_units, "parameter_count": int(parameter_count),
        "train_accuracy": float(row["train_accuracy"]), "val_accuracy": float(row["val_accuracy"]),
        "train_loss": float(row["train_loss"]), "val_loss": float(row["val_loss"]),
        "accuracy_gap": float(row["train_accuracy"] - row["val_accuracy"]),
        "macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "wrong_count": int(np.sum(val_pred != y_val)), "training_seconds": training_seconds,
    }
    return model, pd.DataFrame(history), summary, val_pred, val_proba


def print_comparison(summaries, train_count, val_count, feature_count):
    def cell(text, width, right=False):
        text = str(text)
        display_width = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        padding = " " * max(0, width - display_width)
        return padding + text if right else text + padding

    print(f"\n========== MLP 隐藏层宽度对照：最终第 {EPOCHS} 轮 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；实际特征数={feature_count}")
    print(f"固定：学习率={LEARNING_RATE:g}；L2 正则化 alpha={ALPHA:g}")
    print(f"固定：ReLU、Adam；批次={BATCH_SIZE}；种子={SEED}；不加权")
    print("只改变单隐藏层的神经元数量。")
    print("-" * 54)
    print(cell("指标", 18) + "".join(cell(f"{row['hidden_units']} 个", 12, True) for row in summaries))
    metrics = (
        ("模型参数量", "parameter_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"),
        ("验证准确率", "val_accuracy", ".2%"),
        ("训练交叉熵", "train_loss", ".4f"),
        ("验证交叉熵", "val_loss", ".4f"),
        ("训练-验证差距", "accuracy_gap", ".2%"),
        ("验证错分数量", "wrong_count", "d"),
        ("训练耗时/秒", "training_seconds", ".2f"),
    )
    for label, key, spec in metrics:
        print(cell(label, 18) + "".join(cell(format(row[key], spec), 12, True) for row in summaries))
    print("-" * 54)
    print("评估交叉熵不含 L2 项；模型训练仍使用 L2 正则化。")
    print("耗时只计训练，不含评估及文件读写。")
    # 统一比较最后一轮，不挑各组的历史最高分。
    best = min(summaries, key=lambda row: (-row["val_accuracy"], row["val_loss"], row["parameter_count"]))
    print(f"按最后一轮验证准确率选出的宽度：{best['hidden_units']} 个。")
    print("同分时比较验证交叉熵，再比较参数量。")
    small = min(summaries, key=lambda row: row["hidden_units"])
    large = max(summaries, key=lambda row: row["hidden_units"])
    accuracy_delta = 100 * (large["val_accuracy"] - small["val_accuracy"])
    loss_delta = large["val_loss"] - small["val_loss"]
    print(
        f"{large['hidden_units']} 对比 {small['hidden_units']} 个：验证准确率 {accuracy_delta:+.2f} 个百分点，"
        f"交叉熵 {loss_delta:+.4f}。"
    )
    print(f"训练种子={SEED}，每组轮数={EPOCHS}。", flush=True)


def plot_comparison(output_dir, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for color, (width, history) in zip(("#2166AC", "#D6604D", "#27804F"), histories.items()):
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(history["epoch"], history[f"train_{metric}"], color=color, label=f"{width} 个：训练")
            ax.plot(history["epoch"], history[f"val_{metric}"], color=color, linestyle="--", label=f"{width} 个：验证")
    axes[0].set(title="平均交叉熵（不含 L2 惩罚）", ylabel="交叉熵")
    axes[1].set(title="分类准确率", ylabel="准确率")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("训练轮数")
        ax.set_xlim(1, EPOCHS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"单隐藏层宽度对照：实线训练，虚线验证\n"
        f"学习率={LEARNING_RATE:g}，alpha={ALPHA:g}，批次={BATCH_SIZE}，种子={SEED}", fontsize=12,
    )
    fig.savefig(output_dir / "structure_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False):
    if not HIDDEN_SIZES or len(set(HIDDEN_SIZES)) != len(HIDDEN_SIZES):
        raise ValueError("隐藏层宽度必须非空且不重复。")
    if EPOCHS < 1 or any(not isinstance(width, int) or width < 1 for width in HIDDEN_SIZES):
        raise ValueError("训练轮数和隐藏层宽度必须为正整数。")
    X_fit, X_val, y_fit, y_val, fit_indices, val_indices = load_split()
    print(f"实际训练集：{len(X_fit)} 篇；验证集：{len(X_val)} 篇")
    print(f"本次宽度：{HIDDEN_SIZES}；每组 {EPOCHS} 轮；不加权")
    print(f"共同学习率={LEARNING_RATE:g}；共同 L2 正则化 alpha={ALPHA:g}")
    if check_only:
        print("检查通过，未训练或生成实验结果。")
        return

    # 只在训练部分拟合词表和 IDF，三组共用特征矩阵。
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_fit_tfidf, X_val_tfidf = vectorizer.fit_transform(X_fit), vectorizer.transform(X_val)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"structure_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    common_config = {
        "seed": SEED, "validation_fraction": 0.2, "max_features": MAX_FEATURES,
        "actual_feature_count": X_fit_tfidf.shape[1], "hidden_layers": 1,
        "activation": "relu", "solver": "adam", "learning_rate_init": LEARNING_RATE,
        "alpha": ALPHA, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "training_threads": 2,
        "early_stopping": False, "sample_weight": None,
        "loss_definition": "epoch-end unweighted cross-entropy without L2 penalty",
        "comparison_model": "final epoch only",
        "selection_rule": "final validation accuracy, then lower cross-entropy, then fewer parameters",
        "randomness_caveat": "Same seed is not identical initialization or minibatch order across different shapes.",
        "train_data_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "train_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype="<i8").tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype="<i8").tobytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__, "matplotlib_version": version("matplotlib"),
    }
    summaries, histories = [], {}
    print("开始训练三组；验证集只评估。", flush=True)
    with threadpool_limits(limits=2):
        for hidden_units in HIDDEN_SIZES:
            print(f"\n---------- 开始训练：一个隐藏层，{hidden_units} 个神经元 ----------", flush=True)
            model, history, summary, val_pred, val_proba = train_one(
                hidden_units, X_fit_tfidf, X_val_tfidf, y_fit, y_val,
            )
            group_dir = output_dir / f"hidden{hidden_units}"
            group_dir.mkdir(exist_ok=False)
            history.to_csv(group_dir / "history.csv", index=False, encoding="utf-8-sig")
            config = {**common_config, "hidden_units": hidden_units, "parameter_count": summary["parameter_count"]}
            (group_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            # 完整错例保留在文件中，避免终端被长文本占满。
            with redirect_stdout(io.StringIO()):
                save_validation_analysis(
                    group_dir, X_val, y_val, val_pred, val_proba,
                    model.classes_, val_indices, epoch=EPOCHS,
                )
            summaries.append(summary)
            histories[hidden_units] = history
            del model

    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_config.json").write_text(
        json.dumps({**common_config, "hidden_sizes": HIDDEN_SIZES, "summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 先显示结果，即使绘图失败也能查看对照。
    print_comparison(summaries, len(y_fit), len(y_val), X_fit_tfidf.shape[1])
    plot_comparison(output_dir, histories)
    print("日志、错例和曲线保存于：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较 MLP 单隐藏层的神经元数量。")
    parser.add_argument("--check-only", action="store_true", help="检查数据划分和配置，不训练。")
    main(check_only=parser.parse_args().check_only)
