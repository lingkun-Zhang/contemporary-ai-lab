"""MLP 类别加权对照：调整训练权重，验证集不加权。"""

import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
SEED = 42
MAX_FEATURES = 5000
HIDDEN_UNITS = 50
LEARNING_RATE = 0.0001
ALPHA = 0.005  # 固定已有基线，不随原脚本参数变化。
BATCH_SIZE = 64
EPOCHS = 100
WEIGHT_CLASS = 2  # 基线漏分最多、召回率最低的类别。
WEIGHT_FACTORS = (1.0, 1.5)
REFERENCE_RUN = DATA_DIR / "mlp_results" / "lr0.0001_e100_20261013_003846_215423"


def training_weights(labels, factor):
    """目标类别取 factor 倍权重，再将全局均值归一为 1。"""
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError("权重倍数必须为有限正数。")
    weights = np.where(np.asarray(labels) == WEIGHT_CLASS, factor, 1.0)
    return weights / weights.mean()


def train_one(factor, X_fit_tfidf, X_val_tfidf, X_val, y_fit, y_val, val_indices, run_dir):
    # 同一结构、同一随机状态，只改变训练权重。
    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_UNITS,), solver="adam", learning_rate_init=LEARNING_RATE,
        alpha=ALPHA, batch_size=BATCH_SIZE, early_stopping=False,
        random_state=np.random.RandomState(SEED),
    )
    classes = np.unique(y_fit)
    weights = None if factor == 1.0 else training_weights(y_fit, factor)
    history, training_seconds = [], 0.0
    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        model.partial_fit(X_fit_tfidf, y_fit, classes=classes, sample_weight=weights)
        training_seconds += perf_counter() - start
        train_proba = model.predict_proba(X_fit_tfidf)
        val_proba = model.predict_proba(X_val_tfidf)
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
        if epoch % 20 == 0 or epoch == 1:
            print(
                f"第 {epoch:3d} 轮 | 训练损失={row['train_loss']:.4f} | "
                f"验证损失={row['val_loss']:.4f}\n"
                f"          训练准确率={row['train_accuracy']:.2%} | "
                f"验证准确率={row['val_accuracy']:.2%}",
                flush=True,
            )

    run_dir.mkdir(parents=False, exist_ok=False)
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False, encoding="utf-8-sig")
    config = {
        "seed": SEED, "validation_fraction": 0.2, "max_features": MAX_FEATURES,
        "hidden_units": HIDDEN_UNITS, "solver": "adam", "learning_rate_init": LEARNING_RATE,
        "alpha": ALPHA, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "training_threads": 2,
        "weight_class": WEIGHT_CLASS, "relative_weight": factor,
        "weight_normalization": "training weights have global mean 1; baseline uses no weights",
        "weight_caveat": "MLP normalizes by each batch weight sum; global normalization does not fix every batch regularization scale",
        "validation_weighted": False,
        "loss_definition": "epoch-end unweighted mean cross-entropy without regularization",
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__, "matplotlib_version": version("matplotlib"),
        "train_data_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # 保存完整分析，只隐藏其长篇打印；异常仍会抛出。
    with redirect_stdout(io.StringIO()):
        save_validation_analysis(
            run_dir, X_val, y_val, val_pred, val_proba, model.classes_, val_indices, epoch=EPOCHS,
        )
    summary = {
        "relative_weight": factor,
        "train_accuracy": float(history[-1]["train_accuracy"]),
        "val_accuracy": float(history[-1]["val_accuracy"]),
        "train_loss": float(history[-1]["train_loss"]),
        "val_loss": float(history[-1]["val_loss"]),
        "macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "class2_precision": float(precision_score(y_val, val_pred, labels=[2], average=None, zero_division=0)[0]),
        "class2_recall": float(recall_score(y_val, val_pred, labels=[2], average=None, zero_division=0)[0]),
        "class7_recall": float(recall_score(y_val, val_pred, labels=[7], average=None, zero_division=0)[0]),
        "wrong_count": int(np.sum(val_pred != y_val)),
        "target_class_count": int(np.sum(y_val == WEIGHT_CLASS)),
        "target_class_missed": int(np.sum((y_val == WEIGHT_CLASS) & (val_pred != WEIGHT_CLASS))),
        "target_class_false_positive": int(np.sum((y_val != WEIGHT_CLASS) & (val_pred == WEIGHT_CLASS))),
        "training_seconds": training_seconds,
    }
    return summary, history_df, val_pred, val_proba


def print_error_changes(cases, corrected, introduced):
    """预览少量修正与新增错误，完整文本保存在 CSV。"""
    print("\n========== 错例变化预览 ==========")
    for title, mask in (("修正旧错误", corrected), ("新增错误", introduced)):
        selected = cases.loc[mask].head(2)
        if selected.empty:
            print(f"{title}：无")
        for row in selected.itertuples(index=False):
            print(
                f"{title} | 验证索引={row.validation_index} | 真实类别={row.true_label}\n"
                f"  普通组预测={row.baseline_prediction} -> 加权组预测={row.weighted_prediction}"
            )
            lines = str(row.text).splitlines()
            subject = next((line for line in lines if line.lower().startswith("subject:")), None)
            if subject is None:
                subject = " ".join(str(row.text).split())
            # 按索引取前两例，仅截短终端显示。
            print("  文本：" + subject[:68] + ("..." if len(subject) > 68 else ""))
    print("每种预览最多2条，其他错例见CSV。")


def print_final_comparison(summaries, corrected_count, introduced_count, train_count, val_count):
    """比较最终轮指标及修正、新增错误。"""
    import unicodedata

    baseline, weighted = summaries

    def left_cell(label, width=20):
        display_width = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in label)
        return label + " " * max(0, width - display_width)

    def metric(label, field, spec):
        print(left_cell(label) + format(baseline[field], spec).rjust(12) + format(weighted[field], spec).rjust(12))

    print(f"\n========== MLP 加权对照：最终第 {EPOCHS} 轮 ==========")
    print(f"实际训练集：{train_count} 篇；验证集：{val_count} 篇")
    print(f"学习率={LEARNING_RATE:g}；L2 正则化 alpha={ALPHA:g}")
    print(f"隐藏层={HIDDEN_UNITS}；特征上限={MAX_FEATURES}；批次={BATCH_SIZE}；种子={SEED}")
    print(f"仅调整类别 {WEIGHT_CLASS} 的训练相对权重；验证集不加权。")
    print("-" * 44)
    print(left_cell("指标") + "普通组".rjust(9) + "加权组".rjust(9))
    metric(f"类别 {WEIGHT_CLASS} 相对权重", "relative_weight", ".1f")
    metric("训练准确率", "train_accuracy", ".2%")
    metric("验证准确率", "val_accuracy", ".2%")
    metric("训练交叉熵", "train_loss", ".4f")
    metric("验证交叉熵", "val_loss", ".4f")
    metric("验证错分总数", "wrong_count", "d")
    metric(f"类别 {WEIGHT_CLASS} 漏分数", "target_class_missed", "d")
    metric(f"类别 {WEIGHT_CLASS} 误收数", "target_class_false_positive", "d")
    print("-" * 44)
    print(f"类别 {WEIGHT_CLASS} 共 {baseline['target_class_count']} 篇；误收=其他类别被预测成 {WEIGHT_CLASS}。")
    print("交叉熵为未加权的轮末指标，不含训练中的 L2 项。")
    net = corrected_count - introduced_count
    print(f"修正旧错误 {corrected_count} 条；新增错误 {introduced_count} 条；错分净减少 {net:+d} 条。")
    accuracy_delta = 100 * (weighted["val_accuracy"] - baseline["val_accuracy"])
    loss_delta = weighted["val_loss"] - baseline["val_loss"]
    print(f"验证准确率变化：{accuracy_delta:+.2f} 个百分点；交叉熵变化：{loss_delta:+.4f}。")
    if accuracy_delta > 0 and loss_delta > 0:
        print("结果：错分减少，交叉熵略升。")
    elif accuracy_delta > 0 and loss_delta < 0:
        print("结果：准确率提升，交叉熵降低。")
    elif accuracy_delta > 0:
        print("结果：准确率提升，交叉熵未变。")
    elif accuracy_delta < 0:
        print("结果：整体准确率下降，需结合目标类别指标判断。")
    else:
        print("结果：准确率未变，结合交叉熵与错例判断。")
    print("结果限于这次固定划分。", flush=True)


def plot_comparison(output_dir, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for factor, history in histories.items():
        label = "普通权重" if factor == 1.0 else f"类别 {WEIGHT_CLASS} 相对权重 {factor:g}"
        final = history.iloc[-1]
        axes[0].plot(history["epoch"], history["val_loss"], label=f"{label}（末轮 {final['val_loss']:.4f}）")
        axes[1].plot(history["epoch"], history["val_accuracy"], label=f"{label}（末轮 {final['val_accuracy']:.2%}）")
    axes[0].set(title="验证集分类交叉熵（不含惩罚项）", ylabel="交叉熵")
    axes[1].set(title="验证集准确率", ylabel="准确率")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("训练轮数")
        ax.set_xlim(1, EPOCHS)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(
        f"MLP 加权对照：验证集不加权\n学习率={LEARNING_RATE:g}，alpha={ALPHA:g}，"
        f"隐藏层={HIDDEN_UNITS}，批次={BATCH_SIZE}，种子={SEED}", fontsize=12,
    )
    fig.savefig(output_dir / "weight_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False):
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    texts, targets = frame["text"].astype(str).tolist(), frame["target"].to_numpy()
    X_fit, X_val, y_fit, y_val, _fit_indices, val_indices = train_test_split(
        texts, targets, np.arange(len(frame)), test_size=0.2, random_state=SEED, stratify=targets,
    )
    # 全局均值为1，但各批次权重和仍可能不同，影响L2项的相对尺度。
    weights = training_weights(y_fit, 1.5)
    assert np.isclose(weights.mean(), 1.0)
    assert np.isclose(weights[y_fit == WEIGHT_CLASS][0] / weights[y_fit != WEIGHT_CLASS][0], 1.5)
    assert (LEARNING_RATE, ALPHA, EPOCHS) == (0.0001, 0.005, 100)
    if check_only:
        print(f"检查通过：训练 {len(X_fit)} 篇、验证 {len(X_val)} 篇，目标类别训练 {np.sum(y_fit == WEIGHT_CLASS)} 篇。")
        print("权重均值=1，相对倍数=1.5；未训练或保存结果。")
        return

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X_fit_tfidf = vectorizer.fit_transform(X_fit)
    X_val_tfidf = vectorizer.transform(X_val)
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"weighting_class{WEIGHT_CLASS}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"从头训练两组MLP，各 {EPOCHS} 轮。")
    print(f"比较类别 {WEIGHT_CLASS} 的训练相对权重：1 与 1.5。")
    print(f"固定学习率={LEARNING_RATE:g}；L2 正则化 alpha={ALPHA:g}。")
    print("记录未加权交叉熵；训练含L2，验证集仅用于评估。", flush=True)
    summaries, histories, predictions, probabilities = [], {}, {}, {}
    with threadpool_limits(limits=2):
        for factor in WEIGHT_FACTORS:
            print(f"\n---------- 开始训练：类别 {WEIGHT_CLASS} 相对权重 {factor:g} ----------", flush=True)
            run_dir = output_dir / f"weight{factor:g}"
            summary, history, pred, proba = train_one(
                factor, X_fit_tfidf, X_val_tfidf, X_val, y_fit, y_val, val_indices, run_dir,
            )
            summaries.append(summary)
            histories[factor], predictions[factor], probabilities[factor] = history, pred, proba

    baseline_correct = predictions[1.0] == y_val
    weighted_correct = predictions[1.5] == y_val
    corrected = ~baseline_correct & weighted_correct
    introduced = baseline_correct & ~weighted_correct
    cases = pd.DataFrame({
        "validation_index": np.arange(len(y_val)), "source_index": val_indices,
        "true_label": y_val, "baseline_prediction": predictions[1.0],
        "weighted_prediction": predictions[1.5],
        "baseline_predicted_probability": probabilities[1.0].max(axis=1),
        "weighted_predicted_probability": probabilities[1.5].max(axis=1), "text": X_val,
    })
    cases.loc[corrected].to_csv(output_dir / "corrected_errors.csv", index=False, encoding="utf-8-sig")
    cases.loc[introduced].to_csv(output_dir / "introduced_errors.csv", index=False, encoding="utf-8-sig")
    assert summaries[0]["wrong_count"] - summaries[1]["wrong_count"] == int(corrected.sum() - introduced.sum())
    reference_match = None
    reference_file = REFERENCE_RUN / "validation_predictions.csv"
    if reference_file.exists():
        old = pd.read_csv(reference_file)
        reference_match = bool(
            np.array_equal(old["source_index"].to_numpy(), val_indices)
            and np.array_equal(old["true_label"].to_numpy(), y_val)
            and np.array_equal(old["predicted_label"].to_numpy(), predictions[1.0])
        )
    summary_data = {
        "model": "final epoch only", "weight_class": WEIGHT_CLASS, "epochs": EPOCHS,
        "summaries": summaries, "corrected_count": int(corrected.sum()),
        "introduced_count": int(introduced.sum()), "baseline_matches_saved_predictions": reference_match,
        "caveat": "One validation split and fixed training budget; not proof of stable improvement or fully converged optima.",
    }
    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "summary.json").write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_comparison(output_dir, histories)
    print_error_changes(cases, corrected, introduced)
    print_final_comparison(
        summaries, int(corrected.sum()), int(introduced.sum()), len(y_fit), len(y_val),
    )
    if reference_match is not None:
        print(f"与上次基线预测核对：{'一致' if reference_match else '不一致，请检查数据或环境'}")
    print("错例、日志与对照图目录：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定配置的 MLP 训练样本加权对照。")
    parser.add_argument("--check-only", action="store_true", help="只检查划分、参数和权重，不训练。")
    main(check_only=parser.parse_args().check_only)
