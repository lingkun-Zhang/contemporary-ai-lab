"""固定词表和 MLP 配置，比较普通词频与对数词频。

两组只改变 sublinear_tf。--check-only 检查特征和旧结果，不训练。
"""

import argparse
import hashlib
import io
import json
import shutil
import unicodedata
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits

import MLP_ngram_experiment as base
import MLP_structure_experiment as core
from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
HIDDEN_UNITS = 100
TRAINING_SEED = 42
GROUPS = (("ordinary_tf", "普通词频", False), ("log_tf", "对数词频", True))


def make_features(sublinear_tf, fit_texts, val_texts):
    # 选词依据不变，sublinear_tf 只改变选词后的 TF 数值。
    vectorizer = TfidfVectorizer(
        max_features=core.MAX_FEATURES, ngram_range=(1, 1), sublinear_tf=sublinear_tf,
    )
    # 只在训练部分建立词表和 IDF，验证集沿用它们。
    X_fit, X_val = vectorizer.fit_transform(fit_texts), vectorizer.transform(val_texts)
    names = vectorizer.get_feature_names_out()
    features = pd.DataFrame({"feature_index": np.arange(len(names)), "feature": names, "idf": vectorizer.idf_})
    stats = {
        "actual_feature_count": len(names), "sublinear_tf": sublinear_tf,
        "training_nonzero_values": int(X_fit.nnz), "validation_nonzero_values": int(X_val.nnz),
        "vocabulary_sha256": hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
    }
    return X_fit, X_val, features, stats


def verify_same_features(ordinary, logarithmic):
    # 特征数相同还不够；词表、列顺序和 IDF 也必须一致。
    old_features, new_features = ordinary[2], logarithmic[2]
    if not np.array_equal(old_features["feature"].to_numpy(), new_features["feature"].to_numpy()):
        raise ValueError("两组词表或列顺序不同，不能作为单纯词频缩放对照。")
    if not np.array_equal(old_features["idf"].to_numpy(), new_features["idf"].to_numpy()):
        raise ValueError("两组 IDF 不同，请检查训练数据与特征提取设置。")
    if any(" " in feature for feature in old_features["feature"]):
        raise ValueError("本实验要求只使用单词，不包含双词短语。")
    for old_matrix, new_matrix in zip(ordinary[:2], logarithmic[:2]):
        if old_matrix.shape != new_matrix.shape or not np.array_equal(old_matrix.indptr, new_matrix.indptr) or not np.array_equal(old_matrix.indices, new_matrix.indices):
            raise ValueError("两组稀疏非零结构不同，请检查输入与配置。")


def compare_errors(y_val, ordinary_pred, logarithmic_pred, val_indices, val_texts):
    corrected, introduced = base.compare_errors(y_val, ordinary_pred, logarithmic_pred, val_indices, val_texts)
    names = {"unigram_prediction": "ordinary_tf_prediction", "phrase_prediction": "log_tf_prediction"}
    return corrected.rename(columns=names), introduced.rename(columns=names)


def print_results(summaries, train_count, val_count, corrected_count, introduced_count):
    def cell(text, width, right=False):
        text = str(text)
        size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        padding = " " * max(0, width - size)
        return padding + text if right else text + padding

    ordinary, logarithmic = summaries
    print(f"\n========== MLP 词频缩放对照：最终第 {core.EPOCHS} 轮 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；划分/训练种子均为42")
    print(f"固定：{HIDDEN_UNITS} 个神经元；学习率={core.LEARNING_RATE:g}；alpha={core.ALPHA:g}")
    print(f"固定：批次={core.BATCH_SIZE}；ReLU、Adam、不加权；只使用单词")
    print("词表、列顺序及IDF一致；只改变 sublinear_tf。")
    print("-" * 48)
    print(cell("指标", 20) + cell("普通词频", 14, True) + cell("对数词频", 14, True))
    for label, key, spec in (
        ("实际单词特征数", "actual_feature_count", "d"),
        ("模型参数量", "parameter_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"),
        ("验证准确率", "val_accuracy", ".2%"),
        ("训练交叉熵", "train_loss", ".4f"),
        ("验证交叉熵", "val_loss", ".4f"),
        ("验证错分数量", "wrong_count", "d"),
    ):
        print(cell(label, 20) + "".join(cell(format(row[key], spec), 14, True) for row in summaries))
    print("-" * 48)
    print(f"修正旧错误 {corrected_count} 条；新增错误 {introduced_count} 条；净少错 {corrected_count-introduced_count:+d} 条。")
    accuracy_change = 100 * (logarithmic["val_accuracy"] - ordinary["val_accuracy"])
    loss_change = logarithmic["val_loss"] - ordinary["val_loss"]
    print(f"验证准确率变化：{accuracy_change:+.2f} 个百分点；交叉熵变化：{loss_change:+.4f}。")
    if accuracy_change > 0 and loss_change < 0:
        print("对数词频提高准确率，降低验证交叉熵。")
    elif accuracy_change > 0:
        print("准确率提高，验证交叉熵未下降。")
    elif accuracy_change < 0:
        print("对数词频的验证准确率下降。")
    else:
        print("准确率相同，可结合交叉熵和错例分析。")
    print("词频缩放改变特征值，不改变样本权重或 alpha。")
    print("评估交叉熵不含 L2 项；模型训练仍使用 L2 正则化。")
    print("这里只比较一个训练种子，不能代表所有划分和种子。", flush=True)


def plot_comparison(output_dir, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for color, (name, history) in zip(("#2166AC", "#D6604D"), histories.items()):
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(history["epoch"], history[f"train_{metric}"], color=color, label=f"{name}：训练")
            ax.plot(history["epoch"], history[f"val_{metric}"], color=color, linestyle="--", label=f"{name}：验证")
    axes[0].set(title="平均交叉熵（不含 L2 惩罚）", ylabel="交叉熵")
    axes[1].set(title="分类准确率", ylabel="准确率")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("训练轮数")
        ax.set_xlim(1, core.EPOCHS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("同一单词词表：普通词频与对数词频\n实线训练，虚线验证；种子42，100神经元", fontsize=12)
    fig.savefig(output_dir / "tf_scaling_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False):
    base.validate_core_settings()
    if (base.SPLIT_SEED, base.TRAINING_SEED, base.HIDDEN_UNITS) != (42, 42, 100):
        raise ValueError("共用数据/缓存设置已变，本实验要求划分种子42、训练种子42和100个神经元。")
    fit_texts, val_texts, y_fit, y_val, fit_indices, val_indices = base.load_split()
    feature_sets = {key: make_features(flag, fit_texts, val_texts) for key, name, flag in GROUPS}
    verify_same_features(feature_sets["ordinary_tf"], feature_sets["log_tf"])
    expected = base.cache_expected(feature_sets["ordinary_tf"][3]["actual_feature_count"], fit_indices, val_indices)
    cached = base.find_unigram_cache(expected, y_val, val_indices, len(np.unique(y_fit)))
    print(f"训练 {len(y_fit)} 篇；验证 {len(y_val)} 篇；两组词表、IDF和非零位置均一致。")
    print("对照 sublinear_tf=False/True；本次不用官方测试集。")
    print("原始TF缩放示意：1次 -> 1；10次 -> 1+ln(10)约3.30。")
    print("缩放后仍乘IDF并归一化，这里不是最终TF-IDF值。")
    print(f"需要新训练 {1 if cached else 2} 组，每组 {core.EPOCHS} 轮。", flush=True)
    if check_only:
        print("检查通过，未训练或生成实验结果。")
        return

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"tf_scaling_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, histories, predictions = [], {}, {}
    with threadpool_limits(limits=2):
        for key, name, flag in GROUPS:
            X_fit, X_val, features, stats = feature_sets[key]
            group_dir = output_dir / key
            group_dir.mkdir(exist_ok=False)
            reuse = key == "ordinary_tf" and cached is not None
            if reuse:
                print("\n复用普通词频组第100轮结果。", flush=True)
                summary, history = cached["summary"], cached["history"]
                old_predictions = pd.read_csv(cached["source"] / "validation_predictions.csv")
                val_pred = old_predictions["predicted_label"].to_numpy()
                for filename in ("validation_predictions.csv", "wrong_predictions.csv", "classification_report.txt"):
                    shutil.copyfile(cached["source"] / filename, group_dir / filename)
                source = str(cached["source"])
            else:
                print(f"\n---------- 新训练：{name}，sublinear_tf={flag} ----------", flush=True)
                model, history, summary, val_pred, val_proba = core.train_one(
                    HIDDEN_UNITS, X_fit, X_val, y_fit, y_val, seed=TRAINING_SEED,
                )
                with redirect_stdout(io.StringIO()):
                    save_validation_analysis(
                        group_dir, val_texts, y_val, val_pred, val_proba,
                        model.classes_, val_indices, epoch=core.EPOCHS,
                    )
                source = None
                del model
            history.to_csv(group_dir / "history.csv", index=False, encoding="utf-8-sig")
            features.to_csv(group_dir / "features.csv", index=False, encoding="utf-8-sig")
            config = {
                **expected, **stats, "ngram_range": (1, 1), "split_seed": 42, "model_seed": 42,
                "hidden_units": HIDDEN_UNITS, "parameter_count": summary["parameter_count"],
                "reused": reuse, "source_run_dir": source, "norm": "l2",
                "current_groups_have_identical_vocabulary_and_idf": True,
                "current_training_code_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                "source_note": "Cached old default-unigram run did not record vocabulary/source hashes; recorded data/config/results were checked. New feature hashes describe the current two vectorizers, not proof of the old vocabulary hash.",
                "timing_note": "Reused training time belongs to an older run, not a controlled runtime benchmark.",
            }
            (group_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries.append({**summary, **stats, "group": key, "reused": reuse, "source_run_dir": source})
            histories[name], predictions[key] = history, val_pred

    corrected, introduced = compare_errors(
        y_val, predictions["ordinary_tf"], predictions["log_tf"], val_indices, val_texts,
    )
    corrected.to_csv(output_dir / "corrected_errors.csv", index=False, encoding="utf-8-sig")
    introduced.to_csv(output_dir / "introduced_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_config.json").write_text(json.dumps({
        "common_settings": expected, "summaries": summaries,
        "corrected_count": len(corrected), "introduced_count": len(introduced),
        "vocabulary_and_idf_match_between_current_groups": True,
        "scope": "Single-seed TF scaling screening on a fixed validation split, not a generalization guarantee.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print_results(summaries, len(y_fit), len(y_val), len(corrected), len(introduced))
    plot_comparison(output_dir, histories)
    print("日志、词表、错例和曲线保存于：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较 MLP 的普通词频和对数词频。")
    parser.add_argument("--check-only", action="store_true", help="检查特征和旧基线，不训练。")
    main(check_only=parser.parse_args().check_only)
