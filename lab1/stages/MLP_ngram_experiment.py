"""在 5000 个特征的预算下，比较单词与单词+双词短语。

固定 MLP 配置和种子 42，只改变 ngram_range。
--check-only 检查数据、词表及旧结果，不训练。
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
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

import MLP_structure_experiment as core
from MLP_stability_experiment import validate_cached_group
from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
SPLIT_SEED = 42
TRAINING_SEED = 42
HIDDEN_UNITS = 100
GROUPS = (("unigram", "单词", (1, 1)), ("unigram_bigram", "单词+短语", (1, 2)))


def validate_core_settings():
    # 防止共用脚本的配置改动影响这次对照。
    actual = (core.MAX_FEATURES, core.LEARNING_RATE, core.ALPHA, core.BATCH_SIZE, core.EPOCHS)
    if actual != (5000, 0.0001, 0.005, 64, 100):
        raise ValueError("共用训练配置已改变；本实验要求特征上限5000、学习率0.0001、alpha0.005、批次64、100轮。")


def load_split():
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    return train_test_split(
        frame["text"].astype(str).tolist(), frame["target"].to_numpy(), np.arange(len(frame)),
        test_size=0.2, random_state=SPLIT_SEED, stratify=frame["target"].to_numpy(),
    )


def make_features(ngram_range, X_fit_text, X_val_text):
    # 词表和 IDF 只从训练部分学习，避免验证集信息泄漏。
    vectorizer = TfidfVectorizer(max_features=core.MAX_FEATURES, ngram_range=ngram_range)
    X_fit = vectorizer.fit_transform(X_fit_text)
    X_val = vectorizer.transform(X_val_text)
    names = vectorizer.get_feature_names_out()
    kinds = ["bigram" if " " in feature else "unigram" for feature in names]
    features = pd.DataFrame({
        "feature_index": np.arange(len(names)), "feature": names,
        "kind": kinds, "idf": vectorizer.idf_,
    })
    stats = {
        "actual_feature_count": len(names),
        "unigram_count": kinds.count("unigram"), "bigram_count": kinds.count("bigram"),
        "training_nonzero_values": int(X_fit.nnz),
        "vocabulary_sha256": hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
    }
    return X_fit, X_val, features, stats


def cache_expected(feature_count, fit_indices, val_indices):
    # 旧结构实验使用默认单词表示，但没有记录 ngram_range。
    return {
        "seed": TRAINING_SEED, "validation_fraction": 0.2, "max_features": core.MAX_FEATURES,
        "actual_feature_count": feature_count, "hidden_layers": 1,
        "activation": "relu", "solver": "adam", "learning_rate_init": core.LEARNING_RATE,
        "alpha": core.ALPHA, "batch_size": core.BATCH_SIZE, "epochs": core.EPOCHS,
        "training_threads": 2, "early_stopping": False, "sample_weight": None,
        "loss_definition": "epoch-end unweighted cross-entropy without L2 penalty",
        "comparison_model": "final epoch only",
        "train_data_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "train_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype="<i8").tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype="<i8").tobytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }


def find_unigram_cache(expected, y_val, val_indices, class_count):
    # 按时间查找相容记录，避免按准确率挑结果。
    for run_dir in sorted((DATA_DIR / "mlp_results").glob("structure_*"), reverse=True):
        try:
            common = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
            cached = validate_cached_group(
                run_dir / f"hidden{HIDDEN_UNITS}", common, expected,
                HIDDEN_UNITS, y_val, val_indices, class_count,
            )
            print(f"复用单词基线：{run_dir.name}/hidden{HIDDEN_UNITS}")
            return cached
        except (OSError, ValueError, KeyError, StopIteration) as error:
            print(f"不复用 {run_dir.name}：{error}")
    print("未找到相容基线，将训练两组。")
    return None


def compare_errors(y_val, baseline_pred, phrase_pred, val_indices, texts):
    baseline_pred, phrase_pred = np.asarray(baseline_pred), np.asarray(phrase_pred)
    if any(len(values) != len(y_val) for values in (baseline_pred, phrase_pred, val_indices, texts)):
        raise ValueError("两组预测与验证样本数量不一致。")
    old_correct, new_correct = baseline_pred == y_val, phrase_pred == y_val
    corrected, introduced = ~old_correct & new_correct, old_correct & ~new_correct
    assert int((~old_correct).sum() - (~new_correct).sum()) == int(corrected.sum() - introduced.sum())
    cases = pd.DataFrame({
        "validation_index": np.arange(len(y_val)), "source_index": val_indices,
        "true_label": y_val, "unigram_prediction": baseline_pred, "phrase_prediction": phrase_pred,
        "prediction_changed": baseline_pred != phrase_pred, "text": texts,
    })
    return cases.loc[corrected], cases.loc[introduced]


def print_comparison(summaries, train_count, val_count, corrected_count, introduced_count, dropped_words):
    def cell(text, width, right=False):
        text = str(text)
        size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        padding = " " * max(0, width - size)
        return padding + text if right else text + padding

    baseline, phrases = summaries
    print(f"\n========== MLP 文本表示对照：最终第 {core.EPOCHS} 轮 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；划分/训练种子均为42")
    print(f"固定：100 个神经元；学习率={core.LEARNING_RATE:g}；alpha={core.ALPHA:g}")
    print(f"固定：批次={core.BATCH_SIZE}；ReLU、Adam、不加权；特征上限={core.MAX_FEATURES}")
    print("对照：ngram_range=(1,1) 与 (1,2)。")
    print("-" * 48)
    print(cell("指标", 20) + cell("单词", 14, True) + cell("单词+短语", 14, True))
    for label, key, spec in (
        ("实际特征数", "actual_feature_count", "d"),
        ("单词特征数", "unigram_count", "d"),
        ("双词短语数", "bigram_count", "d"),
        ("模型参数量", "parameter_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"),
        ("验证准确率", "val_accuracy", ".2%"),
        ("训练交叉熵", "train_loss", ".4f"),
        ("验证交叉熵", "val_loss", ".4f"),
        ("验证错分数量", "wrong_count", "d"),
    ):
        print(cell(label, 20) + "".join(cell(format(row[key], spec), 14, True) for row in summaries))
    print("-" * 48)
    print(f"加入短语后，原词表有 {dropped_words} 个单词被替换。")
    print(f"修正旧错误 {corrected_count} 条；新增错误 {introduced_count} 条；净少错 {corrected_count-introduced_count:+d} 条。")
    acc_change = 100 * (phrases["val_accuracy"] - baseline["val_accuracy"])
    ce_change = phrases["val_loss"] - baseline["val_loss"]
    print(f"验证准确率变化：{acc_change:+.2f} 个百分点；交叉熵变化：{ce_change:+.4f}。")
    if acc_change > 0 and ce_change < 0:
        print("准确率提高，验证交叉熵下降。")
    elif acc_change > 0:
        print("准确率提高，验证交叉熵未下降。")
    elif acc_change < 0:
        print("加入短语后验证准确率下降。")
    else:
        print("准确率相同，可结合交叉熵和错例分析。")
    print("评估交叉熵不含 L2 项；模型训练仍使用 L2 正则化。")
    print("这里只比较种子42；结果尚未经过多种子复验。", flush=True)


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
    fig.suptitle("同一5000特征预算：单词与单词+双词短语\n实线训练，虚线验证；种子42，100神经元", fontsize=12)
    fig.savefig(output_dir / "ngram_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False):
    validate_core_settings()
    X_fit_text, X_val_text, y_fit, y_val, fit_indices, val_indices = load_split()
    feature_sets = {key: make_features(ngram, X_fit_text, X_val_text) for key, name, ngram in GROUPS}
    expected = cache_expected(feature_sets["unigram"][3]["actual_feature_count"], fit_indices, val_indices)
    cached = find_unigram_cache(expected, y_val, val_indices, len(np.unique(y_fit)))
    print(f"训练 {len(y_fit)} 篇，验证 {len(y_val)} 篇；本次不用官方测试集。")
    for key, name, ngram in GROUPS:
        stats = feature_sets[key][3]
        print(f"{name} {ngram}：单词 {stats['unigram_count']} 个，双词短语 {stats['bigram_count']} 个。")
    print(f"需要新训练 {1 if cached else 2} 组，每组 {core.EPOCHS} 轮。", flush=True)
    if check_only:
        print("检查通过，未训练或生成实验结果。")
        return

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"ngram_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, histories, predictions = [], {}, {}
    with threadpool_limits(limits=2):
        for key, name, ngram in GROUPS:
            X_fit, X_val, features, stats = feature_sets[key]
            group_dir = output_dir / key
            group_dir.mkdir(exist_ok=False)
            reuse = key == "unigram" and cached is not None
            if reuse:
                print("\n复用单词组第100轮结果。", flush=True)
                summary, history = cached["summary"], cached["history"]
                old_predictions = pd.read_csv(cached["source"] / "validation_predictions.csv")
                val_pred = old_predictions["predicted_label"].to_numpy()
                for filename in ("validation_predictions.csv", "wrong_predictions.csv", "classification_report.txt"):
                    shutil.copyfile(cached["source"] / filename, group_dir / filename)
                source = str(cached["source"])
            else:
                print(f"\n---------- 新训练：{name}，ngram_range={ngram} ----------", flush=True)
                model, history, summary, val_pred, val_proba = core.train_one(
                    HIDDEN_UNITS, X_fit, X_val, y_fit, y_val, seed=TRAINING_SEED,
                )
                with redirect_stdout(io.StringIO()):
                    save_validation_analysis(
                        group_dir, X_val_text, y_val, val_pred, val_proba,
                        model.classes_, val_indices, epoch=core.EPOCHS,
                    )
                source = None
                del model
            history.to_csv(group_dir / "history.csv", index=False, encoding="utf-8-sig")
            features.to_csv(group_dir / "features.csv", index=False, encoding="utf-8-sig")
            config = {
                **expected, **stats, "ngram_range": ngram, "split_seed": SPLIT_SEED,
                "model_seed": TRAINING_SEED, "hidden_units": HIDDEN_UNITS,
                "parameter_count": summary["parameter_count"], "reused": reuse, "source_run_dir": source,
                "tfidf_settings": {"analyzer": "word", "lowercase": True, "min_df": 1, "max_df": 1.0,
                                   "stop_words": None, "sublinear_tf": False, "norm": "l2"},
                "current_training_code_sha256": hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
                "source_note": "Old default-unigram run did not store a vocabulary or source hash; its recorded data/config/results were checked. Saved features describe the current training-only vocabulary.",
                "timing_note": "A reused baseline's training time is from an older run, not a controlled runtime benchmark.",
            }
            (group_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            summaries.append({**summary, **stats, "group": key, "ngram_range": str(ngram), "reused": reuse, "source_run_dir": source})
            histories[name], predictions[key] = history, val_pred

    corrected, introduced = compare_errors(
        y_val, predictions["unigram"], predictions["unigram_bigram"], val_indices, X_val_text,
    )
    corrected.to_csv(output_dir / "corrected_errors.csv", index=False, encoding="utf-8-sig")
    introduced.to_csv(output_dir / "introduced_errors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    old_words = set(feature_sets["unigram"][2]["feature"])
    new_words = set(feature_sets["unigram_bigram"][2].loc[feature_sets["unigram_bigram"][2]["kind"] == "unigram", "feature"])
    dropped_words = len(old_words - new_words)
    (output_dir / "experiment_config.json").write_text(json.dumps({
        "common_settings": expected, "hidden_units": HIDDEN_UNITS, "training_seed": TRAINING_SEED,
        "summaries": summaries, "corrected_count": len(corrected), "introduced_count": len(introduced),
        "dropped_unigram_features": dropped_words,
        "scope": "Single-seed feature representation screening on a fixed validation split, not a generalization guarantee.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print_comparison(summaries, len(y_fit), len(y_val), len(corrected), len(introduced), dropped_words)
    plot_comparison(output_dir, histories)
    print("日志、词表、错例和曲线保存于：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较 MLP 的单词与双词短语特征。")
    parser.add_argument("--check-only", action="store_true", help="检查数据、词表和旧基线，不训练。")
    main(check_only=parser.parse_args().check_only)
