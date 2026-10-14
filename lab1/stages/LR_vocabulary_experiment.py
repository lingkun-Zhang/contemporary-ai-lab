"""扩大词表能否改善逻辑回归的验证表现？

固定 C=100，比较单词 TF-IDF 上限 5000 和 10000，两组从头训练。
--check-only 只检查数据和特征，不训练。
"""

import argparse
import hashlib
import json
import unicodedata
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


DATA_DIR = Path(__file__).resolve().parent
FEATURE_LIMITS = (5000, 10000)  # 只改词表上限，模型设置保持相同。
SEED = 42
C = 100.0
MAX_ITER = 1000
TOL = 0.0001


def load_split():
    # 按脚本位置找数据，避免打开外层文件夹时找不到 CSV。
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    if not {"text", "target"}.issubset(frame.columns) or frame["target"].isna().any():
        raise ValueError("训练CSV必须包含text和无缺失的target列。")
    labels = frame["target"].to_numpy()
    return train_test_split(
        frame["text"].astype(str).tolist(), labels, np.arange(len(frame)),
        test_size=0.2, random_state=SEED, stratify=labels,
    )


def make_features(limit, fit_texts, val_texts):
    # 验证文本沿用训练部分的词表和 IDF。
    vectorizer = TfidfVectorizer(max_features=limit, ngram_range=(1, 1), sublinear_tf=False)
    X_fit = vectorizer.fit_transform(fit_texts)
    X_val = vectorizer.transform(val_texts)
    names = vectorizer.get_feature_names_out()
    features = pd.DataFrame({"feature_index": np.arange(len(names)), "feature": names, "idf": vectorizer.idf_})
    return X_fit, X_val, features


def train_one(limit, X_fit, X_val, y_fit, y_val):
    # 沿用未加权的 LR 设置；max_iter 是优化器迭代上限，不是 epoch。
    model = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER,
        tol=TOL, random_state=SEED, class_weight=None,
    )
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        model.fit(X_fit, y_fit)  # 两组都不加权。
        training_seconds = perf_counter() - started
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    train_pred, val_pred = model.predict(X_fit), model.predict(X_val)
    train_proba, val_proba = model.predict_proba(X_fit), model.predict_proba(X_val)
    for proba in (train_proba, val_proba):
        if not np.isfinite(proba).all() or (proba < 0).any() or not np.allclose(proba.sum(axis=1), 1):
            raise ValueError("预测概率异常，不能作为有效实验结果。")
    train_acc, val_acc = accuracy_score(y_fit, train_pred), accuracy_score(y_val, val_pred)
    summary = {
        "max_features": limit, "actual_feature_count": int(X_fit.shape[1]),
        "parameter_count": int(model.coef_.size + model.intercept_.size),
        "train_accuracy": float(train_acc), "val_accuracy": float(val_acc),
        "train_loss": float(log_loss(y_fit, train_proba, labels=model.classes_)),
        "val_loss": float(log_loss(y_val, val_proba, labels=model.classes_)),
        "accuracy_gap": float(train_acc - val_acc),
        "wrong_count": int(np.sum(val_pred != y_val)), "training_seconds": float(training_seconds),
        "solver_iterations": int(np.max(model.n_iter_)), "converged": converged,
    }
    warning_messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    return model, summary, val_pred, val_proba, warning_messages


def choose_best(summaries):
    # 只在已收敛的组中比较效果，避免把优化没完成当成模型差异。
    candidates = [row for row in summaries if row["converged"]]
    return min(candidates, key=lambda row: (-row["val_accuracy"], row["val_loss"], row["actual_feature_count"], row["max_features"])) if candidates else None


def save_validation_analysis(group_dir, texts, y_val, pred, proba, classes, indices):
    # 概率列按 classes_ 排列，标签不连续时也要正确对应。
    column_by_label = {label: i for i, label in enumerate(classes)}
    pred_columns = [column_by_label[label] for label in pred]
    true_columns = [column_by_label[label] for label in y_val]
    records = pd.DataFrame({
        "validation_index": np.arange(len(y_val)), "source_index": indices,
        "true_label": y_val, "predicted_label": pred, "is_correct": pred == y_val,
        "predicted_probability": proba[np.arange(len(y_val)), pred_columns],
        "true_probability": proba[np.arange(len(y_val)), true_columns], "text": texts,
    })
    records.to_csv(group_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    records.loc[~records["is_correct"]].to_csv(group_dir / "wrong_predictions.csv", index=False, encoding="utf-8-sig")
    report = classification_report(y_val, pred, labels=classes, digits=4, zero_division=0)
    (group_dir / "classification_report.txt").write_text(report, encoding="utf-8")


def compare_errors(y_val, old_pred, new_pred, indices, texts):
    old_correct, new_correct = old_pred == y_val, new_pred == y_val
    records = pd.DataFrame({
        "validation_index": np.arange(len(y_val)), "source_index": indices, "true_label": y_val,
        "prediction_5000": old_pred, "prediction_10000": new_pred, "text": texts,
    })
    return records.loc[~old_correct & new_correct], records.loc[old_correct & ~new_correct]


def print_comparison(summaries, train_count, val_count, corrected_count, introduced_count, added_count, dropped_count):
    def cell(text, width, right=False):
        text = str(text)
        size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        padding = " " * max(0, width - size)
        return padding + text if right else text + padding

    print("\n========== 逻辑回归：词表容量对照 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；划分种子={SEED}")
    print(f"固定：C={C:g}、L2、lbfgs、max_iter={MAX_ITER}；单词、普通词频、不加权")
    print("-" * 54)
    print(cell("指标", 22) + "".join(cell(f"上限{row['max_features']}", 16, True) for row in summaries))
    for label, key, spec in (
        ("实际单词特征数", "actual_feature_count", "d"),
        ("模型参数量", "parameter_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"),
        ("验证准确率", "val_accuracy", ".2%"),
        ("训练交叉熵", "train_loss", ".4f"),
        ("验证交叉熵", "val_loss", ".4f"),
        ("训练-验证差距", "accuracy_gap", ".2%"),
        ("验证错分数量", "wrong_count", "d"),
        ("优化器迭代次数", "solver_iterations", "d"),
        ("训练耗时/秒", "training_seconds", ".2f"),
    ):
        print(cell(label, 22) + "".join(cell(format(row[key], spec), 16, True) for row in summaries))
    print(cell("收敛警告检查", 22) + "".join(cell("通过" if row["converged"] else "未收敛", 16, True) for row in summaries))
    print("-" * 54)
    old, new = summaries
    print(f"5000→10000：验证准确率变化 {100 * (new['val_accuracy'] - old['val_accuracy']):+.2f} 个百分点；"
          f"交叉熵变化 {new['val_loss'] - old['val_loss']:+.4f}。")
    print(f"修正旧错误 {corrected_count} 条；新增错误 {introduced_count} 条；净少错 {corrected_count-introduced_count:+d} 条。")
    print(f"相比小词表，新增词元 {added_count} 个，旧词未保留 {dropped_count} 个。")
    print("词表扩大后，文本归一化也会变化，旧特征的数值不一定相同。")
    best = choose_best(summaries)
    if best is None:
        print("两组都未收敛，暂不选候选模型。")
    else:
        print(f"本次候选：词表上限{best['max_features']}；验证准确率={best['val_accuracy']:.2%}。")
        print("选择顺序：收敛后比较验证准确率，同分看交叉熵和特征数。")
    if any(not row["converged"] for row in summaries):
        print("有未收敛组，比较结论需谨慎；具体警告已保存。")
    print("训练用了 L2，表中损失只统计交叉熵；耗时只计 fit。")
    print("以上是固定验证划分上的结果，未使用官方测试集。", flush=True)


def plot_comparison(output_dir, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x, width = np.arange(len(summaries)), 0.34
    for offset, phase, name, color in ((-width / 2, "train", "训练", "#2166AC"), (width / 2, "val", "验证", "#D6604D")):
        axes[0].bar(x + offset, [row[f"{phase}_accuracy"] for row in summaries], width, label=name, color=color)
        axes[1].bar(x + offset, [row[f"{phase}_loss"] for row in summaries], width, label=name, color=color)
    axes[0].set(title="分类准确率", ylabel="准确率", ylim=(0, 1.05))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(title="训练与验证交叉熵", ylabel="交叉熵")
    for ax in axes:
        ax.set_xticks(x, [str(row["max_features"]) for row in summaries])
        ax.set_xlabel("词表上限")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle("逻辑回归C=100：仅改变TF-IDF词表容量", fontsize=12)
    fig.savefig(output_dir / "vocabulary_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False):
    if FEATURE_LIMITS != (5000, 10000):
        raise ValueError("本次只比较 FEATURE_LIMITS=(5000, 10000)。")
    fit_texts, val_texts, y_fit, y_val, fit_indices, val_indices = load_split()
    feature_sets = {limit: make_features(limit, fit_texts, val_texts) for limit in FEATURE_LIMITS}
    old_features, new_features = feature_sets[5000][2], feature_sets[10000][2]
    old_words, new_words = set(old_features["feature"]), set(new_features["feature"])
    added, dropped = new_words - old_words, old_words - new_words
    print(f"训练 {len(y_fit)} 篇；验证 {len(y_val)} 篇；固定C={C:g}，不加权。")
    for limit in FEATURE_LIMITS:
        print(f"词表上限{limit}：实际特征数={feature_sets[limit][0].shape[1]}。")
    print(f"词表只在训练部分拟合。新增词元={len(added)}，旧词未保留={len(dropped)}。", flush=True)
    if check_only:
        print("检查通过，未训练或保存新结果。")
        return

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "lr_results" / f"vocabulary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print("结果将保存到：")
    print(output_dir)
    common_config = {
        "model": "LogisticRegression", "C": C, "penalty": "l2", "solver": "lbfgs",
        "max_iter": MAX_ITER, "tol": TOL, "random_state": SEED, "class_weight": None,
        "sample_weight": None, "training_threads": 2, "split_seed": SEED, "validation_fraction": 0.2,
        "train_count": len(y_fit), "validation_count": len(y_val),
        "tfidf": {"analyzer": "word", "ngram_range": [1, 1], "sublinear_tf": False, "norm": "l2",
                  "lowercase": True, "min_df": 1, "max_df": 1.0, "stop_words": None,
                  "use_idf": True, "smooth_idf": True, "binary": False, "strip_accents": None,
                  "token_pattern": r"(?u)\b\w\w+\b", "dtype": "float64"},
        "train_data_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "train_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype="<i8").tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype="<i8").tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "selection_rule": "converged groups only: validation accuracy, then cross-entropy, then fewer features",
        "cross_entropy_definition": "unweighted predictive cross-entropy without L2 penalty",
        "scope": "two fresh fits; train-only vocabulary/IDF; no official test, augmentation or weighting",
        "source_index_definition": "0-based data row of loaded train_data.csv, not physical file line",
    }
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps({**common_config, "status": "running"}, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries, predictions = [], {}
    for number, limit in enumerate(FEATURE_LIMITS, 1):
        print(f"\n---------- [{number}/2] 开始训练：词表上限{limit} ----------", flush=True)
        X_fit, X_val, features = feature_sets[limit]
        model, summary, pred, proba, warning_messages = train_one(limit, X_fit, X_val, y_fit, y_val)
        print(f"完成：验证准确率={summary['val_accuracy']:.2%}；交叉熵={summary['val_loss']:.4f}；"
              f"错分={summary['wrong_count']}；收敛警告检查={'通过' if summary['converged'] else '未收敛'}。", flush=True)
        group_dir = output_dir / f"features{limit}"
        group_dir.mkdir(exist_ok=False)
        features.to_csv(group_dir / "features.csv", index=False, encoding="utf-8-sig")
        save_validation_analysis(group_dir, val_texts, y_val, pred, proba, model.classes_, val_indices)
        group_config = {
            **common_config, "summary": summary, "model_parameters": model.get_params(),
            "classes": model.classes_.tolist(), "warning_messages": warning_messages,
            "vocabulary_sha256": hashlib.sha256("\0".join(features["feature"]).encode("utf-8")).hexdigest(),
            "idf_sha256": hashlib.sha256(features["idf"].to_numpy(dtype="<f8").tobytes()).hexdigest(),
        }
        (group_dir / "config.json").write_text(json.dumps(group_config, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(summary)
        predictions[limit] = pred
        pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
        del model
    corrected, introduced = compare_errors(y_val, predictions[5000], predictions[10000], val_indices, val_texts)
    corrected.to_csv(output_dir / "corrected_errors.csv", index=False, encoding="utf-8-sig")
    introduced.to_csv(output_dir / "introduced_errors.csv", index=False, encoding="utf-8-sig")
    new_features.loc[new_features["feature"].isin(added)].to_csv(output_dir / "added_features.csv", index=False, encoding="utf-8-sig")
    old_features.loc[old_features["feature"].isin(dropped)].to_csv(output_dir / "dropped_features.csv", index=False, encoding="utf-8-sig")
    best = choose_best(summaries)
    config_path.write_text(json.dumps({
        **common_config, "status": "completed" if all(row["converged"] for row in summaries) else "completed_with_convergence_warnings",
        "feature_limits": list(FEATURE_LIMITS), "summaries": summaries,
        "selected_max_features": best["max_features"] if best else None,
        "corrected_count": len(corrected), "introduced_count": len(introduced),
        "added_features": len(added), "dropped_features": len(dropped),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print_comparison(summaries, len(y_fit), len(y_val), len(corrected), len(introduced), len(added), len(dropped))
    try:
        plot_comparison(output_dir, summaries)
        print("图像已保存：vocabulary_comparison.png")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"图片未生成，但数据与对照结果已保存：{error}")
    print("汇总、词表、完整错例和图片目录：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定逻辑回归C100，比较5000与10000单词词表。")
    parser.add_argument("--check-only", action="store_true", help="只检查数据/特征，不训练、不保存结果。")
    main(check_only=parser.parse_args().check_only)
