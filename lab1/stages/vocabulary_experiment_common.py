"""MLP 和 SVM 共用的数据准备、结果记录与排版。"""

import hashlib
import json
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


FEATURE_LIMITS = (5000, 10000)
SPLIT_SEED = 42


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_data(data_dir):
    data_dir = Path(data_dir)
    frame = pd.read_csv(data_dir / "train_data.csv")
    if not {"text", "target"}.issubset(frame.columns) or frame["target"].isna().any():
        raise ValueError("训练CSV必须包含text和无缺失的target。")
    labels = frame["target"].to_numpy()
    fit_texts, val_texts, y_fit, y_val, fit_indices, val_indices = train_test_split(
        frame["text"].astype(str).tolist(), labels, np.arange(len(frame)),
        test_size=0.2, random_state=SPLIT_SEED, stratify=labels,
    )
    sets = {}
    for limit in FEATURE_LIMITS:
        # 词表和 IDF 不看验证文本，避免泄漏验证信息。
        vectorizer = TfidfVectorizer(max_features=limit, ngram_range=(1, 1), sublinear_tf=False)
        X_fit, X_val = vectorizer.fit_transform(fit_texts), vectorizer.transform(val_texts)
        names = vectorizer.get_feature_names_out()
        features = pd.DataFrame({"feature_index": np.arange(len(names)), "feature": names, "idf": vectorizer.idf_})
        parameters = vectorizer.get_params()
        parameters["dtype"] = np.dtype(parameters["dtype"]).name
        sets[limit] = {
            "vectorizer": vectorizer, "X_fit": X_fit, "X_val": X_val, "features": features,
            "stats": {
                "max_features": limit, "actual_feature_count": int(X_fit.shape[1]),
                "vocabulary_sha256": hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest(),
                "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
                "tfidf_parameters": parameters,
            },
        }
    metadata = {
        "split_seed": SPLIT_SEED, "validation_fraction": 0.2,
        "train_count": len(y_fit), "validation_count": len(y_val),
        "train_data_sha256": sha256(data_dir / "train_data.csv"),
        "train_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype="<i8").tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype="<i8").tobytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__, "sklearn_version": sklearn.__version__,
        "feature_limits": list(FEATURE_LIMITS), "official_test_used": False, "sample_weight": None,
        "source_index_definition": "0-based record in loaded train_data.csv, not a physical file line",
        "helper_sha256": sha256(__file__),
    }
    print(f"训练 {len(y_fit)} 篇；验证 {len(y_val)} 篇；划分种子42。")
    for limit, group in sets.items():
        print(f"词表上限{limit}：实际 {group['X_fit'].shape[1]} 个特征。")
    return {"fit_texts": fit_texts, "val_texts": val_texts, "y_fit": y_fit, "y_val": y_val,
            "fit_indices": fit_indices, "val_indices": val_indices, "sets": sets, "metadata": metadata}


def validate_prediction_files(group_dir, data):
    """核对样本顺序和文本，防止拿错旧结果。"""
    records = pd.read_csv(Path(group_dir) / "validation_predictions.csv", keep_default_na=False)
    for key, expected in (("validation_index", np.arange(len(data["y_val"]))),
                          ("source_index", data["val_indices"]), ("true_label", data["y_val"]),
                          ("text", data["val_texts"])):
        if not np.array_equal(records[key].to_numpy(), expected):
            raise ValueError(f"旧预测记录的{key}不一致")
    pred = records["predicted_label"].to_numpy()
    if not np.isin(pred, np.unique(data["y_fit"])).all():
        raise ValueError("旧预测中包含未知类别")
    correct = pred == data["y_val"]
    if not np.array_equal(records["is_correct"].to_numpy(), correct):
        raise ValueError("旧预测的正确标记不一致")
    errors = pd.read_csv(Path(group_dir) / "wrong_predictions.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(
        errors.sort_values("validation_index").reset_index(drop=True),
        records.loc[~correct].sort_values("validation_index").reset_index(drop=True), check_dtype=False,
    )
    return records


def new_output_dir(data_dir, model_name):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(data_dir) / f"{model_name}_results" / f"vocabulary_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print("结果目录：")
    print(output_dir, flush=True)
    return output_dir


def finish_comparison(output_dir, data, summaries, predictions, config):
    """保存两组错例变化，分清修正和新增的错误。"""
    truth = data["y_val"]
    for row in summaries:
        pred = np.asarray(predictions[row["max_features"]])
        if pred.shape != truth.shape or not np.isin(pred, np.unique(data["y_fit"])).all():
            raise ValueError("验证预测形状或类别无效")
        if int(np.sum(pred != truth)) != row["wrong_count"] or not np.isclose(np.mean(pred == truth), row["val_accuracy"], rtol=0, atol=1e-12):
            raise ValueError("汇总准确率或错分数量与预测不一致")
    old_correct, new_correct = predictions[5000] == truth, predictions[10000] == truth
    records = pd.DataFrame({
        "validation_index": np.arange(len(truth)), "source_index": data["val_indices"], "true_label": truth,
        "prediction_5000": predictions[5000], "prediction_10000": predictions[10000], "text": data["val_texts"],
    })
    corrected, introduced = records.loc[~old_correct & new_correct], records.loc[old_correct & ~new_correct]
    corrected.to_csv(output_dir / "corrected_errors.csv", index=False, encoding="utf-8-sig")
    introduced.to_csv(output_dir / "introduced_errors.csv", index=False, encoding="utf-8-sig")
    old_words = set(data["sets"][5000]["features"]["feature"])
    large_features = data["sets"][10000]["features"]
    large_features.loc[~large_features["feature"].isin(old_words)].to_csv(output_dir / "added_features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "experiment_config.json", {
        **config, "status": "completed", "summaries": summaries,
        "corrected_count": len(corrected), "introduced_count": len(introduced),
    })
    return len(corrected), len(introduced)


def print_table(title, summaries, metrics, corrected_count, introduced_count):
    def cell(value, width, right=False):
        value = str(value)
        size = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in value)
        padding = " " * max(0, width - size)
        return padding + value if right else value + padding

    print(f"\n========== {title}：词表容量对照 ==========")
    print("-" * 54)
    print(cell("指标", 22) + "".join(cell(f"上限{row['max_features']}", 16, True) for row in summaries))
    for label, key, spec in metrics:
        print(cell(label, 22) + "".join(cell(format(row[key], spec), 16, True) for row in summaries))
    print(cell("结果来源", 22) + "".join(cell("复用旧组" if row["reused"] else "本次训练", 16, True) for row in summaries))
    print("-" * 54)
    old, new = summaries
    print(f"5000→10000：验证准确率变化 {100 * (new['val_accuracy'] - old['val_accuracy']):+.2f} 个百分点。")
    print(f"修正旧错误 {corrected_count} 条；新增错误 {introduced_count} 条；净少错 {corrected_count-introduced_count:+d} 条。")
    print("扩大词表也会改变 L2 归一化后的数值；不一定提高准确率。")
    if any(row["reused"] for row in summaries):
        print("复用组的耗时来自旧运行，不能严格比较速度。")


def plot_svm(output_dir, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x, width = np.arange(2), 0.34
    axes[0].bar(x - width/2, [r["train_accuracy"] for r in summaries], width, label="训练")
    axes[0].bar(x + width/2, [r["val_accuracy"] for r in summaries], width, label="验证")
    axes[0].set(title="分类准确率", ylabel="准确率", ylim=(0, 1.05))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    axes[1].bar(x, [r["wrong_count"] for r in summaries], color="#2166AC")
    axes[1].set(title="验证错分数量", ylabel="错分文本数量")
    for position, row in zip(x, summaries):
        axes[1].annotate(str(row["wrong_count"]), (position, row["wrong_count"]), xytext=(0, 5), textcoords="offset points", ha="center")
    axes[1].margins(y=0.15)
    for ax in axes:
        ax.set_xticks(x, [str(row["max_features"]) for row in summaries])
        ax.set_xlabel("词表上限（不是训练轮数）")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("线性SVM C=100：只改变TF-IDF词表容量")
    fig.savefig(output_dir / "vocabulary_comparison.png", dpi=220)
    plt.close(fig)
