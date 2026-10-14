"""线性 SVM 的 C 参数对照，使用与 LR/MLP 相同的数据划分。

固定 5000 维单词 TF-IDF，比较 C=0.1、1、10、100。
使用 SVC(kernel="linear")，不启用概率校准；相容的旧结果可复用。
"""

import argparse
import hashlib
import json
import shutil
import unicodedata
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# C=100 用于检查原搜索范围的上边界。
DATA_DIR = Path(__file__).resolve().parent
SPLIT_SEED = 42
VALIDATION_FRACTION = 0.2
MAX_FEATURES = 5000
C_VALUES = (0.1, 1.0, 10.0, 100.0)
TOL = 0.001
CACHE_SIZE_MB = 512
# 兼容的历史脚本版本。
LEGACY_SCRIPT_SHA256 = "a8d1faf7c173826d1db6f27544d0e66e22f47afacd18fb080f77bfadc49b652f"
# 仅注释不同的版本。
COMMENT_CLEANUP_BASE_SHA256 = "e141127a08def3d4d9008aa682454695a8f24d03cbeb6af230fd812906401293"
CACHE_MATCH_KEYS = (
    "split_seed", "validation_fraction", "train_count", "validation_count",
    "max_features", "actual_feature_count", "tfidf_settings", "train_data_sha256",
    "train_indices_sha256", "val_indices_sha256", "vocabulary_sha256", "idf_sha256",
    "numpy_version", "pandas_version", "sklearn_version", "model", "kernel",
    "official_test_used", "probabilities_or_cross_entropy_computed",
)
SUMMARY_KEYS = (
    "C", "train_accuracy", "val_accuracy", "accuracy_gap", "wrong_count",
    "support_vector_count", "support_vector_fraction", "training_seconds", "fit_status",
)


def load_split():
    # 相对脚本定位数据，避免工作目录变化。
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    if not {"text", "target"}.issubset(frame.columns):
        raise ValueError("train_data.csv 必须包含 text 和 target 两列。")
    if frame["target"].isna().any():
        raise ValueError("训练标签存在缺失值，请先检查数据。")
    texts, labels = frame["text"].astype(str).tolist(), frame["target"].to_numpy()
    # 与已有实验使用相同分层划分，保留原索引以核验缓存。
    return train_test_split(
        texts, labels, np.arange(len(frame)), test_size=VALIDATION_FRACTION,
        random_state=SPLIT_SEED, stratify=labels,
    )


def make_features(fit_texts, val_texts):
    # 词表和 IDF 仅拟合训练部分，各组 C 共用特征。
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES, ngram_range=(1, 1), sublinear_tf=False,
    )
    X_fit = vectorizer.fit_transform(fit_texts)
    X_val = vectorizer.transform(val_texts)
    return vectorizer, X_fit, X_val


def make_model(c_value):
    # C 越大，间隔违约惩罚越重，正则化相对越弱。
    # 按 tol 收敛，不截断求解器迭代；迭代数不等同于 MLP 训练轮次。
    # 不做概率校准时，SVC 的 random_state 不影响训练。
    return SVC(
        kernel="linear", C=c_value, probability=False, class_weight=None,
        tol=TOL, max_iter=-1, cache_size=CACHE_SIZE_MB,
        decision_function_shape="ovr", break_ties=False, random_state=SPLIT_SEED,
    )


def train_one(c_value, X_fit, X_val, y_fit, y_val):
    model = make_model(c_value)  # 各参数组独立训练。
    started = perf_counter()
    model.fit(X_fit, y_fit)
    training_seconds = perf_counter() - started
    if model.fit_status_ != 0:
        raise RuntimeError(f"C={c_value:g} 未正常收敛，本组不能作为最终候选。")
    # 使用 SVC 的多分类预测规则；决策分数不是概率。
    train_pred, val_pred = model.predict(X_fit), model.predict(X_val)
    train_accuracy, val_accuracy = accuracy_score(y_fit, train_pred), accuracy_score(y_val, val_pred)
    summary = {
        "C": float(c_value), "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy), "accuracy_gap": float(train_accuracy - val_accuracy),
        "wrong_count": int(np.sum(np.asarray(val_pred) != y_val)),
        "support_vector_count": int(len(model.support_)),
        "support_vector_fraction": float(len(model.support_) / len(y_fit)),
        "training_seconds": float(training_seconds), "fit_status": int(model.fit_status_),
    }
    return model, summary, val_pred


def choose_best(summaries):
    # 按验证准确率选择，同分取较小 C。
    return min(summaries, key=lambda row: (-row["val_accuracy"], row["C"]))


def validate_cached_group(group_dir, expected, c_value, outer_summary, y_fit, y_val, val_indices, val_texts):
    """核验配置与完整预测记录，不加载旧模型。"""
    config = json.loads((group_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("旧组配置不是JSON对象")
    allowed_hashes = {LEGACY_SCRIPT_SHA256, COMMENT_CLEANUP_BASE_SHA256, expected["script_sha256"]}
    if config.get("script_sha256") not in allowed_hashes:
        raise ValueError("旧组源码版本未经核验")
    if any(config.get(key) != expected[key] for key in CACHE_MATCH_KEYS):
        raise ValueError("旧组的数据、特征或环境配置不同")
    if config.get("svc_parameters") != make_model(c_value).get_params():
        raise ValueError("旧组SVC参数不同")
    summary = {key: config["summary"][key] for key in SUMMARY_KEYS}
    if any(outer_summary[key] != summary[key] for key in SUMMARY_KEYS):
        raise ValueError("组配置与汇总记录不一致")
    if any(not np.isfinite(value) for value in summary.values()):
        raise ValueError("旧指标不是有限数值")
    if summary["C"] != c_value or summary["fit_status"] != 0:
        raise ValueError("C不同或旧模型未收敛")
    for key in ("train_accuracy", "val_accuracy", "support_vector_fraction"):
        if not 0 <= summary[key] <= 1:
            raise ValueError("旧比例超出范围")
    for key, upper in (("wrong_count", len(y_val)), ("support_vector_count", len(y_fit))):
        if type(summary[key]) is not int or not 0 <= summary[key] <= upper:
            raise ValueError("旧样本计数不合法")
    if summary["support_vector_count"] < 1 or summary["training_seconds"] < 0:
        raise ValueError("旧支持向量计数或耗时不合法")
    if not np.isclose(summary["accuracy_gap"], summary["train_accuracy"] - summary["val_accuracy"], rtol=0, atol=1e-12):
        raise ValueError("旧训练/验证差距不一致")
    if not np.isclose(summary["support_vector_fraction"], summary["support_vector_count"] / len(y_fit), rtol=0, atol=1e-12):
        raise ValueError("旧支持向量比例不一致")
    classes = np.unique(y_fit)
    counts = config["n_support_by_class"]
    iterations = config["solver_iterations_by_binary_problem"]
    if config["classes"] != classes.tolist() or len(counts) != len(classes):
        raise ValueError("旧类别顺序不同")
    if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != summary["support_vector_count"]:
        raise ValueError("旧逐类支持向量合计不同")
    if len(iterations) != len(classes) * (len(classes) - 1) // 2 or any(type(n) is not int or n < 0 for n in iterations):
        raise ValueError("旧求解器记录不合法")
    records = pd.read_csv(group_dir / "validation_predictions.csv", keep_default_na=False)
    required = ["validation_index", "source_index", "true_label", "predicted_label", "is_correct", "text"]
    if records.columns.tolist() != required or len(records) != len(y_val):
        raise ValueError("旧预测表字段或样本数量不同")
    for key, values in (("validation_index", np.arange(len(y_val))), ("source_index", val_indices),
                        ("true_label", y_val), ("text", val_texts)):
        if not np.array_equal(records[key].to_numpy(), values):
            raise ValueError(f"旧预测表的{key}与当前验证集不同")
    predictions = records["predicted_label"].to_numpy()
    if not np.isin(predictions, classes).all():
        raise ValueError("旧预测包含未知类别")
    correct = predictions == y_val
    if not np.array_equal(records["is_correct"].to_numpy(), correct):
        raise ValueError("旧预测正确标记不一致")
    if int((~correct).sum()) != summary["wrong_count"] or not np.isclose(correct.mean(), summary["val_accuracy"], rtol=0, atol=1e-12):
        raise ValueError("旧验证准确率/错分数量与预测不一致")
    wrong = pd.read_csv(group_dir / "wrong_predictions.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(wrong, records.loc[~correct].reset_index(drop=True), check_dtype=False)
    return {"source": group_dir, "summary": summary, "predictions": predictions}


def find_cached_groups(expected, y_fit, y_val, val_indices, val_texts):
    # 按时间查找完整相容的运行，不按成绩挑缓存或跨运行拼组。
    for run_dir in sorted((DATA_DIR / "svm_results").glob("linear_C_*"), reverse=True):
        try:
            common = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
            if not isinstance(common, dict):
                raise ValueError("旧实验配置不是JSON对象")
            if common.get("status") != "completed":
                continue
            if common.get("script_sha256") not in {LEGACY_SCRIPT_SHA256, COMMENT_CLEANUP_BASE_SHA256, expected["script_sha256"]}:
                raise ValueError("源码版本不同")
            if any(common.get(key) != expected[key] for key in CACHE_MATCH_KEYS):
                raise ValueError("数据、划分、特征或环境已改变")
            values = common["C_values"]
            if not values or len(set(values)) != len(values) or not set(values).issubset(C_VALUES):
                raise ValueError("旧C列表不相容")
            summaries = common["summaries"]
            if len(summaries) != len(values) or {row["C"] for row in summaries} != set(values):
                raise ValueError("旧实验汇总不完整")
            cached = {}
            for c in values:
                row = next(row for row in summaries if row["C"] == c)
                cached[c] = validate_cached_group(run_dir / f"C{c:g}", expected, c, row, y_fit, y_val, val_indices, val_texts)
            print(f"读取旧结果：{run_dir.name}；C={values}", flush=True)
            return cached
        except (OSError, ValueError, KeyError, TypeError, AssertionError, StopIteration) as error:
            print(f"不复用 {run_dir.name}：{error}", flush=True)
    print("无相容结果，需重新训练全部组。", flush=True)
    return {}


def save_validation_analysis(group_dir, texts, y_val, val_pred, source_indices, classes):
    predictions = pd.DataFrame({
        "validation_index": np.arange(len(y_val)), "source_index": source_indices,
        "true_label": y_val, "predicted_label": val_pred,
        "is_correct": np.asarray(val_pred) == y_val, "text": texts,
    })
    predictions.to_csv(group_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.loc[~predictions["is_correct"]].to_csv(
        group_dir / "wrong_predictions.csv", index=False, encoding="utf-8-sig",
    )
    # 逐类指标保存到文件，终端表以准确率为主。
    report = classification_report(y_val, val_pred, labels=classes, digits=4, zero_division=0)
    (group_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = pd.DataFrame(confusion_matrix(y_val, val_pred, labels=classes), index=classes, columns=classes)
    matrix.index.name = "true_label / predicted_label"
    matrix.to_csv(group_dir / "confusion_matrix.csv", encoding="utf-8-sig")


def print_comparison(summaries, train_count, val_count, feature_count):
    def cell(text, width, right=False):
        text = str(text)
        size = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
        padding = " " * max(0, width - size)
        return padding + text if right else text + padding

    print("\n========== 线性 SVM：C 参数对照 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；实际特征数={feature_count}")
    print(f"固定：单词 TF-IDF；普通词频；线性核；不加权；划分种子={SPLIT_SEED}")
    print("-" * 72)
    print(cell("指标", 24) + "".join(cell(f"C={row['C']:g}", 12, True) for row in summaries))
    for label, key, spec in (
        ("训练准确率", "train_accuracy", ".2%"),
        ("验证准确率", "val_accuracy", ".2%"),
        ("训练-验证差距", "accuracy_gap", ".2%"),
        ("验证错分数量", "wrong_count", "d"),
        ("支持向量数量", "support_vector_count", "d"),
        ("支持向量占训练比例", "support_vector_fraction", ".2%"),
        ("训练耗时/秒", "training_seconds", ".2f"),
    ):
        print(cell(label, 24) + "".join(cell(format(row[key], spec), 12, True) for row in summaries))
    print(cell("结果来源", 24) + "".join(cell("复用旧组" if row.get("reused") else "本次训练", 12, True) for row in summaries))
    print("-" * 72)
    best = choose_best(summaries)
    print(f"本次候选：C={best['C']:g}；验证准确率={best['val_accuracy']:.2%}；错分={best['wrong_count']}条。")
    print(f"按验证准确率选择，同分取小C；范围为本次划分的{len(summaries)}组。")
    by_c = {row["C"]: row for row in summaries}
    if 10.0 in by_c and 100.0 in by_c:
        old, new = by_c[10.0], by_c[100.0]
        print(f"C=10→100：验证准确率变化 {100 * (new['val_accuracy'] - old['val_accuracy']):+.2f} 个百分点；"
              f"净少错 {old['wrong_count'] - new['wrong_count']:+d} 条。")
    print("支持向量数是样本数，不能等同于模型参数量。")
    print("耗时仅计 fit。")
    if any(row.get("reused") for row in summaries):
        print("缓存耗时来自旧运行，不作严格速度比较。")
    print("SVC无逐轮损失记录，未校准概率，不计算交叉熵。")
    print("图的横轴是C；无标签测试集未参与调参。", flush=True)


def plot_comparison(output_dir, summaries):
    import matplotlib
    matplotlib.use("Agg")  # 直接保存图片。
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    positions = np.arange(len(summaries))
    axes[0].plot(positions, [row["train_accuracy"] for row in summaries], "o-", label="训练准确率")
    axes[0].plot(positions, [row["val_accuracy"] for row in summaries], "s--", label="验证准确率")
    axes[0].set(ylabel="准确率", title="准确率与正则化参数")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend()
    axes[1].bar(positions, [row["wrong_count"] for row in summaries], color="#2166AC")
    axes[1].set(ylabel="错分文本数量", title="验证错分数量（越少越好）")
    for pos, row in zip(positions, summaries):
        axes[1].annotate(str(row["wrong_count"]), (pos, row["wrong_count"]),
                         xytext=(0, 5), textcoords="offset points", ha="center")
    axes[1].margins(y=0.15)
    for ax in axes:
        ax.set_xticks(positions, [f"{row['C']:g}" for row in summaries])
        ax.set_xlabel("C（离散参数组，不是训练轮数）")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("同一训练/验证划分与5000维单词TF-IDF：线性SVM", fontsize=12)
    fig.savefig(output_dir / "svm_comparison.png", dpi=220)
    plt.close(fig)


def make_config(vectorizer, X_fit, y_fit, y_val, fit_indices, val_indices):
    names = vectorizer.get_feature_names_out()
    return {
        "split_seed": SPLIT_SEED, "validation_fraction": VALIDATION_FRACTION,
        "train_count": len(y_fit), "validation_count": len(y_val),
        "max_features": MAX_FEATURES, "actual_feature_count": X_fit.shape[1],
        "tfidf_settings": {"analyzer": "word", "ngram_range": [1, 1], "lowercase": True,
                           "sublinear_tf": False, "norm": "l2", "stop_words": None,
                           "min_df": 1, "max_df": 1.0, "use_idf": True, "smooth_idf": True,
                           "token_pattern": vectorizer.token_pattern, "binary": False},
        "train_data_sha256": hashlib.sha256((DATA_DIR / "train_data.csv").read_bytes()).hexdigest(),
        "train_indices_sha256": hashlib.sha256(np.asarray(fit_indices, dtype="<i8").tobytes()).hexdigest(),
        "val_indices_sha256": hashlib.sha256(np.asarray(val_indices, dtype="<i8").tobytes()).hexdigest(),
        "vocabulary_sha256": hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest(),
        "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__, "matplotlib_version": version("matplotlib"),
        "model": "sklearn.svm.SVC", "kernel": "linear", "C_values": list(C_VALUES),
        "selection_rule": "validation accuracy, then smaller C",
        "randomness_note": "Split is seeded; SVC random_state is ignored with probability=False.",
        "official_test_used": False, "probabilities_or_cross_entropy_computed": False,
        "source_index_definition": "0-based row in loaded train_data.csv, not a physical CSV line number",
    }


def main(check_only=False, reuse=True):
    if not C_VALUES or len(set(C_VALUES)) != len(C_VALUES) or any(not np.isfinite(c) or c <= 0 for c in C_VALUES):
        raise ValueError("C_VALUES 必须非空、不重复，并且全为有限正数。")
    fit_texts, val_texts, y_fit, y_val, fit_indices, val_indices = load_split()
    vectorizer, X_fit, X_val = make_features(fit_texts, val_texts)
    print(f"训练 {len(y_fit)} 篇；验证 {len(y_val)} 篇；单词特征 {X_fit.shape[1]} 个。")
    print(f"本次比较 C={C_VALUES}，各组共用同一份训练/验证矩阵。")
    common_config = make_config(vectorizer, X_fit, y_fit, y_val, fit_indices, val_indices)
    cached = find_cached_groups(common_config, y_fit, y_val, val_indices, val_texts) if reuse else {}
    print(f"复用 {len(cached)} 组；需要新训练 {len(C_VALUES) - len(cached)} 组。", flush=True)
    if check_only:
        print("检查通过；未训练或保存结果。")
        return

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "svm_results" / f"linear_C_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print("\n结果将保存到：")
    print(output_dir)
    print("SVC不输出逐轮进度，各组完成后显示结果。", flush=True)
    names = vectorizer.get_feature_names_out()
    pd.DataFrame({"feature_index": np.arange(len(names)), "feature": names, "idf": vectorizer.idf_}).to_csv(
        output_dir / "features.csv", index=False, encoding="utf-8-sig",
    )
    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps({**common_config, "status": "running"}, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for number, c_value in enumerate(C_VALUES, start=1):
        group_dir = output_dir / f"C{c_value:g}"
        group_dir.mkdir(exist_ok=False)
        if c_value in cached:
            old = cached[c_value]
            print(f"\n---------- [{number}/{len(C_VALUES)}] 复用：C={c_value:g} ----------", flush=True)
            summary = {**old["summary"], "reused": True, "source_run_dir": str(old["source"])}
            # 保留原配置及训练源码哈希。
            shutil.copyfile(old["source"] / "config.json", group_dir / "config.json")
            (group_dir / "reuse_note.json").write_text(json.dumps({
                "source_group_dir": str(old["source"]), "current_script_sha256": common_config["script_sha256"],
                "note": "Original config preserved; validation records checked; training_seconds belongs to the old run.",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            val_pred = old["predictions"]
            classes = np.unique(y_fit)
        else:
            print(f"\n---------- [{number}/{len(C_VALUES)}] 开始训练：线性SVM，C={c_value:g} ----------", flush=True)
            model, summary, val_pred = train_one(c_value, X_fit, X_val, y_fit, y_val)
            summary = {**summary, "reused": False, "source_run_dir": None}
            print(f"完成：训练={summary['train_accuracy']:.2%}；验证={summary['val_accuracy']:.2%}；"
                  f"错分={summary['wrong_count']}；训练耗时={summary['training_seconds']:.2f}秒。", flush=True)
            classes = model.classes_
            group_config = {**common_config, "svc_parameters": model.get_params(), "summary": summary,
                            "classes": classes.tolist(), "n_support_by_class": model.n_support_.tolist(),
                            "solver_iterations_by_binary_problem": model.n_iter_.tolist()}
            (group_dir / "config.json").write_text(json.dumps(group_config, ensure_ascii=False, indent=2), encoding="utf-8")
            del model  # 释放本组模型及核缓存。
        # 根据已核验预测生成分析，不再用旧模型预测。
        save_validation_analysis(group_dir, val_texts, y_val, val_pred, val_indices, classes)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    best = choose_best(summaries)
    config_path.write_text(json.dumps({**common_config, "status": "completed", "summaries": summaries,
                                      "selected_C": best["C"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print_comparison(summaries, len(y_fit), len(y_val), X_fit.shape[1])
    try:
        plot_comparison(output_dir, summaries)
        print("参数对照图已保存：svm_comparison.png")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"绘图未完成，但上面的结果和CSV已保存：{error}")
    print("参数、错例和汇总目录：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="线性 SVM 的 C 参数对照。")
    parser.add_argument("--check-only", action="store_true", help="只核验数据与特征，不训练、不生成结果。")
    parser.add_argument("--no-reuse", action="store_true", help="不复用旧组，从头训练全部 C。")
    args = parser.parse_args()
    main(check_only=args.check_only, reuse=not args.no_reuse)
