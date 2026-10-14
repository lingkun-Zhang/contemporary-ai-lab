"""固定数据划分，用三个训练种子复验 50/100 个神经元的表现。

训练种子影响初始化和样本打乱；符合配置的旧种子 42 结果可复用。
--check-only 检查数据、配置和旧结果，不训练。
"""

import argparse
import hashlib
import inspect
import io
import json
import shutil
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

import MLP_structure_experiment as core
from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
SPLIT_SEED = 42  # 固定划分，单独观察训练随机性的影响。
TRAINING_SEEDS = (42, 43, 44)
HIDDEN_SIZES = (50, 100)


def prepare_data():
    frame = pd.read_csv(DATA_DIR / "train_data.csv")
    texts, labels = frame["text"].astype(str).tolist(), frame["target"].to_numpy()
    X_fit, X_val, y_fit, y_val, fit_indices, val_indices = train_test_split(
        texts, labels, np.arange(len(frame)),
        test_size=0.2, random_state=SPLIT_SEED, stratify=labels,
    )
    # 所有组共用训练部分拟合的词表和 IDF。
    vectorizer = TfidfVectorizer(max_features=core.MAX_FEATURES)
    X_fit_tfidf = vectorizer.fit_transform(X_fit)
    X_val_tfidf = vectorizer.transform(X_val)
    expected = {
        "seed": 42, "validation_fraction": 0.2, "max_features": core.MAX_FEATURES,
        "actual_feature_count": X_fit_tfidf.shape[1], "hidden_layers": 1,
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
    return X_fit_tfidf, X_val_tfidf, X_val, y_fit, y_val, val_indices, expected


def validate_cached_group(group_dir, common, expected, width, y_val, val_indices, class_count):
    """核对旧结果的配置、完整轮数及预测，避免复用不相容记录。"""
    required = ("config.json", "history.csv", "validation_predictions.csv",
                "wrong_predictions.csv", "classification_report.txt")
    if any(not (group_dir / name).is_file() for name in required):
        raise ValueError("缺少配置、日志或错误分析文件")
    config = json.loads((group_dir / "config.json").read_text(encoding="utf-8"))
    for key, value in expected.items():
        if key not in common or key not in config or common[key] != value or config[key] != value:
            raise ValueError(f"配置或数据不一致：{key}")
    count = (expected["actual_feature_count"] + 1) * width + (width + 1) * class_count
    if config["hidden_units"] != width or config["parameter_count"] != count:
        raise ValueError("隐藏层宽度或参数量不一致")
    history = pd.read_csv(group_dir / "history.csv")
    metrics = ("train_accuracy", "val_accuracy", "train_loss", "val_loss", "training_seconds")
    if history["epoch"].tolist() != list(range(1, core.EPOCHS + 1)):
        raise ValueError("日志不是完整且有序的训练轮数")
    if not np.isfinite(history[list(metrics)].to_numpy()).all():
        raise ValueError("日志包含非有限值")
    if not history[["train_accuracy", "val_accuracy"]].map(lambda value: 0 <= value <= 1).to_numpy().all():
        raise ValueError("准确率超出有效范围")
    if (history[["train_loss", "val_loss", "training_seconds"]] < 0).to_numpy().any():
        raise ValueError("损失或耗时为负数")
    if (history["training_seconds"].diff().dropna() < 0).any():
        raise ValueError("累计训练时间不单调")
    summary = next(row for row in common["summaries"] if row["hidden_units"] == width)
    if summary["parameter_count"] != count:
        raise ValueError("汇总参数量不一致")
    final = history.iloc[-1]
    for key in metrics:
        if not np.isclose(summary[key], final[key], rtol=1e-10, atol=1e-12):
            raise ValueError(f"最终日志与汇总不一致：{key}")
    predictions = pd.read_csv(group_dir / "validation_predictions.csv")
    if len(predictions) != len(y_val):
        raise ValueError("预测记录数量不一致")
    for actual, wanted in ((predictions["validation_index"], np.arange(len(y_val))),
                           (predictions["source_index"], val_indices),
                           (predictions["true_label"], y_val)):
        if not np.array_equal(actual.to_numpy(), wanted):
            raise ValueError("验证样本、顺序或真实标签不一致")
    if not (predictions["epoch"] == core.EPOCHS).all():
        raise ValueError("预测记录不是最终轮")
    pred = predictions["predicted_label"].to_numpy()
    correct = pred == y_val
    if not np.array_equal(predictions["is_correct"].to_numpy(), correct):
        raise ValueError("错误标记与预测不一致")
    if summary["wrong_count"] != int((~correct).sum()) or not np.isclose(summary["val_accuracy"], accuracy_score(y_val, pred)):
        raise ValueError("预测重算的准确率或错误数量不一致")
    if not np.isclose(summary["macro_f1"], f1_score(y_val, pred, average="macro")):
        raise ValueError("预测重算的 Macro-F1 不一致")
    true_proba = predictions["true_label_probability"].to_numpy()
    if not np.isfinite(true_proba).all() or np.any((true_proba < 0) | (true_proba > 1)):
        raise ValueError("真实类别概率无效")
    ce = float(-np.log(np.clip(true_proba, np.finfo(float).eps, 1)).mean())
    if not np.isclose(ce, summary["val_loss"], rtol=1e-9, atol=1e-10):
        raise ValueError("真实类别概率重算的交叉熵不一致")
    return {"summary": dict(summary), "history": history, "source": group_dir}


def find_seed42_cache(expected, y_val, val_indices, class_count):
    if 42 not in TRAINING_SEEDS:
        return {}
    candidates = sorted((DATA_DIR / "mlp_results").glob("structure_*"), reverse=True)
    for run_dir in candidates:
        try:
            common = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
            found = {
                (42, width): validate_cached_group(
                    run_dir / f"hidden{width}", common, expected, width, y_val, val_indices, class_count,
                ) for width in HIDDEN_SIZES
            }
            print(f"复用种子42基线：{run_dir.name}")
            return found
        except (OSError, ValueError, KeyError, StopIteration) as error:
            print(f"不复用 {run_dir.name}：{error}")
    print("未找到相容的种子42基线，将训练全部组。")
    return {}


def summarize_runs(rows):
    frame = pd.DataFrame(rows)
    if frame.duplicated(["seed", "hidden_units"]).any():
        raise ValueError("同一种子与宽度有重复记录")
    if set(zip(frame["seed"], frame["hidden_units"])) != {(seed, width) for seed in TRAINING_SEEDS for width in HIDDEN_SIZES}:
        raise ValueError("缺少预先确定的种子或宽度组")
    paired = frame.pivot(index="seed", columns="hidden_units", values="val_accuracy").reindex(TRAINING_SEEDS)
    paired = paired.rename(columns={50: "accuracy_50", 100: "accuracy_100"}).reset_index()
    paired.columns.name = None
    paired["difference_pp"] = 100 * (paired["accuracy_100"] - paired["accuracy_50"])
    aggregates = []
    for width in HIDDEN_SIZES:
        group = frame.loc[frame["hidden_units"] == width]
        aggregates.append({
            "hidden_units": width, "runs": len(group), "parameter_count": int(group["parameter_count"].iloc[0]),
            "mean_val_accuracy": float(group["val_accuracy"].mean()),
            "std_val_accuracy_pp": float(100 * group["val_accuracy"].std(ddof=1)),
            "mean_val_loss": float(group["val_loss"].mean()),
            "std_val_loss": float(group["val_loss"].std(ddof=1)),
        })
    return paired, pd.DataFrame(aggregates)


def print_results(paired, aggregates, expected, train_count, val_count):
    print(f"\n========== MLP 随机稳定性复验：最终第 {core.EPOCHS} 轮 ==========")
    print(f"实际训练集={train_count}；验证集={val_count}；划分种子固定={SPLIT_SEED}")
    print(f"固定：学习率={core.LEARNING_RATE:g}；alpha={core.ALPHA:g}；批次={core.BATCH_SIZE}")
    print(f"固定：特征数={expected['actual_feature_count']}；单隐藏层、ReLU、Adam、不加权")
    print("训练种子影响初始化和样本打乱；验证集只评估。")
    print("-" * 58)
    print("训练种子       50 个准确率     100 个准确率     差值/百分点")
    for row in paired.itertuples(index=False):
        print(f"{row.seed:8d}     {row.accuracy_50:12.2%}     {row.accuracy_100:12.2%}     {row.difference_pp:+10.2f}")
    print("-" * 58)
    for row in aggregates.itertuples(index=False):
        print(f"{row.hidden_units} 个：平均验证准确率={row.mean_val_accuracy:.2%}")
        print(f"       标准差={row.std_val_accuracy_pp:.2f} 个百分点；模型参数量={row.parameter_count}")
        print(f"       平均验证交叉熵={row.mean_val_loss:.4f}；交叉熵标准差={row.std_val_loss:.4f}")
    differences = paired["difference_pp"]
    wins, ties, losses = int((differences > 0).sum()), int((differences == 0).sum()), int((differences < 0).sum())
    print(f"100 对比 50 个：{wins} 次较高、{ties} 次相同、{losses} 次较低。")
    print(f"配对准确率差：平均 {differences.mean():+.2f} 个百分点；标准差 {differences.std(ddof=1):.2f} 个百分点。")
    if wins == len(paired):
        print("各训练种子下，100 个神经元的验证准确率都较高。")
    elif losses == len(paired):
        print("各训练种子下，100 个神经元的验证准确率都较低。")
    else:
        print("不同训练种子的结果并不一致。")
    print("标准差描述训练波动，不是置信区间或显著性检验。")
    print("评估交叉熵不含 L2 项；模型训练仍使用 L2 正则化。", flush=True)


def main(check_only=False):
    if HIDDEN_SIZES != (50, 100) or len(TRAINING_SEEDS) < 2 or len(set(TRAINING_SEEDS)) != len(TRAINING_SEEDS):
        raise ValueError("本实验固定比较 50/100 个神经元，并至少使用两个不重复的训练种子。")
    X_fit, X_val_features, X_val_text, y_fit, y_val, val_indices, expected = prepare_data()
    cache = find_seed42_cache(expected, y_val, val_indices, len(np.unique(y_fit)))
    total = len(TRAINING_SEEDS) * len(HIDDEN_SIZES)
    print(f"训练种子={TRAINING_SEEDS}；每个种子比较 50 和 100 个神经元。")
    print(f"数据划分始终固定为种子 {SPLIT_SEED}；每组 {core.EPOCHS} 轮。")
    print(f"需要新训练 {total - len(cache)} 组；可复用 {len(cache)} 组。", flush=True)
    if check_only:
        print("检查通过，未训练或生成实验结果。")
        return

    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = DATA_DIR / "mlp_results" / f"stability_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    with threadpool_limits(limits=2):
        for seed in TRAINING_SEEDS:
            for width in HIDDEN_SIZES:
                group_dir = output_dir / f"seed{seed}_hidden{width}"
                group_dir.mkdir(exist_ok=False)
                cached = cache.get((seed, width))
                if cached:
                    print(f"\n复用结果：训练种子={seed}，隐藏层={width} 个。", flush=True)
                    summary, history = cached["summary"], cached["history"]
                    for filename in ("validation_predictions.csv", "wrong_predictions.csv", "classification_report.txt"):
                        shutil.copyfile(cached["source"] / filename, group_dir / filename)
                    source = str(cached["source"])
                else:
                    print(f"\n---------- 新训练：种子={seed}，隐藏层={width} 个 ----------", flush=True)
                    model, history, summary, val_pred, val_proba = core.train_one(
                        width, X_fit, X_val_features, y_fit, y_val, seed=seed,
                    )
                    with redirect_stdout(io.StringIO()):
                        save_validation_analysis(
                            group_dir, X_val_text, y_val, val_pred, val_proba,
                            model.classes_, val_indices, epoch=core.EPOCHS,
                        )
                    source = None
                    del model
                history.to_csv(group_dir / "history.csv", index=False, encoding="utf-8-sig")
                config = {
                    **expected, "seed": seed, "split_seed": SPLIT_SEED, "model_seed": seed,
                    "hidden_units": width, "parameter_count": summary["parameter_count"],
                    "reused": cached is not None, "source_run_dir": source,
                    "current_training_function_sha256": hashlib.sha256(inspect.getsource(core.train_one).encode("utf-8")).hexdigest(),
                    "source_code_note": "Old runs did not store a training-function hash; reuse checks recorded configuration, data and result consistency.",
                    "randomness_caveat": "Model seed affects initialization and shuffling; shapes have different random draw counts.",
                }
                (group_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
                rows.append({**summary, "seed": seed, "reused": cached is not None, "source_run_dir": source})

    paired, aggregates = summarize_runs(rows)
    pd.DataFrame(rows).to_csv(output_dir / "runs.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(output_dir / "paired_comparison.csv", index=False, encoding="utf-8-sig")
    aggregates.to_csv(output_dir / "mean_std.csv", index=False, encoding="utf-8-sig")
    (output_dir / "experiment_config.json").write_text(json.dumps({
        "split_seed": SPLIT_SEED, "training_seeds": TRAINING_SEEDS, "hidden_sizes": HIDDEN_SIZES,
        "common_settings": expected, "std_definition": "sample standard deviation, ddof=1",
        "scope": "training randomness on one fixed validation split, not statistical significance",
        "timing_caveat": "Reused seed42 times are from an earlier run, not a controlled runtime benchmark.",
        "runs": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print_results(paired, aggregates, expected, len(y_fit), len(y_val))
    print("日志、来源和均值/标准差保存于：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="复验 MLP 50/100 个神经元在不同训练种子下的表现。")
    parser.add_argument("--check-only", action="store_true", help="检查数据、配置和旧结果，不训练。")
    main(check_only=parser.parse_args().check_only)
