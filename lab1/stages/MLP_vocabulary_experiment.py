"""比较 MLP 在 5000 和 10000 词表下的效果，保持 100 个隐藏层神经元。

5000 词组通过检查后直接复用，只训练 10000 词组。
--check-only 不训练；--no-reuse 重跑两组。
"""

import argparse
import io
import json
import shutil
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import vocabulary_experiment_common as common
import MLP_structure_experiment as core
from MLP_stability_experiment import validate_cached_group
from MLP_experiment import save_validation_analysis


DATA_DIR = Path(__file__).resolve().parent
HIDDEN_UNITS = 100
TRAINING_SEED = 42


def validate_settings():
    """防止共用训练函数的参数改动混进词表对照。"""
    if (common.FEATURE_LIMITS, common.SPLIT_SEED, HIDDEN_UNITS, TRAINING_SEED) != ((5000, 10000), 42, 100, 42):
        raise ValueError("本次固定5000/10000词表、种子42、100个神经元。")
    if (core.MAX_FEATURES, core.LEARNING_RATE, core.ALPHA, core.BATCH_SIZE, core.EPOCHS) != (5000, 0.0001, 0.005, 64, 100):
        raise ValueError("共用MLP配置已改变；要求学习率0.0001、alpha0.005、批次64、100轮。")


def baseline_expected(data):
    meta = data["metadata"]
    return {
        "seed": 42, "validation_fraction": 0.2, "max_features": 5000,
        "actual_feature_count": data["sets"][5000]["X_fit"].shape[1], "hidden_layers": 1,
        "activation": "relu", "solver": "adam", "learning_rate_init": core.LEARNING_RATE,
        "alpha": core.ALPHA, "batch_size": core.BATCH_SIZE, "epochs": core.EPOCHS,
        "training_threads": 2, "early_stopping": False, "sample_weight": None,
        "loss_definition": "epoch-end unweighted cross-entropy without L2 penalty",
        "comparison_model": "final epoch only",
        **{key: meta[key] for key in ("train_data_sha256", "train_indices_sha256", "val_indices_sha256",
                                     "numpy_version", "pandas_version", "sklearn_version")},
    }


def find_baseline(data):
    expected = baseline_expected(data)
    # 按时间找配置一致的旧组，不按最高准确率挑结果。
    for run_dir in sorted((DATA_DIR / "mlp_results").glob("structure_*"), reverse=True):
        try:
            recorded = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
            if not isinstance(recorded, dict):
                raise ValueError("旧配置不是JSON对象")
            cached = validate_cached_group(run_dir / "hidden100", recorded, expected, HIDDEN_UNITS,
                                           data["y_val"], data["val_indices"], len(np.unique(data["y_fit"])))
            records = common.validate_prediction_files(cached["source"], data)
            cached["predictions"] = records["predicted_label"].to_numpy()
            print(f"可复用 MLP 5000 词组：{run_dir.name}/hidden100", flush=True)
            return cached
        except (OSError, ValueError, KeyError, TypeError, AssertionError, StopIteration) as error:
            print(f"不复用 {run_dir.name}：{error}", flush=True)
    print("没有可复用的 MLP 结果，两组都重新训练。", flush=True)
    return None


def plot_comparison(output_dir, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for color, (limit, history) in zip(("#2166AC", "#D6604D"), histories.items()):
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(history["epoch"], history[f"train_{metric}"], color=color, label=f"{limit}词：训练")
            ax.plot(history["epoch"], history[f"val_{metric}"], color=color, linestyle="--", label=f"{limit}词：验证")
    axes[0].set(title="平均交叉熵（统一的预测评估）", ylabel="交叉熵")
    axes[1].set(title="分类准确率", ylabel="准确率")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.set_xlabel("训练轮数")
        ax.set_xlim(1, core.EPOCHS)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("MLP词表容量对照：100神经元，种子42，其余参数固定\n实线训练，虚线验证；比较最终100轮模型")
    fig.savefig(output_dir / "vocabulary_comparison.png", dpi=220)
    plt.close(fig)


def main(check_only=False, reuse=True):
    validate_settings()
    data = common.prepare_data(DATA_DIR)
    cached = find_baseline(data) if reuse else None
    print(f"固定：100神经元、ReLU/Adam、学习率={core.LEARNING_RATE:g}、alpha={core.ALPHA:g}、批次64、100轮。")
    print(f"复用 {int(cached is not None)} 组；需要新训练 {1 if cached else 2} 组。", flush=True)
    print("词表变大，参数也增多；同种子不等于相同初始化和打乱顺序。")
    if check_only:
        print("检查通过；未训练、未生成新实验结果。")
        return

    output_dir = common.new_output_dir(DATA_DIR, "mlp")
    config = {
        **data["metadata"], "model": "MLPClassifier", "hidden_units": HIDDEN_UNITS,
        "model_seed": TRAINING_SEED, "learning_rate_init": core.LEARNING_RATE, "alpha": core.ALPHA,
        "batch_size": core.BATCH_SIZE, "epochs": core.EPOCHS, "activation": "relu", "solver": "adam",
        "training_threads": 2, "early_stopping": False, "comparison_model": "final epoch only",
        "wrapper_sha256": common.sha256(__file__), "training_core_sha256": common.sha256(core.__file__),
        "selection_rule": "final validation accuracy, then lower predictive cross-entropy, then fewer features",
        "baseline_note": "Historical structure run did not record vocabulary/source hashes; its data/config/history/predictions are checked. Current feature hashes are not proof of the historical vocabulary hash.",
    }
    common.save_json(output_dir / "experiment_config.json", {**config, "status": "running"})
    summaries, histories, predictions = [], {}, {}
    with threadpool_limits(limits=2):
        for limit in common.FEATURE_LIMITS:
            features = data["sets"][limit]
            group_dir = output_dir / f"features{limit}"
            group_dir.mkdir(exist_ok=False)
            features["features"].to_csv(group_dir / "features.csv", index=False, encoding="utf-8-sig")
            reused = limit == 5000 and cached is not None
            if reused:
                print("\n复用 5000 词组的第 100 轮结果。", flush=True)
                summary, history, pred = cached["summary"], cached["history"], cached["predictions"]
                for name in ("validation_predictions.csv", "wrong_predictions.csv", "classification_report.txt"):
                    shutil.copyfile(cached["source"] / name, group_dir / name)
                shutil.copyfile(cached["source"] / "config.json", group_dir / "baseline_original_config.json")
                source = str(cached["source"])
            else:
                print(f"\n---------- 新训练：MLP，词表上限{limit}，100轮 ----------", flush=True)
                model, history, summary, pred, proba = core.train_one(
                    HIDDEN_UNITS, features["X_fit"], features["X_val"], data["y_fit"], data["y_val"], seed=TRAINING_SEED,
                )
                with redirect_stdout(io.StringIO()):
                    save_validation_analysis(group_dir, data["val_texts"], data["y_val"], pred, proba,
                                             model.classes_, data["val_indices"], epoch=core.EPOCHS)
                source = None
                del model
            summary = {**summary, "max_features": limit, "actual_feature_count": features["stats"]["actual_feature_count"],
                       "reused": reused, "source_run_dir": source}
            history.to_csv(group_dir / "history.csv", index=False, encoding="utf-8-sig")
            common.save_json(group_dir / "config.json", {**config, **features["stats"], "summary": summary})
            summaries.append(summary)
            histories[limit], predictions[limit] = history, pred
            pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    best = min(summaries, key=lambda row: (-row["val_accuracy"], row["val_loss"], row["actual_feature_count"]))
    corrected, introduced = common.finish_comparison(output_dir, data, summaries, predictions,
                                                    {**config, "selected_max_features": best["max_features"]})
    common.print_table("MLP 最终100轮", summaries, (
        ("实际单词特征数", "actual_feature_count", "d"), ("模型参数量", "parameter_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"), ("验证准确率", "val_accuracy", ".2%"),
        ("训练交叉熵", "train_loss", ".4f"), ("验证交叉熵", "val_loss", ".4f"),
        ("训练-验证差距", "accuracy_gap", ".2%"), ("验证错分数量", "wrong_count", "d"),
        ("训练耗时/秒", "training_seconds", ".2f"),
    ), corrected, introduced)
    print(f"按第 100 轮验证准确率选择词表：{best['max_features']}。")
    print("交叉熵不含 L2 惩罚；训练仍使用 L2 正则化。")
    try:
        plot_comparison(output_dir, histories)
        print("曲线已保存：vocabulary_comparison.png")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"绘图失败，已保存结果和日志：{error}")
    print("结果目录：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定 MLP 100 个神经元，比较 5000 和 10000 词表。")
    parser.add_argument("--check-only", action="store_true", help="检查数据和旧结果，不训练。")
    parser.add_argument("--no-reuse", action="store_true", help="不复用旧结果，两组重跑。")
    args = parser.parse_args()
    main(check_only=args.check_only, reuse=not args.no_reuse)
