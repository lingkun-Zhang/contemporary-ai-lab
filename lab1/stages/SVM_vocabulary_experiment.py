"""固定线性 SVM 的 C=100，比较 5000 和 10000 词表。

5000 词组通过检查后直接复用，只训练 10000 词组。
--check-only 不训练；--no-reuse 重跑两组。
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import vocabulary_experiment_common as common
import SVM_experiment as core


DATA_DIR = Path(__file__).resolve().parent
SVM_C = 100.0


def validate_settings():
    parameters = core.make_model(SVM_C).get_params()
    wanted = {"C": 100.0, "kernel": "linear", "probability": False, "class_weight": None,
              "tol": 0.001, "max_iter": -1, "break_ties": False, "decision_function_shape": "ovr"}
    if (common.FEATURE_LIMITS, common.SPLIT_SEED) != ((5000, 10000), 42) or any(parameters[key] != value for key, value in wanted.items()):
        raise ValueError("本次要求5000/10000词表、固定线性SVM C100、不加权、不启用概率校准。")


def find_baseline(data):
    features = data["sets"][5000]
    expected = core.make_config(features["vectorizer"], features["X_fit"], data["y_fit"], data["y_val"],
                                data["fit_indices"], data["val_indices"])
    # 数据指纹按当前 CSV 计算，源码版本则核对原训练模块。
    for key in ("train_data_sha256", "train_indices_sha256", "val_indices_sha256"):
        expected[key] = data["metadata"][key]
    groups = core.find_cached_groups(expected, data["y_fit"], data["y_val"], data["val_indices"], data["val_texts"])
    cached = groups.get(SVM_C)
    if cached:
        print("可复用 SVM 5000 词、C=100 组。", flush=True)
    else:
        print("没有可复用的 C=100 结果，两组都重新训练。", flush=True)
    return cached


def main(check_only=False, reuse=True):
    validate_settings()
    data = common.prepare_data(DATA_DIR)
    cached = find_baseline(data) if reuse else None
    print("线性 SVM，C=100；不加权，不做概率校准。")
    print(f"复用 {int(cached is not None)} 组；需要新训练 {1 if cached else 2} 组。", flush=True)
    if check_only:
        print("检查通过；未训练、未生成新实验结果。")
        return

    output_dir = common.new_output_dir(DATA_DIR, "svm")
    config = {
        **data["metadata"], "model": "SVC", "svc_parameters": core.make_model(SVM_C).get_params(),
        "wrapper_sha256": common.sha256(__file__), "training_core_sha256": common.sha256(core.__file__),
        "selection_rule": "validation accuracy, then fewer features",
        "probabilities_or_cross_entropy_computed": False,
    }
    common.save_json(output_dir / "experiment_config.json", {**config, "status": "running"})
    summaries, predictions = [], {}
    for limit in common.FEATURE_LIMITS:
        features = data["sets"][limit]
        group_dir = output_dir / f"features{limit}"
        group_dir.mkdir(exist_ok=False)
        features["features"].to_csv(group_dir / "features.csv", index=False, encoding="utf-8-sig")
        reused = limit == 5000 and cached is not None
        if reused:
            print("\n复用 5000 词、C=100 组。", flush=True)
            summary, pred = cached["summary"], cached["predictions"]
            # 从已核对的预测重建报告，避免依赖缺失的旧报告文件。
            core.save_validation_analysis(group_dir, data["val_texts"], data["y_val"], pred,
                                          data["val_indices"], np.unique(data["y_fit"]))
            shutil.copyfile(cached["source"] / "config.json", group_dir / "baseline_original_config.json")
            source = str(cached["source"])
        else:
            print(f"\n---------- 新训练：线性SVM C100，词表上限{limit} ----------", flush=True)
            print("SVC 不显示逐轮进度，训练时请等待。", flush=True)
            model, summary, pred = core.train_one(SVM_C, features["X_fit"], features["X_val"], data["y_fit"], data["y_val"])
            print(f"完成：验证准确率={summary['val_accuracy']:.2%}；错分={summary['wrong_count']}。", flush=True)
            core.save_validation_analysis(group_dir, data["val_texts"], data["y_val"], pred, data["val_indices"], model.classes_)
            common.save_json(group_dir / "solver_details.json", {
                "classes": model.classes_.tolist(), "n_support_by_class": model.n_support_.tolist(),
                "solver_iterations_by_binary_problem": model.n_iter_.tolist(),
            })
            source = None
            del model
        summary = {**summary, "max_features": limit, "actual_feature_count": features["stats"]["actual_feature_count"],
                   "reused": reused, "source_run_dir": source}
        common.save_json(group_dir / "config.json", {**config, **features["stats"], "summary": summary})
        summaries.append(summary)
        predictions[limit] = pred
        pd.DataFrame(summaries).to_csv(output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    best = min(summaries, key=lambda row: (-row["val_accuracy"], row["actual_feature_count"]))
    corrected, introduced = common.finish_comparison(output_dir, data, summaries, predictions,
                                                    {**config, "selected_max_features": best["max_features"]})
    common.print_table("线性SVM C=100", summaries, (
        ("实际单词特征数", "actual_feature_count", "d"),
        ("训练准确率", "train_accuracy", ".2%"), ("验证准确率", "val_accuracy", ".2%"),
        ("训练-验证差距", "accuracy_gap", ".2%"), ("验证错分数量", "wrong_count", "d"),
        ("支持向量数量", "support_vector_count", "d"),
        ("支持向量占训练比例", "support_vector_fraction", ".2%"), ("训练耗时/秒", "training_seconds", ".2f"),
    ), corrected, introduced)
    print(f"按验证准确率选择词表：{best['max_features']}（同分选较小词表）。")
    print("未做概率校准，不计算交叉熵；支持向量数量不等于参数量。")
    try:
        common.plot_svm(output_dir, summaries)
        print("参数对照图已保存：vocabulary_comparison.png")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"绘图失败，已保存结果：{error}")
    print("结果目录：")
    print(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定线性 SVM 的 C=100，比较 5000 和 10000 词表。")
    parser.add_argument("--check-only", action="store_true", help="检查数据和旧结果，不训练。")
    parser.add_argument("--no-reuse", action="store_true", help="不复用旧结果，两组重跑。")
    args = parser.parse_args()
    main(check_only=args.check_only, reuse=not args.no_reuse)
