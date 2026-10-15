# 实验一：文本分类

使用 TF-IDF 比较逻辑回归、MLP 和线性 SVM。实验围绕训练轮数、学习率、正则化、类别权重、网络宽度和词表容量展开。

## 文件说明

- [实验报告](report/report.pdf)：两页，包含训练曲线、错例和实验分析。
- [结果汇总](results/全部实验结果.csv)：29 组配置的末轮结果。
- [实验分析](results/实验结果分析与图像索引.md)：各组对照的结果与解释。
- [每 20 轮摘要](results/MLP每20轮摘要.md)：便于查看；每一轮的数据在各组 `history.csv` 中。
- [邮件头与签名清洗](results/header_cleaning_20261015/README.md)：补充三组对照，含固定规则、逐样本预测、成对错分和图像。
- `results/reproduction_20261014/runs/`：配置、逐轮指标、预测、错例、混淆矩阵和运行日志。
- `figures/`：对照图及图源记录。
- `stages/`：阶段实验脚本，保留实验结束时的参数；统一重跑使用 `scripts/record_all_experiments.py`。
- `final_outputs/prediction.csv`：选定配置重训后的测试集预测。

## 环境和数据

运行环境为 Python 3.10.21，依赖版本见 `requirements.txt`。scikit-learn 1.7.2 支持 MLP 增量训练时传入样本权重。

```powershell
python -m pip install -r requirements.txt
```

课程数据不随仓库分发。将老师提供的文件放在本地 `data/`，保留原始行序：

```text
data/train_data.csv          # 7368 行，text、target 列，标签 0—9
data/test_data_unlabeled.csv # 2457 行，text 列
```

TF-IDF默认使用单词分析器，正则为`(?u)\b\w\w+\b`（长度至少2的词元），转小写、不去停用词，`use_idf=True`、`smooth_idf=True`、`norm="l2"`、`min_df=1`、`max_df=1.0`、`binary=False`。普通词频对应`sublinear_tf=False`；短语和对数词频组只按各自配置改变相应设置。选型LR实际参数见[feature_metadata.json](results/reproduction_20261014/runs/lr_vocab10000/feature_metadata.json)。

早期逻辑回归阶段曾读取并打印无标签测试文本作格式检查，因此不能称测试集从未被查看；未将其用于模型拟合、验证评价或调参。现已从`stages/Logistic_experiment.py`移除测试集读取和打印，当前训练/验证入口只加载训练文件；统一重跑和清洗入口原本就只读训练文件。测试文件由最终重训与预测入口加载。历史源码快照和结果不作追溯改写。

记录保留了软件版本与线程设置，但没有当时的完整硬件快照。耗时仅用于本次实现和预算下的参考比较，不作为跨硬件性能基准。

## 运行

以下命令均在`lab1`目录执行；从仓库根目录进入时先运行`cd lab1`。

检查数据和配置，不训练：

```powershell
python scripts/record_all_experiments.py --check-only
```

运行全部配置：

```powershell
python scripts/record_all_experiments.py --output local_runs
```

数据按 80%/20% 分层划分，种子为 42。词表和 IDF 只在 5894 篇训练文本上拟合。每个配置训练一次，不同对照共用相应的基线结果。MLP 的一轮指遍历一次训练数据，轮末在训练集和验证集上分别计算交叉熵与准确率。评估交叉熵不含 L2 项，训练仍按 alpha 使用正则化；曲线使用全部逐轮点，不作平滑。SVC 未启用概率输出，不记录交叉熵曲线。

检查已有记录，或从逐轮数据重画曲线：

```powershell
python scripts/verify_results.py --results results/reproduction_20261014
python scripts/plot_saved_curves.py --results results/reproduction_20261014 --output local_plots
```

重画17组参数对照图、报告中的两张损失图，或重新抽取每20轮摘要：

```powershell
python scripts/plot_comparisons.py --results results/reproduction_20261014 --output local_plots/comparisons
python scripts/plot_report_figures.py --results results/reproduction_20261014 --output local_plots/report
python scripts/build_epoch_summary.py --records-dir results/reproduction_20261014 --output local_summaries
```

这些命令只读已有记录，不训练，也不覆盖归档图像或摘要。输出目录需为新目录。后处理脚本的来源和适配范围见[图表与摘要生成说明](notes/postprocessing.md)。

新运行保存在 `local_runs/<会话>/`。其中 `private/` 含原文、词表和模型，仅供本地使用。`--resume` 需要本地完整会话，不能从去除私有文件的归档恢复。单独运行 `stages/` 脚本时，课程数据须放在该脚本目录；这些脚本的输出也可能包含原文，已列入 `.gitignore`。

## 重训与预测

当前提交配置是：**原文单词TF-IDF（上限10000词）+ L2逻辑回归，C=100、solver="lbfgs"、max_iter=1000、tol=1e-4，不加权。** `final_outputs/prediction.csv`由这个配置在全部有标签数据上重训后生成，不是MLP、SVM或去来源头版本的预测。

本次选型分为两个阶段，不能把它们写成对所有配置一次性取最高分：

1. 原29组对照中，10000词、C=100的原文LR验证准确率为93.01%，高于所比较的MLP与SVM；与MLP只差2篇，因此同时比较参数量和训练成本，选定较简单的LR。其余参数见 `scripts/train_final.py`，全量重训记录见 `final_outputs/final_training.json`。
2. 后来补做的去来源头组为93.15%、交叉熵0.2569，两项都优于原文组；不能再称原文LR是全部已试配置中验证指标最好的。清洗修正26篇又新增24篇错误，单次划分净收益只有2篇，未做显著性检验或多划分复验。本次暂不因此替换前阶段已经选定并全量重训的模型，也不认定原文模型优于清洗模型。

第二点是补充实验后的保守取舍，不是事先规定的显著性阈值。当前`prediction.csv`始终对应原文LR；不能用清洗组的验证成绩来描述这份预测。所有验证成绩均不是测试集成绩。

```powershell
python scripts/train_final.py --data-dir data --output local_final
```

重训使用全部 7368 篇有标签文本；2457 篇测试文本只做变换和预测。输出按测试原行序排列，每行一个整数类别，无表头和索引。脚本不覆盖已有结果。`--model-output` 可在本地保存模型；测试集无标签，不计算测试准确率。

### 选型配置的留出分类表现

以下是`lr_vocab10000`在5894篇训练、1474篇验证划分中的表现，不是全量重训后的重新评估，也不是测试集成绩。由[逐样本预测](results/reproduction_20261014/runs/lr_vocab10000/validation_predictions.csv)核算，并与[分类报告](results/reproduction_20261014/runs/lr_vocab10000/classification_report.json)、[混淆矩阵](results/reproduction_20261014/runs/lr_vocab10000/confusion_matrix.csv)对应。

| 类别 | 精确率 | 召回率 | F1 | 验证样本数 |
|---|---:|---:|---:|---:|
| 0 | 88.16% | 91.78% | 89.93% | 146 |
| 1 | 91.22% | 91.22% | 91.22% | 148 |
| 2 | 81.17% | 85.03% | 83.06% | 147 |
| 3 | 96.60% | 95.95% | 96.27% | 148 |
| 4 | 96.62% | 95.33% | 95.97% | 150 |
| 5 | 97.39% | 100.00% | 98.68% | 149 |
| 6 | 95.80% | 91.95% | 93.84% | 149 |
| 7 | 87.41% | 84.46% | 85.91% | 148 |
| 8 | 96.60% | 95.95% | 96.27% | 148 |
| 9 | 100.00% | 98.58% | 99.29% | 141 |
| 宏平均 | 93.10% | 93.02% | 93.04% | 1474 |

类别2和7是主要薄弱类：2→7有9篇，7→2有10篇，合计19篇，占103个错误的18.45%。类别5在这次验证中召回率为100%，但精确率不是100%，也不能据此保证未来文本都分对。总体准确率不足以代表每个类别都同样可靠。

## 邮件头与签名对照

入口为 [scripts/header_cleaning_experiment.py](scripts/header_cleaning_experiment.py)，已跑记录在 [results/header_cleaning_20261015](results/header_cleaning_20261015/README.md)，不混入原来的29组汇总。固定原划分、10000词和LR C=100，比较原文、去指定来源字段、再去明确标记的签名；保留Subject、Keywords、Summary和未知字段，不按固定行数截尾。

```powershell
python scripts/header_cleaning_experiment.py --data-dir data --output local_cleaning
```

| 处理 | 验证准确率 / 交叉熵 | 已保存指标 |
|---|---|---|
| 原文复训 | 93.01% / 0.2709 | [original/metrics.json](results/header_cleaning_20261015/original/metrics.json) |
| 去指定来源头 | 93.15% / 0.2569 | [no_source_headers/metrics.json](results/header_cleaning_20261015/no_source_headers/metrics.json) |
| 再去标记签名 | 92.94% / 0.2623 | [no_source_headers_signature/metrics.json](results/header_cleaning_20261015/no_source_headers_signature/metrics.json) |

每组同目录内有`validation_predictions.json`、`classification_report.json`、`confusion_matrix.json`和`cleaning_audit.json`。共同文件包括[固定规则](results/header_cleaning_20261015/protocol.json)、[成对错分与基线复现检查](results/header_cleaning_20261015/summary.json)、[比较图](results/header_cleaning_20261015/comparison.png)及`split.json`、`session.json`、`source_snapshot.py`。

输出包括规则、清洗覆盖率、逐样本验证预测、分类指标、成对错分变化和比较图；不读取测试集、不生成或替换提交用预测。`source_snapshot.py`是该次训练的源码，`figure_source.json`记录后续只调整图例位置的重绘。可用`--redraw --output results/header_cleaning_20261015`从已存指标重画，不会训练。

## 记录来源

早期运行漏记了部分数据，随后按阶段参数统一重跑 29 组配置，报告采用这次记录。`results/history/` 保留旧记录，与重跑分开，不混用耗时。重跑时的源码保存在 `results/reproduction_20261014/record_all_experiments.py`；最终重训源码保存在 `results/final_training_source.py`。当前入口整理过注释和读取检查，并去掉阶段LR的测试读取；历史快照未改，模型配置和已保存训练结果也未改。

公开记录去除了原文列、完整词表和本地路径，处理记录见 `notes/data_export.json`，文件哈希见 `checksums.json`。三次训练种子对照使用同一验证划分，不是三折交叉验证；多次调参后的验证结果也不能替代独立测试。
