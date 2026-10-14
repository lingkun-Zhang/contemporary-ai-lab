# 实验一：文本分类

使用 TF-IDF 比较逻辑回归、MLP 和线性 SVM。实验围绕训练轮数、学习率、正则化、类别权重、网络宽度和词表容量展开。

## 文件说明

- [实验报告](report/report.pdf)：两页，包含训练曲线、错例和实验分析。
- [结果汇总](results/全部实验结果.csv)：29 组配置的末轮结果。
- [实验分析](results/实验结果分析与图像索引.md)：各组对照的结果与解释。
- [每 20 轮摘要](results/MLP每20轮摘要.md)：便于查看；每一轮的数据在各组 `history.csv` 中。
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

## 运行

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

新运行保存在 `local_runs/<会话>/`。其中 `private/` 含原文、词表和模型，仅供本地使用。`--resume` 需要本地完整会话，不能从去除私有文件的归档恢复。单独运行 `stages/` 脚本时，课程数据须放在该脚本目录；这些脚本的输出也可能包含原文，已列入 `.gitignore`。

## 重训与预测

按验证准确率选择 10000 词、C=100 的逻辑回归，其他参数见 `scripts/train_final.py`。其留出验证准确率为 93.01%，不是测试集成绩。

```powershell
python scripts/train_final.py --data-dir data --output local_final
```

重训使用全部 7368 篇有标签文本；2457 篇测试文本只做变换和预测。输出按测试原行序排列，每行一个整数类别，无表头和索引。脚本不覆盖已有结果。`--model-output` 可在本地保存模型；测试集无标签，不计算测试准确率。

## 记录来源

早期运行漏记了部分数据，随后按阶段参数统一重跑 29 组配置，报告采用这次记录。`results/history/` 保留旧记录，与重跑分开，不混用耗时。重跑时的源码保存在 `results/reproduction_20261014/record_all_experiments.py`；最终重训源码保存在 `results/final_training_source.py`。当前入口整理过注释和读取检查，旧源码未改。

公开记录去除了原文列、完整词表和本地路径，处理记录见 `notes/data_export.json`，文件哈希见 `checksums.json`。三次训练种子对照使用同一验证划分，不是三折交叉验证；多次调参后的验证结果也不能替代独立测试。
