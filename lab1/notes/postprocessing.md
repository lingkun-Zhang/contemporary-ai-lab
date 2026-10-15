# 图表与摘要生成

模型训练和后处理是两件事。`scripts/record_all_experiments.py`负责29组训练；下面的脚本只读取已经保存的结果，不会生成新的训练成绩，也不替换归档记录。

| 脚本 | 输入 | 输出 |
|---|---|---|
| `scripts/plot_comparisons.py` | `results/reproduction_20261014/session.json`及各组summary/history | 17组参数对照图及逐文件图源哈希 |
| `scripts/plot_report_figures.py` | 报告对应四组MLP的config/history | 两张完整损失曲线、局部放大与图源记录 |
| `scripts/build_epoch_summary.py` | 全部17组MLP的config/history | 第1轮、每20轮及末轮摘要，共90行 |
| `scripts/archive_history.py` | 用户本地阶段结果目录 | 完整私有归档，以及去文本的公开副本 |

前两个绘图脚本从实验时的`organize_records.py`绘图函数和报告绘图函数整理而来；摘要与归档脚本也来自本地整理工具。这次补入仓库时改了目录参数、相对路径和输出保护，不修改训练算法、原有指标或已归档的图像。它们是适配公开目录的后处理入口，不冒充早期训练时的源码快照。

所有曲线保留完整逐轮点，不平滑。LR与SVM的参数对照图横轴是参数值，不是训练轮次。重绘图片的字节可能随字体或绘图库版本改变，不能因此改写原图的历史哈希；生成的新图有自己的来源记录。

从`lab1`目录运行：

```powershell
python scripts/plot_comparisons.py --results results/reproduction_20261014 --output local_plots/comparisons
python scripts/plot_report_figures.py --results results/reproduction_20261014 --output local_plots/report
python scripts/build_epoch_summary.py --records-dir results/reproduction_20261014 --output local_summaries
```

摘要CSV保留原始数字字符串，Markdown表格只调整显示精度。摘要与重绘使用新目录，不覆盖已有记录。报告正文仍以`report/report.md`和两页`report/report.pdf`为准。

历史归档工具不是复现实验的必需步骤。仅当自己另有阶段结果时使用：

```powershell
python scripts/archive_history.py --lab-dir "你的阶段代码与结果目录" --output local_history
```

该命令不训练，保留成功、失败和复用组。`local_history/history_private`含原文、完整词表及本地路径，不得公开上传；公开副本的删除列和文件取舍会逐项记录。仓库已有去文本历史归档，无须为了复现再运行此步骤。
