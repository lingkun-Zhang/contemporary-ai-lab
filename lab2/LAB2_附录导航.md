# 实验二数据索引

先看[五页报告](lab2_张凌琨_10241900443.pdf)或[报告文本](lab2_张凌琨_10241900443.md)。下表可直接打开各组实验说明和数据。

## 题一

| 研究问题 | 报告 | 数据目录 |
|---|---|---|
| A*实现、样例路径、h=0与曼哈顿及同分排序 | [题一报告](题一/REPORT.md) | [题一/results/](题一/results) |
| 地图结构、障碍密度、地标下界和预处理成本 | [拓展实验](题一/EXPLORATION.md) | [题一/results/exploration/](题一/results/exploration) |
| 地标数量的成本低点 | [参数扫描](题一/LANDMARK_TUNING.md)、[固定候选确认](题一/LANDMARK_CONFIRMATION.md) | [题一/results/landmark_tuning/](题一/results/landmark_tuning)、[landmark_confirmation/](题一/results/landmark_confirmation) |
| 单位费用下BFS与堆搜索的成本 | [BFS对照](题一/BFS_STUDY.md) | [题一/results/bonus_bfs/](题一/results/bonus_bfs) |
| 可采纳但不一致启发为什么需要重开 | [重开消融](题一/REOPEN_STUDY.md) | [题一/results/bonus_reopen/](题一/results/bonus_reopen) |
| 静态连通索引何时值得预处理 | [连通预检](题一/CONNECTIVITY_STUDY.md) | [题一/results/bonus_connectivity/](题一/results/bonus_connectivity) |
| 收缩走廊能否与地标互补，何时回本 | [走廊压缩研究](题一/CORRIDOR_STUDY.md) | [题一/results/corridor_study/](题一/results/corridor_study) |
| 已有输入间的收益差异 | [证据分析](题一/EVIDENCE_AUDIT.md)、[研究概览](题一/BONUS_REPORT.md) | [题一/results/bonus_audit/](题一/results/bonus_audit) |

## 题二

| 研究问题 | 报告 | 数据目录 |
|---|---|---|
| E免费、下界推导、费用账本、tight与排序、费用和位置效应 | [题二报告](题二/REPORT.md) | [题二/results/](题二/results)、[results_supplementary/](题二/results_supplementary) |
| 清理过期堆项能否改善时间和队列峰值 | [阈值扫描](题二/HEAP_TUNING.md)、[固定候选确认](题二/HEAP_CONFIRMATION.md) | [题二/results_heap_tuning/](题二/results_heap_tuning)、[results_heap_confirmation/](题二/results_heap_confirmation) |
| 整数费用表示是否影响桶队列，gcd单位能否修正 | [整数桶研究](题二/BUCKET_STUDY.md) | [题二/results_bucket/](题二/results_bucket) |

## 题三

| 研究问题 | 报告 | 数据目录 |
|---|---|---|
| 数值回传、剪枝、顺序、等号条件及采样排序 | [题三报告](题三/REPORT.md) | [题三/results/](题三/results) |
| 怎样正确缓存截断返回界，何时能减少重复工作 | [界感知缓存](题三/TRANSPOSITION_STUDY.md) | [题三/results_transposition/](题三/results_transposition)、[results_transposition_confirmation/](题三/results_transposition_confirmation) |
| 采样低点是否复现，叶值相关性与有限缓存容量 | [进一步实验](题三/SUPPLEMENTARY_EXPERIMENTS.md) | [题三/results_sampling_confirmation/](题三/results_sampling_confirmation)、[results_correlation_study/](题三/results_correlation_study)、[results_capacity_study/](题三/results_capacity_study) |
| 阈值探测省下的求值是否抵得过重搜 | [Scout研究](题三/SCOUT_STUDY.md) | [题三/results_scout/](题三/results_scout) |

## 题四

| 研究问题 | 报告 | 数据目录 |
|---|---|---|
| 距离矩阵、状态、H2/H4账本、访问顺序与价值边界 | [题四报告](题四/REPORT.md) | [题四/results/](题四/results) |
| 不同地图与宝藏位置下结论是否保留 | [补充实验第2节](题四/SUPPLEMENTARY_EXPERIMENTS.md) | [题四/results_robustness/](题四/results_robustness) |
| 延迟查询何时减少实际寻路与总成本 | [补充实验第3节](题四/SUPPLEMENTARY_EXPERIMENTS.md) | [题四/results_lazy/](题四/results_lazy) |
| 缓存未来收益增量是否安全，宝藏规模何时值得缓存 | [增量缓存研究](题四/INCREMENTAL_CACHE_STUDY.md) | [题四/results_incremental_cache/](题四/results_incremental_cache) |

## 读取数据

`comparison.csv`保存各配置的结果和统计；`timing_batches.csv`保存计时批值，有`timing_calls.csv`的研究还保存逐次调用值。地标整批实验使用`batch_raw.csv`与`candidate_summary.csv`。配对比较文件记录同一输入、同一重复块的方法差值。

`inputs.json`或`reproducibility.json`保存具体地图、树和参数，`protocol.json`保存相应实验的固定比较范围。源码与结果指纹用于核对所运行版本。计时重复、方法配置与独立输入是不同的数量；目录中的复现文件说明各自口径。

## 课程参考

[CS188 参考资料](CS188_参考资料与迁移边界.md) · [课程要求在报告中的位置](LAB2_要求回答与证据核对.md)
