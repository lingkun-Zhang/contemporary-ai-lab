# 题四：竞争环境下的宝藏分配

按老师PDF实现A*距离矩阵与Minimax/Alpha-Beta：A/B/T进入费1，单次选宝算一层，收益为价值减移动费，评价Score_A−Score_B。正式样例H2输出T2/−1，H4输出T3/3，两算法一致。

Python3.9+，仅标准库。Windows双击`run_q4.cmd`即可打印完整矩阵和两深度、两算法的结果；以下命令均在本文件夹执行。

```text
run_q4.cmd
run_q4.cmd --input samples/pdf_sample.json
run_q4.cmd --depth 4 --algorithm alphabeta --details
run_q4.cmd tests
run_q4.cmd experiments
run_q4.cmd robustness
run_q4.cmd lazy
```

也可运行：

```text
python treasure_game.py
python treasure_game.py --depth 2 --algorithm minimax --order label
python -m unittest discover -v
python run_experiments.py
python draw_q4_figures.py
```

当前电脑脚本优先使用已验证的本地Python，换电脑可用`py -3`或设置`Q4_PYTHON`。目前59项测试通过（原有47项，以及收益增量缓存12项）。报告直接回答矩阵、状态、深度差异、剪枝与输入价值边界，见[REPORT.md](REPORT.md)；跨地图与延迟距离查询结果见[补充实验报告](SUPPLEMENTARY_EXPERIMENTS.md)。

```python
from treasure_game import sample_problem, build_distance_matrix, game_search
problem = sample_problem()
distances = build_distance_matrix(problem)
result = game_search(problem, distances, depth=4)
assert (result.best_target, result.best_value) == ("T3", 3)
```

`Problem`保存数字地图、标签、位置与宝物价值；`State`保存双方位置、剩余位掩码、双方累计收益和当前玩家。`apply_action()`只更新行动者。`astar_search()`返回最优费用、路径、逐格账本与统计；`root_action_values()`另作精确根动作值分析。默认标签顺序，同分保留首先访问目标；额外顺序允许返回不同的同值最优目标。

| 文件 | 用途 |
|---|---|
| [treasure_game.py](treasure_game.py) | A*、矩阵、状态转移、Minimax/Alpha-Beta及CLI |
| [reference_tools.py](reference_tools.py) | 独立Dijkstra、全枚举、账本和路径核验 |
| [test_treasure_game.py](test_treasure_game.py) | 25项测试、随机与规则回归 |
| [run_experiments.py](run_experiments.py) | 深度、排序、矩阵启发、路由复用、价值扫描 |
| [draw_q4_figures.py](draw_q4_figures.py) | 读取实测数据生成4张SVG |
| [samples/pdf_sample.json](samples/pdf_sample.json) | 老师完整地图与价值输入 |
| [results/sample_analysis.json](results/sample_analysis.json) | 根真值、后悔值、H2/H4账本及状态遗漏反例 |
| [results/paths.json](results/paths.json) | 每个有序路径的费用、坐标及逐格账本 |
| [results/comparison.csv](results/comparison.csv) | 全部308配置与时间/计数 |
| [results/timing_batches.csv](results/timing_batches.csv) | 2156个原始计时批 |
| [results/protocol.json](results/protocol.json) | 预热前冻结的完整输入、候选和计时边界 |
| [results/reproducibility.json](results/reproducibility.json) | 环境与源码/结果SHA |

实验7重复块×每块3次；有6468次顶层计时调用，不能当作独立输入数。只有一张地形和15组价值。复跑默认另建带时间戳子目录，保留原结果；可用`python run_experiments.py --output-dir results_new`指定。新目录图表用`python draw_q4_figures.py --results-dir results_new`生成。

路由策略实验分`decision_only`和`complete_task`两种输出要求；正式解答始终输出完整6×6矩阵。仅求首步的按需策略不能替代正式矩阵任务。各计时窗包含相应的排序、A*、缓存和预计算开销。T3价值扫描保持Gain公式不变，是自设输入，不是重定义正式题目。

## 补充实验：跨地图及延迟距离查询

```text
python -B run_robustness_study.py --repeats 5 --batch 3
python -B run_lazy_study.py --repeats 5 --batch 3
```

两份脚本各自自动生成图表。默认输出分别为`results_robustness`与`results_lazy`；目录已有结果时另建带时间戳子目录，保留本次数据。也可指定`--output-dir results_robustness_new`或`--output-dir results_lazy_new`。正式计时请串行运行，并等待一项结束后再运行下一项。

| 文件/目录 | 用途 |
|---|---|
| [SUPPLEMENTARY_EXPERIMENTS.md](SUPPLEMENTARY_EXPERIMENTS.md) | 研究问题、同输入对照、数据分析与反例 |
| [run_robustness_study.py](run_robustness_study.py) | 三类30新地图、完整矩阵验证、H1—4与排序/剪枝比较 |
| [lazy_game_search.py](lazy_game_search.py) | 可选的延迟查询与同核预生成对照，保留正式原版 |
| [run_lazy_study.py](run_lazy_study.py) | 原版/同核预生成/惰性，搜索阶段及两种输出要求 |
| [test_lazy_game_search.py](test_lazy_game_search.py) | 12项独立真值、查询计数、子局面与边界测试 |
| [test_robustness_study.py](test_robustness_study.py) | 6项输入生成、协议、实际首步与PV核验测试 |
| [test_lazy_study.py](test_lazy_study.py) | 4项输出要求、访问树、控制条件与配对统计测试 |
| [results_robustness/comparison.csv](results_robustness/comparison.csv) | 780配置；30图720博弈配置与60矩阵配置 |
| [results_lazy/comparison.csv](results_lazy/comparison.csv) | 1620配置；31图搜索与预定7图路由子集 |
| [supplement_manifest.json](supplement_manifest.json) | 补充源码、测试、报告、数据与图表的完整文件SHA |

两组均为5块×每块3次；目录内`timing_calls.csv`保存逐次时间，`timing_batches.csv`保存每块的每调用平均时间。中位数/IQR由5块均值计算；`paired_differences.csv`先在同一块内相减，再计算分位数。总计12000批、36000次顶层计时调用，**只有30张新增独立地图**：第二组复用这30图，并另含原PDF样例；路由7图是求解前固定的子集。

```python
from lazy_game_search import lazy_game_search
result = lazy_game_search(problem, distances, depth=4, order="label")
assert (result.best_target, result.best_value) == ("T3", 3)
assert result.stats.distance_lookups == 46
```

延迟查询保留相同访问树，减少被剪候选的距离查询和Gain准备；各版本实际构造的子状态数相同。需要即时Gain排序时仍先查询所有候选，Minimax无剪枝时也无请求量收益。预计算/按需缓存输出完整矩阵时仍须30次A*，不能把决策子任务的少算用作正式完整任务的提速证据。

## 收益增量缓存与宝藏规模

收益增量缓存沿用叶评价`U=score_A-score_B`，存储未来最优增量`V=F-U`。状态仍记录双方累计收益；缓存键另含双方位置、剩余宝藏、玩家与精确剩余深度。直接删掉分数字段却缓存绝对结果会算错；Alpha-Beta截断返回的界也须与精确值分别处理。

```python
from incremental_cache_search import cache_search
result = cache_search(problem, distances, depth=4,
                      algorithm="alphabeta", order="label",
                      cache_mode="incremental")
assert (result.best_target, result.best_value) == ("T3", 3)
```

```text
python -B -m unittest test_incremental_cache
python -B run_incremental_cache_study.py --repeats 7
python -B summarize_incremental_cache.py --data results_incremental_cache
```

首跑保存在`results_incremental_cache/`；目录已有结果时另建`rerun_*`子目录，汇总时将`--data`指向该目录。18个新输入覆盖4/6/8宝藏、三类地图和每类两个种子，比较原版与同内核无缓存、完整分数键、增量键。计时分别测博弈阶段和“完整有序距离矩阵＋博弈结果/PV”，包含建表、查表和排序成本。四宝藏是预先指定的非终止状态无复用负对照；六、八宝藏属于独立自设拓展，不替代正式四宝藏解答。

具体实现、正确性证明、错误缓存反例和全部损失见[收益增量缓存研究](INCREMENTAL_CACHE_STUDY.md)。`summarize_incremental_cache.py`只用标准库，从原始CSV独立复算中位数、配对差、PV账本和跨图汇总，再生成SVG，不重新计时。

`supplement_manifest.json`中的文档指纹对应历史版本；收益增量缓存实验的协议、源码和结果指纹见`results_incremental_cache/manifest.json`。各结果目录保留对应实验时的记录。
