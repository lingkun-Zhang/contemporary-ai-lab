# 题三：极大极小博弈树与 Alpha-Beta 剪枝

按实验PDF第8—13页实现：根为 MAX，逐层交替 MAX/MIN，默认从左到右访问。正式 JSON 样例输出 `5`。默认接口不使用重排或缓存；另外提供 Minimax 对照、统计、回传轨迹和独立实验。

Python 3.9+，只使用标准库。Windows 可直接双击 `run_q3.cmd`；命令均在本目录执行。

```text
run_q3.cmd
run_q3.cmd --input samples/pdf_sample.json
run_q3.cmd --details
run_q3.cmd --algorithm minimax --details
run_q3.cmd tests
run_q3.cmd experiments
run_q3.cmd caching
```

也可使用 Python：

```text
python alphabeta_tree.py
python alphabeta_tree.py --input samples/pdf_sample.json --details
python -m unittest discover -v
python run_experiments.py
python run_transposition_study.py
```

Windows 上也可以用 `py -3` 代替 `python`，或用 `Q3_PYTHON` 为批处理指定解释器路径。

```python
from alphabeta_tree import TreeNode, alpha_beta, minimax, search

root = TreeNode(children=[TreeNode(3), TreeNode(5)])
assert alpha_beta(root) == 5
assert minimax(root) == 5
result = search(root, trace=True)
print(result.value, result.stats)
```

`alpha_beta(root)`、`minimax(root)`只返回数值；`search()`返回数值、统计和可选轨迹。JSON 接受 PDF 的 `tree_structure` 外层或直接节点，检查可选玩家标记与层次一致。非法叶值、内部节点同时带值、环和不完整的排序回调会报错。公开窄窗口必须满足 `alpha < beta`；默认无限窗口的根结果精确，窄窗口/内部截断返回值可能只是上下界。

文件导航：

| 文件 | 用途 |
|---|---|
| [alphabeta_tree.py](alphabeta_tree.py) | 两种算法、题目接口、结构检查与命令行 |
| [REPORT.md](REPORT.md) | 必答问题、样例回传、实验数据与分析 |
| [test_alphabeta_tree.py](test_alphabeta_tree.py) | 21项测试，含独立迭代真值、随机树及窄窗口界 |
| [run_experiments.py](run_experiments.py) | 318配置、7重复块的配对实验与SVG生成 |
| [draw_sample.py](draw_sample.py) | 样例回传与剪枝示意图生成 |
| [samples/pdf_sample.json](samples/pdf_sample.json) | 完整正式JSON样例 |
| [results/comparison.csv](results/comparison.csv) | 各配置访问计数、完整时间与IQR |
| [results/timing_batches.csv](results/timing_batches.csv) | 每个重复块的原始批平均时间 |
| [results/inputs.json](results/inputs.json) | 所有完整输入，不能把配置数当独立树数 |
| [results/sample_trace.json](results/sample_trace.json) | 正式样例逐步回传、窗口与真实剪枝位置 |
| [results/reproducibility.json](results/reproducibility.json) | 环境、种子、源码及输入SHA、计时边界 |

实验默认7块，每块每配置连续调用3次，保存批平均，再计算7块中位数及IQR。所有树先构建并检查；计时排除输入解析、结构预检、轨迹和文件写入，包含策略准备、排序与采样，另保留不含准备的搜索时间。验证和统计不以唯一对象数替代访问发生次数。

复跑主实验可用 `python run_experiments.py --output-dir ../reruns/q3-main`。单独画样例树可用 `python draw_sample.py --output ../reruns/q3-sample-tree.svg`。

真值优先/逆序是提前完整求解整棵树后的 oracle 参照，其准备工作计入总时间。采样排序只改变访问顺序，额外叶读取单列；本批未观察到相对不排序的时间收益，仍保留该负结果。正式求解接口保持老师规定的左到右访问。

## 界感知缓存及正确性消融

缓存规则和实验分析见[缓存研究报告](TRANSPOSITION_STUDY.md)。`transposition_search.py`提供每次调用新表的`cached_search()`，同内核off、仅精确缓存、界感知缓存均保持左到右；`max_cache_depth`只控制查写层数，不截短搜索。默认正式求解器不变。

```python
from alphabeta_tree import sample_tree
from transposition_search import cached_search
result = cached_search(sample_tree(), mode="bounds", max_cache_depth=None)
assert result.value == 5
```

第一轮缓存实验用同一批展开顺序和叶值，只改变子树共享程度。118 个配置的原始计时在 `results_transposition/`，含无共享树对照和把截断结果误存为精确值的反例。

固定bounds/all后，用新种子的36配置、540批进行独立确认；协议先于预热保存，结果在`results_transposition_confirmation/`。复现：`python run_transposition_confirmation.py`或`run_q3.cmd cache-confirmation`。强共享主比较保留较快方向，纯树均更慢；每组完整配对差及限制见补充报告第8节。


## 低点确认 叶值相关性及有限容量

此前三组实验及适用条件见[SUPPLEMENTARY_EXPERIMENTS.md](SUPPLEMENTARY_EXPERIMENTS.md)：30新树固定q8确认、层级叶值预测质量、LRU有限容量。合计330配置/2250批/6750次计时调用；其中派生输入与计时重复不增加独立输入数。

```text
run_q3.cmd sampling-confirmation
run_q3.cmd correlation
run_q3.cmd capacity
run_q3.cmd extensions
```

最后一条按顺序运行三组，不并行计时。原结果分别在`results_sampling_confirmation/`、`results_correlation_study/`、`results_capacity_study/`；已有输出时复跑另建时间戳子目录。`bounded_transposition.py`是独立LRU模块，原正式算法及早先缓存实现不变。新测试分别在`test_sampling_confirmation.py`、`test_correlation_study.py`、`test_bounded_transposition.py`。

## Scout阈值探测与重搜成本

`scout_search.py` 先对第一个孩子精确求值。后续孩子先通过 TEST 判断能否严格改善当前值，可能改善时才重新求值。

```python
from alphabeta_tree import sample_tree
from scout_search import scout_search
assert scout_search(sample_tree()).value == 5
```

```text
python -B -m unittest test_scout_search
python -B run_scout_study.py
```

题三共有 69 项测试，其中 11 项覆盖 Scout。Scout 实验比较 30 份有序输入、3 种顺序和 3 种实现。结果在 `results_scout/`，记录了排序准备、探测和重搜的时间。

首候选最佳的4份边界输入在左序下，Scout/同模块AB时间比中位为0.733、总读叶相同；后置逐次改善组则为1.980、总读叶比2.900。18随机树左序仅8棵配对更快，不能据此普遍替换正式算法。阶段字段`exact_visits`在AB模式表示AB递归访问，不意味着每次内部返回精确值。详细证明、双基线、失败结果及复核见[SCOUT_STUDY.md](SCOUT_STUDY.md)。
