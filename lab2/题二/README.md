# 题二：加权地形 A* 最小移动代价

本目录按 PDF 完成题二：四方向移动，从 `u` 进入相邻 `v` 时支付地形权值 `w(v)`；进入终点 E 的最后一步免费。正式接口只返回最小总代价，无解为 `-1`。PDF 样例输出 `7`。

另提供最优路径、逐步费用账本、搜索统计和有证据的启发函数对照。题二的设计、实验数据和深入分析见 [REPORT.md](REPORT.md)。

## 1. 环境与目录

Python 3.9 或更新版本，全部程序只使用标准库。当前 Windows 电脑可直接双击 `run_q2.cmd` 运行样例。

```text
weighted_astar.py       正式接口、A*、权重规范化和安全启发函数
run_q2.cmd              当前 Windows 运行入口
test_weighted_astar.py  独立最短路、小图枚举及启发条件验证
test_supplementary_experiments.py  平台计数、费用阈值与配对条件验证
run_experiments.py      对照、拓展和图表生成
run_supplementary_experiments.py  排序、费用比和可达区域的补充实验
heap_compaction.py     保持有效搜索顺序的可选堆清理策略
run_heap_tuning.py     清理阈值扫描、原求解器参照与IQR
test_heap_compaction.py 清理正确性、记录守恒与参数口径验证
summarize_timing_ranges.py 直接整理已有计时的波动范围
run_heap_confirmation.py 先固定ρ1.5再采集新的确认计时
test_heap_confirmation.py 协议、固定输入与确认解释口径
samples/pdf_sample.txt  PDF 样例输入
results/               原始记录、路径账本、环境与 SVG
results_supplementary/  三组补充对照的原始记录、图表与复现信息
results_heap_tuning/    策略参数、原始批值与既有计时范围
results_heap_confirmation/ 固定候选的新计时、协议与配对差
README.md              运行说明
REPORT.md              题二报告内容及实测分析
HEAP_TUNING.md         堆清理参数的详细方法与实测限制
HEAP_CONFIRMATION.md   原提速信号是否重复的独立计时分析
```

## 2. 运行与输入

在本目录的 PowerShell 中：

```powershell
.\run_q2.cmd
.\run_q2.cmd --input samples/pdf_sample.txt --details
.\run_q2.cmd tests
.\run_q2.cmd experiments
.\run_q2.cmd supplementary
.\run_q2.cmd tuning
```

默认只输出整数答案。`--details` 输出 JSON，含费用、路径、步数、每步进入费、累计费和统计。

已配置 Python 的其他环境：

```powershell
python weighted_astar.py
python weighted_astar.py --input samples/pdf_sample.txt --details
python -m unittest discover -v
python run_experiments.py --seeds 10 --repeats 3 --batch 3 --seed 20261028
python run_supplementary_experiments.py
python run_heap_tuning.py
```

自定义实验参数请直接运行脚本；Windows 入口的研究模式不转发后续参数。

输入依次为起点一行、终点一行、可跨行的二维数组，坐标从 0 开始：

```text
(0, 0)
(0, 2)
[
    ['S', 9, 'E'],
    [1, 1, 1]
]
```

该图最少步数路线走 2 步、收费 9，但绕行 4 步只收费 3，程序应输出 `3`。命令行地图中的 `S`、`E`、`X` 可按题面写成不带引号的标记，也可保留引号；直接调用 Python 函数时传入字符串。

## 3. 明确的模型约定

- 地形支持非负整数和数字字符串，如 `3`、`'3'`。负数、小数、布尔值、非法字符和非矩形地图被拒绝。
- 传入起点的初始费用为 0，传入终点进入费用恒为 0，即使端点用数字表示也相同。
- S/E 是可选展示标记；若存在，必须位于指定端点，不能把其他格的 E 当作免费地形。
- 搜索不生成返回起点的边。非负费用路径的环可以删除，不会增加费用，因此此约定不损失从指定起点到终点的最优解，也避免为 S 再进入时人为规定权值。
- X 是无解实验的障碍拓展，0 权普通格是零权实验拓展；PDF 样例不含这些输入，报告分别解释条件。
- 空图、障碍端点或合法但不可达时，`cost=-1`，路径和费用列表为空；起终点相同且可通行时费用为 0、路径只有该点。越界坐标抛出 `ValueError`。
- 输入地图不被修改；搜索在不可变权重快照上运行。

## 4. Python 接口与安全配置

```python
from weighted_astar import astar_cost, astar_search

grid = [['S', 9, 'E'], [1, 1, 1]]
answer = astar_cost((0, 0), (0, 2), grid)  # 3
result = astar_search((0, 0), (0, 2), grid, record_trace=True)
print(result.path)
print(result.path_costs)        # 含起点 0 和终点 0
print(result.cumulative_costs)
print(result.stats)
```

`astar_cost` 符合 PDF 的整数输出要求。扩展接口返回 `WeightedResult`。

| `heuristic` | 含义 |
|---|---|
| `zero` | h=0，Dijkstra/UCS 基线 |
| `corrected`，默认 | `m×max(Manhattan−1,0)`，m 为非起终点通行格的最小进入费 |
| `row_column` | 校正几何下界与行列最小进入费下界取最大值 |
| `row_column_tight` | 两轴均未对齐时补回一次非终点进入费，避免把 E 的免费进入重复放松 |

`tie_break='fifo'` 为默认；`low_h` 只在 f 相同后优先较小 h。命令行只暴露上述安全配置。原始曼哈顿在实验中标为分析对照，不冒充普遍安全的默认选项。

`min_scope='global'` 为默认，在全图非起终点通行格中求 m。可选 `min_scope='reachable'` 先完整扫描 S 可达区域，仅在该区域的非起终点格中取 m，避免不可达的便宜格拖弱启发。它只改变 m，行列最小值仍按全图计算；扫描不提前判无解，相关开销计入整次查询。其启发保证针对 S 可达子图。返回值中的 `min_scope` 记录选择，`component_size` 记录该模式扫描的格子数，默认模式为 `None`。

```powershell
python weighted_astar.py --heuristic row_column --tie-break low_h --details
python weighted_astar.py --heuristic corrected --tie-break low_h --min-scope reachable --details
Get-Content -Raw samples/pdf_sample.txt | python weighted_astar.py --stdin
```

Python 中可使用 `astar_search(start, end, grid, heuristic='corrected', tie_break='low_h', min_scope='reachable')`。该模式的可达遍历也会扫描与目标无关的区域，不保证比默认更快。

## 5. 正确性验证与统计口径

本次通过46项测试方法，包括15项核心、4项补充口径、9项堆清理、5项独立确认协议及13项整数桶测试。200 张随机图×4 个安全启发×2 个 m 范围×2 个并列策略，共 3200 个最短性配置；48 张小图逐点验证下界、逐边验证一致性，其中正权与零权各 24 张，global 检查全图，reachable 仅检查 S 可达区域；40 个 3×3 图再用简单路径穷举核对。另有 21×21 三条高权整行专例验证 tight 的起点下界为 63，等于独立最优费用，而旧行列下界为 62。参考 Dijkstra 不复用核心的转换与邻居逻辑。反向距离计算使用原边目的格费用，避免把加权方向弄反。

补充口径测试检查平台统计排除终点、无解无数值阈值、K=11 的费用切换、配对图的可达问题不变，以及可达遍历经过 E 后仍继续并保留真实零权。

| 字段 | 含义 |
|---|---|
| `expanded_nodes` | 有效出队后实际生成邻居的次数，不含终点 |
| `generated_nodes` | 成功入堆次数，含起点和改进后重复入堆 |
| `popped_nodes` | 全部弹出次数，含过期记录与终点 |
| `max_frontier` | 物理堆峰值，含尚未弹出的过期项 |
| `stale_pops` | 因已有更小 g 而跳过的旧记录 |
| `reopened_nodes` | 已扩展格以更优 g 再次实际扩展的次数 |
| `unique_reached` | 已发现的不同格子数 |

`trace` 和 `expanded_order` 仅在 `record_trace=True` 时保存。计时关闭轨迹，包含完整输入规范化、共同预处理和路径回溯，不包含参考求解、条件检查、绘图与写文件。

补充排序实验按独立最优费用 C* 将有效的非终点扩展分成 `f<C*`、`f=C*` 和 `f>C*`；三类之和须等于 `expanded_nodes`。无解时不定义 C* 平台。可达区域实验把扫描格子数与 A* 扩展数分开记录，整次查询计时包含扫描，避免用减少的 A* 节点代替总工作量。

## 6. 结果文件

正式对照有 228 条输入记录、200 个不同地图与端点组合，比较 4 个安全配置与原始曼哈顿分析配置，共 1140 次配置结果；912 个安全配置全部达到参考最优值。固定带状场景在无障碍时会跨种子重复，缩放倍数 1 也与样例重复，未将这些记录称为独立地图。

`row_column_tight` 在 31×31 高费行列带上把扩展数从旧行列版本的 811 降至 60，在三条高权整行上从 306 降至 40；m=0 时则退化为旧行列版本，均匀图与随机权重图也存在节点不变、耗时不获益的情况。完整消融、限制和计时见报告。

| 文件 | 用途 |
|---|---|
| `comparison.csv`、`safe_summary.csv` | 所有配置原始结果与安全版本分层汇总 |
| `sample_path_ledger.csv`、`sample_trace.csv` | 样例逐步收费、累计代价与出队轨迹 |
| `heuristic_checks.csv` | 可采纳性、一致性与真实距离差异 |
| `bfs_vs_cost.csv` | 最少步数路线与最小费用路线的区别 |
| `weight_scaling.csv`、`high_cost_gap.csv` | 权重缩放、同数量高权格的缺口位置实验 |
| `outcomes.json`、`reproducibility.json` | 主要结论、完整地图、参数、环境和源码哈希 |
| SVG 图 | 路径费用、BFS 反例、启发函数与受控参数可视化 |

CSV 带 UTF-8 BOM，可在 Excel 中查看；SVG 可在浏览器中打开。所有实验仍遵循四方向规则；分析对照若返回非最优结果，会明确保存费用差异而非当作正确答案。

## 7. 排序、费用与可达区域实验

- 排序交叉对照：同一地图比较 4 种安全启发 × FIFO/low_h，结合最优 f 平台解释节点变化，区分启发增强与同分排序的作用。
- 相对费用比：固定 21×21 地形、高费格数量与位置，只改变高费 K；以 `min(18+K,29)` 预测直行与绕行的切换，并保留 K=11 时的多最优情形。
- 可达区域下界：配对改变隔离格的权值，比较全图 m 与可达区域 m；保留没有节点收益、可达区域已有零权及无解等对照，并比较包含分量扫描的总耗时。

正式补充运行有 95 条输入记录、89 个不同地图与端点组合；排序组 576、费用比组 72、区域组 56，共 704 个配置全部达到独立参考最优费用，352 条启发条件检查全部通过。带状无障碍图跨种子的重复保留为输入记录，不称为独立地图。

补充结果限定了原优化收益：高费带的 tight 在 FIFO 下扩展960个状态，在low_h下60个；K超过11后本构造改走缺口；区域扫描在均匀图恢复精确下界后可获益，在带状图、原本没有不可达零权或可达区域已有零权时也保留了无收益与额外开销。详细数据、证明、反例和本组时间见报告第5.6—5.8节。

运行 `run_q2.cmd supplementary` 可重新生成 `results_supplementary/`。基础实验与补充实验分别保存环境、输入、参数、原始计时和对应源码哈希；两次运行的时间应在各自同轮对照内解释。

| 补充目录文件 | 内容 |
|---|---|
| `tie_break_comparison.csv`、`tie_break_paired.csv` | 完整交叉结果及同输入同启发的排序配对 |
| `cost_ratio.csv` | 9 个 K 值的费用、路线类别、指定带进入次数与搜索量 |
| `component_minimum.csv` | global/reachable 配对、扫描访问数、完整查询和独立预处理计时 |
| `heuristic_checks.csv` | 按对应保证范围逐状态与逐边检查 |
| `outcomes.json`、`reproducibility.json` | 核验结论、完整输入/路径、随机计时顺序和源码哈希 |
| SVG 图 | 排序交互、f 平台、费用阈值、分量开销与阈值两侧路径 |

预处理耗时来自独立 `prepare_terrain` 测量，包含规范化及行列数组，不能用两个中位数相减宣称精确的纯搜索时间。补充计时每批按固定种子打乱配置顺序；重复批次仍不是独立地图样本。

## 8. 堆清理阈值调参

运行 `run_q2.cmd tuning` 或 `python run_heap_tuning.py`，生成 `results_heap_tuning/`。默认扫描ρ=1.05/1.2/1.5/2/4/8/16/32/inf，与原求解器同批参照；9个重复块、每批3次完整查询，顺序随机交错。方法、实测、内部低点的限制和负对照见 [HEAP_TUNING.md](HEAP_TUNING.md)，正式报告的“结论与适用范围”概括了堆清理的取舍。

可选接口为 `from heap_compaction import compacted_astar_search`，调用参数兼容 `astar_search`，末尾增加 `rho=2` 等阈值。它只删除过期记录，保持有效优先级、路径和最优费用；同内核inf不清理，但仍有open维护成本，不能代替原求解器基线。峰值统计包含清理前瞬时堆长，不代表总内存字节。

堆清理的9项测试含3840个随机配置和重开/无解/无过期项/参数边界检查。正式调参70配置全部核验最优费用、路径与有效顺序一致；630个计时批值保存IQR与全范围。阈值效果依赖工作负载，报告同时保留未观察到稳定内部最优及维护成本超过收益的结果。

`previous_timing_ranges.csv` 从旧基础/补充记录直接整理1844组已有计时的中位数、IQR和范围，`previous_timing_sources.json` 保存来源哈希；它不会重跑或覆盖旧实验。可单独运行 `python summarize_timing_ranges.py` 更新该汇总。

## 9. 预定参数的独立计时确认

运行 `run_q2.cmd confirmation` 或 `python run_heap_confirmation.py`。关注参数预先固定为ρ=1.5，对照original、同内核inf和有限左右邻居1.2/2；ρ1.05只作过积极清理的成本对照。四个原组合逐格核对输入，不因新种子重抽地图。首个预热前保存protocol并在结束核对SHA。

首跑结果在`results_heap_confirmation/`，已有结果时另建`run_*`时间戳子目录。24配置/360计时批/1080计时调用均完成，最优费用、路径与有效序列保持。原发现批的提速在本次主比较未重复，严格内部低点也未确认；未改换候选来挑选有利结果。新5项测试检查协议、输入、固定配对、无动作对照和复跑保护。详见[独立确认报告](HEAP_CONFIRMATION.md)，正式报告的“结论与适用范围”概括了复验结果，必答问题的位置见[要求回答与证据核对](../LAB2_要求回答与证据核对.md)。

## 10. 整数桶与公共费用单位

新增可选模块`bucket_astar.py`，正式默认算法保持原样。循环f桶与同内核堆使用相同的安全下界、松弛和完整优先级；`bucket_gcd`仅将桶索引按进入费最大公约数换单位，原始费用账本不变。只支持已证明一致的整数安全下界，超过32768个槽时精确回退堆。

```text
python -B -m unittest test_bucket_astar
python -B run_bucket_study.py
python -B summarize_bucket_study.py --data results_bucket
```

完整记录在`results_bucket/`，包含预热前协议、12张不同基础图、84个费用倍率输入、672配置、4704计时批与9408调用。已有结果时实验另建`rerun_*`子目录；汇总图命令需相应调整`--data`。所有费用、路径、账本及同序轨迹均核验，包含空桶扫描、槽数、回退和新准备工作的完整成本。

零权随机图q=256时，归一化桶相对原始桶时间比的三图中位为0.657，3/3图配对更快；但全部168个输入—排序配对中均未快于原入口。详见[整数桶研究](BUCKET_STUDY.md)，不把表示修正或避免回退包装为普遍提速。
