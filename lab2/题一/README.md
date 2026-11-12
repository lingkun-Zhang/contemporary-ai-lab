# 题一：使用 A* 进行迷宫寻路

本目录完成《实验二要求_new(1).pdf》的题一：在二维迷宫中沿上、下、左、右四个方向寻找从起点到终点的最短路径。坐标采用从 0 开始的 `(行, 列)`，`S` 表示起点，`E` 表示终点，`.` 表示可通行格，`X` 表示障碍。每次移动的代价为 1。返回的路径包含起点和终点；无路可走时返回 `[]`。

样例路径、扩展节点数与 BFS 核验见 [题一报告](REPORT.md)。障碍密度、同分排序和地标实验见 [拓展实验](EXPLORATION.md)，BFS、节点重开和连通性预检的结果见 [补充实验](BONUS_REPORT.md)。

## 1. 环境与目录

需要 Python 3.9 或更新版本，只用标准库。以下命令均在本目录执行。Windows 可以双击 `run_q1.cmd` 运行样例。

```text
maze_astar.py          A* 算法、题目接口与命令行入口
run_q1.cmd             Windows 运行入口，可双击运行样例
run_experiments.py     对照实验与 CSV、SVG 生成
run_explorations.py    五组拓展实验、统计汇总与 SVG 生成
run_bonus_experiments.py 四组成本与正确性研究的入口
run_landmark_tuning.py 地标数整批调参，原40查询复测与重复计时
run_landmark_confirmation.py 固定k1与0/2的独立计时确认，先写协议
test_landmark_confirmation.py 原输入、协议不可覆盖与固定比较口径
bfs_baseline.py        普通 BFS 与同内核匹配堆对照
reopen_probe.py        节点重开正确性及故意错误的冻结消融
connectivity_index.py  静态快照的连通分量预检
run_*_study.py         BFS、重开和连通预检的实验脚本
run_evidence_audit.py  按地图配对的 bootstrap 证据检查
landmark_heuristic.py  固定地标选择、BFS 距离表和障碍感知下界
test_maze_astar.py     最短路、边界条件及随机场景验证
test_landmark_heuristic.py 地标下界、适用条件及随机查询验证
samples/
  pdf_sample.txt      PDF 样例输入
results/              运行对照实验后生成的数据和图
results/exploration/   拓展实验的原始数据、汇总、地图与图表
results/bonus_*/       四组成本与正确性研究的数据和图
README.md             运行与复现说明
REPORT.md             题一报告内容及实测结果分析
EXPLORATION.md        五组研究问题、控制变量、数据和适用范围
BONUS_REPORT.md       成本与正确性研究的主要结论及导航
*_STUDY.md            三组补充研究的完整分析
EVIDENCE_AUDIT.md     统计审计方法、区间与证据限制
LANDMARK_TUNING.md   真正整批地标数调参、局部低点与波动分析
LANDMARK_CONFIRMATION.md 固定候选的新计时确认与判据说明
```

## 2. 运行题目样例

在 PowerShell 中运行：

```powershell
.\run_q1.cmd
.\run_q1.cmd --input samples/pdf_sample.txt --details
```

若在已安装 Python 的其他电脑上运行，可使用以下普通命令。

直接运行，以 PDF 样例作为输入，标准输出仅包含路径：

```powershell
python maze_astar.py
```

从文件读取同一样例：

```powershell
python maze_astar.py --input samples/pdf_sample.txt
```

在 PowerShell 中也可以通过标准输入传入：

```powershell
Get-Content -Raw samples/pdf_sample.txt | python maze_astar.py --stdin
```

输入格式依次为：起点一行、终点一行、二维迷宫列表。迷宫列表可以跨多行。例如：

```text
(0, 0)
(2, 3)
[
    ['S', '.', 'X', '.'],
    ['X', '.', 'X', '.'],
    ['.', '.', '.', 'E']
]
```

该例的一条最短路径为：

```text
[(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)]
```

路径坐标数为 6，实际移动步数为 `6 - 1 = 5`。多条最短路径同时存在时，输出其中一条即可，路径长度仍应最短。

## 3. 在 Python 中调用

题目接口：

```python
from maze_astar import astar

grid = [['S', '.', 'X', '.'], ['X', '.', 'X', '.'], ['.', '.', '.', 'E']]
path = astar((0, 0), (2, 3), grid)
print(path)
```

若需要统计信息，可使用扩展接口：

```python
from maze_astar import astar_search

result = astar_search(
    (0, 0), (2, 3), grid,
    heuristic="manhattan",  # 或 zero / chebyshev / euclidean
    tie_break="fifo",       # 或 lifo / low_h
    record_trace=True,
)
print(result.path)
print(result.cost)
print(result.stats)
```

`SearchResult` 中的 `path` 是路径，`cost` 是移动步数，`stats` 是统计字典。仅在 `record_trace=True` 时保存 `trace` 和 `expanded_order`，用于检查搜索过程。空地图、端点为障碍或合法但不可达的地图返回空路径；起终点相同且可通行时返回只含该点的路径。非矩形地图、非法字符和越界坐标抛出 `ValueError`。

扩展接口也支持 `heuristic(node, goal)` 可调用对象，以及四个基本方向的任意排列 `neighbor_order`。调用者需保证自定义启发函数返回有限非负数且具有可采纳性、一致性；接口不会自动证明这些性质。下例使用本项目已证明适用的固定地标下界，索引可在同一张静态地图的多次查询间复用：

```python
from landmark_heuristic import build_landmark_index

index = build_landmark_index(grid, max_landmarks=4)
result = astar_search((0, 0), (2, 3), grid,
                      heuristic=index.heuristic(4), tie_break="low_h")
print(index.metadata())  # 含地标、预处理耗时与距离表条目数
```

地标预处理仅依赖地图，改变障碍后必须重建。它适用于本题的无向、单位代价图；不能直接将距离绝对差公式迁移到第二题的特殊进入代价。

起点、终点以传入的坐标为准，`S`、`E` 在通行判断中与 `.` 相同。详细接口在无解时的 `cost` 为 `-1`；题目要求的 `astar` 接口仍只返回 `[]`。

## 4. 查看详细统计

默认只输出路径，加入 `--details` 后输出包含路径、代价和统计信息的 JSON：

```powershell
python maze_astar.py --input samples/pdf_sample.txt --details
python maze_astar.py --input samples/pdf_sample.txt --heuristic zero --details
python maze_astar.py --input samples/pdf_sample.txt --heuristic euclidean --tie-break low_h --details
python maze_astar.py --input samples/pdf_sample.txt --heuristic manhattan --tie-break low_h --details
```

三种对照方法为：

| 方法 | 优先顺序 | 作用 |
|---|---|---|
| `zero + fifo` | 先比较 `g`，相同则按入队顺序 | 不使用到终点的距离信息，作为正确性与节点数对照 |
| `manhattan + fifo` | 先比较 `g+h`，相同则按入队顺序 | 题一的基础 A* 实现 |
| `manhattan + low_h` | 先比较 `g+h`，再比较 `h`，仍相同则按入队顺序 | 检查同 `f` 节点的处理方式是否影响扩展数 |

`low_h` 始终把 `f` 放在第一优先级，只在 `f` 相同的情况下优先选择离终点更近的节点；不能把它理解成只比较 `h` 的贪心搜索。

## 5. 复现对照实验与验证

以下命令把复跑数据写到 `../reruns/`。基础实验、拓展实验和地标扫描默认会写入仓库已有的结果目录，因此复跑时请保留命令中的输出参数。

```powershell
python -B -m unittest discover -v
python run_experiments.py --repeats 7 --batch 10 --seed 20261022 --output-dir ../reruns/q1-base
python run_explorations.py --output-dir ../reruns/q1-exploration
```

63 项测试包含独立 BFS 核验、路径合法性、地标下界、计时协议和走廊压缩。用固定种子生成的地图还覆盖不同启发函数、同分规则和邻居顺序。

复现四组成本与正确性研究：

```powershell
python run_bfs_study.py --output-dir ../reruns/q1-bfs
python run_reopen_study.py --output-dir ../reruns/q1-reopen
python run_connectivity_study.py --output-dir ../reruns/q1-connectivity
python run_evidence_audit.py --output-dir ../reruns/q1-audit
```

最后一条读取仓库已有的实验结果重新汇总。方法和数据位置见 [补充实验](BONUS_REPORT.md)。

复现地标参数的整批重复测量：

```powershell
python run_landmark_tuning.py --landmarks 0 1 2 3 4 8 --queries 40 --repeats 9 --seed 20261029 --output-dir ../reruns/q1-landmark-tuning
```

仓库数据在 `results/landmark_tuning/`。调参脚本读取 `results/exploration/reproducibility.json`，复用两张地图的原查询。实验设置和计时见 [地标数量扫描](LANDMARK_TUNING.md)。

长墙在本批已测集合中 k=1 成本最低，九个重复块都快于 k=0 与 k=2；完美迷宫 k=1/2 区间重叠、配对排名波动，报告为近似平台；开阔图仍以不建表的 k=0 最低。新总值直接计完整批次，不能与原逐查询中位数求和估计混用或声称通用最佳 k。

仓库中的基础实验结果位于 `results/`：

| 文件 | 内容 |
|---|---|
| `comparison.csv` | 14 张地图 × 3 种设置的路径步数、节点数、堆峰值和计时原始数据 |
| `reproducibility.json` | Python 与系统信息、实验参数、全部地图、各方法路径、源码 SHA256 |
| `sample_trace.csv` | PDF 样例有效出队顺序及每一步的 `g/h/f`，含最后取出的目标 |
| `sample_search_comparison.svg` | PDF 样例的路径与扩展范围 |
| `open40_search_comparison.svg` | 开阔 40×40 地图的路径与扩展范围 |
| `detour_search_comparison.svg` | 强制绕路场景的路径与扩展范围 |
| `expanded_nodes_comparison.svg` | 全部场景的扩展节点数对照图 |

CSV 带 UTF-8 BOM，可直接用 Excel 打开；SVG 可用浏览器查看。具体实验参数和实测分析见 `REPORT.md`。

拓展实验将输入地图、逐次结果、分层汇总、选取的地标与源码哈希保存在 `results/exploration/`。默认参数为 `--seeds 20 --repeats 3 --batch 3 --seed 20261022`。主规模组为 300 张地图、1200 个配置；有解与无解分开统计。其余组包括 51×51 结构地图、相同障碍数的缺口移动、24 种邻居排列和 1/2/4 地标、同地图 40 次独立查询。探索脚本需比样例和基础实验运行更久，详细方法和数据索引见 [EXPLORATION.md](EXPLORATION.md)。

本次 PDF 样例得到 7 步最短路径，三种设置的扩展数依次为 `15 / 13 / 7`；开阔 40×40 地图为 `1599 / 1599 / 78`。这说明曼哈顿距离与同 `f` 处理需要分别分析，不能只根据算法名称判断效率。

统计口径如下：

| 字段 | 含义 |
|---|---|
| `expanded_nodes` | 实际展开邻居的次数，不包含取出后直接结束的目标节点 |
| `generated_nodes` | 入堆记录数，包括起点及每次成功松弛产生的新记录；可能重复计同一位置 |
| `popped_nodes` | 弹出的堆记录总数，包含目标与过期记录 |
| `max_frontier` | 物理堆中的记录数峰值，包含尚未弹出的过期记录 |
| `stale_pops` | 因已有更小 `g` 而跳过的过期出堆记录数 |
| `reopened_nodes` | 先前已扩展的状态以更小 `g` 再次实际扩展的次数 |
| `unique_reached` | 找到过有效 `g` 值的不同格子数，包含起点 |

同一格子在发现更短路径时可能再次加入堆，旧记录从堆弹出时不应重新扩展。因此，堆操作次数不能直接当作扩展节点数。目标节点在取出后立即结束搜索，不再展开邻居。图中的最终路径与已经搜索的格子分别标注。邻居按固定的右、下、左、上顺序检查，便于复现 PDF 样例路径。

节点数主要反映本算法的搜索工作量；小迷宫上的时间容易受运行环境影响。每个场景的每种设置先预热 1 次，再运行 7 批、每批 10 次搜索，记录各批平均单次耗时的中位数与最小值，单位为毫秒。计时调用关闭追踪记录，包含输入校验和少量循环开销，不包含 BFS 核验、地图生成、绘图和文件读写；垃圾回收保持解释器默认设置。每次输入校验都会扫描整张地图，因此扩展数下降不意味着耗时会按同比例下降。

## 6. 实现依据

题目规则和验收要求以 PDF 为准。本代码独立实现，网上资料只用于理解算法与设计对照实验：

- [Berkeley CS188：Informed Search](https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html)
- [Red Blob Games：A* Implementation](https://www.redblobgames.com/pathfinding/a-star/implementation.html)
- [Asai 与 Fukunaga（AAAI 2016）：Tiebreaking Strategies for A* Search](https://ojs.aaai.org/index.php/AAAI/article/view/10071)
- [Corrêa 等（IJCAI 2018）：Analyzing Tie-Breaking Strategies for the A* Algorithm](https://www.ijcai.org/proceedings/2018/655)
- [Goldberg 与 Harrelson：Computing the Shortest Path: A* Search Meets Graph Theory](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/soda05.pdf)

第三篇文献用于了解同优先级处理问题，论文重点研究的含零代价动作场景与本题不同；本实验只检查本题单位代价迷宫中的实际变化。

## 7. 固定候选的独立计时确认

运行 `python run_landmark_confirmation.py --output-dir ../reruns/q1-landmark-confirmation`。在三张原地图上各复用 40 个查询，对地标数 0、1、2 重新计时 15 批。事先固定比较 (k=1) 与 (k=0,2)。

仓库结果在 `results/landmark_confirmation/`。长墙上建一个地标的总耗时更低，迷宫中一个和两个地标的差别较小，开阔图则以不建地标更快。逐批数据和比较方法见 [地标复测](LANDMARK_CONFIRMATION.md)。


## 精确走廊压缩与多查询回本

`corridor_astar.py` 把单位代价迷宫中的走廊压缩成边，搜索后还原为逐格路径。地图修改后需重建索引。

```python
from corridor_astar import build_corridor_index
index = build_corridor_index(grid)
result = index.search(start, end, heuristic="manhattan", tie_break="low_h")
print(result.path, result.cost, result.stats)
```

复现完整实验及派生汇总：

```text
python -B run_corridor_study.py --output ../reruns/q1-corridor
python -B summarize_corridor_study.py --data ../reruns/q1-corridor
```

仓库结果在 `results/corridor_study/`。输出目录已有结果时，脚本会生成 `rerun_*` 子目录，汇总时把 `--data` 指向该子目录。完整计时包括建图、接入端点、搜索和还原路径。地图、查询数量与计时结果见 [走廊压缩实验](CORRIDOR_STUDY.md)。

新增13项正确性测试包括全部512种3×3障碍布局及46080次搜索，另有长并行走廊、纯环、内部双端点与输入快照检查。宏点展开数与原网格格点不是同一单位，不能直接当作格子访问或内存节省；同内核identity图用于区分压缩与实现开销，现有k=1地标是强基线。
