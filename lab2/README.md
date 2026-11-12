# 实验二 搜索算法与博弈

[五页实验报告](lab2_张凌琨_10241900443.pdf) · [实验与数据索引](LAB2_附录导航.md)

## 运行老师的样例

需要 Python 3.9 或更新版本，使用标准库。在 `lab2` 目录运行：

```sh
python 题一/maze_astar.py
python 题二/weighted_astar.py
python 题三/alphabeta_tree.py
python 题四/treasure_game.py
```

| 题目 | 样例输出 | 输入与参数 |
| --- | --- | --- |
| 迷宫寻路 | 最短路径 7 步，返回含起终点的坐标 | [样例 TXT](题一/samples/pdf_sample.txt) · [题一说明](题一/README.md) |
| 加权地形 | 最小费用 7 | [样例 TXT](题二/samples/pdf_sample.txt) · [题二说明](题二/README.md) |
| 极大极小博弈树 | 根节点得分 5 | [样例 JSON](题三/samples/pdf_sample.json) · [题三说明](题三/README.md) |
| 宝藏分配 | 6×6 距离矩阵。深度 2 选 T2，收益差 −1。深度 4 选 T3，收益差 3 | [样例 JSON](题四/samples/pdf_sample.json) · [题四说明](题四/README.md) |

自定义输入可照样例写，在命令末尾加 `--input 文件路径`。题二的 `S`、`E` 可以照题面省略引号。四题均可用 `--details` 查看过程和统计。Windows 也可双击各题目录的 `run_q1.cmd` 至 `run_q4.cmd`。

## 测试与实验数据

进入任一题目录运行 `python -B -m unittest discover -v`。四题共 239 项测试。

各题说明列出了实验命令。对应的 `results*` 目录保存输入、参数和计时记录。复现某个对照时，先看该组的 `protocol.json` 或 `reproducibility.json`，确认哪些步骤计入耗时。报告中的耗时来自 Windows 11 和 Python 3.12.14。

重跑时将结果写到 `lab2/reruns/` 下。输出参数用 `--output-dir` 或 `--output`，可用脚本的 `--help` 查看。在 `lab2` 目录可以这样重跑题一的基础对照：

```sh
python 题一/run_experiments.py --output-dir reruns/maze
```

题二后续更新了命令行输入解析。核对早期结果记录中的源码指纹时，请取用[当时的求解器](https://github.com/lingkun-Zhang/contemporary-ai-lab/blob/da202e1ce8bb497fb32f8098db3235d957ec7cba/lab2/%E9%A2%98%E4%BA%8C/weighted_astar.py)。
