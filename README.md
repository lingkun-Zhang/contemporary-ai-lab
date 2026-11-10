# 当代人工智能Lab

- [Lab 1：文本分类](lab1/README.md)
- [Lab 2：搜索算法与博弈](lab2/README.md)

## Lab 1 材料

使用TF-IDF比较逻辑回归、MLP和线性SVM，保留有效与未奏效的尝试。原29组参数对照与后来补做的三组清洗实验分开记录，没有只保留最好的结果。

- [两页实验报告](lab1/report/report.pdf) · [可编辑正文](lab1/report/report.md)
- [运行说明与代码入口](lab1/README.md)
- [原29组结果汇总](lab1/results/全部实验结果.csv) · [完整参数](lab1/results/全部实验参数.csv)
- [训练记录与逐样本验证预测](lab1/results/reproduction_20261014) · [对照图](lab1/figures)
- [MLP每20轮摘要](lab1/results/MLP每20轮摘要.md)
- [邮件头与签名清洗：三组结果、规则及双向错分](lab1/results/header_cleaning_20261015/README.md)
- [最终测试预测](lab1/final_outputs/prediction.csv) · [全量重训记录](lab1/final_outputs/final_training.json)
- [图表与摘要的生成说明](lab1/notes/postprocessing.md)

代码、可公开的指标记录、图像、报告和预测均在仓库中。老师提供的原始数据、含完整文本的记录、私有模型和本地备份不公开；取得课程数据后可按Lab 1说明运行。

## Lab 2 材料

四道题分别实现单位费用A*、加权地形A*、Minimax与Alpha-Beta，以及结合寻路的双人宝藏博弈。代码只使用Python标准库，包含正式样例、237项测试、固定种子输入、原始计时记录和专题分析。

- [五页实验报告](lab2/lab2_张凌琨_10241900443.pdf) · [可编辑版本](lab2/lab2_张凌琨_10241900443.docx)
- [运行与输入格式](lab2/README.md) · [全部实验与数据导航](lab2/LAB2_附录导航.md)
- [走廊压缩](lab2/题一/CORRIDOR_STUDY.md) · [免费终点与地形下界](lab2/题二/REPORT.md)
- [Scout探测与重搜](lab2/题三/SCOUT_STUDY.md) · [收益增量缓存](lab2/题四/INCREMENTAL_CACHE_STUDY.md)

实验同时保留未提速的输入和方法，区分搜索节点数与包含准备工作的总耗时。随机输入、参数和计时重复分别统计；课程要求PDF不随仓库发布。
