# A549共同72小时终点：顺铂暴露时长的有限挑战

日期：2026-10-10。**这里已有真实共同观察终点下的两个暴露方案，不再只是“不同采样时点”的替代比较。** 标签是发表图上的相对cleaved caspase-3信号，不能解释成Annexin阳性比例，也没有接入旧G头。

原文：Hastings et al., *Analysis of pulsed cisplatin signalling dynamics identifies effectors of resistance in lung adenocarcinoma*，eLife 2020，[DOI](https://doi.org/10.7554/eLife.53367)。主代理亲读[原XML](../pulse/PMC7282820.xml)的s2-1、s4-2、s4-5、s4-6，以及Fig1 supplement1/2；查看[处理时间原图](../pulse/elife-53367-fig1-figsupp1-v1.jpg)与[蛋白读数原图](../pulse/elife-53367-fig1-figsupp2-v1.jpg)。

| 字段 | 短时脉冲 | 持续暴露 |
|---|---|---|
| 细胞背景 | A549 | A549 |
| 药物与源浓度 | cisplatin 5 µg/mL | cisplatin 5 µg/mL |
| 开始处理 | 0 h | 0 h |
| 暴露结束 | 2 h，原图明确去药 | 72 h |
| 去药后观察 | 70 h | 0 h |
| 共同观察终点 | 72 h | 72 h |
| 图示均值数字化估计 | 约1.27 | 约2.91 |
| 本次像素读图包络 | 1.2025–1.3361 | 2.8448–2.9784 |

两行的观测ID、源单位、坐标和协议见[observations.csv](observations.csv)。小数位保留数值复算信息，不表示原实验有四位小数精度。独立目视范围分别为1.10–1.40、2.75–3.10，均包含本次结果。两者范围均仅描述读图精度，**不是作者SD、置信区间或生物学变异**。原图报告n=3、mean±SD；本包未取得这两个方案的逐重复值，也未数字化其SD。

## 可使用范围

可以对冻结模型在该来源、剂量、两种时长下的相对蛋白信号预测计算误差，并检验能否复现已发表均值的顺序。两条观测提供一个局部、可证伪的时长挑战，不能证明方法在新药、新剂量或其他细胞状态中的普遍性能。

这是回顾性资料；图像和结果已在研究设计阶段阅读。不能事后改称新的盲测，不从两个均值计算新的生物学显著性。若选择目标是Annexin阳性比例，本包不能作为该目标的现成标签；若使用这个相对蛋白读出作辅助目标，必须明确另一个终点定义。

原图确认脉冲去药，但未在本次所读文字中量化洗涤次数及残留清除。图的线性relative坐标保留原样；Methods s4-5的median-0h归一化和log用于后续数据/PCA处理，未把该log操作擅自套给本图。图中每点精确归一化分母尚未独立恢复。

论文的[作者蛋白源表](../pulse/elife-53367-fig1-data1-v1.xlsx)对应Fig1D的脉冲面板，不能把它说成同时提供了上述两方案的原始数值。两类数据分开登记。该研究没有给本包配套RNA测量；FUCCI资料另见[可得性核验](../pulse/Hastings_FUCCI_availability.md)。

## 复算与绑定

[提取脚本](../../../../../../scripts/extract_hastings_duration_challenge.py)使用原始3981×5264图像，按颜色区分72h红圆点/灰方块，以填充标记中心而非误差棒顶端读取。线性标尺与±4像素中心、±2像素坐标轴包络记录在[calibration.json](calibration.json)。输出后不覆盖已有版本。

[manifest.json](manifest.json)绑定5项输入和3项输出，包括两个原图、PDF、XML及脚本；[contract.json](contract.json)记录共同时间原点、两个暴露窗与尚未支持的结论。代码和数据准备不等于模型已训练；`formal_training_ready=false`。
