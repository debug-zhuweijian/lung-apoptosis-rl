# A549 暴露时长与共同观察终点：第五轮定向来源复核

核验日期：2026-10-10。责任范围只含本目录；未改已有冻结包、模型、训练代码或全局结论。主代理另在兄弟目录 `cleaved_caspase_challenge_v1/` 登记 Hastings 图示均值，本目录不重复生成该图的正式数字化结果。

## 本轮结论

**已核实一个核心来源支持 A549 在共同 72 h 终点比较不同顺铂暴露时长。** Hastings 2020 使用同一浓度的 2 h 脉冲与持续暴露，公开图给出相对 cleaved-Caspase-3 信号。它补上“暴露时长选择有实测凋亡相关比较”的一个有限实例；不能被扩展为多状态、多药、任意时长或 Annexin 比例的完整监督。

第二个来源 Hirai 2020 有 2/4/8 h 七氟烷暴露、统一结束后 24 h 的 Annexin-V 相对荧光结果。其图示为**错开给药起点、共同结束给药及观察**，因此应定义为暴露窗口/等待加暴露的动作，而不能记成相同初始时刻立即给药、只改变停止时刻。它属于麻醉剂的补充实验，未自动纳入课题当前抗癌药动作域。

没有把“未找到更多来源”写成不存在。全文排除了一条与药物撤除无关的放射性标记 pulse-chase 线索，详见检索记录。

## 1. Hastings et al.：顺铂 2 h 脉冲与持续暴露

- 标题：Analysis of pulsed cisplatin signalling dynamics identifies effectors of resistance in lung adenocarcinoma。
- 版本：eLife 2020，官方 API version 1，DOI [10.7554/eLife.53367](https://doi.org/10.7554/eLife.53367)，[PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7282820/)。
- 主原件：[PMC7282820.xml](PMC7282820.xml)、[官方 PDF](elife-53367-v1.pdf)、[官方 API 完整记录](eLife53367_api_record.json)。
- 时间图：[Figure 1—figure supplement 1A](elife-53367-fig1-figsupp1-v1.jpg)，2660 × 4315 像素。
- 结局图：[Figure 1—figure supplement 2](elife-53367-fig1-figsupp2-v1.jpg)，3981 × 5264 像素，末行右侧 Relative Cleaved Caspase 3。

### 可核验的条件

| 字段 | 实际证据与登记 |
|---|---|
| 细胞 | A549，正文 `s2-1`、补图标题/图注 |
| 细胞培养 | Methods `s4-2`：Advanced RPMI，1% FCS，1% GlutaMAX，5% CO2、20% O2 |
| 剂量 | 两臂均 cisplatin 5 µg/mL；保留原单位，不额外推定摩尔浓度 |
| 脉冲 | Fig1-supp1A：t=0 开始，t=2 h 去药，t=72 h 观察 |
| 连续 | 同图：t=0 开始，持续至 t=72 h 观察 |
| 共同终点 | 以开始处理为 t=0，T*=72 h；脉冲后恢复 70 h。依据图示，不把部分正文“following a pulse”的宽泛表述机械换成 T=74 h |
| 去药细节 | drug removed 有图示；换液/洗涤次数、洗液、残留药清除程度未说明，登记 unknown。Methods 的 PBS 清洗为固定/抗体染色步骤，不能借用为去药流程 |
| 凋亡相关 Y | Fig1-supp2 相对 cleaved-Caspase-3，多重蛋白检测；图中 0/24/48/72 h，有红色 pulse 与灰色 continuous 两条曲线 |
| 作者重复 | 图注 n=3，mean±SD；透明报告称除特列实验外为 biological triplicates，此 multiplex 比较不在例外名单 |
| 观察层级 | 公开发表均值图；原始三次独立测量未在已核实附件中提供 |
| 单位/分母 | y 轴为 Relative Cleaved Caspase 3。基线图示约1；分母的精确定义未由该图注明，不能自行写成 Annexin fraction 或细胞死亡百分比 |
| 变换 | Methods `s4-5` 中 median 0 h 归一化与 log 用于后续 PCA 数据分析；不能因此断言这张线性 relative 图已经取 log，或已确认它的具体分母 |

**独立读图交叉核验（不替代主代理正式数字化）：**72 h 灰色 marker 约 2.9，保守位置读图区间 [2.75, 3.10]；红色 marker 约 1.25，区间 [1.10, 1.40]。这些区间仅表示图像读取容差，**不是 SD、SE 或 CI**。不据此生成原始重复或显著性检验。

### 官方 source data 实际包含什么

[elife-53367-fig1-data1-v1.xlsx](elife-53367-fig1-data1-v1.xlsx) 已成功下载并读取。官方透明报告最后一节明确将 source data 对应到 **Figure 1D（供 Figures 2/3 分析）**。Raw Means 与 Standard Deviations 各有六个细胞系、每系 12 个时点；没有标明连续暴露的额外 A549 实验臂。其余 Normalised to median、Log transform、Z scores 三表含整理/转换数据，不能仅凭 sheet 名称把所有列都认作完成相应变换。

- A549 在 Raw Means / Standard Deviations 的第 3–14 行；Time 为 0、2、4、6、8、12、16、20、24、36、48、72 h。
- 原始列 C:BC 共 53 个通道标题，含重复蛋白/位点来源与缺失的 ACTIN；不是 53 个独立转录基因。
- A549 凋亡通道：V=Active Caspase-9，X=Active Caspase-8，AH=Active Caspase 3，BB=Cleaved PARP。
- 通过 [extract_hastings_source.py](extract_hastings_source.py) 原样导出均值、SD、精确 Excel 坐标及缺失状态，不进行归一化。
- [Hastings_A549_pulse_proteins_author_summary.csv](Hastings_A549_pulse_proteins_author_summary.csv)：636 条（含缺失）；[凋亡相关四通道](Hastings_A549_pulse_apoptosis_channels_author_summary.csv)：48 条。
- 原始通道单位未在工作簿明确印出，登记 `not_explicit_in_workbook`；不凭 MAGPIX 名称推成绝对蛋白浓度。

**不允许的拼接：**用这组 Figure1D pulse 数值替代 Figure1-supp2 中的 pulse 臂，再与补图 continuous 臂比较。二者的结果表和实验块身份未得到证明，且可见数值并不相同。

### 状态与分子观测的边界

同研究 pulse 蛋白面板含 t=0 群体蛋白读数及药后时程，可成为该脉冲实验的低维分子参考。它不能证明 duration 对照的两个实验臂有逐个样本配套的给药前 RNA，也不能替代课题完整 RNA 表示的迁移核验。

在已读取的正文、官方 API dataSets/additionalFiles、Source Data 和相关补充中，未核实到本实验配套 RNA-seq 或转录组沉积。表述限于已审查范围。不同细胞系的六条 pulse 时程也不能改名为 A549 的六种状态。

Methods `s4-10` 另有先拍给药前 FUCCI 细胞周期状态、再逐细胞跟踪的实验，但它对应 cisplatin/dactolisib 的独立图组；不把该单细胞身份转移到此处群体蛋白 pulse/continuous 表。

主代理后续要求的同细胞轨迹载体复核已另存 [Hastings_FUCCI_availability.md](Hastings_FUCCI_availability.md)：官方 Fig5 和 A549 p53-knockdown 补图已下载，保留逐细胞图形证据；官方清单没有列出逐细胞数表或原始影片。该补查不改变本节 duration 对照的实验块身份。

### 不能混淆的指标

| 来源 | 实际测量 | 本轮用途 |
|---|---|---|
| Fig1-supp2 | 相对 cleaved-Caspase-3，多重蛋白读数 | 共同终点时长比较的凋亡相关 Y |
| Fig1-supp1B | 活细胞 PI uptake | 膜完整性/死亡辅助读数，不能单独等同特异性凋亡 |
| Fig1B、Methods s4-3 | 固定细胞 PI DNA 含量的 sub-G1 群体 | 与活细胞 PI uptake 不同；该图为 pulse 跨系比较 |
| Methods s4-6 / Fig3 等 | NucView 488 caspase-3 底物 | 另一个活细胞凋亡相关读数；不强行赋给 Fig1-supp1B |
| Fig1-supp1C | MTS 增殖/代谢 | 不能作为凋亡率监督 |

## 2. Hirai et al.：统一撤药时间的 2/4/8 h 麻醉剂窗口

- 标题：Differential effects of sevoflurane on the growth and apoptosis of human cancer cell lines。
- DOI：[10.1007/s00540-019-02701-w](https://doi.org/10.1007/s00540-019-02701-w)，2019-10-31 online，Journal of Anesthesia 34:47–57（2020）。
- 主原件：[名古屋大学机构库作者稿](Hirai2020_Sevoflurane_author_manuscript.pdf)，官方 URL 及 SHA256 见 [下载台账](Hirai_download_log.json)。
- 关键定位：PDF p6–9（细胞培养、暴露、Annexin/7-AAD）；p13–14（结果）；p23（Fig3图注）；p28（Fig3原图）。[p28渲染图](Hirai_Figure3_page28.png)。
- 出版商更正记录仅说明作者撤回 Open Choice、版权变化；已读记录未提示本次时间图或结果的生物学更正。没有将 ResearchGate 展示的旧许可当成当前出版许可。

### 协议核验

A549 及非 MCF10A 细胞使用 DMEM +5% FBS +1% penicillin-streptomycin，37°C、5% CO2。Annexin 实验接种 8×10^4 cells/well（6-well）。暴露条件为 1% v/v sevoflurane，5% CO2 in air，流速 0.5 L/min；结束后转入正常培养箱，继续培养 24 h。

**Fig3a 对时间原点很重要：**三种灰色暴露块在同一时刻结束；2 h、4 h、8 h 的开始时刻错开。相对最早暴露起点，可重表达为：

| 方案 | 给药开始 h | 给药结束 h | 暴露 h | 撤药后恢复 h | 观察 h |
|---|---:|---:|---:|---:|---:|
| 8 h | 0 | 8 | 8 | 24 | 32 |
| 4 h | 4 | 8 | 4 | 24 | 32 |
| 2 h | 6 | 8 | 2 | 24 | 32 |

以上坐标是**本次依 Fig3a 的等价重表达**，不是作者逐组文字列出的绝对时间。作者 Methods 的泛称“接种24h后处理”不足以覆盖错开图示的每一臂，故原图与文字一起保留，不改成同时开始。

### 结果与用途

- Fig3b 轴为 relative intensity (Annexin V)，图注为 Annexin-V fluorescence 相对未暴露对照，mean±SD，n=4；Methods 说 n 为独立 assay 次数。
- 正文 p14 将 A549 增幅描述为 4.1 fold，并使用 ratio of apoptotic cells 字样；保守按图表的相对荧光指标登记，不反推出绝对阳性细胞百分比。4 h 与 8 h 仅图示均值，若计算应明确数字化。
- Fig3c 是代表性单通道直方图；没有公开逐重复 FCS 或数表得到核实。Methods 另测7-AAD，不意味着此图已经给出 Annexin×7-AAD 四象限比例。
- Fig4 为 cleaved-Caspase-3 WB，代表性条带不能直接作为有均值/SD的定量凋亡率。
- 该来源只有 A549 的一个常规背景，缺少对应初始转录组。七氟烷属于挥发麻醉剂、浓度为气相 v/v；不能与常规 µM 抗癌小分子剂量直接混合。
- 可支持“等待后暴露不同窗口、共同观察时刻”的有限实测挑战；不能独立证明现有状态模型/RL的收益。

## 3. 重要排除：药物 pulse 与放射性 pulse 不是一回事

[PMC8131322.xml](PMC8131322.xml)：Early nucleolar responses differentiate mechanisms of cell death induced by oxaliplatin and cisplatin。全文 Methods 的 Pulse-chase 指 ^32P orthophosphate 放射性 RNA 标记，chase 又回到含药培养基；Figure2/3 衡量 rRNA/NPM1。它不是不同抗癌药物暴露时长在共同终点的 Annexin/Caspase 挑战，不能纳入本轮已通过来源。

## 文件验证及未完成项

- 所有新内容限定在本目录；下载台账记录 URL、字节数与 SHA256。
- 官方 Hastings API 仅枚举一个 source-data XLSX；相关 supplement 1/2/3 与 transparent reporting form 均已读取。supp4/5 为患者/另一细胞面板，未为本轮问题扩大下载。
- 尚缺 Hastings pulse-vs-continuous 原始重复数表、明确洗液流程及该图 relative 分母；现有发表均值仍可形成有限挑战，须保留这些界限。
- 尚缺上述 duration 挑战对应的多 A549 初始状态与 RNA 表示。发现有限 duration 数据不等于完整数据对齐目标已经完成。

