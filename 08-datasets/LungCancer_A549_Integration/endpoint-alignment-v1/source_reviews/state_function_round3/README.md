# A549 多状态 × 小分子候选：第三轮源数据核验

日期：2026-10-10。写入范围仅本目录；未改前两轮文件、未训练、未联系作者。

## 1. 结论与本轮真正增加的内容

本轮定向读取了 **10 篇新候选的相关原文段落**，核验两套 GEO 元数据并实际下载表达文件，另下载一个官方 Source Data XLSX 和一个流式原图 PDF。**新增了可读的 NSH76 三状态／三种小分子相关流式象限数值，但仍未取得剂量与端点定义均完整、可直接支撑多状态候选排序的公共数值集。** 这是当前已查范围内的结论，不能延伸为“所有公共数据都不存在”。

不将“没有同细胞 t0 RNA”设为否决理由。来源明确的群体状态、平行参考表达、给药前/平行 qPCR 标志都可以是合法的有限观测。本轮 NSH76 的三种派生状态可识别，确有基线 qPCR 图示；其主要未闭合处是**流式单图剂量不完整、重复源值缺失、作者概括的 apoptosis 与 Annexin 右侧象限之和不等价**。完整 RNA 表示是否能迁移仍是另一项未验证问题。

主要交付：

- `NSH76_FigS2_quadrants.csv`：**14 张显示的代表性流式图、56 个印出的象限百分比**，并计算 Q2+Q3 的 Annexin 总阳性；来源是原图直接数字，不是柱高估读。
- `GSE246690_A549_raw_counts.csv.gz`：**60,728 基因 × 16 个 A549 样本**，通过 GSM `description` 与原始工作簿列 ID 精确连接；包含 Pro、doxorubicin、palbociclib、INK128 四种状态。
- `GSE246690_samples.csv`、`GSE56742_samples.csv`：保留状态、来源样本编号及原协议；不自行扩增生物学重复。
- `primary_evidence_sections.md`：NSH76 关键 Methods 原文及图注定位。
- `round3_validation.json`、`source_inventory.json`、`query_log.csv`：读取验证、文件 hash、来源及查询台账。

## 2. NSH76：三状态可识别，但原始端点必须按象限保存

原文：*NSH76: a selective inhibitor of RRN3 and RNA polymerase I transcription with potential for cancer therapy*，[DOI 10.1186/s12967-025-06588-y](https://doi.org/10.1186/s12967-025-06588-y)，[PMC12538875](https://pmc.ncbi.nlm.nih.gov/articles/PMC12538875/)。原文为 `PMC12538875.xml`；原始 Supplementary Fig S2 为 `12967_2025_6588_MOESM4_ESM.pdf`（只有 1 页），可直接核对 `NSH76_Supplementary_FigS2_render.png`。主图为 `NSH76_Figure6.png`。

### 2.1 状态来源及可观测信息

Methods 的 `Sec7`、`Sec9`：A549 来自 **NCCS, Pune, India**，作者自行用 6 个月递增药物建立 A549-CisR 与 A549-DoxR；初始暴露分别为 cisplatin 1 µM、doxorubicin 0.2 µM ×48 h，逐步增加至 10 µM／1 µM。培养为 RPMI-1640、10% heat-inactivated FBS、1% penicillin–streptomycin。

**不是 RCCL A549rCDDP2000。不能与 Sarin 按“都是 A549 耐顺铂”拼成同来源 RNA—功能样本。** 维持药物和再挑战前额外无药洗脱时间未单独写清，保留缺失。

Fig 6f、g 为三种状态相对亲本的 RRN3、47S rRNA qPCR；图注与横轴没有另加 NSH76，故可作为作者报告的基线群体标志参考。Fig 6h 才是 NSH76 后 47S 的变化，**不能把 h 当给药前特征**。原文没有公开这三状态的高维转录组矩阵或逐样本 qPCR 源表；f、g 柱图未印出逐重复数值。本轮不从柱高捏造精确 RNA 特征。

合法的有限状态输入可以是三种来源明确的 `state_id` 加诱导史；它不能证明整个 RNA 编码器/表达表示的跨状态迁移成立。

### 2.2 候选药物、时间与剂量的真实可知程度

Methods `Sec23` 给出 Annexin V/PI 暴露 **24 h**，每个样本 1.5×10^5 细胞，BD Accuri C6 Plus、FlowJo。只有 trypsinization 的收集描述，没有证明将之前脱落到培养液中的所有细胞一并收集。

| 原图 panel | 明确比较的状态 | 小分子候选 | 对本 panel 确定的剂量信息 |
|---|---|---|---|
| S2a / Fig6d | A549、A549-CisR | cisplatin vs control | `Sec10` 列出 5、10、15 µM，但单张流式图未标明究竟使用哪一个 |
| S2b / Fig6e | A549、A549-DoxR | doxorubicin vs control | `Sec10` 列出 0.5、1、1.5 µM，但单张流式图未标明具体值 |
| S2c / Fig6k | A549、A549-CisR、A549-DoxR | NSH76 vs control | `Sec10` 默认说使用 IC50；Fig6i 分别报告 7.38、2.85、2.63 µM，但 S2c 没有直接标剂量。CSV 单列保存 IC50 参考，不伪装为 panel 已印出的浓度 |

这里是**稀疏的状态×候选覆盖**：未测/未显示的 CisR×doxorubicin、DoxR×cisplatin 保持缺失。控制组在 a、b、c 是各自显示的不同流式图，不能任选一张重复分母去覆盖其他 panel。不能因为 viability 图有多个剂量，就说 Annexin 已有相同剂量曲线。

### 2.3 象限定义与源值

原图横轴 Annexin V、纵轴 PI。因此 Q1=PI+/Annexin−，Q2=PI+/Annexin+，Q3=PI−/Annexin+，Q4=双阴性。操作性定义的 **Annexin 总阳性=Q2+Q3**；Q2 可以单独报告为双阳性。Q1 单列，不能直接加入 Annexin 总阳性。

| 药物 panel | 状态 | Q1：PI+/Annexin− % | Q2：双阳性 % | Q3：Annexin+/PI− % | 计算的 Q2+Q3 % |
|---|---|---:|---:|---:|---:|
| cisplatin S2a | A549 | 1.72 | 36.2 | 16.8 | 53.00 |
| cisplatin S2a | A549-CisR | 1.31 | 17.5 | 12.5 | 30.00 |
| doxorubicin S2b | A549 | 19.7 | 38.7 | 14.9 | 53.60 |
| doxorubicin S2b | A549-DoxR | 8.78 | 6.36 | 20.2 | 26.56 |
| NSH76 S2c | A549 | 3.43 | 24.2 | 15.1 | 39.30 |
| NSH76 S2c | A549-CisR | 32.9 | 21.8 | 4.61 | 26.41 |
| NSH76 S2c | A549-DoxR | 35.6 | 23.9 | 3.88 | 27.78 |

所有显示的 control 象限也已录入 CSV，共 14 图。每图四象限之和在印出精度内接近 100%，最大绝对差不超过 0.10 个百分点。

**来源内部需特别警惕的端点差别**：主图 Fig6k 标题为 apoptosis，柱图约为 43%／61%／63%，而 S2c 的 Annexin 右侧总和为 39.30%／26.41%／27.78%。将 Q1 也加入得到约 42.73%／59.31%／63.38%，在形态上更接近主图。这里不能确定作者所有重复的具体门控计算，但可以确定：**S2c 中 Annexin 总阳性不能按主图概括直接写为耐药株更高。** 本轮保留每个象限，不替作者重定义指标，也不因结果方向不同而删数据。

论文说全部实验至少 3 个生物学重复；可取得的 PDF 却只显示每条件一张流式图。不能将一张图生成三个重复，也不能把源图数值当作作者三重复均值。没有取得 FCS 或完整重复表。当前 CSV `training_eligible=false`，用于证据核验和有限案例说明，等待主代理确定可接受的正式使用边界。

## 3. GSE246690：新的四状态表达参考，功能端点未闭合

论文：*H4K20me3-Mediated Repression of Inflammatory Genes Is a Characteristic and Targetable Vulnerability of Persister Cancer Cells*，[PMC7617193](https://pmc.ncbi.nlm.nih.gov/articles/PMC7617193/)，DOI [10.1158/0008-5472.CAN-24-0529](https://doi.org/10.1158/0008-5472.CAN-24-0529)。[GSE246690](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246690) SOFT 与官方 raw-count XLSX 已实际下载。

| A549 状态 | GEO 原始诱导协议 | 采样 |
|---|---|---|
| ctr | untreated/proliferating | 7-day study reference |
| doxo | doxorubicin 100 nM；72 h 后换无药培养基 | 开始处理后 day7 |
| pb | palbociclib **5 µM（A549）**；72 h 后换无药培养基 | day7 |
| ink | INK128 **200 nM（A549）**，换液时补药，约每 3 天换液 | day7 |

不能把 SK-MEL-147 的 palbociclib 1 µM／INK128 100 nM 混用于 A549。矩阵总共 32 个样本、两细胞系；本轮从同一个原文件导出 A549 的 16 列、60,728 个 ENSG 行。每列与 GSM description 精确对应。

每状态有 4 个样本文件，但 A549 的 `2` 与 `2-2` 标题在 GEO 均填写 `replicate: 2`，独立性尚未解释；不直接声称 4 个独立生物学重复。官方 XLSX 的 worksheet dimension 错标为 A1，实际 XML 有完整矩阵；读取时 `reset_dimensions()` 恢复流式遍历，未改原文件。GEO processing 文字出现 `mouse reference genome GRCh38`，与同条目的 Homo sapiens/GRCh38/ENSG 及论文人类样本描述不一致；保留原字符串，不擅改为已核实的准确比对流程。

该论文的 navitoclax 再挑战 48 h 主要报告 **cell viability**，不能作为凋亡标签。Annexin/PI 测定是在 INK128 诱导阶段检验死亡低，补充 S1D 的定义还是 Annexin+/PI+，不是完整的多状态×候选凋亡矩阵。所有其它 raw data 声明为向作者索取。故可增加**参考状态表达覆盖**，但不能自动与本轮其它研究的随意“senescent A549”对接。

## 4. 其余新候选的实际边界

| 来源 | 可用信息 | 未补齐的关键部分 |
|---|---|---|
| [PMC12119881](https://pmc.ncbi.nlm.nih.gov/articles/PMC12119881/), *BH3 mimetics targeting BCL-XL have efficacy in solid tumors with RB1 loss and replication stress*, DOI 10.1038/s41467-025-60238-x | Fig4H 原文含 A549：nolatrexed/raltitrexed 48 h 后 navitoclax 6 h Caspase3/7；官方 `41467_2025_60238_MOESM4_ESM.xlsx` 已下载 | 实际 Figure 4H 工作表仅能定位 ZR75、RKO、A375P 三块；整个工作簿字符串未找到 A549，不能把别细胞块赋给 A549。RNA 分析/机制重点在其他背景，未获得对应 A549 预处理 RNA |
| [PMC11889205](https://pmc.ncbi.nlm.nih.gov/articles/PMC11889205/), *Mitochondrial priming and response to BH3 mimetics in “one-two punch” senogenic-senolytic strategies* | A549 多种诱导状态、BH3 候选；有 baseline Annexin 图及 BH3 profiling | 药物 dose response 为 alamarBlue，BH3 profiling 为 JC-1 去极化；不是药后 Annexin 或 caspase 阳性率。转录组描述引用 SENESCopedia，未提供本研究各状态新的转录组 |
| [PMC7189993](https://pmc.ncbi.nlm.nih.gov/articles/PMC7189993/), *Galacto-conjugation of Navitoclax...* | 明确定义的 control/cisplatin-senescent A549，Navitoclax 与 Nav-Gal **各10 µM**，Fig4b 为 Annexin 阳性时程；3 个生物学重复，每个另有3技术重复 | 是非常贴近的两状态×两候选真实实验，但公开展示曲线，不是逐重复数表；data availability 为 upon request，未获得状态转录组/数字表。这里不以缺同细胞t0否决，只保留数值可获得性缺口 |
| [PMC8717125](https://pmc.ncbi.nlm.nih.gov/articles/PMC8717125/), A549/ACR20 mitochondrial DNA study | RIKEN A549 与 ACR20，顺铂80 µM×24 h Annexin，n=6，Fig1；有状态 qPCR/蛋白信息 | NAC 24 h预处理后顺铂的比较用 **crystal violet viability 72 h**，不是 Annexin。公开 LC655995/LC657585 是序列 accession，不能当 RNA 表达矩阵；当前真实凋亡仅一药剂量对照 |
| [PMC4632318](https://pmc.ncbi.nlm.nih.gov/articles/PMC4632318/), surrogate death receptor signaling | 自建 A549 R1/R2，多项 caspase/PARP，TRAIL 与 cisplatin 组合 | TRAIL为蛋白配体；未取得状态转录组及逐重复 caspase源表，不能当SMILES小分子完整候选集 |
| [PMC8169122](https://pmc.ncbi.nlm.nih.gov/articles/PMC8169122/), eLife cell-cycle-dependent platinum resistance | GSE161800；细胞周期可观测、真实单细胞命运及 cisplatin±DNA repair inhibitor；存在弱或阴性死亡差异 | 源 RNA 是顺铂后72 h按大小分选，不同于同步后给药前细胞周期参考；死亡跟踪不能直接升级为专一凋亡标签。可服务细胞状态/命运问题，未补本轮Annexin/Caspase候选矩阵 |
| [PMC4504487](https://pmc.ncbi.nlm.nih.gov/articles/PMC4504487/), Abraxane-resistant A549/Abr | GSE56742 SOFT＋5样本 FPKM 实际下载；亲本未处理2、亲本100 nM×4 h 1、耐药未处理1、耐药100 nM×4 h 1 | 原论文多药敏感性主要为活力/IC50，未找到匹配 Annexin/Caspase 数据；不能因文字提到 apoptosis-related genes 就填凋亡标签 |
| [PMC4929484](https://pmc.ncbi.nlm.nih.gov/articles/PMC4929484/), KLF12 and anoikis | PAR/SUS 状态、Nimblegen array 描述；KLF12 knockdown/悬浮状态的 Annexin | doxorubicin dose response 为活力；aphidicolin0.1µg/mL×24h后悬浮48h Annexin 是另一有限设计。未取得阵列矩阵和多药源值，不能混为一般多候选状态筛选数据 |

## 5. 现有数据最合理的有限设计（建议，未执行）

1. **状态输入分层**：保持来源明确的群体标签、参考表达向量和少量 qPCR 标志为不同输入层级。可用 NSH76 的三种来源状态做类别条件化；有完整矩阵的 Sarin/GSE246690 支撑各自表达参考。不能因为都叫 A549、resistant、senescent，就合并成同一状态。
2. **功能目标分层**：Annexin 总阳性、Annexin/PI 双阳性、caspase活性fold、一般死亡和代谢活力分开。不能建立一个数值上混合这些端点的“真实凋亡率”目标。
3. **比较只使用已经测过的共同条件**：同研究、相同候选剂量/时长、明确端点的比较才构成有效排序边。缺失格不补0，不用模型自己的预测填作独立真值，不将代表图扩增为重复。NSH76 的 dose 不确定条目只保留证据，不用于完整剂量模型的确定监督。
4. **表达预测与功能判断分别验证，再验证连接**：参考状态本身可合法使用，但RNA预测改善不自动证明促凋亡决策改善。现有局部链可以检验具体条件下的相对反应；不能用局部验证代替整个状态条件化多候选主问题。

本轮停止无边界扩搜，并交回已下载来源、可计算部分和缺口。不宣布原课题已经不可做，也不宣布数据已经完全对齐。若后续继续，最有价值的是取得这些已定位研究的公开数值补充/明确 panel protocol 或另有明确源数据的新线索，避免反复把已有 viability 曲线包装成 apoptosis。

## 6. 验证记录

`extract_round3_evidence.py` 已运行：14流式图的56个印出象限值、16列A549计数矩阵和GSM连接已导出。所有计数为非负整数、gene ID唯一；象限和的差异仅在原始印出精度内。源 XLSX、PDF、GEO gzip 通过实际读取；原文件保留未改。未创建训练任务、未联系作者、未将局部数据纳入正式训练。
