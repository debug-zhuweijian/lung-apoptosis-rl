# A549 状态表达—凋亡功能来源定向审查，第 5 轮

日期：2026-10-10。状态：本轮资料包已完成；**三个重点候选均未通过完整的“可追溯初始状态表达 × 多个药物凋亡结果”联合数据资格审查**。这表示本轮不能新增合格联合训练/测试行，不等同于不存在任何可用公开来源。其中 Nature Aging 2026 已列于 `../state_function/README.md` 第 101 行，本轮是对其 ENA 样本与状态对应的窄复核，**不计为新论文发现**。

本轮增加了两套可定位的公开表达资源元数据（PRJEB52271 的 12 个 A549 样本、GSE116192 的 16 个样本），并明确了原文中容易被误配的状态、检测对象和单位。未下载 FASTQ/整库表达数据，未训练，未修改既有主表、化学包、状态输入或源码。

## 三个重点候选

| 来源 | 表达资源和本轮实际读取状态 | 真正功能终点 | 当前结论与具体缺口 |
|---|---|---|---|
| González-Gualda 2026，Nature Aging，[DOI 10.1038/s43587-025-01054-2](https://doi.org/10.1038/s43587-025-01054-2)，[PMC12920146](https://pmc.ncbi.nlm.nih.gov/articles/PMC12920146/) | ENA [PRJEB52271](https://www.ebi.ac.uk/ena/browser/view/PRJEB52271)；36 个 run 中 12 个为 A549，control/cisplatin/docetaxel/palbociclib 各 3 个。已读 run 表及每状态一个原始 sample XML，未读表达数值 | Fig.3b：接受 control-CM/cisplatin-senescent-CM 的 A549，vehicle/galunisertib 50 μM/staurosporine 0.5 μM；Annexin V 信号按 confluence 归一化 | RNA 四状态主要是**产生 CM 的细胞**；Fig.3b 检测的是**接受 CM 的细胞**。不能直接配对。可能适合作为状态资源或独立功能来源，尚不能合并为同状态多药物矩阵 |
| Statello 2021，NAR Cancer，[DOI 10.1093/narcan/zcab002](https://doi.org/10.1093/narcan/zcab002)，[PMC8209975](https://pmc.ncbi.nlm.nih.gov/articles/PMC8209975/) | [GSE116192](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE116192)；16 GSM，8 个条件各 2 个生物重复。亲本/耐药、±cisplatin、耐药细胞±SCAT7 LNA。已读所有 GSM 元数据；官方列有处理后 FPKM 文件，未下载矩阵 | Supplementary Fig.S1D：亲本 A549 中 SCAT7 KD ±cisplatin 15 μM、24 h 的 cleaved-PARP western；不将其换成 Annexin 比例 | RNA 的 SCAT7 KD 在 **A549/cDDP**，该 cleaved-PARP 图在 **亲本 A549**。缺相同 KD 状态的表达或耐药状态的对应功能结果，不能组成所需状态效应检验；并有 GEO 15 mM 与原文 15 μM 的单位矛盾 |
| Liao 2021（DOI 卷年为 2020），Front Cell Dev Biol，[DOI 10.3389/fcell.2020.609285](https://doi.org/10.3389/fcell.2020.609285)，[PMC7843937](https://pmc.ncbi.nlm.nih.gov/articles/PMC7843937/) | 本文做了 A549/A549DDP RNA-seq；正文仅报告 304 DEG，数据声明为作者可提供；本轮未找到其可验证公共矩阵或 accession | Fig.3A–D：A549/A549DDP，control、DDP、DDP+Cor50/100/200，48 h Annexin-V-FITC/PI；Fig.3E/F 为 cleaved-caspase-3 等 western | 研究设计接近目标，但公开表达缺口没有补上；Fig.3 未直接注明 DDP 浓度及 Cor 数字单位。不能从其他实验的 IC50 或剂量表静默补出。同名药物 GEO 命中 GSE81727 属于另一研究，不能代填 |

`candidate_review.csv` 是上述三个候选的机器可读判定。`function_condition_inventory.csv` 只登记原文明确可见的功能条件，不是数值标签或合格配对表。所有条目 `joint_dataset_ready=false`。

## PRJEB52271 的实际发现及 Nav-Gal 边界

- `PRJEB52271_A549_runs.csv` 保留 12 个 A549 的 sample/run 关系。原始 ENA 全表保存在 `raw/PRJEB52271_runs.tsv`，另 24 个为小鼠 L1475(luc)，没有把小鼠受体 RNA 当成人 A549 RNA。
- ENA 列出的 A549 paired FASTQ 合计为 48,894,323,694 字节（约 45.54 GiB）；这是**登记文件大小**，未下载、未测量本地 RNA 处理耗时。四份 sample XML 主要有样本名、物种及发布日期，没有补齐剂量、诱导时长或 RNA 收取阶段；这些必须回查原文。
- 不是既有 GSE246690。本文提及的 GSE15622 是临床 microarray 二次分析，不是这 12 个 A549 RNA 样本。
- 不能用本资源给已经冻结的 Nav-Gal 挑战直接补“同源状态”：2026 原文 Sec20 为 **ATCC A549、15 μM cisplatin、7 天**；2020 Nav-Gal 原文 4.4 为 **ECACC A549、15 μM cisplatin、10 天**。作者重叠、药名相同均不能抹去细胞来源与诱导时长差异。若未来用于跨来源迁移，必须作为待检验迁移关系。
- 原文还有内部不一致：docetaxel 为 Sec20 的 100 nM、Fig.1c 图注的 150 nM、Results Sec3 的 200 nM；Fig.3b 图注 72 h、Sec21 凋亡方法 48 h。当前均保留，不择一个值自动放行。
- 实际复用了既有 `../state_function/43587_2025_1054_MOESM3_ESM.xlsx`，亲读 `Fig 3b_A549 n=3`：6 个条件各有 3 个数值，另存带单元格坐标的 18 行小表 `Nature2026_Fig3b_A549_source_cells.csv`。这是既有来源的复核，不计新增发现。上述 donor/recipient 差异依旧存在。本轮补充 PDF 下载失败（Springer CDN SSL EOF、Europe PMC supplementaryFiles 读超时），尚未读取补图；失败记录不表示源文件不存在。

## GSE116192 的条件及可读出处

`GSE116192_sample_metadata.csv` 是完整 16 行 GSM 字段小表；原始 SOFT 在 `raw/`。八个条件为：

| 条件 | GSM |
|---|---|
| A549 WT，vehicle | GSM3212803–GSM3212804 |
| A549 WT，cisplatin | GSM3212805–GSM3212806 |
| A549/cDDP，vehicle | GSM3212807–GSM3212808 |
| A549/cDDP，cisplatin | GSM3212809–GSM3212810 |
| A549/cDDP，control LNA、vehicle | GSM3212811–GSM3212812 |
| A549/cDDP，control LNA、cisplatin | GSM3212813–GSM3212814 |
| A549/cDDP，SCAT7 LNA2、vehicle | GSM3212815–GSM3212816 |
| A549/cDDP，SCAT7 LNA2、cisplatin | GSM3212817–GSM3212818 |

亲本和耐药的来源关系由主文 Methods “Generation of cisplatin-resistant A549 cells”（`SEC2-3`）明确：由亲本历时 8 个月构建。RNA 元数据药物处理 24 h；LNA 条件在转染后 48 h 收取。**这些是群体条件，不是逐个细胞的纵向配对。**

补充 PDF 已从 Europe PMC 原始补充 ZIP 中实际提取：`raw/SCAT7-TOP1_Suppl File_Rev1-Size1.pdf`。第 1 页是 Fig.S1 原图，第 2 页图注 D 直接说明亲本 A549、SCAT7 KD、15 μM cisplatin、24 h，并称至少 3 个生物重复的代表性 western。原图分别是 SCAT7-sh2 和 SCAT7-LNA1，而 GEO 标题为耐药细胞 L2 条件。该图可作独立 cPARP 端点证据，当前不能转写为有 RNA 对应的耐药状态凋亡率。

官方 FPKM 文件链接见 `raw/GSE116192_series.soft` 的 `Series_supplementary_file`；只确认存在官方登记，未把“列有链接”写成已读数值矩阵。GEO 的 `15 mM` 原字段完整保留；主文 `SEC2-1` 与 Supplementary Fig.S1D 是 `15 μM`，不能从这项冲突推定所有处理后 RNA 剂量都已纠正。

## Liao 原图、公共数据查找及余下缺口

已亲读原 Fig.3 并保存 [Fig.3 原始图](raw/Liao2020_Fig3_original.jpg)。每个细胞背景各 5 个面板：control、DDP、DDP+Cor50、DDP+Cor100、DDP+Cor200。原图既含代表性流式图，也含均值±SD柱图；本文称 3 次独立实验。**未把代表图百分比当 3 次重复的数值源表。**

主文 `S2.SS1` 说明 A549DDP 培养维持 DDP 2 μg/ml；RNA 收样前是否撤药及撤药多久不明确。`S2.SS9` 只给 RNA-seq 分析方法，未给公共样本 ID。`S2.SS6` 给功能检测时长 48 h，但未为 Fig.3 指定 DDP 浓度。

GEO 定向查询 `A549 AND (cordycepin OR 609285) AND gse[ETYP]` 返回 1 条：GSE81727。实际读取 series 后确认它属于另一作者研究、只有 DMSO/cordycepin/cordyceps 3 个数组样本，不能替代本文 A549/A549DDP baseline RNA。查询原始 JSON 与 series SOFT 均已保存。此结果只覆盖本次查询，不能据此宣称全网没有本文数据。

补充 Table 1 DOCX 的 PMC 访问返回 403，实际 Content-Type 为 `text/html; charset=UTF-8`；错误回应保存为 `raw/Liao2020_Table1_attempt.http-error.html`，source JSON 同时保留请求文件名与实际保存名。Europe PMC 补充包读超时。没有从未读到的表格推断完整表达或生物重复数。

## 初筛排除及避免重复

| 来源 | 原文核到的排除/保留原因 |
|---|---|
| Sun2019，10.1002/cam4.2190，PMC6558586 | GSE6914/GSE77209 的 RNA 来自 Calu3、H1299、H1355；A549 是后续验证。不能按论文整体名义给 A549 配其他细胞系表达 |
| Lee2020 SEZ6L2，10.3390/biomedicines8110500，PMC7697537；其引文 Jeon2016，10.1038/emm.2016.129，PMC5192074 | GSE158638 确有亲本/CR A549 各 1 样本；但原文功能主要是存活细胞、迁移/侵袭、成球。前文里的 7-AAD 与 TMRE 不等于 Annexin/caspase 凋亡终点，不选作本轮核心联合来源 |
| Increasing cisplatin exposure...，10.1007/s00432-025-06164-3，PMC11953189 | 3 个耐药层次有 RNA-seq 叙述，未见可核公共 accession；Data Availability 声明“无数据产生或分析”与正文 RNA-seq 描述不相容。未定位合格凋亡标签表，不选作核心来源 |
| Non-canonical NOTCH1...，10.1038/s41419-025-08355-9，PMC12780219 | 本轮取到原文后检出已在 `state_function_round2/README.md` 审核；停止深入，不计新发现或重复推进 HRA007891 |
| Genome-scale modeling...，10.1038/s42003-024-07408-7，PMC11681178 | 确有 A549 EMT、Annexin V/PI 双阳性及 caspase-3/7，但关键干预是 ENO1/ENO3 siRNA，且 siRNA 在 TGFβ 诱导前施加；不是既成状态的候选小分子挑战。可作方法旁证，不当药物筛选联合监督 |

已遵照分工不重新研究 Sarin/GSE108214、Nav-Gal 功能提取、Jochems/GSE246690、NSH76、Hastings eLife53367、cFLIP/SENESCopedia 和 RNF113A。为确认 Nav-Gal 与新 ENA 是否同协议，仅复读其 4.4 方法，没有改动已接受的化学包。

## 文件与验证

- `candidate_review.csv`：3 个重点候选及所缺字段。
- `GSE116192_sample_metadata.csv`：16 个 GSM 原字段摘要，含有单位矛盾的原字符串。
- `PRJEB52271_A549_runs.csv`：12 个 A549 sample/run 与官方 FASTQ 登记大小。
- `function_condition_inventory.csv`：条件级证据索引；尚非可训练数值标签。
- `Nature2026_Fig3b_A549_source_cells.csv`：复用既有 XLSX 的 18 个原始数值，保留单元格坐标；未连接 donor RNA。
- `primary_evidence_index.json`：供主代理亲读的原始 XML section/figure 及 PDF 页码定位。
- `search_log.json`：检索入口、定向查询、筛选、去重与失败记录。
- `raw/*.source.json`：成功/失败 HTTP 回应的 URL、状态、时间、字节数、SHA256。原始文件和从 ZIP 提取的文件以 `manifest.json` / `validation.json` 核验。

本轮可交付的是来源资格判定与原件证据。没有新增已通过联合训练/评测条件，没有将这些候选标为“数据对齐已解决”。
