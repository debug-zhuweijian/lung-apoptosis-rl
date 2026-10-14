# A549 初始状态—药后功能终点：原始证据核验

核验日期：2026-10-10。任务范围是补齐当前课题的状态功能证据，不将研究目标改成一般细胞活力预测。所有新文件均在本目录；未修改现有训练资产。该记录由 AI 辅助检索并逐项读取原文、图和源文件完成。

## 结论

**已实际取得 A549 不同群体状态 × 后续处理的凋亡相关源数值，并取得可作为状态表征的公开表达。尚不能把这些文件全部登记成无条件可训练的精确配对集。** 最强的非遗传状态证据是 Wang 2022 的增殖/alisertib 诱导衰老 A549 在 Lz-TRAIL 处理后的 Caspase-3/7 阳性比例；最接近小分子连接的来源是 Shostak 2020 的 RNF113A 操作状态与顺铂 Annexin，但其原始元数据有未解决的条件冲突。

三个粒度要分开：同一真实细胞纵向命运配对、同一研究/明确状态的独立群体对应、跨研究借用状态表达。本次取得后两类的证据，没有取得第一类。

## 1. Wang 2022：明确状态交叉的 Caspase 终点

- 原论文：*cFLIP suppression and DR5 activation sensitize senescent cancer cells to senolysis*，Nature Cancer 3, 1284–1299，DOI [10.1038/s43018-022-00462-2](https://www.nature.com/articles/s43018-022-00462-2)。
- 原文公开网页：`Wang2022_cFLIP_article.html`；Fig. 3 官方图页：`Wang2022_Fig3.html`；原图：`43018_2022_462_Fig3_HTML.png`。
- 亲读作者论文集 Chapter 6：`Jochems2023_Senescence_As_Steppingstone_author_thesis.pdf`，PDF 第 170 页 Fig. 3；第 184–185 页 Drug treatments、Colony formation assay、IncuCyte cell proliferation assays、Cell viability measurement。作者存储库：[PDF](https://pure.umcutrecht.nl/ws/portalfiles/portal/310861847/proefschrift_6516cffcd3ebd.pdf)。这是作者学位论文中的原论文版本，不能将其排版页码写作期刊页码。

### 实验与实际取得值

A549 在 RPMI、10% FBS 培养背景下用 0.5 µM alisertib 诱导 7 天衰老。Fig. 3b 比较增殖/衰老状态、对照/200 ng/ml Lz-TRAIL，测量 activated Caspase-3/7 positive cells（百分比），不是 Annexin 百分比。源表 `43018_2022_462_MOESM7_ESM.xlsx` 的 `Fig 3b` 页 D:F 三列明确标为 R1–R3，图注说明三次独立实验。

| 源表行 | 初始群体状态 | 再处理 | 三个源数值（%） |
|---|---|---|---|
| 3 | 增殖 | 对照 | 0.4889056, 0.8267879, 1.1180680 |
| 4 | 增殖 | Lz-TRAIL | 5.7324841, 3.6540804, 4.5560748 |
| 5 | alisertib 诱导衰老 | 对照 | 21.7948718, 26.3513514, 17.5757576 |
| 6 | alisertib 诱导衰老 | Lz-TRAIL | 79.3650794, 88.1818182, 75.3731343 |

这是实测状态依赖的功能差异，能够用于检查模型是否需要状态输入。当前模型并未在该来源上验证。存在明显的状态基线差异，不能只比较处理后两个柱高而不使用各自对照。

### RNA 连接与必须保留的限制

1. 原文将 GSE102639 用于 A549 alisertib 的表达分析。已取得 `GSE102639_family.soft.gz` 与完整 `GSE102639_A549_Etoposide__Aurora_treated.txt.gz`，有未处理、tozasertib、barasertib、alisertib、etoposide，每组两个 RNA 样本。
2. **GEO 元数据写 alisertib 1 µM × 7 天，而 Wang 2022 图注写 0.5 µM × 1 周。** 不能直接把该表达视为同剂量精确初态。
3. 同团队 SENESCopedia 的 E-MTAB-9970 有明确 0.5 µM alisertib × 7 天的 A549 RNA，SDRF 与 processed log2CPM 已实际取得，见下一节。其 RNA 测量前撤药 24 小时；Wang 2022 的 Caspase Fig. 3b 没有单独声明该等待步骤，故属于有协议差异的状态代表，不能伪造同批样本对应。
4. 作者论文集说明 colony/CellTiter assays 在 7 天诱导后 trypsin/reseed、停止诱导处理，再给后续药物 5 天；**该 5 天不能自动赋给 Fig. 3b 的 Caspase 快照**。IncuCyte 方法只写染料和成像计量，未明确 Fig. 3b 取点小时数，故当前保留 `challenge_duration_h=null`。
5. Lz-TRAIL 是蛋白配体；conatumumab、mapatumumab 是抗体。它们不能伪装成当前只编码小分子 SMILES 的 P/Actor 已支持动作。
6. Navitoclax 的相关 Fig. 4e 是 CellTiter-Blue IC50。Fig. 6 与扩展图中的小分子/组合大量读数是活力或 colony assay，不能改名为 Annexin/Caspase 比例。

## 2. Jochems 2021 SENESCopedia：实际得到状态表达

- 原论文：[PMC8333195](https://pmc.ncbi.nlm.nih.gov/articles/PMC8333195/)，DOI 10.1016/j.celrep.2021.109441；本地 `PMC8333195.xml`。
- 实际取得 `E-MTAB-9970.json`、`E-MTAB-9970.sdrf.txt`、`E-MTAB-9970.idf.txt`。
- 通过 [Europe PMC 补充文件接口](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8333195/supplementaryFiles) 取得 `Jochems2021_supplementaryFiles.zip`，其中原始 `mmc5.xlsx` 提取为 `Jochems2021_mmc5.xlsx`。官方说明 Table S6 为 processed log2CPM。

| RNA 列名 | 明确状态 | 批次 | 相应 ENA run |
|---|---|---|---|
| A549_P_batch3 | 未处理增殖 | 3 | ERR5023186 |
| A549_Ali7_batch3 | alisertib 0.5 µM，7 天 | 3 | ERR5023183 |
| A549_Eto7_batch3 | etoposide 2 µM，7 天 | 3 | ERR5023185 |
| A549_Ali14_batch1 | alisertib 0.5 µM，14 天 | 1 | ERR5023184 |

RNA 方法明确：诱导后撤药 24 小时再提 RNA。每个上述状态只有一个已列 RNA sample，不是三生物重复；Ali14 与其余 A549 样本不同测序批次。不能把四列当纵向跟踪同一群细胞或批次严格匹配的时间曲线。

已提取 `Jochems2021_A549_state_log2CPM.csv`：20,029 个 Ensembl 基因 × 4 个 A549 RNA 样本，保留 gene_name 和 HGNC symbol。此为作者归一化 bulk 表达，不能当单细胞 counts 或直接输入原单细胞分类器解释成百分比。

Fig. 2/Methods 中的 ABT-263 终点是诱导 7 天、重播种后 5 天的 CellTiter-Blue 测量，三生物重复，每次技术三重复。该来源本身提供状态依赖的活力任务，**不能单独证明凋亡比例预测**。

## 3. Shostak 2020 RNF113A：有无药 RNA 与顺铂 Annexin，条件冲突未消除

- 原论文：[PMC7062854](https://pmc.ncbi.nlm.nih.gov/articles/PMC7062854/)，DOI [10.1038/s41467-020-15003-7](https://www.nature.com/articles/s41467-020-15003-7)；本地 `PMC7062854.xml`。
- 亲读 Fig. 2b/c、RNA-seq 方法、Apoptosis and ferroptosis assays、Transfection and lentiviral infections、Data availability。
- `41467_2020_15003_MOESM13_ESM.zip` 已下载；其中 `Source Data.xlsx` 已提取到 `RNF113A_source_data/Source Data.xlsx`。未解压不需要的 Western blot PPTX。
- 表达原始元数据：`GSE133029_family.soft.gz`。12 个作者提供的 gene-symbol count 文件全部已下载，每个 19,713 行，合并为 `GSE133029_gene_symbol_counts.csv.gz`；`GSE133029_samples_review.json` 保留源 URL、sample title、hash。

有 shControl 无药 3 样本、shRNF113A 无药 3 样本，以及各自加 cisplatin 的 3 样本。无药 RNA 是对应工程细胞群的表达参照；`untreated` 以 title/characteristics 为准，不能因所有样本共享 treatment protocol 文本而误认为均给了药。

Fig. 2c 的 Annexin 源表有 shControl、shRNF113A #1(09)、#2(81)，源表 0h 与 24h 每组各三个数值。已原样提取 18 个值，保留状态与每个单元格。**亲看原图 `41467_2020_15003_Fig2_HTML.png` 后确认柱图为 late apoptosis，即 Annexin-V+/PI+；不能解释成早期+晚期总比例。** 原图多处明确标为 25 µM、20 h，仍需保留其与源表的 24h 冲突。

以下缺口阻止精确配对训练：

- 论文 Annexin 方法写 **25 µM、20 h**；源表 Fig2C 写 **0h/24h**；GEO protocol 写 **25 µg/ml、20 h**。µg/ml 与 µM 是不同量纲，不能擅自把其中一个判成笔误。
- RNA 的 shRNA 字段只写 `shRNF113A`，未核实对应 #1(09) 还是 #2(81)。两个 clone 的终点不能复制使用同一个 RNA 而宣称精确对应。
- Fig2C 每组三列，正文图注写 two independent experiments。不能把三个数值无条件称作三个独立生物重复。
- 0h 基线不能自动视为与 20/24h 给药组同一时间采样的 vehicle control。
- 这是明确的遗传操作背景，不能作为自然状态异质性的同义替代。

这些不一致已登记在 `state_function_endpoint_source_cells.csv` 的 blocker 字段中；原始文件不改写，尚未开启训练标记。

## 4. Triana-Martínez 2019：真正小分子状态差异，但未取得同状态 bulk RNA

- 原论文：[PMC6803708](https://pmc.ncbi.nlm.nih.gov/articles/PMC6803708/)，DOI 10.1038/s41467-019-12888-x。
- 初态：A549 增殖与 bleomycin 20 µM × 5 天诱导衰老；后续给 digoxin。
- 亲读 Fig. 2 与 Apoptosis analysis，原图 `41467_2019_12888_Fig2_HTML.png`；源表 `41467_2019_12888_MOESM3_ESM.xlsx` 的 `FIGURE 2`。
- Fig2a 有四条件 × 三个 Annexin-V 百分比，共 12 个源值。Fig2b 实际非空 active-Caspase3 源值为 10 个：两种 control 各只有两个值，drug 各三个值。不能用图注概称 n=3 补造缺失的两个值。
- 可作为明确定义群体状态的功能证据。当前未核实 Fig2a/b 的 digoxin 特定剂量/测量时长，不把补充图 S2 的 digoxin 0.1 µM 条件转移过来。
- 未取得本文匹配的全转录组；补充图 S3a 的 ATP1A1 qPCR 不是全基因初态。不能拿其它诱导方案的“senescent A549”RNA 替代精确初态。

## 5. 必须撤回的错误候选解释与其它查证

### EphB1 / PMC9675789 不是已经核实的再挑战表

原文：[DOI 10.1038/s41419-022-05385-5](https://www.nature.com/articles/s41419-022-05385-5)。初筛时结果段的“drug resistance”与 Annexin 图看似支持再挑战；**亲看 Fig3F 后撤回此判断**。

原图 `41419_2022_5385_Fig3_HTML.png` 的横轴为 0d/7d/21d/35d，展示初次顺铂处理后状态过程；没有标出“先得到 DOR/REA 再重新给药”的剂量、暴露时长。图中纵轴是 7-AAD、caption 却写 PI。该图不可当作多个初态共享新处理的监督样本。

RNA PRJNA730205 的 9 个 runs 已用 ENA 实际枚举并保存 `PRJNA730205_ena_runs.tsv`；本文附表3是 ATAC motif，附表6是 pathway，附表7只有 SOX2/NANOG/OCT4 的三状态 RNA 与部分 ATAC，不是完整 Annexin 重复源表。不能因出现 “NC/Dormancy/Reactivation” 的工作表就认定找到整套功能配对数据。

### 其它具有边界的线索

- Deryabin 2021，PMC8629786 / DOI 10.1007/s00018-021-03980-x：A549 etoposide 2 µM ×3天，诱导后至少7天才分析；ouabain 1 µM 后有 Fig7 Annexin-V/DAPI 与 Caspase3/7。作者借用 GSE102639 的无药/etoposide RNA。公开附件主要是 TIFF 图和 DEG/pathway CSV，未取得可复算的逐重复功能源数值；不将其与 Jochems 的 etoposide 7 天持续暴露视为同协议。
- Nature Aging，PMC12920146 / DOI 10.1038/s43587-025-01054-2：已取源 XLSX 和全文。Annexin 的方法是 **接受对照或衰老细胞 conditioned medium 的受体 A549** 后续 galunisertib/staurosporine，而 bulk RNA 主要是产生 conditioned medium 的原始衰老细胞。不能把 donor RNA 与 recipient Annexin 拼作同状态测量。
- Fauveau 2026，PMC13356641 / DOI 10.1016/j.omton.2026.201279：有 A549 3D spheroid 14 天 cisplatin10µM+pemetrexed250µM 后的 scRNA、再挑战高剂量组合及撤药恢复实验，但再挑战定量终点是 CellTiter-Glo ATP。该资源提供新的状态/历史响应线索，未作为凋亡补齐依据。
- PMC7617193 的 INK128 persister 研究有 A549 状态 RNA，但总体数据声明把其余原始资料留作 upon request。本任务未联系作者。

## 6. 生成文件与检查

`extract_state_function_source_tables.py` 已实际运行，输出：

- `state_function_endpoint_source_cells.csv`：52 个真实的、逐单元格定位的功能源值；cFLIP 12、Triana 22、RNF113A 18。不是 52 个独立实验，也不是 52 个可训练条件。
- `Jochems2021_A549_state_log2CPM.csv`：20,029 基因、4 个 RNA 样本。
- `GSE133029_gene_symbol_counts.csv.gz`：19,713 基因、12 个 RNA 样本。
- `extraction_checks.json`：范围、唯一源单元格、数值非空与 [0,100] 检查；所有正式训练开关保持 false。

本任务所做的是取得真数据并显式建立候选连接与拒绝理由。缺少精确时间、同状态协议、clone 身份或生物重复说明的项目，不能通过代码自动补全。要将其变成正式评测，需逐个确定可允许的任务粒度：状态固定的功能差异检验可以采用来源自身的固定协议；药物—剂量—时长泛化仍必须补齐相应协议字段。任何有限状态案例都不能单独证明全药物泛化或同细胞命运预测已成立。

## 7. 检索覆盖与失败回退

检索围绕 A549 + initial/cell state、RNA-seq、Annexin、Caspase、dormancy、senescence、rechallenge、live-cell、drug-tolerant persister 多条路线。网页搜索用于发现，Europe PMC XML、Nature 官方 source-data XLSX、GEO SOFT/processed counts、BioStudies SDRF、ENA run metadata用于核验；同一论文的多个入口不计多份证据。

PMC 网页附件直接下载部分返回403；SENESCopedia 使用 Europe PMC supplementaryFiles 官方 ZIP 成功取得。cFLIP 期刊正文非开放，公开图注/源表成功读取，全文使用作者公开学位论文存储库；researchinformation.umcutrecht.nl 首地址403，搜索结果给出的 pure.umcutrecht.nl 官方镜像成功。未绕过访问控制、未联系作者、未将全文失败说成数据不存在。

