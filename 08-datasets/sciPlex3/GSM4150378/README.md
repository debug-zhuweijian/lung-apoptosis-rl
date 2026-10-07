# sci-Plex3 A549 剂量—时间数据审计

日期：2026-09-24。官方 accession：GSE139944 / GSM4150378。本目录是独立候选来源，**尚未加入现有 prepared/v1 或 training-v1**。

## 当前状态

2026-09-24 22:25更新：**完整3,096,224,606字节的官方压缩表达矩阵已下载并通过全流校验；244,281个A549表达已提取并全部回读通过。** 共检查1,007,419,688条稀疏记录，799,317个源细胞的计数和全部匹配官方元数据，无不符项。没有运行模型训练，也没有将本源加入现有训练包。早先200,000条前缀审计保留为历史检查。

原始文件名、来源、大小、SHA256及状态见[source_inventory.json](source_inventory.json)。当前完整产物见[表达提取说明](extracted/v1/README.md)和[独立回读审计](extracted/v1/readback_audit.json)。

| 文件/目录 | 内容及状态 |
|---|---|
| `sources/*pData.txt.gz` | 799,317行、39个字段的全细胞注释；R导出空格分隔，含额外行名，不是TSV |
| `sources/*hashTable_metadata.txt.gz` | 4,992个处理孔对应的细胞系、时间、重复、药物目录号、浓度、CAS及SMILES |
| `sources/*gene.annotations.txt.gz` | 110,983行基因坐标，含人和鼠；58,347行为人基因注释 |
| `sources/*cell.annotations.txt.gz` | 表达矩阵列顺序，799,317个细胞，与pData按sample+cell全部一一对应 |
| `sources/*hashSampleSheet.txt.gz` | 样本哈希标签表 |
| `sources/*screen_UMI.count.matrix.gz` | 全部官方计数流；SHA256为`7d632716aa6ed0fc1780996003d0abc440ff78340609bceaaa4c8ade9345d00a`；gzip EOF/CRC与源细胞计数和通过 |
| `extracted/v1/` | 202个A549细胞分片，每片同时保留58,347人基因原始轴和32,461共同轴；整数计数、未归一化、未建训练划分 |
| `sources/*079639c.R`、`sources/*PMC7289078.xml` | 本轮亲读的作者代码和官方论文正文副本，来源见[audit/paper_code_sources.json](audit/paper_code_sources.json) |
| `audit/expression_prefix_200000.tsv` | 文件顺序前缀，不是随机样本、完整表达文件或可用于效果评估的子集 |

## A549 条件覆盖

质量规则只复核并执行作者[Notebook3第49–66行](https://github.com/cole-trapnell-lab/sci-plex/blob/079639c50811dd43a206a779ab2f0199a147c98f/large_screen/Notebook3_preprocess_data.R#L49-L66)：两种top oligo非缺失、两种hash UMI均≥5、两种首/次标签比均≥5、rt_well不等于1。**下表不是声称复现了作者最终所有筛选步骤。**

| 作者报告的处理时长 | A549筛选前细胞 | 按上述hash规则保留 | 药物目录条目（不含Vehicle） | 药物—剂量条件（含Vehicle） | 生物重复标签 |
|---|---:|---:|---:|---:|---|
| 24 h | 162,171 | 143,015 | 188 | 753 | rep1 / rep2 |
| 72 h | 82,110 | 67,443 | 47 | 189 | rep1 / rep2 |

药物四档浓度为10、100、1000、10000 nM，即0.01、0.1、1、10 μM；Vehicle为0。它们与Tahoe的0.05、0.5、5 μM没有相同浓度档，不应强制对齐数值。

47个72 h药物均有24 h记录，共188个药物—剂量组合。要求两个时点的两个重复分别至少有1/2/20/50个hash合格细胞时，分别保留187/185/156/71个组合。这些是**覆盖敏感性统计**，不是已冻结的筛选阈值，也不代表统计效力已经充足。低回收条件保留原记录及缺失原因，不应删除后只展示容易预测的药物。

72 h正文报告64,440个细胞，与此次67,443相差3,003；原因未闭合。正文“48 compounds”与此表47药物＋Vehicle也分别保留原口径，不强制改写。原文见[官方PMC全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7289078/)，实际取得版本见本地XML。

## 基因、化学身份与表达读取

- 按Ensembl ID去版本号、**保留`_PAR_Y`后缀**后，现有32,461个基因全部找到唯一注释，覆盖32,461/32,461。没有缩小模型基因轴。
- 另有29条`_PAR_Y`行对应当前基因轴内的基础ID。不能用简单按句点切分把它们自动合并。当前提取版在人基因原始轴完整保留PAR_Y，共同轴仅按原ID映射，不汇总后缀；没有改变旧训练基因轴。全A549的PAR_Y记录共2个UMI，保留原值。
- sci-Plex188种记录的SMILES均可解析。与Tahoe比较，15种的完整结构编码相同，其中2种同时有24/72 h（Roxadustat、Rucaparib phosphate）。相同完整编码仍不等于独立核实了试剂的立体化学、盐型及混合物组成。
- 放宽到最大结构片段有18种候选；进一步忽略立体信息有25种候选。后两类**仅供身份复核，不自动并表或合并训练划分**。尤其Tahoe存在不同名称共享缺失立体信息编码的情况。
- 原始矩阵是无Matrix Market头的`gene_index、cell_index、UMI`三列、1-based坐标，不能直接以带头的MTX格式读取。全流已检查坐标严格有序且不重复、计数为正整数、源细胞计数和匹配；人基因和共同轴矩阵之间的计数对应全部回读一致。
- 数据完整性和注释对应不代表生物学效力、跨平台可比性或模型精度通过。原始源包含鼠基因；本版同时记录全源、人基因和共同轴UMI总量，尚未拟合任何归一化。

## 协议与适用范围

24/72 h分别有16/4块培养板，培养板嵌套于时长；但两个时点共用`Chem3_Screen2`，并在全部4个RT板、4个ligation板、2个PCR板中都有细胞。不能称为所有技术批次完全分离。

当前可研究来源内的有限药物—剂量—时间条件表达分布；保持各时间的Vehicle及重复/板信息。作者报告处理时长，换液、补药、洗脱和收样时间安排仍未取得足够方法证据。记录未知值，不能默认为“没有”。

没有同一细胞24→72 h配对、真实凋亡标签或换药历史。作者代码`viability`由回收数量的拟合比值得到，不是Annexin/caspase凋亡率。不能将72 h比24 h的表达差异解释为共同72 h终点上的时长治疗优势。科研判断详见[剂量时间研究增量](G:/研究生科研文件/毕业论文/sciPlex剂量时间审计与先导接口进展_2026-09-24.md)。

## 可复查入口

- [统计摘要](audit/audit_summary.json)
- [药物—剂量—时间—重复条件表](audit/a549_condition_coverage.csv)
- [两时间四重复单元覆盖](audit/shared_dose_time_rep_counts.csv)
- [基因映射](audit/shared_gene_axis.csv)、[额外PAR_Y行](audit/par_y_annotation_rows.csv)
- [化学结构编码交集](audit/tahoe_chemical_encoding_overlap.csv)
- [表达前缀逐细胞校验](audit/expression_prefix_cell_checks.csv)
- [培养板×时间](audit/plate_time_cross_tab.csv)、[样本×时间](audit/sample_time_cross_tab.csv)
- [元数据/前缀获取脚本](../../../scripts/fetch_sciplex_audit_sources.py)、[审计脚本](../../../scripts/audit_sciplex_a549.py)

研究脚本只写本目录新增审计结果；不覆盖旧训练数据、划分或默认配置。完整源文件与筛选前A549表达保留，hash筛选结果为可选标记，没有物理删除低质量记录。新时间训练版本仍需协议/QC差异处理、训练内预处理、独立划分和模型读取接入；`extracted/v1`只代表提取完成，不是训练就绪。
