# Sarin / GSE108214 两状态表达：基因身份映射与原值核验

核验日期：2026-10-10。对应当前研究架构 §2 的群体状态输入与 §4.3 的统一基因身份；只完成坐标与来源连接，没有更换研究目标或宣称功能监督已闭合。

**本轮已经把 GSE108214 的全部 29,729 个微阵列探针、22 个样本和当前 32,461 个共同基因建立了可追溯映射。** 22,901 个探针可解析至共同轴中的单一基因，覆盖 16,852 个不同 ENSG。12,070 个基因恰有一个已解析探针，已形成保持原值的 `22 × 32,461` 矩阵；其余坐标为 `NaN`，同时保存显式布尔 `mask`。多个探针映射同一基因的原值全部保留，但没有擅自平均、求和或选一个探针。

这属于 **gene identity alignment verified**。该材料是 bulk 微阵列，不能直接作为当前计数型单细胞模型的已验证输入；也没有解决跨平台观测模型、Sarin 的相对倍数端点或全候选筛选验证。

## 1. 实际输入与来源

| 来源 | 版本及用途 | 本地核验 |
|---|---|---|
| [GSE108214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108214)，所附 GPL17077 | Agilent 039494 / SurePrint G3 Human GE v2；原平台明确 hg19 / GRCh37，没有记录 Ensembl release | 回读 `../state_function_round2/GSE108214_family.soft.gz` 的全部平台和 22 张样本表；与既有探针注释、654,038 个 VALUE 文本逐项一致 |
| [HGNC 官方完整表](https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt) | 下载日期快照 `hgnc_complete_set_2026-10-10.txt`；HTTP Last-Modified 为 2026-10-09；45,233 条 Approved 记录 | 用原平台 `LOCUSLINK_ID` 精确连接官方 `entrez_id → ensembl_gene_id`；不靠 symbol 猜测，不宣称这就是原平台历史注释 |
| [Ensembl GRCh37 release 75 GTF](https://ftp.ensembl.org/pub/release-75/gtf/homo_sapiens/Homo_sapiens.GRCh37.75.gtf.gz) | 2014-02-07 文件；基因组构建兼容的固定注释，用来检查原 ENST 身份；不是已证明的原平台精确发行版 | 39,344,043 字节；gzip 全量读取通过；GNU `sum` 为 `00743 38422`，与官方 `CHECKSUMS` 完全一致 |
| `prepared/v1/gene_map.parquet` | 当前共同轴，32,461 个唯一 ENSG 与连续坐标 | 按 ENSG 身份连接并验证坐标顺序；不借用同名基因替代编号 |

下载 URL、HTTP 状态、时间、大小、SHA256 在 [annotation_download_manifest.json](annotation_download_manifest.json)。所有原始来源、注释、轴和派生产物的 SHA256 绑定在 [manifest.json](manifest.json)，脚本副本也已冻结。

## 2. 映射规则

1. 原平台 `ENSEMBL_ID` 实际是 ENST 转录本编号，不能直接解释为 ENSG。读取该列及 `ACCESSION_STRING` 中**全部明确 ENST**，保留原文本。
2. 先用原 `LOCUSLINK_ID` 通过 HGNC 官方映射取得候选 ENSG。若有原 ENST，再从固定 GRCh37 GTF 取得全部候选 ENSG 作交叉核对。
3. 一个 Entrez 或多个 ENST 指向多个基因时隔离；Entrez 与转录本编号指向不同 ENSG 时隔离。不以多数表决或基因名称覆盖冲突。
4. 没有可用 Entrez 身份桥时，仅在**全部源 ENST 均能解析且共同指向一个基因**时采用转录本桥。部分转录本缺失时不擅自补全；未知转录本版本也不静默去版本。
5. 解析出的 ENSG 必须在共同轴中。原 symbol 与共同轴 symbol 是否相等仅作为诊断字段，不用于选择基因。
6. 多探针基因保留完整探针层观测。当前单值基因矩阵只采用一个已解析探针对应一个基因的情况；这是一份保守的无聚合视图，并不表示其他探针已被删除或生物学上无用。

这段规则是本课题为避免身份猜测而实现的工程规则，不是原作者使用的探针汇总算法。旧平台与当前注释的冲突可能来自注释演变或探针歧义；本轮没有未经核查地判定哪一个来源“错误”。本轮依据官方身份注释，没有重新比对探针序列或验证杂交特异性。

## 3. 实测匹配数量

| 探针状态 | 数量 |
|---|---:|
| 无歧义解析至共同轴的单一基因 | 22,901 |
| 可解析基因但不在当前共同轴 | 3,495 |
| 没有已核验的编号桥 | 1,990 |
| 官方注释涉及多个基因 | 738 |
| Entrez 与 ENST 注释冲突 | 524 |
| 只有部分 ENST 可解析且没有 Entrez 桥 | 81 |
| 合计 | 29,729 |

共同轴内 22,901 个探针中，21,971 个使用 Entrez/HGNC 主桥，930 个使用完整 ENST/Ensembl 桥。源平台共涉及 36,156 个不同 ENST；Ensembl 75 找到其中 32,969 个。未找到不等于不存在，只表示该固定注释没有提供本轮所需连接。

映射覆盖 16,852 个共同轴基因：12,070 个为唯一已解析探针；4,782 个由合计 10,831 个探针覆盖，保留为多探针材料。单值矩阵因此有 12,070 个可用列和 20,391 个未填列，**未用零填充**。

可定位的冲突实例：探针 `A_33_P3300975` 原记录为 Entrez `3221` / ENST00000430889，当前 HGNC 与 GRCh37 release 75 分别给出 ENSG00000198353 与 ENSG00000273266。账本保留两个结果并隔离，未因源 symbol 为 HOXC4 而直接选一方。

## 4. 样本、状态和真实时间

下表从原 GEO 样本身份核验。所有 22 个 GSM 的源 treatment protocol 相同，明确记载过夜贴壁、4 h 血清饥饿、随后在无补充 IMDM 中处理 24 h；亲本、耐药株剂量与各自样本 characteristics 一致。

| 群体状态 | 药物/剂量 | 真正取样时点与角色 | 样本数 |
|---|---|---|---:|
| A549 | 无药培养基 | 平行 24 h 对照参考 | 4 |
| A549 | cisplatin 11 µM | 24 h 药后表达 | 3 |
| A549rCDDP2000 | 无药培养基 | 平行 24 h 对照参考 | 5 |
| A549rCDDP2000 | cisplatin 11 µM | 24 h 药后表达 | 5 |
| A549rCDDP2000 | cisplatin 34 µM | 24 h 药后表达 | 5 |

完整 GSM、source replicate suffix、背景、剂量、时长、角色在 [GSE108214_aligned_samples.csv](GSE108214_aligned_samples.csv)。亲本来源 ATCC、耐药亚系来源 RCCL 以及实验协议见源论文 [Sarin 2018 §4.2–4.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877628/)；原 XML 已绑定 hash。相同 suffix 不提供跨论文重复配对证据。

**没有实测给药前 t=0 RNA。** 对照为 `time_matched_control_reference`，不是同一培养孔或单细胞的给药前状态。耐药两个剂量共享同一组 5 个参考样本，未重复计数或扩增样本。

## 5. 观测尺度与凋亡连接边界

原作者和 GEO 说明 VALUE 已作背景校正与分位数归一化，但没有明确标注 log base。本轮不新增变换，654,038 个原值范围为 **−0.816772203 至 19.28334696**；负值也原样保存。因此不能把它登记为 raw counts，不能直接套单细胞文库归一化或计数损失。

Sarin 2017 的 Annexin 数据是相对于各自未处理对照的 **fold change**，不是绝对 Annexin 阳性比例。与本表达资料的连接限于两个研究中同名明确细胞来源、协议、药物、剂量和时长；没有共享培养孔或重复 ID。具体凋亡值与条件连接仍以 [上一轮原始审查](../state_function_round2/README.md) 为准，本轮未改写也未扩增。

这组资料只有两种状态、一种药物和三个处理条件，能作为有限状态对照的表达材料；不能独自支持多药候选排名、真实序贯处理、同一细胞命运、RL 优势或完整课题数据可行性。跨平台输入如何建模以及相对倍数如何单独评估仍需研究设计和验证。

## 6. 文件与验证

| 文件 | 内容 |
|---|---|
| [probe_gene_mapping.csv.gz](probe_gene_mapping.csv.gz) | 全 29,729 探针的源身份、全部转录本、两种编号桥、冲突/缺失原因与共同轴坐标 |
| [transcript_gene_annotation_subset.csv.gz](transcript_gene_annotation_subset.csv.gz) | 实际命中的官方 GTF 转录本→基因条目，含源 GTF 行号与位置 |
| [GSE108214_original_probe_values.npz](GSE108214_original_probe_values.npz) | `expression` 为 probe×sample float64，全部原探针与 GSM ID 原序保留 |
| [GSE108214_common_gene_values.npz](GSE108214_common_gene_values.npz) | `expression` 为 sample×common_gene float64，另含 mask、gene_ids、sample_ids、value_scale |
| [gene_matrix_columns.csv.gz](gene_matrix_columns.csv.gz) | 单值基因矩阵每个已填列对应的唯一源探针 |
| [validation.json](validation.json) | 实际构建的匹配数、原值检查、尺度、样本分组与未通过事项 |
| [manifest.json](manifest.json) | 输入来源与输出 hash、注释版本边界、完整验证结果 |

实际运行入口为项目 [align_sarin_state_expression.py](../../../../../scripts/align_sarin_state_expression.py)。已经运行成功；它拒绝覆盖存在 manifest 的冻结输出。复算需建立新输出目录并带入本轮四个官方注释文件及 `annotation_download_manifest.json`，随后使用 `--output` 指定该目录，不覆盖这里的版本。

验证结果：全部源 VALUE 文本与原 SOFT 一致；注释行与原平台一致；两个 NPZ 完整回读一致；12,070 个填充基因列逐值等于相应原探针，缺失区为 NaN；4 个针对冲突、多探针、缺失转录本版本、非有限数和轴身份的测试通过。测试是工程校验，不是生物学有效性或模型性能证据。

未修改旧源材料、冻结训练包、公共数据模块或训练配置；未启动训练、修改 Zotero、更新 Memory、提交或推送 Git。
