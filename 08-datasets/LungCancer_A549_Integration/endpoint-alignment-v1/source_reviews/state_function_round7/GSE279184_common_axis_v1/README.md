# GSE279184：六列公开计数到共同 ENSG 轴

日期：2026-10-10。**已完成全部 100,020 个公开计数的回读和可追溯基因映射：15,697 个基因进入当前 32,461 ENSG 轴，输出 `6 × 32,461` 表达矩阵及显式 mask。** 缺失坐标为 NaN，没有补零、归一化、训练或建立 RNA—凋亡配对。

## 来源、尺度和处理表范围

来源为作者上传至 [GSE279184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279184) 的 `GSE279184_Gene_counts_replicates_UT_and_salinomycin_treated.csv.gz`，本地原件保留在 [induced 目录](../induced/GSE279184_Gene_counts_replicates_UT_and_salinomycin_treated.csv.gz)。SHA256：`45dd9a31edaec75bc70f344bc128ab9ae16c05c1cfed7d340206017b7c6a837d`。

亲读 [PMC11670111 原 XML](../induced/PMC11670111.xml) 的 §s14、§s21、Fig. 2C 和 Fig. 3B/3C，以及系列和全部六个 GSM SOFT。作者说明为 bulk RNA-seq 的去重复 gene counts：UMI-aware 重复标记使用 Rumidup，GenSum 采用 Ensembl release 102 transcript GTF。此处保留作者公开表原值，未自行进行文库大小归一化或 log 变换。

公开表为 **16,670 个 GeneSymbol 行、17 列**。只有以下六列作为计数：`ut_r1`、`ut_r2`、`ut_r3`、`tr_r1`、`tr_r2`、`tr_r3`。其余十个 `baseMean`、`log10BaseMean`、`log2FoldChange`、`stat`、`pvalue`、`padj`、`log10BaseMeanA`、`log2FoldChangeFitted`、`log2FoldChangeMinFitted`、`foldChange` 均按原文本保存为独立作者统计表，没有当作表达样本或 counts。

**范围只标记为 `author_deposited_processed_table`。** 本轮没有建立作者筛选规则或完整未经加工矩阵的证据。实际六列数值均为有限非负整数，最小值 1、最大值 317,090；有 9,275 个值为 1，没有值为 0。没有据此推断作者一定做了何种过滤/伪计数，也没有擅自减 1。“本轮没有再归一化”不等于已经独立证明此公开处理表从未经过任何作者侧处理。

机器可读的 manifest、validation、样本表和 NPZ 统一登记 `measurement_scale=author_deposited_processed_count_table`、`raw_unmodified_count_matrix_verified=false`、`transform_applied=none`；manifest 和两 NPZ 另有 `joint_single_cell_input_ready=false`。作者方法中的计数名称单独保留为 `author_scale_description`，不把本表直接接入 joint 单细胞 raw-count 文库归一化。通用映射器中的 `bulk_raw_counts` 参数仅用于非负整数数值域检查，不作为本数据的外发观测尺度标签。

## 六个 GSM 与状态、时点

列对应从每个 GSM 的**完整标题处理组和末尾 `biological replicate 1/2/3`**核验；标题中间的 `_R1` 在所有六个样本均相同，未拿它作为列重复编号。每个 GSM 的 `Sample_data_row_count` 为 0，因此不能声称又取得了独立的每 GSM 数值表来验证条码对应；这是作者命名所支持的列级对应。

| 公开计数列 | GSM | 作者分组与本次登记 |
|---|---|---|
| ut_r1 | GSM8563730 | untreated senescent，biological replicate 1 |
| ut_r2 | GSM8563731 | untreated senescent，biological replicate 2 |
| ut_r3 | GSM8563732 | untreated senescent，biological replicate 3 |
| tr_r1 | GSM8563733 | salinomycin-treated senescent，biological replicate 1 |
| tr_r2 | GSM8563734 | salinomycin-treated senescent，biological replicate 2 |
| tr_r3 | GSM8563735 | 原题拼为 `Saliomycin`，biological replicate 3；原拼写保留 |

六组均为 **alisertib 诱导衰老 A549**。原文 §s14、Fig. 2C 与 GEO series 支持 alisertib 0.5 µM、约 7 d 诱导；RNA 处理组为 **salinomycin 2.5 µM × 48 h**。UT 是该衰老群体的未处理平行参考，**不是未诱导亲本 A549，也不是同一样本的实测 t=0**。UT 的单独采样持续时间没有从其 GSM 独立明确，因此 `salinomycin_treatment_h` 留空，不伪填 0 或 48。

原文 §s14 描述追加处理前洗 PBS、trypsinize、重铺；这里不将计数列重复编号解释为跨处理的同一孔配对。完整来源与语义见 [samples.csv](samples.csv)。

Fig. 3B 的凋亡等死亡通路 WB 使用 **2 µM × 48 h**，与 RNA 的 2.5 µM 不同。因此本次没有生成同条件 RNA—WB/Annexin 标签连接，更没有逐细胞未来命运标签。

## 唯一身份映射规则

复用现有 [共同基因映射器](../../../../../../scripts/align_state_function_expression.py)所使用的 `src/lung/data/external_gene_axis.py`，以 ENSG 将最终确认的编号投影至共同轴。基因身份参考为当前 `prepared/v1/gene_map.parquet` 及已冻结的官方 HGNC 2026-10-10 快照；**该 HGNC 快照不是作者 Ensembl 102 注释文件**。

1. 先检查共同轴同名 symbol 与 HGNC 当前 approved symbol 的明确 ENSG；两者冲突或指向多个 ENSG 则隔离。
2. 没有上述直接候选时，才使用官方 `prev_symbol` / `alias_symbol`；仅全部支持候选合计一个 ENSG 时采用，不挑选恰好在共同轴内的一方来掩盖歧义。
3. 占位符、缺失或不支持的名称不猜测。多个源行即使名称不同，只要落到同一 ENSG，也由现有映射器隔离，不能求和、平均或任选一行。
4. 只有唯一已支持身份且处于目标轴中的行写入矩阵；不在轴中及隔离行仍在原始计数文件和映射账本中保留。

| 结果 | 源行数 |
|---|---:|
| 成功映射 | 15,697 |
| 已解析 ENSG 但不在当前共同轴 | 969 |
| 身份冲突或歧义 | 3 |
| `NO_GENE_NAME` 占位符 | 1 |
| 合计 | 16,670 |

成功映射包含 15,659 个直接身份和 38 个唯一官方旧名/别名连接。本次没有重复源 symbol 或多个源行最终碰撞同一 ENSG 的实际记录；该边界已用合成测试验证。

三个冲突为 `ARMCX5-GPRASP2`、`CCL3L1`、`NPIPA9`，当前 HGNC 与共同轴 symbol 分别指向不同 ENSG。两个候选都保留在台账，没有把某个注释版本未经核查地判为“错误”。最终共同轴有 **16,764 个未观测基因列**，均为 NaN/mask=false，不能解释为测得零表达。

## 文件与数值验证

| 文件 | 内容 |
|---|---|
| [aligned_counts.npz](aligned_counts.npz) | sample×gene 表达、observed_mask、GSM、原列名、完整 ENSG 顺序及每目标基因对应的原行号 |
| [original_six_column_counts.npz](original_six_column_counts.npz) | 六列全部 100,020 个原整数，包含未匹配基因行 |
| [gene_mapping.parquet](gene_mapping.parquet) / [CSV](gene_mapping.csv.gz) | 16,670 源行的 symbol、HGNC ID、各组候选 ENSG、身份依据、坐标和最终状态 |
| [author_statistics_not_counts.csv.gz](author_statistics_not_counts.csv.gz) | GeneSymbol 与十个作者统计列的原文本，未用作 counts |
| [validation.json](validation.json) | 实测匹配数、原值回读、范围和未解决边界 |
| [manifest.json](manifest.json) | 原计数、GSM/series、论文、轴、官方注释、映射器与派生文件的 SHA256 |

实际用第二个独立 CSV 解析器重读 gzip 原件全部 100,020 个数字并比较；原始 NPZ 完整回读一致；所有 94,182 个已映射数值与其原行、原样本对应计数精确一致；对齐 NPZ 的表达和 mask 完整回读一致；缺失区全部 NaN。未匹配值没有丢弃，仍保留在原始六列 NPZ 中。

复算脚本：[align_gse279184_expression.py](../../../../../../scripts/align_gse279184_expression.py)。冻结副本及所用通用映射器副本也保存在本目录。运行：

```powershell
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_gse279184_expression.py -q
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/align_gse279184_expression.py
```

五项合成边界测试通过（最终语义字段复核后为 0.62 s），真实小表映射与回读用时约 2.90 s；没有读取旧大矩阵或影响联合预处理任务。脚本拒绝覆盖已有版本，重新构建应指定新 `--output` 目录。交付前将观测尺度字段明确化，计数与 mask 未改；最终 manifest 记录该次 metadata revision，不使用此前临时 manifest 的 hash 作为最终交付版本。

本轮仅新增脚本、测试和本目录。`induced/` 原文件、主索引、其他源码与训练配置未改。该产物符合研究架构的共享基因坐标与来源观测边界；它不是单细胞数据、配对凋亡训练集、已验证模型输入效果或完整研究可行性证明。
