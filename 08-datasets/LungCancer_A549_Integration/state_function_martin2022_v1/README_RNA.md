# Martin 2022 / GSE193258：source-preserving bulk RNA

2026-10-11。仅下载官方两个 processed 矩阵，完整读取和数值/轴/元数据验证通过。论文为 DOI 10.1158/2767-9764.CRC-22-0066，RNA协议亲读 PMC10035388 Methods sec2-9；逐GSM字段以GSE193258 SOFT及60样本JSON核验。

实际19712 gene-symbol行×60 RNA列；两个矩阵行列和顺序完全相同，无重复symbol/column。源矩阵不含Ensembl ID或版本后缀；GEO data_processing声明hg38/Ensembl v79，不凭此伪造源ID。

| 官方原文件（raw/） | bytes | SHA256 |
|---|---:|---|
| GSE193258_RNAseq_estimated_counts.tsv.gz | 8413632 | 51c92720b7bf4a0d7d29f6b7f33e304bd9f858ba8182572260ddf7f1c4de6d33 |
| GSE193258_RNAseq_log2TPM_abundance.tsv.gz | 8364455 | c4bd1b3d7e158783347a7b678df70bec2c590728a59486d67fcb3a8e21750949 |

下载前官方HEAD 200，Content-Length与完整文件一致；Last-Modified为2022-01-07。官方来源：[GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258)，文件URL及哈希在rna_manifest.json。未下载SRA/FASTQ。

单位：estimated_counts为Salmon/tximport countsFromAbundance=lengthScaledTPM的估计基因计数，非原始UMI或必须为整数的raw count；log2TPM为log2(TPM+0.01)，允许负值。counts全finite、非负，有992546非整数值；log2TPM全finite，419290负值合法。两矩阵各1182720值使用独立csv/float解析与float64 NPY逐项精确比较，源TSV继续保留为数值真相源。

## 样本与时钟

4背景PC9、H1975（NCI-H1975）、HCC827、HCC2935，各5状态×3 biological RNA replicates。rna/rna_samples.csv按矩阵列顺序绑定GSM，rna/geo_samples_source.json保留全部源字段。

| 状态 | 来源协议 | 定位边界 |
|---|---|---|
| parental_DMSO | 亲本无药培养21d，vehicle DMSO24h | 平行reference；12 RNA样本=4背景×3生物重复，并非同一细胞t0 |
| acute_post_treatment | 亲本无药培养21d，再500nM osimertinib24h | 已处理后的RNA，不能冒充该次给药初态 |
| DTP | 500nM osimertinib21d后立即RNA | 已经历诱导的状态 |
| short_washout | DTP21d，两次PBS洗涤、无药24h | 保留真实诱导与恢复历史 |
| long_washout | DTP21d后无药至恢复指数增殖：PC9 7d、H1975 3d、HCC827 4d、HCC2935 10d | 时长按cellline，不能统一为一个天数 |

RNA biological replicates不迁移为功能生物重复。原文未声明same-well RNA+功能，不附功能Y、不把均值复制到3份RNA。无A549样本；bulk不变single-cell，不进入现有单细胞预处理或归一化。

## 资产与共同轴

- rna/estimated_counts.npy、rna/log2TPM.npy：原source gene-symbol顺序×原60列，float64；无标准化/转置/聚合。
- rna/rna_genes.csv：source_row_index/source_gene_id/source_gene_symbol。source_gene_id在这里原样保留symbol字符串，不代表Ensembl。
- rna/rna_samples.csv：column_index/matrix_column/gsm/source_title/cell_line/state/biological_rna_replicate及协议字段；functional_y=not_attached。
- rna/parallel_baseline_samples.csv：仅12个parallel parental DMSO RNA，四个背景各3；不是12个独立状态。
- rna/common_axis_coverage.csv：对joint32461 genes.parquet的精确、双方唯一symbol映射；16167覆盖（49.8044%），16294缺失。歧义joint symbol行0；quarantined_axis_symbols.csv保留空表结构。
- source_observed=false的source_row_index为空，缺失不是实测0；仅轴覆盖表，不生成补零表达。精确symbol对应不证明v79与当前Ensembl跨版本生物学等价，alias/reannotation未核实。
- rna_manifest.json登记源/派生文件bytes/SHA256、数值校验、共同轴固定哈希和全部边界。

## 验证与研究架构一致性

运行`F:/anaconda/envs/study/python.exe scripts/materialize_martin2022_rna.py`与`F:/anaconda/envs/study/python.exe -m pytest tests/test_martin2022_rna.py -q`：14 passed。覆盖非法数值、重复轴、GSM缺失/重复、cellline/long washout时钟错误、歧义symbol隔离、源值readback与缺失mask。

已对照毕业论文研究架构：这是已立项state-conditioned LUAD apoptosis主线的cross-background来源候选资产，不改A549优先验证和主线，不宣称A549内状态增量或功能桥已通过。未做模型/训练/forward/实际预测。功能挑战需独立source-specific assay域、候选块与配对资格审查；该RNA包本身不提供Y。重复层级、same-well配对、跨版本gene映射、实际功能时间与RNA诱导对应尚未完成，不能自动进入核心训练。
