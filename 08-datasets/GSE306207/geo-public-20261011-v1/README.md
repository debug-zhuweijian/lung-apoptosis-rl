# GEO public 2026-10-11 来源快照

来源 [GSE306207](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306207)，BioProject PRJNA1309681，平台 GPL29480。GEO 发布 2025-12-22、记录更新 2026-02-06；本轮实际取得的 family 文件 HTTP Last-Modified 为 2026-08-08，TPM 工作簿为 2025-08-22。来源分别保留版本字段与文件字节 SHA，不把 HTTP 修改时间当成实验或运行日期。

官方原文件：`GSE306207_family.soft.gz`、`GSE306207_A375_Seq_All_Samples_TPM.xlsx`；辅助原件为官方 full text/目录 HTML、SequenTx 出版页 HTML及其链接的 `42256_2026_1192_MOESM1_ESM.pdf` 和 Reporting Summary `MOESM2_ESM.pdf`。没有下载 SRA/FASTQ。出版页 DOI `10.1038/s42256-026-01192-1`，2026-03-04 正式论文使用 SequenTx 名称；GEO摘要沿用 AlphaTherapy，不擅自修改源字段。

[派生包 manifest](processed/source-observations-joint-axis-v1/manifest.json)登记原单位 `source_RSEM_TPM_GRCh38`、`base_transform=none`、`role=unassigned`及文件 SHA。

- `source_tpm.npz`：8×62,710，source gene ID、GSM ID与 float64 TPM。
- `joint_tpm.npz`：8×32,461，冻结有序共同基因 ID、float64 TPM、bool observation mask；本来源全部匹配、mask全真。测得的零值不是缺失。
- `sample_annotations.json`：8个唯一 GSM 的原 title/description 对矩阵列的绑定、4个历史条件、时钟/单位/未知字段及 SOFT 行号。
- `gene_axis_ledger.tsv`：每个 joint gene 的确切源行，不做 symbol、旧 ID 或多对一猜配。

原工作簿 sheet 为 `A375_Seq_All_Samples_TPM`，62,710 行和 `gene` 加8列。可在已安装的 `F:/anaconda/python.exe`（pandas2.3.3、openpyxl3.1.5）读取；研究 study 环境缺 openpyxl 的首轮失败已保留，没有为此安装依赖或重复下载。规范获取与读取脚本位于[本轮证据目录](../../../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/)。

这是新的独立来源快照，未接入已有 A549 joint 训练包或冻结职责划分。基因身份相同不能证明 TPM 与 single-cell counts/既有 latent 或 decoder 的兼容；源适配与真实模型评价另行验证。
