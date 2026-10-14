# GSE306207：A375 有限真实序贯表达

本轮新来源版本为 [geo-public-20261011-v1](geo-public-20261011-v1/README.md)。已从 [GEO 官方记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306207)取得并实际读取 family SOFT 和作者 TPM 工作簿，完成来源字段—样本列—共同基因轴逐值核验。

数据为 A375 bulk RNA，**4 个条件各 2 个作者报告的生物重复**：DMSO、I-BET726 2µM/72h、同一前处理后撤药 72h、同一前处理后 Oxaliplatin 10µM/72h。源矩阵 62,710 基因×8样本；项目共同轴 32,461 基因全部 exact Ensembl 匹配。TPM、浮点数和观察 mask 原样保留，未归一化、拟合或分配全局角色。

只授予已测 A375 历史终点表达的来源资格。DMSO 时间、洗脱和前药保留细节未知；跨条件没有配对细胞/培养批次证明。无 Annexin 标签、A549 迁移或完整序贯决策效果资格。当前模型保持停止。

详细范围和原文定位见[专属报告](../../docs/DATA_ALIGNMENT_SEQUENCE_SOURCE_ROUND22_2026-10-11.md)，实际证据见[独立读回](../../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/independent_actual_readback_receipt.json)。
