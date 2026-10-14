# HM97662 A549 bulk 表达精确基因轴对齐 v1

原始 GSE333828 表按六个 A549 GSM 的官方列名读取，保留作者 Salmon gene-level abundance estimates 的 float64 值，未归一化、拟合或四舍五入。

sample_gene_expression.npy 为 6×32,461：前三行 untreated，后三行 HM97662；具体 RNA 重复和 GSM 见 samples.json。gene_mapping.tsv 按冻结共同 Ensembl 轴排序，gene_observed.npy 标明 30,268 个有源值的基因；2,193 个缺失基因保留 NaN，不能当零计数。

RNA 与 Fig.3A Annexin 具有同来源、A549、标称剂量和 144 h 观察时钟；RNA 每 72 h 换液，flow 换液、漂浮细胞收集与重复一对一未说明。RNA_Annexin_condition_links.json 只记录条件关系，不含凋亡数值或学习角色。

作者报告该 A549 条件无显著凋亡诱导；ns 不证明零效应。此来源只有一个活性药物—剂量—时长组合；参考 RNA 为同期对照，不是 t0 或同细胞前后配对。本版本仅完成表达坐标对齐，不授予模型、功能或完整 A/B/C/D 验收资格。
