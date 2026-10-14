# HMOX1耐药研究：主文功能读出不补成凋亡标签

2026-10-10定向核验。[Heme oxygenase 1 (HO-1) is a drug target for reversing cisplatin resistance in non-small cell lung cancer](https://pmc.ncbi.nlm.nih.gov/articles/PMC12869221/)。主代理取得官方XML并读取细胞培养、耐药构建、细胞增殖、组合药物、转录组及功能结果。源文件及SHA256见`source.json`。

这篇确有A549/A549-R和HMOX1过表达的转录组研究，但主文[Cell proliferation assay及组合药物方法](PMC12869221_text.md:93)使用CCK-8；后续重点是铁死亡相关验证。抑制剂救援、GSEA中出现apoptosis或电镜中的apoptotic body，都不能替代可对应的定量Annexin或caspase终点表。

文中复用GSE108214和GSE84146是外部表达验证，不等于本研究自建A549-R与既有耐药株是同一材料。A549-R培养基含维持耐药的cisplatin；不能只按“A549 resistant”名称跨研究配对。

本轮仅审主文，未宣称所有补充附件都已核验。没有把CCK-8或铁死亡结果写入凋亡训练表；本源未关闭当前缺口。已有Sarin/GSE108214保持其原始材料与实验条件，不因本文再次引用而重复计为新增。
