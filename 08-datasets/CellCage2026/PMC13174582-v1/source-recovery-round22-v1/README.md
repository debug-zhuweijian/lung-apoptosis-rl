# CellCage A549 来源恢复，metadata 子版本

状态为 **metadata_only_experimental_RNA_and_CCE_mapping_pending**。`metadata/` 包含25个实际HTTP响应，24次200与1次403；[manifest.json](manifest.json) 保存请求身份、文件大小、SHA与内容类型。响应完整性及JSON结构实际读回通过，不能据此将实验矩阵登记为已下载或已对齐。

对应论文 DOI `10.64898/2026.05.05.723030`，v1。官方 bioRxiv metadata 与旧审查所用2249字节响应SHA相同。重新读取和核验31个旧原文/补充审查文件，全部SHA匹配，未复制旧大型补充包。

实际新内容包括：bioRxiv版本、Github公开仓库搜索及3个完整tree、GEO/BioProject/SRA查询、ENA study查询、EuropePMC数据链接响应、Zenodo精确DOI查询及不同稿件受限记录、官网论文页/Demos/Products真实链接、一个被排除的GEO/BioProject候选的SOFT/XML。目录不包含原始影像、视频、FASTQ、模型参数、合成表达或零占位矩阵。

标题查询返回的 GSE338040/PRJNA1492146 实际为 MCF10A/HCT116 的 PENNE 辐射实验，18个样本；它不是本 CellCage A549 数据。Zenodo17584933 的题名/作者不等同于本论文，`access_right=restricted`，官方文件列表403；未申请权限或联系作者。Github安装manifest也不是RNA/CCE来源。

本源实际读取：RNA矩阵0，RNA数值0，CCE–影像–RNA映射表0，已核源CCE ID0。共同轴的32,461基因metadata另有读回，但源gene ID未取得，映射数及缺失mask保持null。fit/selection/final未分配，真实PT/NN/forward/fit全部停止。

- [实际读取回执](../../../../experiments/20261011-data-alignment-round22/cellcage_source_recovery/actual_readback_v1.json)
- [来源合同与所需文件](../../../../experiments/20261011-data-alignment-round22/cellcage_source_recovery/data_contract_v1.json)
- [候选逐项排除证据](../../../../experiments/20261011-data-alignment-round22/cellcage_source_recovery/candidate_source_classification_v1.json)
- [完整定向报告](../../../../docs/DATA_ALIGNMENT_CELLCAGE_SOURCE_ROUND22_2026-10-11.md)

此版本供后续取得实际processed矩阵与行表时核对来源，不建立本源模型消费资格，不把终末CCE汇总RNA改成初态或单细胞RNA，不把四类影像表型直接改成核心Annexin-positive fraction。
