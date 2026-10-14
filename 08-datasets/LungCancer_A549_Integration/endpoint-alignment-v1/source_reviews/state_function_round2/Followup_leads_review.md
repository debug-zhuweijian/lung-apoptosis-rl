# 第二轮追加的三个指定线索

核验日期：2026-10-10。此处仅记录原文／原始公开页面实际支持的内容；不能以论文做过某个实验代替公开数据已取得。

## 1. YAP/TEAD：生物学设计接近，当前公开数值仍缺

正式发表：Brosseau et al., *YAP/TEAD involvement in resistance to paclitaxel chemotherapy in lung cancer*, [DOI 10.1007/s11010-024-04949-7](https://doi.org/10.1007/s11010-024-04949-7)，2024-03-01 online，2025 年卷期。

已取得作者 [Research Square v1 PDF](https://assets-eu.researchsquare.com/files/rs-3363457/v1/3c0a6dc7-da70-4306-a527-99be23733901.pdf?c=1697104347) 为 `Brosseau2023_YAP_TEAD_rs3363457v1.pdf`，27 页，2,825,639 字节。`Brosseau2023_YAP_TEAD_rs3363457v1_extracted.md` 按 PDF 页码提取，少量连字解析为缺字；确切文字应回看 PDF。正式出版页已保存 `Brosseau2025_publisher_page.html`。

| 证据定位 | 亲读事实 | 对对齐的影响 |
|---|---|---|
| v1 PDF p5, Cell lines and culture reagents | A549／耐紫杉醇 A549R 均来自 Kent RCCL；耐药亚克隆经长期 20 nM paclitaxel 培养；RPMI、10% FBS、10 mM glutamine、1% penicillin/streptomycin | 是明确定义来源的群体状态，不能与别处同名 A549R 随意连接 |
| v1 p7, RNA sequencing analysis | 亲本／耐药 A549，3×10^6 起始细胞、约 50% 汇合，IV#6 暴露 24 h；重复 3 次、间隔两代。有 ±IV#6 的转录组分析 | 描述了同研究状态表达，但本轮未取得原始／处理后矩阵；也不是同一芯片实验实测 t=0 |
| v1 p8, 3D-culture on-chip；p13 Results | A549 包埋在 2.3 mg/mL collagen；CellEvent Caspase-3/7 Green，每小时成像，共 72 h；STAMP 计数 | 是实际凋亡检测。RNA 在常规培养描述，功能在 3D collagen，不能隐去环境变化 |
| v1 p13 Results | 处理为 paclitaxel 100 nM、IV#6 20 µM、两药联合；亲本对照 DMSO，耐药对照 **paclitaxel 20 nM**；n=3 | 耐药对照不能标成“完全无药”，必须保留维持药背景 |
| v1 p13, patient-derived assay | 后面的患者样本检测为 DRAQ7 全因死亡，48 h | 不能混为 A549 CellEvent 数据，也不能统一为 caspase 阳性率 |
| v1 p14 Discussion | IV 系列化合物化学结构仍保密 | 对当前依赖 drugSMILES 的动作表示是额外缺口；只证明 v1 如此，未推断正式版是否另公开结构 |
| v1 p9 Statistical analyses；正式页 Data availability | v1 为向通讯作者索取；正式页仍写数据可获得性应向作者询问；正式页补充仅两张 PNG | 本轮没有取得 RNA 矩阵／逐重复 Caspase 源数字。未联系作者，不能标为公共数据通过 |

本线索可保留为高度相关实验设计参考。正式全文受订阅限制，细节核验依据是公开 v1，不能假定 v1 与正式版每一个剂量或步骤完全相同。

## 2. Nobiletin / A549-ADR：有亲本—耐药转录组描述，未取得矩阵且凋亡功能主要在耐药株

原文：[PMC6316077](https://pmc.ncbi.nlm.nih.gov/articles/PMC6316077/)，*Nobiletin Enhances Chemosensitivity to Adriamycin through Modulation of the Akt/GSK3β/β–Catenin/MYCN/MRP1 Signaling Pathway in A549 Human Non-Small-Cell Lung Cancer Cells*，DOI [10.3390/nu10121829](https://doi.org/10.3390/nu10121829)。全文 `PMC6316077.xml` 已取得。

- §2.1：作者在自有 A549 亲本上用 adriamycin 0.03→0.5 µM、3 个月建立 A549/ADR，培养为 Ham’s F-12K＋10% heat-inactivated FBS。
- §2.3 Transcriptome Analysis：确有亲本与耐药株的 RNA-seq，描述 TRIzol、TruSeq、STAR、Trinity、RSEM、edgeR；§3.2 报告 400 DEGs。**不能说该论文没有 RNA-seq。** 但本轮全文未发现 GEO/SRA/ArrayExpress accession，未得到样本级表达矩阵。
- Fig 7：明确是 **A549/ADR**；ADR 0.5 µM、NBT 50 µM及联合；SubG1 与 Western cleaved caspase-3/PARP。§3.6 文字给出 cleaved caspase-3 3.41 fold、c-PARP 4.20 fold；这不是 caspase 阳性细胞比例。
- §2.7 cell-cycle 方法给出 48 h；不能把它自动扩展为缺乏独立注明的所有 Western 条件。
- Supplementary Materials 明确只列 Figure S1（其它 phytochemicals + ADR 的 MTT）和 Figure S2（MYCN/MRP1 Western）；本轮未下载该补充 PDF。没有据此假造 RNA 数表。

缺口：未取得 transcriptome 值；缺同条件亲本与耐药株的多药凋亡功能对照；Western fold 的分母及实验时长需逐 panel 固定。不能把 MTT 扩充为 apoptosis 标签。

## 3. Early cisplatin persister：RNA 为筛过幸存者的药后表达，未看到 A549 再挑战闭环

原文：[PMC10616253](https://pmc.ncbi.nlm.nih.gov/articles/PMC10616253/)，*Transcriptional signature of early cisplatin drug-tolerant persister cells in lung adenocarcinoma*。全文 `PMC10616253.xml` 已取得。

- §2.3 Annexin-V/PI：暴露后的浮游细胞与消化的贴壁细胞混合，分两份用于 Annexin 和 caspase；只呈现 24 h 结果；2 个独立实验、各 3 个重复，15,000 个 events。这里需要区分独立实验与孔重复，不能写成 6 个生物独立实验。
- §2.5 Caspase：caspase-3/7 enzymatic activity，按蛋白量归一化，结果为相对各细胞系未处理对照的 fold，**不是 caspase 阳性百分比**。
- §2.7 RNA-seq：在顺铂之后**丢弃上清死亡细胞，反复洗去漂浮／死亡细胞**，随后消化贴壁存活细胞取 RNA；活力>95%。这是有选择的 DTP 药后群体表达，不能代表原始所有细胞或先验初始状态。
- Fig 1 的不同敏感性来自 A549、H1299、H1573、3B1A 等不同细胞系；不能把它们换名为 A549 的多个状态。后续 SOCS1 knockdown 功能主要针对 3B1A，不能冒充 A549 预处理—再挑战。
- 本轮没有从此论文建立“已经形成的 A549 DTP 初始状态 → 新一轮多药 Annexin/Caspase”的源数据连接。相关同次暴露终点可以服务另一种终点分析，但不能覆盖本轮缺口。

## 4. 本轮追加检索日志

以下是本次追加实际执行的定向查询，不声称覆盖整个领域或完成系统综述：

| 查询／入口 | 实际结果与下一步 |
|---|---|
| `"YAP/TEAD" "paclitaxel" "A549" resistance` | 找到 PMID38427166、正式 DOI、作者 Research Square v1；正式页与 v1 原文均已读取 |
| `"PMC6316077" RNA nobiletin` | 联合工具结果未提供有用独立来源；改为 Europe PMC 的 PMC6316077 原 XML 核验 |
| `"PMC10616253" cisplatin RNA persister` | 返回原 PMC；下载 XML，逐读 Methods 与 Fig1 |
| `"Brosseau" "A549R" "GSE"` | 返回不相关 YAP 工作，未发现可用 accession；这不是“没有数据”的证明 |
| `"YAP/TEAD involvement" "data" sequencing` | 定位公开 v1 的 upon-request 声明；回到正式出版页 Data availability 复核 |
| 正式 Springer URL 与 Research Square 页面 | Springer 可读摘要／数据声明／补充列表；Research Square HTML 工具错误，改取同平台官方 PDF 成功 |

这 3 个指定候选均已去重并进行了关键全文段落核查，但没有新增可导出的完整状态×小分子凋亡矩阵。完成度事实与检索阴性结果分开保留。
