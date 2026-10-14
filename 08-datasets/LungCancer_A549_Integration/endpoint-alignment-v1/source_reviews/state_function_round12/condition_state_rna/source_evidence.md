# 核心原件定位

核验日 2026-10-10。本文件为定位表；解释与可用性见 README.md。XML 的段落 id 可直接定位，不把网页的动态行号当论文页码。

| 本地原件 | 直接定位 | 核实内容 |
|---|---|---|
| PMC5527746.1.xml | p#P10；siRNA transfection and RNA-seq data analysis | 72 h 转染后 RNA；两个独立 siRNA；后续实验改等量 pool |
| PMC5527746.1.xml | p#P13；Cisplatin treatment and apoptosis assay | 转染后24 h 起加药，72 h 暴露，Annexin-V/PI |
| PMC5527746.1.xml | p#P14；Statistical analysis | mean±SD |
| PMC5527746.1.xml | p#P22 | 顺铂20 µM/72 h 的 A549 两个正文均值；组别顺序语义问题 |
| PMC5527746.1.xml | fig#F5/caption | early/late apoptotic cells、重复三次；正确语义图应匹配实际 f6 文件 |
| GSM2355674.soft.txt | Sample_title / characteristics / extract_protocol / data_processing | 唯一已下载 A549 control、72 h、edgeR normalized counts、hg19 |
| GSM2355675.soft.txt / GSM2355676.soft.txt | 同上 | RNA 两个 siRNA 分别处理，不能当 pool 重复 |
| RNF20_pubmed_observed_figure_links.json | 各 figure_id/href | 真实 DOM URL；图内容与编号错位，不猜 URL |
| nihms874355f6.jpg | 面板 c/d；A549 行/柱 | 凋亡图实际内容、图例与方向；非精确来源均值表 |
| PMC8050912.1.xml | p#Par21 | Annexin V/FACScelesta，未披露详细母门 |
| PMC8050912.1.xml | p#Par39 / Fig2 caption | A549 三背景、cisplatin 5/10 µM 48 h，n≥3，SEM |
| 13046_2021_1921_MOESM1_ESM.docx | RNA-Seq lines 段 | 四份 A549 siNFKBIA、两份 scramble；未列逐样本 clone key |
| IKBa_supplement_text.md | “RNA-Seq lines”后的段落 | 从 DOCX document.xml 实际提取，不是二手摘要 |
| PMC8050912.1.xml | Availability of data and materials | 仅未来可用声明 |
| PMC10701619.1.xml | Fig4C–E、Cytotox Green 方法 | 死亡对象计数，不能视为 Annexin；RNA 时点与药物前48 h准备不直接重合 |
| PMC12230637.1.xml / CTM2-15-e70392-g002.jpg | Fig2I、5.14、data availability | shNC/shFTO Annexin-FITC 阳性图，RNA/MeRIP 自有项目；详细时钟未核齐 |
| PRJNA1196085_bioproject.md | accession 下的项目说明/Project Data | A549 FTO knockdown mRNA；12 experiments、6 BioSamples，不当12生物重复 |
| PRJNA1196111_bioproject.md | 同上 | MeRIP；6 experiments/6 samples，不能替代普通表达输入 |

primary_evidence_locator.json 是 RNF20/IκBα 的部分原文段落，含原 XML 文件名和 id。RIF1_RBIS_FTO_fulltext_bundle.md 为一次 Exa 返回的三篇全文串联，标题和每段 URL 保留，不能只按文件首篇标题归属全部 accession。PMC7756122_fulltext.md 只有可返回的摘要/元数据，不声称本包已阅读全文。

