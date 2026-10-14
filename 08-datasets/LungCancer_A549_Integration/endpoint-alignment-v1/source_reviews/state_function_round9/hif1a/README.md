# HIF1A / BRG1：A549 RNA 与功能材料的定向对应审查

日期：2026-10-10。状态：取得官方 RNA/ChIP 元数据及功能协议证据；尚未形成可接入的高维初态—凋亡候选块。

## 原始来源与版本

- HIF1A 正式论文：[10.1016/j.omton.2025.201049](https://doi.org/10.1016/j.omton.2025.201049)，PMCID PMC12493206。主代理亲读 [官方 XML](PMC12493206.xml) 的方法、Fig2 图注及相邻结果；[提取文本](PMC12493206_text.md:63)方便定位。`source.json` 记录官方 Europe PMC URL 与原文件哈希。
- 引用的 BRG1 作者预印本：[10.1101/2025.05.01.651609](https://doi.org/10.1101/2025.05.01.651609)，2025-05-07 版本。主代理通过[作者全文副本](https://www.researchgate.net/publication/391552844_BRG1_targeting_overcomes_ABCC-based_multidrug_resistance_induced_by_paclitaxel)亲读方法、Fig3–6 图注与相邻结果；尚未完成图像内部数值审核。官方 bioRxiv PDF 获取返回429，未反复重试。
- 正式版 DOI 为 [10.1016/j.jpet.2025.103772](https://doi.org/10.1016/j.jpet.2025.103772)，PMID41371092，版本元数据保存在 `brg1_version_search.json`。**正式版全文未读**，不把预印本审查写成正式版复现。

## 实际读取的 RNA 元数据

[PRJNA1095909 八个 run](PRJNA1095909_run_metadata.tsv)中，四个为 A549 RNA；其他四个为 MDA-MB-231 RNA/ChIP。[PRJNA1096152 三个 run](PRJNA1096152_run_metadata.tsv)均为 MDA-MB-231 ChIP。两份表来自官方 ENA，下载记录与哈希见 `run_metadata_manifest.json`。

| A549 run | experiment | BioSample | 官方 experiment title 对应状态 |
|---|---|---|---|
| SRR28539878 | SRX24139907 | SAMN40738598 | A549 RNA-Seq，未标耐药 |
| SRR28539877 | SRX24139908 | SAMN40738599 | A549 resistant to paclitaxel RNA-Seq |
| SRR28539876 | SRX24139909 | SAMN40738600 | A549 resistant to paclitaxel siCTRL RNA-Seq |
| SRR28539875 | SRX24139910 | SAMN40738601 | A549 resistant to paclitaxel siBRG1 RNA-Seq |

这是四个测序 run 的身份表，不能自动计作四个独立生物重复。当前没有读取 FASTQ 或完整表达矩阵值；这些字段本身也没有锁定急性药物挑战、采样时钟及相应培养重复。

## 功能读出及连接限制

HIF1A Fig2K 比较 siCTRL/siHIF1A、MK571 及其组合背景下的化疗反应。原文方法与图注明确为 **Caspase-Glo 3/7 活性除以 resazurin 读出**；它不是 Annexin 阳性比例。RNA 表中的 siBRG1 不能换名为 siHIF1A。Fig2J 的给药时钟也不直接移填 Fig2K。

BRG1 预印本 Fig3G 的 siBRG1 多药比较是 resazurin 活力；Fig6A–C 的球体 Annexin/PI 是荧光强度。Fig6D 明确 A549-PTX、PFI3 2.5 µM/72 h 与 cleaved-caspase-3 成像，但当前没有锁定该 panel 的化疗剂量、完整观察时钟及可读取数值。不能从 Fig6E/F 活力实验补填。Fig4 的 PFI3 后 ABC 基因 qPCR可作为局部状态线索，不能替代四个高维 RNA 样本的真实处理身份。

以上为原文限定的实验口径。**本课题判断**：当前可以保留“已取得 A549 亲本/耐药/siRNA 元数据”和“局部状态—功能桥接候选”；不能据“BRG1 相关”把不同遗传/化学扰动强配，也不能把强度或代谢归一活性并入比例型 G。

## 当前未取得与下一步

功能补充统计表、原图条件和具体时钟需继续核验；原文列出的附件入口不等于附件内部已读。只下载与当前功能定义有关的小附件，不启动原始 RNA 全库处理。本轮没有新增正式凋亡标签或 RNA—凋亡样本配对，没有训练、联系作者或改动 Zotero。

实际获取限制：Europe PMC `supplementaryFiles` 的内容读取超时；按XML文件名解析的PMC单附件地址返回404，两次均未保存有效附件，记录为 `supplement_fetch_failure.json` 和 `supplement_individual_fetch.json`。PMC网页工具还返回验证页面，未尝试绕过。上述只证明本次获取失败，不证明附件不存在。Cloud独立BRG1审查已完成并归档于[交付提取](../../../../../../experiments/20261010-cloud-dot-coordination/cloud_brg1_delivery.md)；未因作者预印本和正式版标题变化而合并阅读状态。
