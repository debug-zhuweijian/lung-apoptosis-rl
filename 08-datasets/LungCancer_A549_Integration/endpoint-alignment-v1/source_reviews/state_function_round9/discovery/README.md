# 第九轮定向补证与去重记录

日期：2026-10-10。目的：补查处理前 A549 状态、真实凋亡功能与多候选之间的具体数据连接；不按论文数或细胞数宣称完整可行性。

## 已执行入口

- SciSpace：完整问题与10条返回元数据保存在 [query_results.json](query_results.json)。SCCP 的预印本和正式版本按同一工作去重。
- Sider：定向检索返回需要重新认证；错误保留在同一 JSON。不把失败表述成无文献，也未为此扩大权限。
- 主代理网页查询涵盖 A549 quiescent/Annexin/RNA-seq，以及 pretreatment/transcriptomic/apoptosis/resistant；随后沿 HIF1A 原文的 BRG1 引文做定向版本与全文核查。
- Cloud：三组限定查询、两篇深入核查及随后 BRG1 定向审查。完整查询和返回边界见 [Cloud 第九轮原文交付](../../../../../../experiments/20261010-cloud-dot-coordination/cloud_round9_delivery.md)。其首次将 Frontiers2023 当新来源，后已按本地 Round2 记录去重。

## 来源资格

| 来源 | 本轮核查及处置 |
|---|---|
| SJB2-043，10.3390/cimb47030155 | 实读 Fig4、方法及补充；真实24h剂量均值与代表图分开，名义材料核定；RNA文件未取得。详见 [SJB审查](../sjb/README.md)。 |
| SCCP，10.1021/acs.analchem.2c00413 | 主代理实读正文：未来死活来自蛋白组聚类与晚期群体的解释，不是同细胞未来命运标签；不进入凋亡桥。详见 [SCCP审查](../sccp/README.md)。 |
| HMOX1，PMC12869221 | 已读正文涉及活力与铁死亡，当前不能作 Annexin/caspase 数据；不外推为所有附件均无此数据。详见 [HMOX1审查](../hmox1/README.md)。 |
| HIF1A 与 BRG1 | 官方 ENA 中确有4个 A549 RNA run；高维 RNA 的 siBRG1 与功能 siHIF1A/PFI3 不互换。详见 [定向审查](../hif1a/README.md)。 |
| IJO2019，10.3892/ijo.2019.4690 | 主代理亲读[官方正文](https://www.spandidos-publications.com/10.3892/ijo.2019.4690?text=fulltext) SILAC 方法、Results 与 Fig3图注：二群体0/10µM顺铂24h Annexin；蛋白比较同样在给药后。保留固定药物状态挑战线索，未提取图内比例，未新增高维初态或多药标签。 |
| Frontiers2023/PMC10616253、NOTCH1/PMC12780219、Gonzalez-Gualda2026/PMC12920146 | 既有 Round2/5 来源，按 DOI/PMCID 去重；不重复计作新增数据。 |

## 个人库核对与停止边界

[Zotero只读核对](zotero_check.json)通过 Local API 取得60个顶层条目，返回总数同为60；记录的五组关键词未命中。结论仅针对这些关键词，不表示所有新增 DOI 已全库精确查尽。`01-papers/index.md` 对本轮已查术语没有命中，未新增精读笔记或写回 Zotero。

后续[精确核对](zotero_directed_followup_check.json)再次实际读取全部60个顶层条目，BRG1预印本/正式版、HIF1A、IJO2019的四个DOI及BRG1标题词均未命中；论文索引的对应编号也未命中。没有因此重复创建任何条目。

本轮判断限于上述检索和原件实际读取范围。公开数据缺少处理时钟、检测定义或共同样本键时保留缺口；接口可构造、局部标签可用与整体验证成立分别报告。
