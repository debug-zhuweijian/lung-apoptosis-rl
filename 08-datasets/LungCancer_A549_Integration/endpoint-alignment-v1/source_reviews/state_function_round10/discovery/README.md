# 第十轮定向发现与去重记录

日期：2026-10-10。问题是能否取得初态/参考表达、实际处理方案与真实功能结局之间的有效连接，保留已确认的A549主线。本轮没有重新开展无限制选题，也没有将搜索摘要当全文证据。

## 查询与筛选

- SciSpace自然语言查询：Which primary studies in A549 cells link pretreatment transcriptional states or sister-cell transcriptomes to subsequent apoptosis under multiple drugs or doses, with publicly accessible lineage, imaging and gene-expression data that can test whether initial state improves treatment selection?
- 网页查询包括 `A549 pre-existing transcriptional state apoptosis cell fate tracking RNA sequencing sister cells`、`A549 single cell RNA "Annexin" "clones" cisplatin`，随后精确定位CellCage标题及官方公司页。
- SciSpace返回10项论文元数据。逐项初筛/去重不等于10篇全文已读；ReSisTrace等不同体系的结果只作线索，sci-Plex/SCCP等与既有记录核对，没有重复计数。
- 数字DOI尾号723030的宽查询出现气象等无关返回，已丢弃，没有作为证据。
- 检查当前来源目录及paperstudy论文/笔记索引中的标题、DOI和方法关键词。新重点来源为CellCage A549论文；HIF1A、BRG1沿前轮线索补附件；CellCage伴随论文沿已观察的官方链接单独核验。

[原始检索返回](query_results.json)保存SciSpace查询/十项元数据和精确标题网页返回。来源角色和实际读取状态：

| 来源 | 角色 | 本轮决定性核查 |
|---|---|---|
| HIF1A，10.1016/j.omton.2025.201049 | 既有线索的数值/协议补证 | 官方PMC Cloud原表、图、方法；取得80个比值，缺时钟/相容RNA |
| CellCage，10.64898/2026.05.05.723030 | 新的功能—表达连接候选 | 官方XML、Fig5、Supplement 1、官网PDF及数据入口；实际矩阵/CCE表未取得 |
| BRG1，10.1101/2025.05.01.651609 | Cloud线索的限定原文复核 | 作者预印本文字；原图像素、补充file09和正式版全文未取得 |
| Dynamic caging companion，10.64898/2026.05.05.723043 | 定向引文/伴随论文补漏 | 作者PDF数据声明与方法；无A549对象，数据仅未来公开承诺，排除为直接补缺来源 |

没有将不相容体系的功能指标改成A549 Annexin，没有因为访问失败就声称数据不存在。当前停止点是四项定向来源的用途已可判定；没有宣称全领域不存在可用资料。

## Zotero与项目索引

Zotero Local API只读全库顶层条目检查成功，共读60项。按上述三篇HIF1A/CellCage论文的DOI与标题匹配，返回0项；没有只扫描某个collection。结果见[zotero_exact_dedup.json](zotero_exact_dedup.json)。此前三个标题关键词的有效API查询也保存于[zotero_lookup.json](zotero_lookup.json)。

`paperstudy/01-papers/index.md`与`02-notes/index.md`对应标题/DOI关键词未命中。没有向Zotero建条目、写标签、改附件，也没有修改论文索引或项目记忆。正式精读登记另按用户安排进行；本轮只保留执行数据审查来源。
