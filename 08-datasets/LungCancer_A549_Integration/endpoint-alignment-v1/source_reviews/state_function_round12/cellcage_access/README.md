# Round12 CellCage 数据入口复核：本支线停止下载

日期：2026-10-10。论文：Khurana 等，*Scalable longitudinal imaging and transcriptomics of cells in dynamic enclosures*，DOI `10.64898/2026.05.05.723030`，v1。

**本次先判定是否存在值得下载的新文件：结果为 0 个具体 A549 数据候选入口。停止此支线，不重复下载论文，不把论文实验描述升级为“已取得配对训练数据”。**

本轮重新读取第十轮 README 和保存的原始响应，并核验其 **3 个原始输入、28 个审查文件 hash**。这些入口在同一天已实际访问、保存和读取，因此本轮复用仍有效的证据，不重新抓取相同 PDF/网页。没有新网络请求或数据下载，没有运行模型。可逐入口查看 [entry_audit.json](entry_audit.json) 中的实际链接和响应。

## 能打开的入口具体通向什么

| 已核实入口 | 实际内容 | 原始/处理 RNA 矩阵或 CCE 行映射 |
|---|---|---|
| 论文 fullTextXML | 正文、图、补充 DOCX；没有 accession、数据仓库或具体 R 包地址 | 无入口 |
| bioRxiv 官方 metadata | 返回 v1 和原文 JATS XML 地址 | 无入口 |
| EuropePMC `databaseLinks` | 成功响应但没有数据记录 | 无入口 |
| EuropePMC `labsLinks` | 只有 Altmetric | 无入口 |
| EuropePMC `supplementaryFiles` | 13 个成员：12 个正文图片和 Supplement 1 DOCX；DOCX 为 Fig. S1–S7 和抗体表 | 没有数据表/代码 |
| 官方论文专页 | PDF 按钮和 DOI；`View on bioRxiv` 也指向同一个 PDF | 无入口 |
| 官网 Products | 说明可以导出 `.rds`、提供 R packages/vignettes | 没有对应下载地址 |
| 官网 Technology | 介绍 CCE ID 将影像和终末 RNA 连接的产品能力 | 没有该论文 A549 行级数据 |
| 官网 Demos & Data | 视频和联系表单 | 没有该论文 A549 行级数据 |

上述实际网页锚点、PDF 全部 URI 和补充包成员名均写入机器可读入口清单；本次没有依靠网页标题“Data”或 `hasData=Y` 推断已经公开原始数据。PDF 中 `pending:yes` 对应 Supplementary Figures 链接，也不是可访问的数据仓库或数据发布承诺。

原出版页的 403/429、PMC 浏览器挑战记录被保留为访问限制。本次没有绕过或重试这些限制；可用的官方 PDF 与 XML 已足以核对正文是否提供数据地址。**这些限制与“已读来源未列出行级数据入口”分别记录，不能统称为需要账号或付费。**

## 实际可读数据数量与仍缺的字段

截至本次复核，在这套已核实入口中：

- A549 原始或处理表达矩阵：0。
- CCE—capture spot／RNA matrix 列—影像标签对应表：0。
- 已读取并核验的 A549 CCE ID：0。
- 作者公开 R 包源码／分析 vignette：0。
- 新出现的可下载 A549 子集或元数据 URL：0。

没有源表，因此不能声称已经核查其列名、唯一键、缺失值或逐行配对。继续之前需要具体的 matrix/export 地址，以及 CCE 到 RNA 列和时序影像表的映射；随后才可能验证最小 A549 子集。

## 时间与死亡选择的现有边界

这些是第十轮已亲读原文得到的实验边界，本轮没有新的行级资料可以增强它们：

| 环节 | 原文层面的已知内容 | 行级未得信息 |
|---|---|---|
| 0 h | 给药前影像基线；未测同 CCE 初态 RNA | 原始基线表、细胞实例 ID 与 CCE 映射 |
| 随访 | 8/16/24/32 h 影像 | 各时点 Annexin、live/dead、细胞计数和跟踪缺失表 |
| 32 h 末端 | 最后影像后裂解，RNA 按 CCE 汇总；CCE 可包含多个后代 | 表达列 ID、实际终末细胞构成、与影像的精确 join |
| 死亡标签 | 四类表型利用活/死数量和历史 | Annexin 数值 gate、标签脚本及互斥优先级 |
| RNA 选择 | 按转录本、capture overlap 和线粒体比例做 QC | A549 数值阈值、逐 CCE 排除清单、死亡相关 RNA 丢失比例 |

因此，当前既不能用末端 RNA 冒充初态，也不能证明 RNA 捕获对死亡结局无选择偏倚。论文支持 CCE 级实验连接的候选价值，尚未提供本任务已可读取的训练接口。

## 停止条件与后续触发

此次满足主代理给出的停止条件：**没有任务相关的实际新增文件入口**。不扩成其他细胞系或 companion 论文，不再以相同原文下载充当数据接入进展。

仅在出现新的、可定位的公开 A549 数据文件/代码入口，或经用户明确授权获得资料后，才重新打开本支线。这个结论限定于本次所读版本与已列出的官方入口；不证明作者从未采集数据、任何地方都不存在数据或未来不能开放。

本轮所有文件只写入 `state_function_round12/cellcage_access/`。旧 CellCage 包、核心代码、训练、Zotero、Memory 和 Git 提交均未改动。

## 来源与校验

- 前轮主证据：[Round10 README](../../state_function_round10/cellcage/data-review/README.md)。
- 前轮 manifest SHA256：`6b1c0b1598d76353d2a627f02d755888605ed6c653b66cd9980181f9a7e965e3`。
- 本轮可重查原始入口清单：[entry_audit.json](entry_audit.json)。
- 本轮输入/输出 hash：`manifest.json`；核验结果：`verification.json`。

这里报告的是同日保存响应的本地重核，未宣称再次完成一轮全网搜索或新的在线访问。
