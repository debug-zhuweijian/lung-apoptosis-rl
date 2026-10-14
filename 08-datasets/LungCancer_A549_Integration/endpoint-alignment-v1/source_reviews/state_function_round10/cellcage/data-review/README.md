# CellCage：CCE–影像–终末 RNA 独立数据读取审查

日期：2026-10-10。限定论文：Khurana 等，*Scalable longitudinal imaging and transcriptomics of cells in dynamic enclosures*，DOI `10.64898/2026.05.05.723030`，PMID `42146668`，PMC `PMC13174582`。bioRxiv 官方 API 本轮返回一个记录：v1，2026-05-08，`published=NA`。以下结论针对该版本，不把平台营销文字或其他论文当成本实验结果。

**结论：论文的确描述并分析了“同一个 CCE 小室的纵向影像表型 ↔ 该 CCE 的终末 RNA”连接，是有实测功能关联的候选来源。但本轮尚未取得带 CCE ID 的公开 RNA/影像对应表，不能将其登记为已可读取、已对齐或可训练数据。CCE 可能包含多个后代，RNA 按整个小室汇总；该 A549 实验没有初态 RNA，只有一种药物的一个剂量与对照。**

本轮只写本 `data-review/`；没有改原 XML/MD/元数据、核心代码或已有数据包，没有整库下载、训练、联系作者、提交表单或创建账户。

## 1. 亲读证据和定位

| 证据 | 核查内容与位置 |
|---|---|
| [原始 XML](../PMC13174582.xml) / [原文 MD](../PMC13174582.md) | XML `S4`：capture spot 与 CCE 对应；`S6`：A549 结果；`S15`：RNA 聚合及 QC；`S20`：A549 培养、给药、影像和分类方法 |
| [Fig. 5 原图](figure5.jpg) | 来自官方 supplementaryFiles 返回包，已亲看；A 面板显示 −24、0、8、16、24、32 h，部分 0 h 小室已经有多个细胞；D/G 包含 dead 等表型 |
| [官方公开 PDF](official_preprint.pdf) | 从 Cellanome 该论文页面的真实 PDF 链接取得，32 页 bioRxiv 网页打印版；第 19 页 QC 段、第 21–22 页 A549 段、第 24 页 supplementary link 再核对 |
| [Supplement 1](media-1.docx) / [提取文本](supplementary_text.md) | 34 个非空段落、1 个抗体表；内容为 Supplementary Fig. 1–7，没有原始 CCE/RNA/影像对应表，也无外链关系 |
| [原文片段及链接审计](source_contract_excerpts.json) | 保留精确 XML section/paragraph ID、非参考文献外链、源文件 hash |

在线原始全文来源：[EuropePMC fullTextXML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13174582/fullTextXML)。在线补充材料来源：[EuropePMC supplementaryFiles](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13174582/supplementaryFiles)。上述链接均直接对应这篇论文；补充包包括正文图片及 DOCX，不是原始实验数据集。

## 2. CCE 到底是什么单位

平台以位置/几何关系将 capture spot 分配给 CCE，一个 spot 不分配给两个 CCE；一个 CCE 可覆盖一个或多个 spot。STARSolo 先生成基因 × capture spot 的转录本计数，再将属于同一 CCE 的所有 spot 按基因求和。**最终表达对象是 CCE，不是保证只含一个终末细胞的对象。**出处：XML `S4`、`S13`、`S15`。

A549 的 daughter-cell-resistant 类明确要求末次影像中同一 CCE 内既有至少一个活细胞、又有至少一个死细胞；proliferation 也可能有多个细胞。因此，至少这些记录的 RNA 是小室内保留下来的物质共同形成的汇总，无法凭现有说明分配到其中某一个后代。图 5A 还显示部分 0 h 给药前基线 CCE 已含多个细胞；不能假定每个给药初态都恰好是一个细胞。

原文的平台目标包括围绕单个细胞形成 CCE，但未给出 A549 全体小室的单细胞入室 QC 表、基线细胞数表或谱系表。不能把所有 CCE 一概称作已经核验的单克隆，也不能把捕获 spot 数当作细胞数。

## 3. 什么是真实对应，什么没有测

| 问题 | 本文证据与边界 |
|---|---|
| 是否有同一实验单位的影像与 RNA？ | 论文方法和图 5 支持 CCE 级连接；正文按该连接比较了四类影像表型和终末表达。实际 CCE/barcode 行连接尚未取得，当前为论文级证据 |
| RNA 在何时？ | 最后一次 32 h 影像之后裂解，做 3′ mRNA-seq；不属于 0 h 初态 RNA |
| 0 h 有什么？ | Annexin V-FITC 参与的基线影像，之前在 fibronectin flow cell 培养 24 h；没有同一 CCE 的给药前 RNA 测量 |
| 中间时间点有什么？ | 每 8 h 刷新培养液后拍摄，持续至 32 h；是纵向影像，不是反复测 RNA |
| 是否可叫同期 RNA–Annexin？ | 可谨慎称为“终末 RNA 与临近末次 Annexin 影像在同 CCE 内关联”，但四类标签还使用细胞数量和历史；没有证据把它们直接等同于单细胞 Annexin 二分类或群体 Annexin 阳性比例 |
| 是否能提供初态表达→未来凋亡监督？ | 该实验不能，因为初态表达未测；终末 RNA→先前/末端表型关联不能反向宣称初态预测效果 |
| 是否有多个药物或剂量？ | 此 A549 实验仅 Olmutinib 10 µM 与未加药对照；正文无第二药物、剂量梯度或撤药时长对比 |

图 4 的其他细胞系在不同时间点采 RNA 是另一个实验，且作者明确说明不同 lane 裂解、并非同一细胞纵向 RNA；不能借此给 A549 补造 0 h RNA。

## 4. 培养、lane 与重复

Methods `S20` 给出 A549 单细胞悬液、CellTrace Far Red 标记、fibronectin coating、培养 24 h 后加入 Annexin V-FITC 并拍摄基线。随后四个 lane 中三个加 10 µM Olmutinib（MedChemExpress），一个不加药作对照；每 8 h 刷新培养液并拍摄，最后一次成像前加入 PE anti-EGFR，再裂解。

分析段明确三条处理 lane 来自**同一个 flow cell**，并在模块分数回归中把 lane 作为协变量。本文没有把这三条 lane 证明为三个独立生物重复，也没有提供多个独立 flow cell 的 A549 重复。对照只有一条 lane，处理条件与 lane 存在设计上的关联；数万个 CCE 不能当作数万个独立给药实验。

不同处理时长的停止/洗脱方案、多个药物和多剂量比较没有在这个 A549 实验中出现。平台可以设计这些实验的产品能力，不能证明本数据已经包含这些条件。原文未提供本实验药物目录号或物料批号，本轮不另查数据库给它补造独立化学验证。

## 5. 数量、标签和筛选

| 项目 | 论文报告 | 是否已用原始数据行复核 |
|---|---:|---|
| 最终处理组 RNA 数据集 | 10,604 CCE | 否 |
| 最终对照组 RNA 数据集 | 4,337 CCE | 否 |
| 影像无法可靠分类而排除 | 2,328 CCE | 否 |
| 若按以上数字相减的分类候选量 | 8,276 CCE | 只是算术差，不是下载表的实测行数 |

作者用影像得到 cell-cycle-arrest、proliferation、dead、daughter-cell-resistant 四类。部分分类受分割误差影响，因此先检查 segmented cell 的平均面积，方法中给出 15,000 pixels 的面积阈值。原文 proliferation 描述与其他类的文字条件存在重叠；没有作者脚本及标签表时，不自行发明分类优先级来重建标签。

RNA QC 包括 CCE 总转录本下限、CCE 与捕获 spot 重叠率下限和线粒体 read 比例上限；作者说这些阈值由用户和具体实验确定，**未在该段给出 A549 的实际数值**。同样没有取得 Annexin positive/live/dead 的数值 gate 或逐小室 QC 排除日志。

最终分析中出现 dead 类，说明不能说死细胞小室被全部删除。但死细胞降解、低转录本或高线粒体比例导致多少 RNA 记录丢失、剩余 RNA 究竟来自哪些后代，本轮没有实测数据可以量化。因而不能假设终末 RNA 对所有死亡结局具有无偏覆盖。

## 6. 实际公开数据入口审查

1. 原文 XML 完整扫描了 section 标题、正文和非参考文献外链：没有 Data/Code availability 标题或 GEO/SRA/Zenodo/Figshare/GitHub accession/link。方法只说处理在 open-source R package 中实现，没有包名或链接。
2. [bioRxiv 官方 API](https://api.biorxiv.org/details/biorxiv/10.64898/2026.05.05.723030) 返回 v1 及真实 `jatsxml` URL，已保存 [biorxiv_record.json](biorxiv_record.json)。原发布网页返回 403，API 所列 source.xml 返回 429/Cloudflare；没有绕过。
3. EuropePMC metadata 的 `hasData=Y,hasSuppl=Y,supporting_data` 不是原始训练数据可读证明。[databaseLinks](https://www.ebi.ac.uk/europepmc/webservices/rest/MED/42146668/databaseLinks?format=json) 本次没有返回数据记录；[labsLinks](https://www.ebi.ac.uk/europepmc/webservices/rest/MED/42146668/labsLinks?format=json) 只有 Altmetric。
4. 官方 supplementaryFiles 成功返回 13 个归档成员：12 个正文图片和 1 个 DOCX。DOCX 是 Supplementary Fig. 1–7，唯一表格为抗体表；不含 CCE/RNA/Annexin 元数据或程序文件。
5. 额外按主代理指定的三个官网页面及它们的实际可见链接核对。Demos & Data 是视频/联系入口，未给本论文下载表；Products 提及 Seurat `.rds` 导出、R packages 和 vignettes，但没有提供论文数据或对应包的真实下载链接。[官方 Demos & Data](https://www.cellanome.com/resources-demos-data)、[官方产品页](https://www.cellanome.com/products)、[官方技术页](https://www.cellanome.com/technology)。
6. 官网 Publications → 该论文专页提供可下载的公开 PDF，本轮已经读取。PDF 是 32 页网页打印版，正文也未见数据/代码可用性声明或数据仓库链接。其 `pending:yes` URI 位于 supplementary figures 链接，**不能解释成作者承诺数据将发布**。[官方论文专页](https://www.cellanome.com/publications/scalable-longitudinal-imaging-transcriptomics-dynamic-enclosures)。

当前读取的**原始表达矩阵数=0、CCE–影像–RNA 对应表数=0、已核验源 CCE ID 数=0、已取得作者 R 包/分析 vignette 数=0**。因此本报告没有“实际源表列名”，任何后续接口字段只能标成需求，不能伪装为源 schema。公开入口不足不等于数据不存在、永不公开或必然付费；本轮只能说在已核查的官方入口没有取得它。

网络记录见 [official_entry_fetches.json](official_entry_fetches.json)、[linked_data_metadata_fetches.json](linked_data_metadata_fetches.json)、[original_page_fetches.json](original_page_fetches.json)、[official_website_audit.json](official_website_audit.json)、[official_paper_page_and_pdf.json](official_paper_page_and_pdf.json)。第一次 PDF 文本解析因当前 ML 环境没有 PyMuPDF 未执行，随后使用应用捆绑环境的 pypdf 成功完成 32 页解析；成功结果在 [official_pdf_readback.json](official_pdf_readback.json)，没有安装包或改变训练环境。

## 7. 最小缺失文件与可支持的接口

要把本文变成可训练实证来源，至少需取得：

- CCE ID 到 lane/field/处理条件的索引，RNA matrix 列 ID，以及 capture-spot/barcode 到 CCE 的映射。
- CCE × time 的细胞实例计数、Annexin 测量、live/dead gate、四类 phenotype 标签及无法分类标志。
- 每个 CCE 的 RNA QC 数值、A549 实际阈值与包含/排除清单。
- flow cell/lane/batch 等重复信息和实际作者分析脚本或对应 R package/vignette。

这些字段是**所需接口清单，不是本轮观察到的源列名**。取得后可逐项验证连接唯一性、重复/缺失、RNA 与影像的单位一致性、死细胞选择偏倚和严格留出单位。

若这些数据可取得，可评估限定 Olmutinib 10 µM 场景中的“终末 CCE 表达与同 CCE 实测影像表型”的监督/校准关系；若还有可靠 Annexin 逐实例数和分母，才进一步定义 CCE 层面的 Annexin 比例目标。四分类目标需要单独命名，不能直接冒充项目现有的群体 Annexin-positive fraction。

它仍不能单独证明初态表达驱动的药物选择、多药/多剂量最优干预、同一终末单细胞 RNA 的完整时间轨迹。当前对课题的价值是**真实功能–表达连接的有条件候选**，不是对全任务数据缺口的完成声明。

## 8. 文件与校验

| 文件 | bytes | SHA256 |
|---|---:|---|
| 官方 `supplementary.zip` | 4,882,597 | `72a01e1b9fcf0c2050ca15300c430a8db44af95b32e6e4ad0b2d01a0013141da` |
| 提取 `media-1.docx` | 3,439,974 | `4b43e7bf202b4e2339ef242cbdc8f13431651a04c4815f147588ef89af748eda` |
| 官方 `official_preprint.pdf` | 1,505,817 | `13a428f45a3a2dc49536f2a59d28a274703c58ec4d78f4430746634c414f4421` |
| `biorxiv_record.json` | 2,249 | `480d89030862e05de5cd957d4b42a0c8179e4fc0097903d3622d588e1b40963e` |

完整本地审查文件校验列于 `manifest.json`；父目录三个原始源件作为只读输入绑定。结构化结论见 [data_contract_review.json](data_contract_review.json)。
