# CellCage A549 来源定向恢复，Round22

**本轮仍未取得本论文的真实表达矩阵或 CCE–时间–处理–影像命运行表，关键实验数据缺口未关闭。** 已完成25个小型公开HTTP响应的实际读取、字段与SHA核验，核31个旧原文/补充/审查SHA一致，取得了准确的入口与权限证据，并排除了一个标题近似而实验对象不符的公开RNA数据集。未下载原始影像、视频、FASTQ或全测序，未运行真实PT、NN、forward、拟合或优化器，未改原角色、协议、模型或奖励。

限定来源为 Khurana 等，*Scalable longitudinal imaging and transcriptomics of cells in dynamic enclosures*，DOI [10.64898/2026.05.05.723030](https://doi.org/10.64898/2026.05.05.723030)，PMID42146668，PMC13174582。官方bioRxiv metadata仍返回v1／2026-05-08／published=NA，响应SHA `480d89030862e05de5cd957d4b42a0c8179e4fc0097903d3622d588e1b40963e` 与旧审查一致，不将其称作正式同行评议版本。

## 原文定位、数据单位与真实连接

本轮亲读并SHA复核旧 [PMC XML](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/cellcage/PMC13174582.xml)、[原文MD](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/cellcage/PMC13174582.md) 与 [Supplement文字](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/cellcage/data-review/supplementary_text.md)，复用已核的官方PDF/补充内容。XML S4、S6、S13、S15、S20实际段落及全部31文件复核在 [original_source_readback_v1.json](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/original_source_readback_v1.json)。在线对应 [EuropePMC原始全文](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13174582/fullTextXML)。

| 连接／要素 | 原文支持与定位 | 尚未取得或不能推导的内容 |
|---|---|---|
| 药物／细胞／协议 | S20与Fig5：A549，Olmutinib10µM，3处理lane／1未处理lane；先培养24h，基线Annexin影像后给药，每8h换液成像至32h，末次前加PE anti-EGFR，再裂解 | 实际每行lane／事件／时钟／材料批次未读取；3lane没有被核验为3独立生物重复 |
| capture spot→CCE RNA | S4/S13/S15：spot先生成基因计数，同CCE所属spot按基因求和 | barcode/spot→CCE映射和真实gene×CCE矩阵缺失 |
| 同CCE纵向影像→终末RNA | S6/S20与Fig5：影像史定义表型，最后影像后测3′mRNA | 行级CCE ID、真实时间戳、影像特征、QC排除与RNA列连接未读取 |
| 初态RNA→未来功能 | 本A549实验基线为影像；RNA仅在末端裂解 | 没有同CCE给药前RNA，不支持初态表达→未来凋亡监督；其他lane时间点采RNA也不能补成同一单位纵向RNA |
| CCE→单终末细胞 | daughter-cell-resistant小室同时含活、死后代；proliferation也可多细胞 | 不能将每CCE汇总RNA假定为一个终末单细胞，也未得到逐后代RNA分配或完整谱系 |
| 四类影像→Annexin | S20列cell-cycle-arrest/proliferation/dead/daughter-cell-resistant；使用细胞数、历史与活死判断 | 四分类不是项目群体Annexin-positive fraction；缺原始数值、gate、分母和分类优先级，不自行重建阈值 |
| 单药／剂量→联合决策 | 本实验只描述Olmutinib10µM与未处理对照 | 不授多药、多剂量、时长联合选择、多步滚动或真实序贯资格 |

原文报告10,604处理CCE、4,337对照CCE，2,328处理CCE无法可靠分类；它们是论文数字，**本轮读取的实际源CCE行数仍为0**。不把算术差8,276登记成实际下载行数，不把CCE数或lane数当成独立给药实验数。死亡小室仍可能存在RNA QC选择偏倚，当前不能量化其覆盖；原分类描述存在重叠，没有脚本／表时不发明优先级。

## 本轮实际入口与反例

[全部读取回执](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/actual_readback_v1.json) 核25原件大小／SHA／实际JSON结构。24次HTTP200和1次403仅描述访问结果，不等于24份实验数据。

| 入口 | 实际结果 | 能说明的范围 |
|---|---|---|
| bioRxiv DOI metadata | 200；仍v1／published=NA，旧SHA相同 | 版本未见变化；未给矩阵入口 |
| GEO／BioProject／SRA来源名及DOI查询 | 三个200，count0，保留API querytranslation/warning | 该实际查询未找到；不证明全库无数据 |
| ENA study_title CellCage/Cellanome | 200，实际空数组 | 限定字段查询为空 |
| EuropePMC databaseLinks | 200，仅request/version字段，无返回数据链接 | `hasData`或论文关联材料不等于可用矩阵 |
| GitHub cellanome org/public search | 官方命名org公开repos=[]；search有5项；亲读3个完整未截断tree | 完整tree分别为网站代码、两视频、Scoop安装manifest；另外2条只是第三方API简介／内部安装仓库metadata，未取得本论文R包、vignette或数据源身份 |
| 官网论文／Demos & Data／Products | 三个200，实际锚点保存；论文仅DOI/PDF，产品介绍.rds导出能力 | 未发现该论文矩阵/行表下载地址；云端能力宣传不是公开实验文件 |
| Zenodo精确本论文DOI | 200，hits.total=0 | 该DOI查询为空 |
| Zenodo广义CellCage查询 | 一条17584933，Orcutt-Jahns，题名为另一稿件，access_right=restricted，files=[] | 未确认该记录属于Khurana论文或A549；不把题名相近当作同数据 |
| 该Zenodo官方files链接 | **403，Permission denied.**；48字节原始响应保存 | 是未授权文件访问证据，不是数据不存在；没有request-access提交／凭证／作者联系 |
| 标题查询GEO/BioProject唯一候选 | 实际SOFT与XML：GSE338040/PRJNA1492146，PENNE，MCF10A/HCT116／10Gy辐射，18样本 | 明确排除为本源；不下载其RNA。NCBI曾自动丢掉Olmutinib并拆分标题词，实际querytranslation保留，不能凭search命中就建桥 |

对应原始链接：[GitHub公开org](https://api.github.com/orgs/cellanome/repos?per_page=100&type=public)、[GEO候选源表](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE338040&targ=self&form=text&view=full)、[Zenodo17584933](https://zenodo.org/records/17584933)。详细分类、公开repo commit/tree范围与被排除字段见 [candidate_source_classification_v1.json](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/candidate_source_classification_v1.json)。原文不含本实验数据仓库／R包地址的旧结论在31SHA一致时复用，没有重抓相同PDF/补充包或绕过403。

检索亦返回JITC会议摘要[178，A194](https://jitc.bmj.com/content/13/Suppl_2/A194)，描述Olmutinib或Cetuximab-MMAE／3日影像。未取得其对应matrix/accession，不把该摘要与本v1的10µM／32h实验合并，不据此增加药物覆盖或Annexin资格。

## 来源子版本、共同基因轴与模型边界

新增 [CellCage2026 metadata 子版本](../08-datasets/CellCage2026/PMC13174582-v1/source-recovery-round22-v1/README.md)，[manifest](../08-datasets/CellCage2026/PMC13174582-v1/source-recovery-round22-v1/manifest.json) 状态为metadata_only，matrix_shape／source_gene_IDs／source_observation_IDs均null。实际新文件只有public metadata／HTML／SOFT／XML，包含失败原件，不创建合成表达或零占位。既有共享README、索引、模型和数据包未修改。

[共同基因metadata读回](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/joint_gene_axis_readback_v1.json) 读取32,461个唯一Ensembl ID，genes.parquet SHA `4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc`；这只是现项目的预期轴。CellCage源轴、计数／normalized实际量纲、可测基因、映射数、缺失mask和真实投影都未得到。不能说已完成CellCage共同轴对齐；原文Seurat流程也不证明其输出可直接套用项目单细胞library10000标准化。

本轮实际RNA矩阵0，RNA数值0，CCE映射表0，源CCE ID0；fit／selection／final未分配。AnnData矩阵／obs／var检查仅作为取得真实结构化数据后的读取方式，没有伪称read_h5ad已执行。采集脚本 [audit_cellcage_public_source.py](../scripts/audit_cellcage_public_source.py) 只做公开GET、小型metadata保存和独占文件写入；回执逐源hash已经实读，无模型入口。不要直接重跑同版本覆盖旧原件；新查询另取唯一receipt/file名或新子版本。

## 最小后续依赖与科学判断

[来源合同](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/data_contract_v1.json) 明确区分原文声明、实际观察列与所需材料。继续实数据对齐至少需要以下来源原件：

1. processed终末gene×CCE矩阵，保留gene ID／CCE列ID及其实际计数或normalized单位。
2. CCE→flowcell／lane／field／barcode/capture-spot映射，以及原QC与包含/排除表。
3. CCE×time实例/特征表，保留原时间戳、live/dead／Annexin数值和gate、四类标签及无法分类标志。
4. A549实际RNA QC阈值、分割/分类规则与优先级，以及原药物/剂量/换液/时钟事件记录。

这些是待取得材料需求，**不是伪造的源列名**。单看当前原文，此来源有条件支撑“终末CCE表达与同CCE实测影像表型”的有限监督研究；当前行级数据访问还不满足实际开展条件。即使取得它，也不能补出初态RNA、完整单细胞未来凋亡预测、多药联合选择或序贯实验。最小外部事项是取得上述processed导出与原metadata的有效共享入口／相应访问权限；本任务没有发送作者请求或申请账户，不能自动突破该外部状态。保留完整主目标未完成状态，不以本来源缺口降低核心研究范围。
