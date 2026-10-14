# Nav-Gal 状态 RNA 参考的定向核查

日期：2026-10-10。状态：新取得 GSE200541 官方元数据，但它不是 Nav-Gal 实验中的同源状态；未新增 RNA—Annexin 配对。旧 GSE102639、PRJEB52271 已去重，不重复计作新发现。

## 本次核查的精确目标

既有 Nav-Gal 挑战有 ECACC A549 对照与顺铂诱导衰老两个背景及真实图像 Annexin。需要查明有无可追溯的表达参考，不能只看到“A549 senescent”就连接样本。

主代理重新读取 [2020 原始 XML](../../state_function_round3/PMC7189993.xml) 的 “Cells and reagents” 和 “Cell viability and apoptosis analysis”：衰老诱导为 cisplatin 15 µM、10天；Annexin 挑战的两组铺板密度不同，图示/方法剂量10/11 µM的既有冲突仍保留。这些是身份/时钟核查，并非新实验操作建议。本文没有改写原挑战。

## 新取得的 GSE200541

- 官方 [GEO记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200541)，[SOFT元数据下载](https://ftp.ncbi.nlm.nih.gov/geo/series/GSE200nnn/GSE200541/soft/GSE200541_family.soft.gz)。网页访问失败后，使用独立公开的官方FTP接口成功取得2,268字节的SOFT压缩包。
- 8个GSM分为 scr/CSE knockdown × non-senescent/senescent，名义每组合2个replicate。`rep`是来源标题标记，本轮没有确认其独立培养批。
- 共同 treatment_protocol 字段描述衰老诱导为 **7 Gy irradiation，72 h后收取**。这段通用方法也出现在non-senescent条目里；不能据此把non-senescent样本也改成照射样本，须保留各自condition字段。
- 原字段称GRCh38.p13、STAR2、featureCounts，系列登记一个全样本raw gene counts文件。**表达矩阵尚未下载或读取**，不报告其基因数或映射完成。
- GEO登记PMID36372112，对应 [Gupta et al., 10.1016/j.bbamcr.2022.119388](https://pubmed.ncbi.nlm.nih.gov/36372112/)。本轮亲读该记录的摘要，CSE指cystathionine gamma-lyase，并非搜索结果中同缩写的cigarette smoke extract。期刊全文未取得，不能由摘要判断不存在凋亡实验。

| 组合 | 已核验的GSM |
|---|---|
| scr、non-senescent | GSM6037501、GSM6037502 |
| scr、senescent | GSM6037503、GSM6037504 |
| CSE knockdown、non-senescent | GSM6037505、GSM6037506 |
| CSE knockdown、senescent | GSM6037507、GSM6037508 |

原字段提取见 [GSE200541_metadata_readback.md](GSE200541_metadata_readback.md)，机器表见 [GSE200541_sample_metadata.csv](GSE200541_sample_metadata.csv)。

**资格判断：** 该资源可以继续作为另一种A549状态表达候选；当前不能填进Nav-Gal挑战并声称状态精确匹配。照射72h、基因敲低与顺铂10天是不同来源状态。若以后研究跨诱导方式的迁移，必须另定义该任务与验证，不能把迁移假设登记成同源配对事实。本轮因此未下载整套表达，未生成状态向量或新增功能标签。

## 复用的旧来源和停止边界

[PRJEB52271已核记录](../../state_expression_function_round5/README.md)是ATCC A549、cisplatin15 µM、7天的来源状态，RNA主要来自产生conditioned medium的细胞，其Annexin来自接受培养基的另一组细胞。此前已取得12个A549 RNA样本元数据；本轮没有把这一旧发现再算一次，也没有将其与Nav-Gal的ECACC/10天条件合并。

[GSE102639与SENESCopedia旧审查](../../state_function/README.md)已记录不同诱导材料/时钟和其他功能定义，同样不自动补成cisplatin10天的RNA。

两个检索入口为SciSpace完整问题语义检索与网页定向检索，查询及返回元数据见 [discovery_record.json](discovery_record.json)。SciSpace返回10条元数据，多数为非A549或综述；它们不计作10篇全文审查。数字PMID的网页查询产生无关结果，未作为科研依据。当前结论限于这些查询和已读官方资料，不能写成“所有公开数据中都不存在相应RNA”。

本支线到此停止重复搜索相同Nav-Gal RNA。实际新增为官方8样本元数据、条件排除依据与来源记录；真实模型推理、训练、Zotero写入、联系作者、提交和记忆写入均未执行。
