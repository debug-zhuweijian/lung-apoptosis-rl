# 第十八轮：真实协议张量接入与原始 RNA 恢复

2026-10-11。接续[第十七轮](DATA_ALIGNMENT_ROUND17_2026-10-11.md)，保持[毕业论文研究架构](G:/研究生科研文件/毕业论文/研究架构.md)的 A549 优先、状态表达增量、未见条件、功能一致性及联合选择范围。专项实现与独立复审使用用户指定 GPT-6.1-sol。上一轮属于实质进展；本轮继续完成真实数据准备，不恢复正式模型训练、真实模型 forward/inference 或 PT 张量加载。

## 1. 本轮实质结果

**真实完整处理记录已能转换成协议编码器的输入张量；A549 历史 RNA 的原始恢复也已从前置查证进入实际数据任务。** 这分别关闭有限 source→tensor 缺口与本机工具不可执行的阻碍，仍没有补出 RNA—Annexin 配对或整个模型效果。

| 工作 | 本轮真实状态 | 保留的边界 |
|---|---|---|
| 完整协议 source→tensor | Viel20、Bothe4，共24候选/70事件实际读取和验证；当前有效v2 | 全部base condition kwargs、fullproducer、来源decoder及权重仍pending |
| Windows原生工具 | 作者Subread2.1.1实际取得、版本/合成gzip/path/index/align/count smoke成功 | 与作者STAR流程不同；合成smoke不是原始RNA结果 |
| 实际注释 | 99,517源transcript→29,634选择gene，23,125可测gene/207,555唯一区间；完整独立验证 | 新current snapshot、primary/both-strands方法，非精确作者历史注释 |
| 实际参考 | hg19.fa.gz 948,731,419bytes完整取得、publisher MD5通过 | genome验证与注释验证分别登记 |
| 实际数据任务 | 完整human genome index已完成，exit0，6文件共6,031,893,097bytes独立hash通过；双端raw先导仍按live handle下载 | 索引尚未实际用于RNA比对；未完成raw不称RNA/counts就绪；当前不是P/G/Q执行 |
| 新功能来源 | FOSL1/CPT2026的6RNA run及8代表双阳性流式图值已核实 | RNA6h/si、功能48h/sh或普通A549，不直接配对、不增加核心G |

主代理本轮 **44 个新定向检查通过**：source映射17、注释21、下载完整性6。独立复审另外核对了缓存/资产反例、250逐碱基案例与全注释NumPy oracle。[主测试记录](../experiments/20261011-data-alignment-round18/pytest-main.txt)保留范围；测试数量不替代生物学有效性。

## 2. 真实协议怎样转换成模型张量

[新增 reader](../src/lung/data/full_protocol_inputs.py)先核对冻结源manifest、递归bindings与原条件表，再构建有顺序的事件；模型张量入口只接受真实候选ID。它对接已冻结的 `FullPhysicalProtocolBatch`，没有改旧编码器字节。当前有效资产为[full-protocol-inputs-v2](../experiments/20261011-data-alignment-round18/full-protocol-inputs-v2/manifest.json)。

| 来源 | 候选 | 事件 | 显式时钟 |
|---|---:|---:|---|
| Viel连续24h | 10 | 10 | drug interval 0→24，共同T24 |
| Viel6h洗脱+18h恢复 | 10 | 40 | drug0→6，wash/media转移点6，recovery6→24 |
| Bothe四历史 | 4 | 20 | drug/vehicle0→4，wash2/media转移点4，recovery4→28，第二drug/vehicle28→32 |

Bothe12个RNA样本是四历史各三重复，未扩成12候选。指纹张量为 `[24,5,2048]`；20个Viel药物事件有源绑定指纹，Bothe指纹仍未知。dose/start/end/exposure/wash_count/prior_retention/media_change/recovery八字段的known计数为 `[24,70,70,28,4,0,14,14]`。

Viel洗次数未知，Bothe明确PBS两次；stock form、细胞内前药retention未知。Bothe vehicle只写0.1%，未确认v/v，原数值保留metadata，tensor dose/unit保持unknown。wash/media的start=end是事件转移时钟约定，不是测得操作耗时为零。状态使用合法source语义，不把协议起点叫作实测细胞t0。

这些值由主代理和[独立 source mapper 复审](../experiments/20261011-data-alignment-round18/independent_protocol_review/technical_review.md)核对原文Viel Fig3C/5C及Washout Methods、Bothe Fig3A/E及re-induction Methods/SDRF。源中含Y的上游文件只作bytes hash，不反序列化为模型输入，也不宣称历史盲法。

初版存在实例缓存可被修改的P2：改dose/role或删source bindings后仍可产tensor。现每次API从固定源重新构建并核对完整缓存/谱系；改records并同步改caller hash也不能绕过。旧v1保留，因consumer identity变化明确拒读；v2成为当前版本。17tests及独立反例复验关闭。**真实source→tensor已通过，full model condition/生产资格没有因此通过。**

## 3. 原始 RNA 恢复为何现在能够推进

上一轮只在PATH中没有确认STAR/featureCounts或Linux。本轮从作者[Subread官方release](https://sourceforge.net/projects/subread/files/subread-2.1.1/)取得Windows portable2.1.1，48,084,495bytes ZIP，SHA256 `16f04778c1ade1bfca0c594016527a7e73eaecd1b6724f4eace00c674921ab3e`。解压目标边界逐项核验，GPLv3和官方guide已读；未全局安装/改PATH或安装WSL/Docker。

subread-buildindex、subjunc和featureCounts版本实际运行。固定合成12kb genome、两对100bp gzip FASTQ在中文absolute和cwd+ASCII relative路径均成功；SAM四records的独立预期位置/flags/CIGAR/MAPQ逐项一致。初始超长单行FASTA被拒绝，80bp换行后通过，失败证据保留。[portable回执](../experiments/20261011-data-alignment-round18/bothe_portable_runtime/README.md)与[smoke记录](../experiments/20261011-data-alignment-round18/bothe_portable_runtime/NATIVE_SMOKE.md)明确它们是合成检查。

这个路线使用Subjunc，原论文使用STAR2.7.0a；它必须作为本课题新processing版本。原始12样本来自同一已核验实验，重处理技术选择与生物实验条件不是同一个概念。不能因为程序可执行就宣称精确复现原作者，也不能因为算法名字不同就把公开RNA判成不可获得。

实际计数还发现两项会改变结果的边界：

- `featureCounts -Q10` 在本次tiny实测中使用pair至少一端通过，mixed9/20被计入；不能直接替代作者逐read过滤MAPQ<10。
- 只去掉MAPQ9记录而保留原mate flags后，`-B`仍接纳孤儿，count为2；真正按实际完整pair联合过滤后为1。后续必须实际核验QNAME/R1/R2、两端mapped与质量，再冻结整对规则。

secondary/supplementary、proper-pair、duplicate、NH、strandness和fragment策略还需明确。作者当前源码snapshot已读取并绑定commit，尚未证明与release2.1.1逐字一致；文档与源码的`-C`描述差异分别标注，不把未经核验的文字当运行行为。[探针/源码回执](../experiments/20261011-data-alignment-round18/bothe_portable_runtime/mapq_orphan_smoke_receipt.json)保留具体证据。

## 4. 注释和基因坐标已经实际完成什么

官方UCSC当前下载的ncbiRefSeq及RefSeqLink、SQL schema、下载manifest、chrom.sizes已固定版本与hash；SQL仅用于读字段定义，没有执行。GeneID从RefSeqLink.locusLinkId取得，不仅依赖symbol。源genePred坐标为0-based半开区间。

[新版unique introns](../08-datasets/Bothe2021/E-MTAB-9923/reference/unique-introns-v1/manifest.json)按sum(exon length)选择最长spliced transcript，声明并列规则；primary范围内跨chr/strand和不相交locus隔离，alternate记录保留台账。去除所有source transcript exons及多gene intronic交叠，采用保守both-strands定义；SAF转换为1-based闭区间，并保留feature→stable NCBI GeneID映射。

| 分母 | 实际数量 |
|---|---:|
| 全部源transcript行 / primary行 | 99,517 / 90,444 |
| 选择gene / 有可测feature的gene | 29,634 / 23,125 |
| 唯一SAF区间 | 207,555 |
| 保留总长度 | 1,206,619,076bp |
| primary跨chr/strand冲突gene / 不相交locus gene | 27 / 3，隔离而非拼接 |
| 同时有primary/alternate源记录gene | 1,842，不因alternate直接合并 |

没有可测feature的gene保留不可测身份，后续不能当成实测零。23,125是该处理定义的gene数量，不是32,461共同Ensembl轴，也不是新增单细胞数量。

21固定测试通过；独立审查用250逐碱基案例和另一完整NumPy端点算法，逐项重算全部207,555区间、gene/strand、坐标、总长度、源/输出hash和染色体边界，全部一致。[独立注释报告](../experiments/20261011-data-alignment-round18/independent_annotation_review/technical_review.md)。原作者历史RefSeq和strandness未知，当前产物标为新重处理注释；它仍不是RNA counts。

## 5. 两个实际数据任务与可恢复状态

主代理已完整取得UCSC初始hg19.fa.gz，948,731,419bytes，发布方MD5 `806c02398f5ac5da8ffd6da2d1d5d1a9`通过，SHA256 `ac73947d38df63ccb00724520a5c31d880c1ca423702ca7ccb7e6c2182a362d9`。不是`latest`指向的p13.plusMT版本，参考与注释各自的范围保留。[genome declaration](../08-datasets/Bothe2021/E-MTAB-9923/reference/current-ucsc-hg19-20261011/genome_declaration.json)与[index run](../08-datasets/Bothe2021/E-MTAB-9923/reference/subread-hg19-index-v1/run.json)保留散列。

原始RNA先导下载session28793仍经主代理实际poll确认运行；完整human data index session22687已经返回exit0并关闭。下载先导选官方最小双端run ERR5000305（EtOH→EtOH rep2），仅按元数据字节选择，不按Y。首个run双端声明3,301,778,336bytes，不代替12样本；完整genome已验证，RNA`.part`不当作完整FASTQ。下载器严格验证Content-Range续传、bytes和publisher MD5，保留失败partial，不重复覆盖完整文件。

data index使用实际portable buildindex，`-M8000`MB/default gapped，新prefix拒绝覆盖，启动前再次核源MD5及工具hash。主代理已收到真实进程exit0，完整日志报告成功；另一个验证器重新读取全部6个索引文件，大小及SHA256与runner清单完全一致，总6,031,893,097bytes。[完成验收](../experiments/20261011-data-alignment-round18/index_completion_validation.json)与[验收源码](../experiments/20261011-data-alignment-round18/verify_index_completion.py)保留证据。原[validation](../experiments/20261011-data-alignment-round18/validation.json)中的running快照保留，以新增完成验收更新当前状态；没有覆盖历史观察。真实RNA比对可用性尚未实际验证。后续见[接续交接](../experiments/20261011-data-alignment-round18/CONTINUATION_HANDOFF_2026-10-11.md)：仅对仍运行的raw任务poll同handle，不重启已完成索引或覆盖prefix。

本轮未完成RNA比对/count或12列矩阵/QC，更未训练模型。下载/index属于数据准备，不能计作P/G/Q效果。

## 6. 新来源是否解决了凋亡配对

本轮按缺口定向检索网页与Europe PMC两个入口，阅读并保存四个候选原文，按DOI/accession去重。新决定性核查为：*FOSL1 promotes homologous recombination repair and camptothecin resistance via TCOF1-mediated ribosome biogenesis in non-small cell lung cancer*，DOI [10.1016/j.tranon.2026.102701](https://doi.org/10.1016/j.tranon.2026.102701)。主代理亲读[原始XML](../experiments/20261011-data-alignment-round18/a549_function_source/PMC12924127.xml) Methods sec0009/sec0011、Data Availability及Fig2H/8H图注，亲看两图，并核官方project XML/6-run表。

RNA PRJNA1332077为siNC/siFOSL1各三重复，CPT30nM/6h药后。Fig2H为stable shGFP/shFOSL1、CPT30nM/48h；Fig8H为普通A549、CPT30nM与BMH21 0.2µM单药/组合48h。八个转录数值是代表流式图的Annexin V+/DAPI+右上象限百分比，不是三重复原表或发表均值，也不直接改写为early+late总Annexin比例。没有把siRNA当shRNA、6h当48h或药后RNA当t0。

该源能补充有限功能状态证据，**没有关闭本课题高维参考→多药/剂量/时长共同覆盖的功能连接**。Icariside II2022未定位公开表达accession；Zhu2018的RNA/功能剂量、时钟不一致且accession位置空白；补充docx获取失败明确未读。[来源报告与检索记录](../experiments/20261011-data-alignment-round18/a549_function_source/README.md)保留实际覆盖及淘汰理由，不将获取失败说成资料不存在，不纳入核心42 Annexin/17输入分母。

## 7. 对“缺未见条件RNA就做不了模型”的修正

[bulk/source设计核查](../experiments/20261011-data-alignment-round18/bulk_native_design/design_report.md)区分四项：真实工程执行、零样本/迁移预测身份、中间表达验证、最终功能验证。**不要求每个未见drug/dose/time都有自身训练RNA，才允许开展预测研究。** 原架构本来就检验未见条件的可泛化性。

现有联合A549 metadata确有osimertinib mesylate和venetoclax各三剂量0.05/0.5/5µM、24h；主代理再次读取真实条件表确认。osim完整盐形式与Martin parent不同，venetoclax同full key仍不认证材料/跨来源配对。四药均无joint sciPlex命中，AZD4320/K-975在当前joint名义/全key未命中；它们不是整个公共领域不存在的结论。

现有bulk RNA支持已测osim500nM急性/历史条件的来源出口与表达监督；四药72h未有直接target RNA。缺该直接监督限制“已验证中间表达”的主张，不自动否定有显式假设、真实输入与独立功能Y的迁移研究。未来可以另建真正执行、明确零样本/迁移身份的预测产物，完整候选的独立功能Y证伪整体效用；它不能唯一证明中间高维RNA正确。当前typed bulk出口、匹配权重及可信producer未实现，未靠删除守卫将caller向量授为native。

设计中的f/f′反例只说明现有loss不能唯一识别未测条件或无额外假设给保证，不能推出统计学习不能泛化。本轮没有因此降低原A/B/C/D或同一链要求，也没有预设方法必须胜出。

## 8. 当前验收与接续

本轮已关闭：24有限source候选到协议tensor的可核验接入、相应缓存完整性缺陷、Windows portable工具可执行性、全量unique intron注释与独立源读回、实际genome取得和完整human index创建/文件验收。实际raw下载仍按handle推进；counts、source outlet/base kwargs/fullproducer、匹配模型与整链实测评价尚未完成。旧架构、frozen模型/输入/joint/Martin桥未覆盖，旧dirty工作树保留；无commit/push、Memory/Zotero写入或新Cloud/Dot派单。

下一依赖是完整raw和已完成index→明确paired过滤/真实比对/count→12样本QC与轴桥→来源表达出口和真实执行→受控功能评价/Q。全过程保留失败与迁移范围；数据仍不相容的部分继续补原始证据，不用程序通过替代完整可行性结论。
