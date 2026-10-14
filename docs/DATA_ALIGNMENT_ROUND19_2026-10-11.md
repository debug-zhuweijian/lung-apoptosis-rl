# 第十九轮：严格双端计数链与名义药物输入补齐

2026-10-11。继续用户要求的完整数据对齐，由 GPT-6.1-sol 专项代理实现与独立复审，主代理读取原件、源码并运行验收。[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)保持原字节与 A549 优先、A/B/C/D、同一算法职责链。当前工作为数据准备，没有恢复正式训练、真实模型 forward/inference 或 PT 加载。

## 1. 已完成的结果与未完成的部分

**本轮实际跑通完整 hg19 索引上的 Subjunc→严格双端过滤→featureCounts→完整注释解析链，并补齐 Bothe 四个 Dex 事件的名义结构输入。** 比对链的输入是参考基因组生成的人工读段，真实 RNA 正在下载；两类证据分别登记。

| 项目 | 本轮已完成 | 仍需完成 |
|---|---|---|
| 完整 human index | 前轮真实创建，exit0；本轮六文件再核 hash，并实际用于比对 | 真实 RNA 与更复杂拼接读段的表现、全样本运行成本 |
| 双端联合过滤 | 真实 R1/R2、双端 MAPQ、互逆坐标等检查；异常输入及发布问题复审关闭 | 全量真实 SAM 的吞吐、QC 与所选新处理规则的影响 |
| count matrix 接口 | 固定12样本和完整注释轴；零、缺失、不可测分开；Windows native 注释编码兼容 | 原始样本计数、专用真实 executor 接入及12列 QC |
| 新完整协议 v3 | 24候选/70事件；四 Dex nominal 指纹新增，known FP20→24 | 状态参考、base kwargs、来源 decoder、fullproducer及匹配权重 |
| 原始 RNA 获取 | 先导继续下载；另外两进程处理其余11个run | 双端完整 bytes/MD5、FASTQ内容核验与实际计数 |
| 凋亡连接 | 保留已核验来源与原研究问题 | 没有增加 RNA—Annexin 实测配对，整链功能评价仍未完成 |

主代理先完成127项合并检查，随后新增独立下载完成验收11项，**共138项通过**：本轮94项（双端35、矩阵31、v3协议11、下载协调6、完成验收11），相关既有44项回归（v2协议17、注释21、下载器6）。独立复审的反例、故障注入及化学逐位复算另有回执，不重复加到138的分母。[原127项验收](../experiments/20261011-data-alignment-round19/validation.json)保留，[追加完成验收](../experiments/20261011-data-alignment-round19/completion_boundary_validation.json)补充当前范围。

## 2. 双端过滤补齐了哪项实际问题

前轮已实际发现，`featureCounts -Q10`采用pair至少一端通过，`-B`依赖mate flags，不能保证去掉低质量记录后两端仍存在。本轮[过滤程序](../src/lung/data/paired_alignment_filter.py)使用 SQLite 磁盘分组，处理非相邻 QNAME；每个保留模板必须实际包含各一条主 R1/R2，两端 mapped，MAPQ≥10，mate chromosome/POS、strand flags和TLEN互逆。整对保留或拒绝，保留源 header 与记录顺序。

这属于本课题的新处理版本。作者已知的 MAPQ、primary/bothEnds/noChimera 描述与下列补充选择分开：

- secondary/supplementary记录不输出，合格primary可以保留；不额外要求proper-pair，不去duplicate，分别计数。
- NH出现且不为1时拒绝；缺NH保留并登记unknown，不将缺标签当唯一定位证明。
- MAPQ255的质量不可用，本版拒绝；没有额外fragment长度截断。
- 原来源strandness未明确，本版显式unstranded计数，不认证原作者方向性。

规范依据为 [SAM v1.6，b5341fb，2025-08-12，§1.3–1.5](https://samtools.github.io/hts-specs/SAMv1.pdf)；具体实现和新处理取舍见[policy报告](../experiments/20261011-data-alignment-round19/paired_filter/README.md)。

独立复审发现并关闭三项P2：非法H/S clipping顺序被接受、header重复tag被吞掉、输出或回执写失败后留下正常名称的未完成SAM。现使用明确unqualified临时文件，完成校验后发布，回执发布失败时回退SAM名称。该机制不是两文件的断电事务；消费方仍核complete receipt及匹配hash。[最终复审](../experiments/20261011-data-alignment-round19/independent_pair_filter_review/technical_review.md)记录35测试及6独立反例/故障探针，原失败证据保留。

## 3. 样本、基因和零值如何保持对应

[矩阵接口](../src/lung/data/bothe_count_matrix.py)固定前轮真实 SDRF/ENA 的12样本、12run、四历史各三重复；R1/R2属于同一实验重复。注释固定为29,634选择gene、23,125可测gene、207,555 SAF features；逐字段核对Geneid、Chr、Start、End、Strand、Length，再按stable NCBI GeneID汇总。

| 情况 | 矩阵表示 | 含义 |
|---|---|---|
| 样本尚未取得完整count文件 | 空值、missing_samples登记 | 该列没有测量结果 |
| gene无可测unique intron feature | 空值、annotation status登记 | 在该观测定义下不可测 |
| 可测gene的合法输出count为0 | 整数0 | 已读取该样本，实测该处理输出为零 |

列必须与执行入口的sample对应，不按列名猜测；重复/缺feature、未知样本列、非负整数与聚合溢出均检查。`parse_featurecounts()`可以校验真实文件结构，也可以校验合成输出；**解析通过本身没有认证数据的实际实验或执行身份**。当前受控real matrix构建入口仍待专用executor，audit mock默认拒绝，partial不能变成12列ready。这个工程接口尚未接上，不表示公开来源在科学上不可用。

主代理还从前轮真实Windows工具输出发现新P2：`# Program`注释中的中文exe路径为CP936，整文件UTF-8解码先失败，无法跳过注释。修复先按原始bytes跳过注释，再严格解码表头和数据；没有转写源文件或忽略非法数据。实际native合成输出读回count2，源hash不变；非法UTF-8数据仍拒绝。31项检查与独立复验关闭此项。[矩阵复审](../experiments/20261011-data-alignment-round19/independent_count_matrix_review/technical_review.md)及[编码修复证据](../experiments/20261011-data-alignment-round19/independent_count_matrix_review/native_comment_postfix_results.json)。

## 4. 完整索引上实际验证了什么

[先前索引完成验收](../experiments/20261011-data-alignment-round18/index_completion_validation.json)固定六文件6,031,893,097bytes。本轮[probe入口](../scripts/probe_bothe_reference_index.py)重新核六文件SHA256、exact completed run、annotation manifest、准备的FASTQ与实际exe，再由主代理启动单进程T1。

从hg19 chr1合法SAF introns流式提取十对75bp forward/reverse-complement人工read，使用独立synthetic命名；没有使用真实RNA样本ID。Subjunc在完整human index上运行，strict filter和完整SAF featureCounts随后运行，两个native命令均exit0。十对全部符合预定R1/R2坐标、方向及75M；完整207,555-feature输出严格解析，四个非零GeneID的count为1、1、6、2，总10，与坐标oracle一致。[实际result](../experiments/20261011-data-alignment-round19/reference_index_probe/result.json)和[执行绑定](../experiments/20261011-data-alignment-round19/reference_index_probe/execution_binding.json)保留命令、hash与合成性质。

native SAM的非UTF-8 header按bytes保留，alignment data严格解码。工具日志对极小数据给出Phred offset提示及不足以推断fragment长度的提示；源FASTQ确为75bp和Phred33的固定I质量，坐标与count核验通过，保留提示，没有将tiny运行当常规RNA QC。实际峰值RAM没有连续采样，不能由索引文件大小或前后内存快照推算。准备阶段的RAM保守未执行记录保留，后续root真实exit0/result补充当前状态。

这个结果支持实际索引与计数链的工程可执行性；人工无剪接读段的通过不证明真实RNA质量、所有splice junction、多重定位规则的影响、12样本计数或作者STAR精确复现。真实RNA counts仍为0。

## 5. 名义地塞米松结构怎样进入新协议版本

主代理读取保存的Bothe原XML Methods s13/Fig3、IDF/SDRF及[官方PubChem原始JSON](../experiments/20261011-data-alignment-round19/bothe_dexamethasone_source/dexamethasone_properties.json)。支持名义Dex100nM、4+24+4h/T32；参考化学图CID5743、完整InChIKey `UREBDLICKHMUKA-CXSFZGCWSA-N`。sodium phosphate是不同完整图，没有替代采用；实验厂家/货号/批次/stock form/储备液浓度/溶剂与active-moiety依据仍未认证。

按项目既定Morgan radius2、2048bits、includeChirality=True，现有RDKit2026.03.6实际计算。原始SMILES和SDF独立解析完整key一致，53个on-bits及2048位逐位复算一致。[来源增补说明](../experiments/20261011-data-alignment-round19/bothe_dexamethasone_source/receipt.md)。

[新v3 reader](../src/lung/data/full_protocol_inputs_v3.py)单独复用v2张量机制，先核原source与固定增补manifest/全部登记files，再按source history重建四个Dex事件：Dex→Dex的第0/4事件、Dex→EtOH第0、EtOH→Dex第4。vehicle/wash/recovery不取得Dex指纹。新[full-protocol-inputs-v3](../experiments/20261011-data-alignment-round19/full-protocol-inputs-v3/manifest.json)保持24候选/70事件，known FP20→24，scalar known仍`[24,70,70,28,4,0,14,14]`，其余张量与旧版逐项相同。

旧consumer和v1/v2没有修改，v2实际仍可读取。新版本每次API重新核固定来源并与缓存比较，不因caller改值、重写hash或删除bindings获得资格。11 tests及11独立反例通过，[独立v3复审](../experiments/20261011-data-alignment-round19/independent_protocol_v3_review/technical_review.md)。**新增known指名义化学bit向量，stock/material、EtOH百分比basis、参考RNA、base kwargs、来源decoder、fullproducer和权重资格没有随之通过。**

## 6. 真实RNA下载和具体接续

原pilot ERR5000305双端共3,301,778,336bytes，主代理session28793继续运行。[新下载协调器](../scripts/download_bothe_remaining.py)session5491以两独立进程依次处理其余11run，共56,754,947,648bytes；加pilot即前轮声明的24 FASTQ/60,056,725,984bytes。它排除已被pilot占用的run，不重复写其目录；按源字节排序，未利用Y选择。实际最先处理ERR5000296和ERR5000298，两份log均已有持续bytes输出。[任务清单](../experiments/20261011-data-alignment-round19/raw_remaining_acquisition/jobs.json)固定源映射和两个script hash。

每run使用现有严格Content-Range/bytes/publisher MD5下载器；partial保留，失败run不由协调器反复重启。新协调器6项mock测试只验证排除pilot、11run分母、拒覆盖/所有权冲突、失败不重试和错误receipt拒绝，未下载测试数据。两handle的live观察是有时间的快照，后续必须重新poll；进程域读取失败不能代替root handle终态。实际观察与续接见[handoff](../experiments/20261011-data-alignment-round19/CONTINUATION_HANDOFF_2026-10-11.md)。

最后的下载协调复审发现一项完成payload校验不足：其摘要只核来源/两端身份/状态，没有逐端核actual bytes和MD5；mock错字段仍可被摘要记complete。固定下载器自身确会先实际校验完整bytes/MD5，反例没有证明当前真实文件错误。正在运行的协调器及旧回执保留，**旧协调器complete摘要不能单独作独立完成资格**。

新增[只读完成验收](../scripts/verify_bothe_downloads.py)由固定12run映射重核每端URL、文件路径、sample/history、expected/actual bytes、expected/actual MD5和SHA，再对真实完整文件重新计算size/MD5/SHA。metadata-only明确不认证实际文件；12run都通过才登记complete_files_verified，仍不登记FASTQ内容/同步、counts或模型资格。11项实际tiny/mock检查通过；合法JSON但错误payload形状逐run记录拒绝，整轮可继续。首次[audit1](../experiments/20261011-data-alignment-round19/download_completion_audit_1.json)及修后[audit2](../experiments/20261011-data-alignment-round19/download_completion_audit_2.json)均为0verified、12pending、0reject，没有把partial或旧摘要当完成。[新消费边界独立复审](../experiments/20261011-data-alignment-round19/independent_acquisition_review/technical_review.md)关闭；旧协调器内部弱摘要未改，后续真实executor从该受控验收路径消费完整raw。

具体下一步是完整双端raw→MD5/FASTQ一致性→专用真实比对/filter/count executor→12列QC与可测mask→来源观测出口/基因轴桥→base condition kwargs及同一响应/功能/选择链。Bothe提供的是四历史T32群体intronic RNA，不提供同细胞t0/t28、Annexin或Viel6+18/T24匹配表达。

## 7. 与研究架构一致的当前结论

架构SHA256保持`34e08ddb0f48e035fc6b3ba4f6db4f9a172bb5e159a30bc9ca8c8ffff5fb188e`。本轮支持§4.2的名义药物/完整history输入及§4.3的来源观测准备；§6的独立科学验证仍未完成。

状态表达增量、未见条件可靠性、真实凋亡功能连接、联合选择收益均保留。未见drug/dose/time不必各自有训练RNA才允许开展泛化研究；可信输入/实际执行、迁移身份、中间RNA验证、最终功能验证分别登记。独立功能Y能够证伪整体预测或选择收益，不能唯一证明中间高维表达。

本轮没有新增真实RNA—Annexin配对或核心G标签，未训练或实际执行P/G/Q；数据与整链可行性尚未全部通过。目标继续active；新source与工程实质进展不标为反复无进展的阻塞。没有修改研究架构、旧冻结数据/模型/桥或用户已有改动；无commit/push、Memory/Zotero写入或新Cloud/Dot派单。
