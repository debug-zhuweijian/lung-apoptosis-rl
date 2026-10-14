# 数据对齐第二十轮：真实表达、凋亡连接与顺序计数

日期：2026-10-11。主目标仍为“完整实现数据对齐，确保整个课题在数据上的可行性”。本轮使用 6.1-sol 专项执行与独立复审。**真实数据连接有新增进展，完整课题数据可行性仍未验收通过。模型训练保持停止。**

## 1. 现在可以明确回答什么

“每条凋亡实验都必须同时具有药后 RNA，否则模型无法完成”并不是必要条件。表达预测器 P 可以在实际表达数据上训练，再用合法参考状态和候选条件产生预测表达特征；终点模型 G 以这些特征、真实处理协议和实测 Annexin 条件结果校准。该路径不制造同一细胞或同一实验孔的配对。

原始先例 CSG2A 的 Fig.1d、§3.2.2、§3.3.2 使用预测表达与真实 log IC50 监督下游头，并通过细胞系连接基准表达。主代理本轮重读原文及固定源码 `CSG2A_finetune.forward`。这是预测特征连接功能监督的依据，**IC50 不是 Annexin，原论文也没有验证本课题的 A549 撤药方案**。原文固定 10 μM／72 h 是特征约定，不能补填本课题实际药物协议。具体出处见[原件与源码核查](../experiments/20261010-data-alignment-method-search/endpoint_transfer_review.md)，[论文正文，§3.2.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC11256952/)。

本轮还取得同论文 A549、同标称剂量与时钟的 RNA／Annexin 来源 HM97662，并已将真实 RNA 原值对齐当前基因轴。它消除了“公开来源中只有完全不同条件的表达和功能”这一笼统说法，但只有一个活性药物—剂量—时长组合，不能因此宣布完整状态条件化筛选成立。

数据上仍需分别回答：初态或参考是否合法、处理条件是否准确、生成表达是否可靠、实验定义的功能预测是否准确，以及候选选择是否改善。这些要求不等同于每个未见候选必须有自己的训练 RNA。

## 2. 六种对应关系不能混淆

| 对应关系 | 本轮实际情况 | 可支持的说法 |
|---|---|---|
| 基因坐标 | HM97662 有 30,268 个 exact Ensembl ID 对齐共同 32,461 轴 | 这部分原表达值可按冻结轴读取；缺失基因保留 mask |
| 样本与条件 | 六个 A549 GSM 与原表六列一一核对 | RNA 的条件和生物重复身份明确 |
| 药物、剂量、时间、完整协议 | HM 的 A549／1 μM／144 h 相容；RNA 换液明确，flow 换液未说明 | 同来源条件桥成立于已核字段；不能补成完整协议完全相同 |
| RNA 与实测功能 | HM 提供平行群体来源；Viel 可走预测表达—真实 Annexin 条件连接 | 条件层级功能监督，不是逐细胞未来命运标签 |
| RNA／Annexin 生物重复 | 没有证明 HM RNA rep1 对应 Annexin exp1 | 不配成三条伪独立 RNA—Y 记录 |
| 模型预测与候选 ID | v2 metadata 桥核 28 输入／24 活性 Y 身份／14 标准请求 | 可准备实际生产；P、G、Q 结果目前均为 0 |

## 3. 新 HM97662 来源及实际表达对齐

来源是 [Lee 等，Neoplasia 2026，DOI 10.1016/j.neo.2026.101349](https://doi.org/10.1016/j.neo.2026.101349)，[GEO GSE333828](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE333828)。主代理亲读 `PMC13476540.xml` 的 flow 方法、RNA 方法、RNA analysis、Results para0028 和 Fig.3A；同时读取 A549 的 GSM9774923、GSM9774926 并核原表，独立 reviewer 复核全部六个 GSM 与全部输出。

| 项目 | 已核事实 |
|---|---|
| 官方表达表 | 57,905 个基因、18 个样本，包含 A549、H841、SBC5 |
| 本轮 A549 范围 | GSM9774923–25 未处理三重复；GSM9774926–28 HM97662 三重复 |
| RNA 处理 | 1 μM、6 天；同期未处理亦 6 天；RNA 方法每 3 天换液 |
| 量化 | 原文 STAR／Salmon；GSM 说明 Salmon v1.10.3 的 gene-level quantification |
| 数值 | 六 A549 列 347,430 个值，其中 41,656 个非整数；保持 float64，不四舍五入 |
| 当前共同轴 | 32,461 个 Ensembl ID；30,268 个精确对应，覆盖 93.2442% |
| 未对应 | 2,193 个；输出 NaN＋显式 mask，不当作零表达 |
| 实际输出 | 6×32,461 NPY、gene mask、source-row mapping、GSM/sample 表、条件连接和 manifest |
| 功能定义 | Annexin V-FITC／PI flow，Fig.3A 定义 Annexin V-positive cells；3 个独立实验、mean±SD |
| 结果性质 | 作者报告 A549 没有显著凋亡诱导；ns 不能证明效应等于零 |

正式数据和原件在[HM97662 数据目录](../08-datasets/HM976622026/GSE333828/README.md)。实际对齐在[processed/joint-axis-v1](../08-datasets/HM976622026/GSE333828/processed/joint-axis-v1/README.md)，[准备与实际读回程序](../experiments/20261011-data-alignment-round20/hm97662_joint_alignment/prepare_verify.py)退出 0。独立从 gzip 解析原数值并逐元素比较 30,268×6 个对应值，差异为 0；2,193 个缺失基因全部 NaN，源／轴／输出哈希一致，见[独立审查](../experiments/20261011-data-alignment-round20/independent_hm97662_review/technical_review.md)。

这里的 untreated 是同期对照 RNA，不是给药前 t0。RNA 与 Annexin 没有已核同孔、同 batch、同细胞或逐重复配对。flow 换液和浮游细胞收集未说明，不把 RNA 每 72 h 换液复制给 flow。已提取下述近似 Annexin 发表均值，但没有注册学习角色或纳入核心 G。表达表不是 UMI，也不直接适用 Bothe 整数 intron-fragment count 接口。

本来源可提供一个有真实表达与真实功能出处的限定负例。它不能独自支撑 A549 内不同初始表达状态的功能增量、多药多剂量多时长选择或序贯效果；H841／SBC5 也不能替代 A549 的证明。

### 3.1 两条实测功能观测已恢复为机器可读近似表

| A549 条件 | 发表 Annexin V-positive 均值，近似 | 像素读图包络 | 对应 RNA 组 |
|---|---:|---:|---|
| 同期 NT，144 h | 15.9% | 14.7–17.1% | GSM9774923–25 |
| HM97662，1 μM／144 h | 14.0% | 12.8–15.2% | GSM9774926–28 |

读取的是 Fig.3A 右上量化图的 A549 第二对柱子，未用左侧代表性 FACS 象限代替组均值。原图 731×427；原尺寸零基纵轴 ticks y144／123／103／82／61／40 对应 0／10／20／30／40／50%，NT／HM 柱顶 y111／115。主代理亲看原图并以原像素重算，得到 15.9336%／14.0079%；保存值取一位小数。独立 reviewer 以边缘梯度和标定复核，源与产物 hash、百分比／fraction 转换和 RNA group 关系通过。

这些包络是保守的图像读数范围，不是生物 CI、SD 或显著性区间；多种检测共用同一原图，也不是独立生物证据。原三次实验未恢复为原始重复，SD 保持 null。近似均值差与原图 ns 不能证明 HM 降低凋亡或等效于对照。

数据在[Annexin 近似组表](../08-datasets/HM976622026/GSE333828/processed/annexin-published-means-v1/README.md)，[独立审查](../experiments/20261011-data-alignment-round20/independent_hm_annexin_review/technical_review.md)关闭已核范围。仅两个 source observations、两个 RNA groups，每条观测 weight=1；没有复制为六条 Y，role=unassigned，完整协议等价及模型资格仍 false。精确补表／PDF 的定向访问遇 403／404，未取得内容，不能说不存在更精确数值。

## 4. 新 α-hederin／sEV 来源的边界

已取得 [Frontiers in Pharmacology 2024，DOI 10.3389/fphar.2024.1257941](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2024.1257941/full) 的 PRJNA946809 官方 RNA run／BioSample 关系和公开 Excel。两个 sheet 为 463＋274，共 **737 个唯一差异基因×4 原 count 列**，不是全转录组。早期只读第一 sheet 的 463 已纠正。

RNA 是 sEV 48 h 条件；Fig.2D 功能是 sEV 与 PTX／α-hederin 或组合共同处理 48 h。未确认先测 sEV 状态后再挑战。Fig.2H 的 A549T 是另一背景，不能直接接该 RNA。该来源保留作表达参考与功能研究线索，不配成已验证的给药前状态—后续药物结局。

差异基因由全样本选出，不能当作无泄漏的完整高维输入。正式数据及具体条件见[数据说明](../08-datasets/AlphaHederin2024/PRJNA946809/README.md)；[来源专项报告](../experiments/20261011-data-alignment-round20/a549_function_bridge_sources/README.md)记录其他检索来源的排除理由和全文访问缺口。

### 4.1 KDM1B 未读全文项已完成限定补核

此前 EuropePMC API500 的 [KDM1B，DOI10.1002/jcb.70115](https://doi.org/10.1002/jcb.70115)现从正式 PMC HTML 取得全文。主代理亲读 Discussion末段（`#jcb70115-sec-0300`，[本地原文转写第1298行](../experiments/20261011-data-alignment-round21/kdm1b_source/PMC13434360_text.txt:1298)），作者明确本研究没有进行 RNA-seq等全局转录组分析。又核官方 [GSE33532](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33532)及GSM835268：20患者、100肿瘤/正常肺组织微阵列，是组织而非A549 WT/KO表达。

因此该项由“全文未核”转为“排除作为公开A549高维状态—功能RNA桥”。论文的真实Annexin图仍是限定功能线索，不补患者组织RNA，不扩增核心Y，也不宣称缺失补充中没有功能表。补充获取仍受限；取得原件、源字段与失败访问的完整记录见[第二十一轮限定来源审查](../experiments/20261011-data-alignment-round21/kdm1b_source/README.md)。这一排除不会将整个课题判为失败，但也没有填补A549高维初态对功能的增量证据。

只读个人库核对已完成：Zotero全库顶层分页60条，精确DOI与完整标题均无匹配，paperstudy论文索引亦无KDM1B/该DOI条目；这是当前个人库状态，不限制公开文献结论。见[2026-10-11核对记录](../experiments/20261011-data-alignment-round21/kdm1b_source/library_crosscheck.json)，没有写回Zotero或建立精读笔记。

### 4.2 耐药 A549 的新增定向补漏

实际执行四组 web 与两组 EuropePMC 查询，围绕顺铂／taxane 耐药 A549，不带RL／L1000限制；去重后收敛两篇新来源。主代理亲读两份正式XML的RNA段和数据声明，并重核17个原件hash；这轮没有取得新增的可读高维RNA，也没有增加核心Y。

| 新来源 | 原文实际RNA对象 | 数据与配对边界 |
|---|---|---|
| [KLF15/PGC1α，2026](https://www.nature.com/articles/s41420-026-03235-3) | `PMC13572412.xml:Par28`：A549/DDP siKLF15 vs control siRNA | `notes-type=data-availability`为按请求提供；Fig2主要功能为KLF15过表达，与此RNA knockdown不同。未把图注50µg/L改单位，也不借CCK8时间补flow |
| [NSUN5/GLUT1，2026](https://doi.org/10.1016/j.redox.2026.104193) | `PMC13137025.xml:p0100`及Fig3：A549/DDP shNC vs shNSUN5 RNA/Bis-seq | `p0030`明确RNA/Bis-seq需请求；§5.21 Annexin方法48h。本轮未取得公开矩阵和完整挑战剂量，不用维持耐药浓度代替challenge |

官方GSE213102原记录为顺铂24h后early-persister与untreated比较，是已核的另一来源，不是这两篇自建耐药背景RNA。按请求提供的声明不等于全领域没有公开来源；结论限定为本轮两篇及实际检查范围。KLF15小DOCX补充已读，补图PDF取得但未读；未联系作者。

[完整来源与覆盖记录](../experiments/20261011-data-alignment-round21/a549_resistance_source_search/README.md)保留原XML、方法和未核字段；[root源校验与个人库核对](../experiments/20261011-data-alignment-round21/a549_resistance_source_search/root_source_and_library_check.json)显示17文件一致、Zotero全top60条精确DOI/标题无匹配，论文索引精确DOI亦无匹配。保留为限定功能线索，不能把它们写为已经补齐A549高维状态—功能数据。

## 5. 预测表达—实测功能的可执行 metadata 桥

最新有效版本为[功能连接 v2 设计](../experiments/20261011-data-alignment-round20/predicted_expression_function_bridge/v2/design_report.md)及[执行程序](../experiments/20261011-data-alignment-round20/predicted_expression_function_bridge/v2/metadata_bridge.py)。它实际读取、重建并验证：

- 28 个候选输入映射 24 个活性观测身份；Nav 10／11 μM 两种解释共享 4 个 Y，不扩为 8 个实验。
- 14 个标准 P 请求；20 个 Viel 候选具有完整 physical 输入。
- 固定 Tahoe A549 proxy 的 2,431 个 fit 对照细胞索引。它是统一外部参考，不是 Viel 或 Nav 实测初态。
- 10 折×2 分支×24 个角色单位，共 480 条角色 metadata；每折 16 fit／2 selection／2 final／4 external。
- 真实指纹、剂量、物理时钟、候选／观测／状态／来源／请求／view 逐项对应；未反序列化 Y 值。

初版只将部分 source hash 写回执，未固定审计身份；独立反例可让自更新 role hash 的错误 state、fit/final 交换或错误 reference source 进入审计。v2 固定已审核 PLAN／FULL／reference 的字节和 JSON 身份及相关 consumer，三个反例全部拒绝。初版保留为历史快照，见[独立复审](../experiments/20261011-data-alignment-round20/independent_function_bridge_review/technical_review.md)。

该桥只有 metadata／设计产物，没有生产消费或资格 API。所有 P artifact 路径仍为空。真实生成与训练必须走受控 producer／cache／角色接口；metadata 通过不能给 caller 数组贴“P 已生成”的标签。

### 5.1 已确认的实际代码缺口

旧 standard P 将同药同剂量的 pulse 和 continuous 合并成连续 24 h 请求，实际撤药协议主要进入 G。它可作对照，不能宣布完整协议已经进入 P。完整实现需要匹配版本的 `FullProtocolLatentAdapter.sample_latent`、来源 decoder、真实表达摘要生成与 G 消费接口。旧模型权重不能仅改名字后当作已训练完整协议。

Nav 的检测定义、源状态与 reserved external scope 不兼容现有 Viel G／评价。四个 reserved Y 不挪作校准后仍称独立测试。需要新的明确域迁移方案或独立校准来源；这是未关闭的研究条件，不能用接口自动通过来代替。

既有每折 final 的独立候选块是完整 pulse／continuous pair。不同折有不同 G，不能拼成单一冻结模型对 20 个候选的独立排行榜。当前 metadata 合同为后续执行提供合法连接，不等于完整 D 已验证。

## 6. Bothe 真实 RNA→counts 的实际执行推进

上一轮完成完整 human hg19 index、unique-intron 注释及十对参考 DNA 合成数据处理；它们不计为真实 RNA。本轮完成新[FASTQ 验证器](../src/lung/data/paired_fastq_validation.py)、[受控计数 driver](../src/lung/data/bothe_count_executor.py)、[生产读回](../src/lung/data/bothe_count_production.py)和[CLI](../scripts/process_bothe_rna.py)。

固定处理链为：注册 12-run 来源／正式下载回执 → 每端 bytes／MD5／SHA → gzip 全文 CRC／EOF 和 FASTQ 双端同步 → 固定 Subjunc → 严格双端 SAM 过滤 → 固定 featureCounts → 全 feature 解析 → NCBI gene count 矩阵。只接受 registered runs 和新 version，不接 caller raw 路径、runner、count 表或资格回执。

FASTQ 不强制每条 75 bp，记录实际长度分布和偏离；不凭字符猜 Phred offset。主代理发现并亲读 [ENA archive-generated FASTQ 说明](https://ena-docs.readthedocs.io/en/latest/retrieval/file-download/archive-generated-files.html)，专项核 12 run／24 文件均 `GENERATED_FILE` 且链接与固定源一致，因此 registered 来源声明为 Phred+33；任意 audit 文件仍保持偏移未知。

独立审查关闭两项 FASTQ P2：扫描文件必须直接与下载绑定逐端相同，不能 A→B→A；registered header 额外 token 不能隐去错误端标识。最终 validator SHA 为 `c669c1cb8b3ef38dd8d3ea9d97551b00e7264b4f09d5ac9619962022faf330da`，driver 已锁定。[FASTQ 复审](../experiments/20261011-data-alignment-round20/independent_fastq_review/technical_review.md)与[计数复审](../experiments/20261011-data-alignment-round20/independent_count_executor_review/technical_review.md)均关闭已证实 P1/P2。

矩阵保留 12 sample×29,634 selected gene。未取得 sample、没有可测 feature 的 gene 使用 NA；观测到且可测的零 count 才是整数 0。全部 12 run 才为 counts_complete，少于 12 为 prepared_partial。该域是 bulk total-RNA unique-intron fragment counts，不是单细胞 UMI；counts 完成也不自动授予模型或功能资格。

### 6.1 Root 数据进程：首次先导已失败，新先导从零重试

| Root session | 实际任务 | 本轮已核状态 |
|---|---|---|
| 28793，已终止 | 首次 genome／ERR5000305 pilot 下载 | genome 完整验证；R1 收满声明长度但 MD5 错误，exit1；失败文件保留，未发布回执 |
| 54677，新重试 | 原登记路径从零重取先导双端 | 保留坏副本后使用原下载器／官方 MD5；21:55:50 UTC root 轮询返回非终止句柄，R1重新传输，未完成 |
| 5491 | 其余 11 run，两下载 worker | 同批 root 轮询非终止；0296已通过R1守卫并进入R2，0298R1继续；其他排队 |
| 10921 | registered-rna-counts-v1，全部 12 run 顺序处理 | 已实际启动，完整 index 再核后等待 pilot 正式回执和双完整 gzip；未开始真实 RNA native stage |

计数进程使用 `--execute --wait-for-downloads`。每个 run 到齐后进入原严格验证和真实处理；缺端或 `.part` 不放行。等待不增加 counts，不授予任何资格，不设总耗时上限；native 同一时刻串行。状态文件只是快照，进程是否仍运行须以 root 句柄再次读取；不以 PID／日志时间猜测存续。

下载器、正在运行的采集协调器、旧 SAM filter、旧 count parser、索引、注释与模型保持其原字节。后续不要编辑在跑 driver／production／CLI／validator；执行最后会再次核 source bindings。最新过程证据见[接续说明](../experiments/20261011-data-alignment-round20/CONTINUATION_HANDOFF_2026-10-11.md)。

首次先导的稳定副本虽然达到 1,507,908,000 bytes，但 actual MD5 与官方值不同，root 追加的 gzip 容器检查也在流末尾发现 CRC 不符。ENA 两次当前声明与固定映射一致，远端五处各1 KiB局部相同不能证明完整远端内容正确；具体损坏来源未定位。错误副本已同目录保留 rename，54677 在原登记路径从零重下，原校验值与回执合同不变。完整诊断、保留和新任务证据见[下载故障与恢复记录](../experiments/20261011-data-alignment-round20/pilot_md5_diagnosis/README.md)。没有把坏副本送入计数，也没有据此认定数据集缺少样本。

### 6.2 官方 GeneID→Ensembl 桥及真实计数消费接口

实际取得 NCBI human-specific `gene_info.gz`，5,194,256 bytes；[官方 README](https://ftp.ncbi.nlm.nih.gov/gene/README)说明 GeneID、taxonomy 与 `dbXrefs` 的 database:value 结构，并明确这里的 Ensembl xref 与 gene2ensembl 可不完全一致。文件含 193,925 条 tax9606 和 73 条其他 taxonomy；后者显式隔离，不能按人类文件名假设全部为 9606。

对 Bothe 全 29,634 个 selected genes，结果如下；状态按规定顺序互斥，原 annotation status 仍保留。

| 源 GeneID 状态 | 数量 |
|---|---:|
| 可测且取得唯一 joint 对应 | 17,107 |
| 官方对应不在 joint 轴 | 7,039 |
| 没有 Ensembl dbXref | 2,705 |
| 对应在 joint，但本注释不可测 | 2,337 |
| 多个当前官方 GeneID 指同一 Ensembl | 437 |
| 当前官方文件缺席，退休未确认 | 8 |
| 一对多 | 1 |

完整 joint 32,461 行、source 29,634 行、transcript scope 99,517 行和 taxonomy 排除台账全部保存。只用官方 exact ID，不靠 symbol 猜，不合并歧义，不追未确认替换。该当前 snapshot 不等于作者 2021 年注释年代。独立重建全部关系、轴顺序与 NA 一致，见[正式桥](../08-datasets/Bothe2021/E-MTAB-9923/reference/gene-axis-bridge-v1/README.md)。

初版 loader 可接受 caller 替换的 source 和自报 hash、缺失台账或资格字段。修后固定 8 项 source hashes／登记路径，并对照冻结 annotation 输出；四份 ledger 和 summary 必须从实际官方源重建，raw／counts／model／function 等资格严格 false。原 spoof 及其他反例拒绝，[独立复审](../experiments/20261011-data-alignment-round20/independent_bothe_gene_bridge_review/technical_review.md)已关闭。

新固定接口 `project_production_counts(version)` 内部只调用已审生产 loader 和固定 ID 桥，不接 caller counts、array、路径或 hash。将实际已验收 Bothe count 投影为 joint32,461×12 的整数 tuple 和 observed mask；未映射／无 feature／缺 sample 为 None，真实 0 为整数 0，保留 int64 精度。CLI `build_bothe_gene_axis_bridge.py --mode project --version <version>` 只读，不覆盖数据。**接口与隔离测试通过，真实生产矩阵仍未完成，因此真实投影尚未执行**；不能将 tiny fixture 当 RNA counts。

## 7. 对完整研究架构的核对

[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)字节保持 `34e08ddb0f48e035fc6b3ba4f6db4f9a172bb5e159a30bc9ca8c8ffff5fb188e`。本轮没有改主线、奖励、角色或候选成功标准。

| 研究方面 | 本轮贡献 | 当前仍需完成 |
|---|---|---|
| A 状态信息增量 | 保留已有真实表达监督；HM 新表达精确对齐 | 一个 A549 背景或统一 proxy 不能证明状态对功能的增量 |
| B 未见条件可靠性 | 保留 source／compound／protocol 留出身份和真实角色 | 实际合格 P 及留出预测未执行；时间迁移不等于已验证 |
| C 真实功能一致性 | 预测特征—实测 Y 路径具体化；新同来源 RNA／Annexin 条件桥 | qualified P、G 校准、Annexin 数值／范围和独立评测仍需执行 |
| D 完整候选选择 | 原完整 candidate、实际协议与角色连接保留 | 同一实际算法链的 Q／枚举／简单对照及独立可比 Y 仍未完成 |
| E／D／P／C／G／Q 同链 | 新数据量纲与职责分开，不把代理或 caspase 换成 Annexin | 来源 decoder、base kwargs、full producer、匹配权重、实际同链产物仍 pending |

“能构造合法监督和证伪实验”与“实验效果会成功”不同。表达配对、功能条件关系与候选评测有不同证据粒度；不将新负例、工程通过或内部高分写成完整课题已保证成功。

## 8. 验证与交付边界

- Root 最终 **131 passed in 12.14s**：FASTQ 33＋executor 37＋旧 parser 31＋ID 桥／生产消费30。此前101是count链子阶段，新增30另经独立复审通过。汇总的metadata/source／11文档263本地链接／git diff-check通过，在跑count五个源码hash未改，见[完整验证](../experiments/20261011-data-alignment-round20/validation.json)、[测试stdout](../experiments/20261011-data-alignment-round20/final_test_result.json)和[文档检查](../experiments/20261011-data-alignment-round20/document_validation.json)。独立反例不重复计入单测总数，也不与上一轮138简单相加。
- HM 实际小矩阵读取、对齐、NPY 读回退出 0；全部对应值和缺失 mask 独立重核通过。
- HM 两条近似功能组均值、坐标／单位／两 RNA group 关系独立核验通过；不增加独立实验或恢复 SD。
- 功能 metadata v2 重建与保存一致，原三个篡改反例拒绝；实际 P／G／Q 数量仍为 0。
- 新来源与 Zotero 全库 60 个顶层条目穷尽分页核对，精确 DOI／HM 标题未匹配；论文索引也未命中相关 DOI／名称。只覆盖这些字段，不排除别名；[查询证据](../experiments/20261011-data-alignment-round20/library_crosscheck/result.json)。未写回 Zotero。
- 数据／NPY／Excel／gzip 本体由 git-ignore 留在本地。原脏工作树保留，没有 commit／push、记忆写入、作者联系或模型训练恢复。

## 9. 下一步仍围绕数据与真实执行

1. 维护三个 root 数据句柄，完整下载到齐后观察真实 FASTQ／native/count 阶段及终止码；失败保留具体证据，不以等待认定失败。
2. 真实 Bothe 生产完成后，使用已完成的官方 ID 桥／固定消费接口核真正的共同轴 count 值和 mask；保留来源观测域，继续准备来源出口。
3. 将新 HM 的限定条件、近似功能值、量纲和完整协议 unknown 逐项纳入后续数据角色审查，不把一个来源直接充当全部验证。
4. 完成 full producer、来源 decoder 和 G／Q 实际消费依赖；模型执行与训练继续遵守停止状态，不借数据准备自动恢复。
5. 在同一程序、合法候选和独立真实终点上检查 A/B/C/D；完整目标保持活动，当前不标记验收完成。
