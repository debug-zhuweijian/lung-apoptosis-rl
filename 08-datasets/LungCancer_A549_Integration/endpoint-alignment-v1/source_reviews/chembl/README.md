# ChEMBL A549 Annexin 原文核验子集

核验日期：2026-10-10。此次完成 **5 篇原始来源的实质审查**，对应 12 个带来源的单药处理记录；其中 5 个记录的剂量、24h 时长和已发表数字可由原文确认，另有明确排除或条件不完整的记录。当前全部 `use_for_training=false`，等待主代理亲读关键原文和整合审查。没有修改 ChEMBL 候选池、prepared/training manifest 或训练代码。

本目录只保存原始来源缓存、逐文 JSON 和核验说明。上游数据库快照为 `08-datasets/ChEMBL_A549_Annexin/ChEMBL_37/audit_2026-10-09/`。数据库 activity 是文献的二次抽取；下表数字由原文正文或原图亲读后单独确认。

| 来源及逐文记录 | 可确认的 A549 单药终点 | 处理与数字证据 | 审查结论与边界 |
|---|---|---|---|
| [Puri 2023](CHEMBL5316296_Puri2023.json)，DOI [10.1021/acs.jmedchem.2c01612](https://doi.org/10.1021/acs.jmedchem.2c01612) | compound24，5µM，24h：early1.60%、double-positive5.08%，合计6.68%；untreated0.51% | Results sec2.2.6、Methods sec4.2.8、原Fig9。原图为3独立实验中的代表实验。作者CSV的24号SMILES与ChEMBL完整InChIKey实算相同 | 原文处理、数字和作者结构对应已核验。没有3个逐重复数值/FCS，没有核验与Tahoe表达相接 |
| [Chen 2018](CHEMBL4138167_Chen2018.json)，DOI [10.1021/acs.jnatprod.7b00578](https://doi.org/10.1021/acs.jnatprod.7b00578) | compound14，2µM24h：14.8+3.8=18.6%；4µM24h：26.5+4.4=30.9%；control1.43+2.97=4.40% | 原始作者稿Results直接给出两门数值、Fig2图注写24h及3独立实验，Methods明确FITC-Annexin/PI和仪器。8µM原值仍保留，超出本次0.05–5µM优先域 | 已确认正文标量。右上门原文称late apoptotic and necrotic，应保存为Annexin/PI双阳性。原Fig2/逐重复/SEM数字未取得，结构尚需逐原子核验 |
| [Wang 2014](CHEMBL3352466_Wang2014.json)，DOI [10.1021/jm5003588](https://doi.org/10.1021/jm5003588) | compound9a，100nM24h：0.7+3.0=3.7%，CTL1.1%。同药100nM48h：1.6+32.5=34.1%，CTL2.0% | Results sec2.3.2、Methods sec4.18、原Fig3B上下两行均已视读。100nM=0.1µM | 24h标量可纳入后续核验，48h必须单列。未报告明确的凋亡实验n，不套用细胞毒性triplicate。结构与候选目录连接未完成 |
| [Mostafa 2018](CHEMBL4270553_Mostafa2018.json)，DOI [10.1016/j.ejmech.2018.07.004](https://doi.org/10.1016/j.ejmech.2018.07.004) | compound19明确2µM24h：3.27+3.31=6.58%，正文四舍五入为6.6%。5FU原图确有6.37+6.59=12.96%，control0.52+0.33=0.85% | 作者上传的原始论文Results2.2.5及Methods4.2.4.2可读；Elsevier官方高清Fig6B已下载视读 | 19的处理与数字有原文支持；5FU的2µM24h仅由ChEMBL记录，本轮未独立确认参考药完整条件，不把19条件隐式移给5FU。四象限总和分别98.04/98.18/98.42%，原文缺完整分母/排除门说明，保留质量标记且不重新归一化 |
| [Klingbeil 2016](CHEMBL5712891_Klingbeil2016.json)，DOI [10.1038/cddis.2016.271](https://www.nature.com/articles/cddis2016271) | JQ1 0.5/2.5µM原Fig1d均total2.7%，DMSO3.6%；OTX015 S1C为0.5µM1.8%、2.5µM2.6%，DMSO2.1% | 主Fig1d、官方S1原图、Methods和PDF第3页已读。纵轴为Annexin，横轴为PI | **主Fig1d明确48h，不能使用ChEMBL的24h标记。ChEMBL将early和PI单阳性门对调。OTX图未给时长，暂不放入24h集** |

## 与后续数据对齐的关系

本轮可确认剂量与24h的5个处理条件来自4篇论文和4种编号化合物。多个门不是独立条件；同一化合物的不同剂量也不是独立药物。5个条件的Annexin阳性公开标量范围为3.7–30.9%，提供低值和较高值来源，但没有在本轮证明这些新结构已对应Tahoe中的实测表达。

`eligible_for_priority_005_to_5um_24h_review` 只表示满足本轮时间/剂量优先规则并已读到相应发表数字，**不是监督闭环或训练准入字段**。`inclusion_decision` 另保留分母、结构或处理条件的审查限制。Mostafa19仍带分母未完全核验的条件限制；5FU不计入5个明确条件。没有把发表代表图、均值或数据库值转成不存在的原始重复。

可以复用的下一步连接是：先用 `document_chembl_id + primary_compound_id + molecule_chembl_id` 保留来源身份，再对母体/盐型/完整InChIKey进行结构审查，明确剂量和真实暴露时长，最后连接候选目录。新增结构只能进入已有方法所支持的未见化合物评测或校准路线；缺对应实测RNA时，不称表达—Annexin真配对。原始重复与跨来源实验背景的不确定性需要进入评测设计。

## 本轮读到的范围与未读边界

`raw_manifest.json` 登记所有缓存的来源URL、大小、SHA256及是否含全文body。`PMC4111373.xml`、`PMC6448791.xml`、`PMC5059870.xml`、`PMC9841531.xml` 有相关正文，已读与本任务对应的Methods/Results。Mostafa论文通过作者上传的原文网页读取，直接本地下载遇到403；本地保存的是官方Fig6原图，JSON保留具体章节和可读原文链接。

为在已有131篇候选中定位可读来源，进行了精确DOI的PMC元数据检索。缓存中的其余8个PMC XML **只含元数据/摘要、没有body，不能称全文已读或数据已通过**。没有逐篇读取131篇全文，也没有把未检索到PMC误写成不存在全文。

当前没有取得任何新的逐重复Annexin数组或FCS。Puri/Chen的n=3为作者声明，Wang/Mostafa/BET单药的凋亡n未明确。Puri、Wang补充PDF已检查相关端点，未发现逐重复Annexin表；作者结构CSV不能当凋亡数据表。没有联系作者或读取新的大规模表达本体。

## 文件校验

5份逐文JSON均可解析；12个单药记录的early+double-positive计算与total一致；时间/剂量优先字段和纳入判断分别记录；所有引用的本地原文文件存在。药物SMILES经RDKit重算完整InChIKey与保留的ChEMBL标识一致；其中Puri24另与作者CSV结构进行独立比对，其余来源不据此声称完成原始结构逐原子核验。

未调用训练；未编辑正式候选池或其他代理的文件。仅本目录有新增缓存、审查记录和README。

