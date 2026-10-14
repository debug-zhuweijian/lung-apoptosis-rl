# GSE306207 真实序贯表达来源读回

本轮已解决毕业论文 §5.1 保留的 GSE306207 来源尚未在 I 项目落实真实矩阵的问题。实际取得并读取 **A375 bulk TPM，4条件×2个作者报告的生物重复，62,710源基因×8样本**；完成8列对唯一GSM的字段绑定及 **32,461共同基因的259,688个值逐值读回**。只授已测 A375 历史终点表达的来源资格，未运行模型、建立全局角色、生成凋亡标签或更改奖励。

## 原件、版本与可定位事实

原始入口为 [GEO GSE306207](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE306207)，本地[新数据版本](../08-datasets/GSE306207/geo-public-20261011-v1/README.md)。GEO public 为2025-12-22，record last update 为2026-02-06；BioProject PRJNA1309681、GPL29480、Homo sapiens/taxid9606。family SOFT 3,342B，gzip完整读取；官方工作簿3,005,866B、SHA256 `5ec2d9ded179e501a194376b39110e3ccf01dd6457d26c42b0c8c60f14239e01`，ZIP CRC各成员通过。[SOFT解压原文](../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/GSE306207_family.soft)第7–14行含标题、日期、DOI、summary和完整 Overall design，第35行指向唯一处理工作簿。

[原出版页](https://www.nature.com/articles/s42256-026-01192-1)将该 accession 关联至 Chen等2026 *Reinforcement learning-based design of sequential drug treatment targeting the evolving tumour landscape with SequenTx*，Nature Machine Intelligence8:351–371，DOI10.1038/s42256-026-01192-1。实际保存其HTML和链接PDF；GEO摘要沿用AlphaTherapy旧名，字段不改。出版页 Data availability 将GSE306207定义为序贯机制研究的RNA来源，不能用论文的其它来源训练标签或其它细胞功能实验填成本accession标签。

Reporting Summary [原PDF](../08-datasets/GSE306207/geo-public-20261011-v1/42256_2026_1192_MOESM2_ESM.pdf)5页、3,911,416B，SHA256 `50d02ed654518a0b088ace6b769a273407ab3c35a5f06e21fa2e52d345f0ff3e`。第2页 RNA-seq analysis 支持GRCh38、STAR2.7.9a、RSEM1.3.1及TPM，与GEO一致；第3页 Replication 明确RNA-seq双重复，并将数据报告为biological replicates，其它实验多于三次不套入RNA。PDF文本提取只有少量特殊字符，本轮渲染并亲读原第1–4页，关键[第3页图像](../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/reporting-summary-page-3.png)保留。第4页Flow cytometry描述固定乙醇/RNase/PI的细胞周期检测，没有为本RNA包提供Annexin fraction标签。

## 真实样本—历史设计

| 真实GSM | 源矩阵列 | 已报道处理 | 表达采样时钟 |
|---|---|---|---|
| GSM9194773/4774 | S1/S2_A375_NC | DMSO control | GEO所读字段未明确；保留unknown |
| GSM9194775/4776 | S1/S2_A375_IBET726 | I-BET726 2µM，72h | 处理结束，名义72h |
| GSM9194777/4778 | S1/S2_A375_IBET726W | 同一2µM/72h前处理，随后without drug 72h | 最后阶段结束，所报道阶段合计名义144h |
| GSM9194779/4780 | S1/S2_A375_726_Oxa | I-BET726 2µM/72h后，Oxaliplatin10µM/72h | 最后阶段结束，所报道阶段合计名义144h |

上述给药顺序与长度来自 SOFT第14行 Overall design及各sample treatment protocol，例如第67行。第68行 growth protocol 为DMEM/10%FBS/1%penicillin-streptomycin、37℃、5%CO2。不能由“followed by”推定精确洗脱操作、前药是否与第二药继续共存或额外转换间隙为0；名义144h是两段72h的设计合计，不补绝对时间戳或分钟级处理误差。

完整列绑定来自sample title与description：GSM9194773的54/73行、4774的96/115行、4775的138/157行、4776的180/199行、4777的222/241行、4778的264/283行、4779的306/325行、4780的348/367行。各记录的processing字段，例如第74–76行，说明Cutadapt1.9、GRCh38/STAR2.7.9a、RSEM1.3.1、TPM及XLSX格式。sample注释逐项保存原行号。重复编号1/2不证明跨处理是同一培养批次或同一细胞的配对采样。

## 表达、共同轴与实际验收

官方XLSX的单sheet `A375_Seq_All_Samples_TPM` 有62,710个唯一、无版本后缀的 exact Ensembl gene IDs，8个样本列float64、无空值且全部有限非负。列总TPM约1,000,000；没有重算、取整或变为raw counts。例如ENSG00000000003的I-BET726 S1/S2为12.12/9.59，DMSO S1/S2为13.22/14.23，序贯S1/S2为9.82/12.04，均可直接在原工作簿定位。

共同轴采用冻结 `training-joint-sciplex-v1/genes.parquet`，SHA256 `4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc`。32,461个轴基因全部在源中，无重复猜配，另外30,249个源基因保留在source矩阵但不投影到joint。没有fit normalization、source latent、decoder或A375到A549迁移验证。

[实际派生包](../08-datasets/GSE306207/geo-public-20261011-v1/processed/source-observations-joint-axis-v1/manifest.json)保存8×62,710 source TPM、8×32,461 joint TPM、bool mask与 gene ledger；本源missing=0、mask全真，测得零值仍为有效观察。读取与投影不改变原XLSX，source矩阵仍保留全部源基因。原始值和共同轴字段通过独立openpyxl逐格oracle核对：**501,680个源值及259,688个joint值完全相同**，8个GSM/title/description/A375字段逐项闭合。来源SHA在读前后相等。[独立实际回执](../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/independent_actual_readback_receipt.json)包含原PDF、family、轴和矩阵绑定；不能只看manifest的status推定验收。

[9项投影边界测试](../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/pytest_projection_v1.txt)实际0.93s通过，新I basetemp `experiments/gseqr1`先检查不存在，验证浮点与零/缺失区分、重复ID、不合法值及源字段映射。这些测试只验证软件；真实文件逐值读回另列。首轮study环境缺openpyxl、只读oracle重复解压效率故障及其自身进程精确停止记录保留；复用本机base环境、一次读入数组后实际读回通过，没有动其它live作业或删除失败。

## 能监督什么与仍缺什么

这组数据提供单药后72h的实际群体表达，以及同一前处理后撤药/第二药两条分支的实际末端表达。可以为对应A375固定历史和剂量的来源内表达预测、撤药恢复及有限两阶段路径提供监督或小规模外部诊断。前处理72h群体可作为后一阶段的可观测population landmark；跨条件没有同细胞或同培养批次配对证明，因此不能创建逐细胞轨迹，也不能把S1/S2跨组人工配成真实个体转移。

当前不含反向Oxaliplatin→I-BET726、Oxaliplatin单药RNA、持续I-BET726到144h、明确分别匹配72/144h的DMSO、更多剂量/间隔/路径或连续中间时间点。因此不能凭此独立识别所有顺序效应、任意时间规律、通用Markov状态、六步真实滚动或完整策略收益。withdrawal与sequential两个末端有共同前处理和名义总阶段长度，但转换方法/保留前药未知仍需保留，不能声称完整物理协议等价已认证。每条件n=2，统计功效有限。

没有本包匹配Annexin或其它已登记功能reward；RNA和PI细胞周期不转换成促凋亡率。A375是黑色素瘤，基因轴完全覆盖也不改变背景或TPM观测域，更不替代主A549、状态增量、独立终点、联合选择和真实序贯完整目标。已有模型配置的背景/源解码及输入预处理另需适配，单细胞counts的library10000转换不能直接套用源TPM。本包所有样本role保持unassigned，没有更改原global split、奖励或正式训练停止安排。

## 接续与原始访问限制

普通GEO web入口先遇reCAPTCHA，publisher图页/部分PDF的web工具遇redirect500；同一官方text/FTP与publisher链接通过requests实际200获取并读回。该差异及首轮错误见[访问限制](../experiments/20261011-data-alignment-round22/science_evaluation/sequence_GSE306207/web_access_limitations.json)；未以失败或空返回推断资料不存在。所存主出版页为subscription preview，完整主文Methods尚未取得；supplement1实际读取但没有补足所缺处理细节，Reporting Summary完成原图阅读。没有联系作者、使用凭证或下载原始测序。

与当前[研究架构](G:/研究生科研文件/毕业论文/研究架构.md) §5.1、§6一致：完整目标继续保留，该来源只推进原已确认的A375辅助序贯表达层。后续先复核此source manifest与原件，独立制定A375源适配/角色及可识别评价；模型停止期间继续数据和证据工作。需要真实NN、PT、forward或fit时另有具体授权，当前未执行。

## 追加：两个来源名称的名义结构与FP

先核既有冻结identity-v3：Oxaliplatin已有CID9887053完整立体图、官方属性原响应及第25行含手性FP；I-BET726未在该目录的exact name行中找到。没有重复构建或改写全局identity表。本次在[独立chemical-identity-v1](../08-datasets/GSE306207/geo-public-20261011-v1/chemical-identity-v1/README.md)保存这两个源名称的官方名义结构核验。

2026-10-11实际读取 [PubChem I-BET726](https://pubchem.ncbi.nlm.nih.gov/compound/52912222) 与 [PubChem Oxaliplatin](https://pubchem.ncbi.nlm.nih.gov/compound/9887053) 的官方PUG REST complete-name→CID、CID properties和synonyms JSON。两个原名称各唯一解析为CID52912222/9887053；原返回CID和官方synonym同词均闭合（Oxaliplatin同词小写），没有人工换别名或挑其它parent。

| 源名称 | 完整立体InChIKey | PubChem图组分 | FP置位 |
|---|---|---|---|
| I-BET726/CID52912222 | FAWSUKOIROHXAP-NPMXOYFQSA-N | 1，2个定义立体中心 | 50 |
| Oxaliplatin/CID9887053 | DRMCATBEKSVAPL-BNTLRKBRSA-N | 3，2个定义立体中心 | 17 |

完整立体SMILES、InChI、公式及响应版本保存在[nominal_chemical_identity.json](../08-datasets/GSE306207/geo-public-20261011-v1/chemical-identity-v1/nominal_chemical_identity.json)和原`raw_pubchem/CID*_properties.json`。RDKit从完整SMILES得到的InChIKey与官方一致；未定义原子/键立体数均0。Oxaliplatin保留PubChem原铂/草酸/二胺三个组分及电荷，没有strip salt、largest-fragment选择或另造配位图。

按照项目既有 `GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)`计算2×2048 uint8 FP，记录RDKit2026.03.6。Oxaliplatin的结构/Key/InChI与旧官方`properties_0200.json`（SHA256 `8d77794471a2edac0fa7b0ad18ef92946a609a42ecd7d850d9b532cbc3fecb62`）一致，FP与冻结identity-v3第25行逐位相同。独立旧Morgan API再次核全部位相同。7项边界tests实际0.74s通过、新I basetemp `experiments/gseqchem1`先检查不存在，拒绝多CID、不同CID、官方同名缺失、立体信息丢失及以二胺片段替代Oxaliplatin整图。

[真实化学读回](../08-datasets/GSE306207/geo-public-20261011-v1/chemical-identity-v1/actual_chemical_readback_receipt.json)与[源阶段链接](../08-datasets/GSE306207/geo-public-20261011-v1/chemical-identity-v1/source_protocol_event_links.json)将8个源样本中的6次I-BET726及2次Oxaliplatin阶段对到名义图/FP行；无药阶段没有假结构，原dose/单位/时钟/未知协议和unassigned角色保留。该连接只落实来源名义化学图。所读GEO/作者原件未给本GSM stock/批号/具体盐型；完整主文Methods也未取得，不能将名义PubChem图当实际实验材料鉴定。

RNA原包及其manifest SHA保持不变，本化学附属版本单独绑定source manifest和冻结chemical资产。旧source delivery回执中的文档SHA适用于追加前报告；新的化学交付回执记录追加后的报告SHA，旧回执不覆盖。当前真实NN/PT/forward/fit均0，没有更新全局药物表、roles、奖励、P模型或其它来源资格，完整目标继续未完成。
