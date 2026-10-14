# A549 状态—凋亡功能连接：第二轮源数据核验

核验日期：2026-10-10。范围：A549 亲本／明确来源的耐药派生状态、同源转录组及药后实测 Annexin/Caspase。只保存源文件、解析脚本和核验记录；未训练、未写入正式训练集，也未改变上一轮冻结材料。

## 1. 本轮结论

取得一条此前没有完成的可复算连接：**Sarin 2017 的 A549／A549rCDDP2000 顺铂后 Annexin 来源值，与 Sarin 2018 的 GSE108214 表达数据，在指定细胞来源、培养协议、药物、剂量、24 h 时长上可做跨研究条件级对应。** 这比仅凭泛称 A549/DDP 连接两个研究严格得多；两文明确采用 RCCL 的 A549rCDDP2000，同团队后文还直接引用前文实验。

但它仍然**不是实测给药前 t=0 RNA → 同一实验未来命运**：无药 RNA 是平行培养 24 h 后的对照，两个研究没有共享样本／培养孔／重复 ID。正确角色是 `time_matched_control_reference`，不是 `observed_initial_state`。它提供两种已知群体状态的参考表达，以及相同协议的末端表达与凋亡诱导倍数，不能独自完成原课题的完整数据闭环。

数据规模为 **2 种细胞状态 × 1 种药物的 3 个处理条件**；表达矩阵 **29,729 个探针 × 22 个样本**，Annexin **10 个源数据值**。10 个观测不能写成 10 个独立状态或 10 个不同干预。也不能将 RNA 重复与 Annexin 表中同序号重复配对，或将所有 RNA × 所有 Annexin 交叉扩增为样本量。

## 2. 原始来源与连接依据

| 来源 | 已取得材料 | 作用及精确定位 |
|---|---|---|
| Sarin 2017, *Cisplatin resistance in non-small cell lung cancer cells is associated with an abrogation of cisplatin-induced G2/M cell cycle arrest*, DOI [10.1371/journal.pone.0181081](https://doi.org/10.1371/journal.pone.0181081), [PMC5528889](https://pmc.ncbi.nlm.nih.gov/articles/PMC5528889/) | `PMC5528889.xml`；`pone.0181081.s008.docx` | Cell lines：XML `sec004`；Apoptosis assay：`sec009`；Statistical analysis：`sec012`；Apoptosis induction：`sec018`；Fig 5a；S8 DOCX 第 6 张表、第 4–7 行、第 2–4 列 |
| Sarin 2018, *Key Players of Cisplatin Resistance: Towards a Systems Pharmacology Approach*, DOI [10.3390/ijms19030767](https://doi.org/10.3390/ijms19030767), [PMC5877628](https://pmc.ncbi.nlm.nih.gov/articles/PMC5877628/) | `PMC5877628.xml` | §4.1 Drugs、§4.2 Cell Lines、§4.3 Microarray；参考文献 6 明确为前一篇论文 |
| [GSE108214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108214) | `GSE108214_family.soft.gz`，含 GPL17077 注释及全部 22 张样本表 | 每个 GSM 的 sensitivity、treatment conc、growth/treatment protocol、VALUE 处理说明；包含亲本与指定耐药株的无药及顺铂处理表达 |
| [PLOS 原始 S8 文件](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181081.s008&type=supplementary) | 98,060 字节 DOCX，直接解析 `word/document.xml` | 原始数字，不是柱状图像素估读；第 6 表标题为 Fig 5 Apoptosis induction |

供主代理亲读的原文段落另外导出为 `Sarin_primary_evidence_sections.md`。该文件是原 XML 摘取，解释和边界以本 README 为准，不能把摘录本身当作跨研究同实验的证明。

### 2.1 两文相同的细胞来源和暴露协议

- 亲本为 ATCC A549；耐药亚系为 RCCL collection 的 **A549rCDDP2000**，经递增顺铂适应至 2,000 ng/mL。
- 常规培养为 IMDM，4 mM L-glutamine、10% 胎牛血清、100 I.E./mL penicillin、0.1 mg/mL streptomycin；耐药株维持培养另外含 **2 µg/mL cisplatin**。
- 实验细胞过夜贴壁，经过 **4 h 血清饥饿**，在**不含补充物的 IMDM** 中处理 **24 h**。
- 亲本接受 **11 µM cisplatin**；耐药株接受 **11 或 34 µM**。各自有无药平行对照。
- 2018 文明确每 10 代解冻新备份以保证重复性。不能据此假设 2017/2018 样本是同一瓶或相同代次。
- 从维持顺铂培养转入实验的额外预先洗脱时长没有单独给出。保留原文所写的 4 h 血清饥饿和无补充培养基处理，不自行补一个“长期无药洗脱”步骤。

2017 Annexin 方法使用 BD Pharmingen FITC Annexin V / PI 试剂盒、FACSCalibur。每个 6-well 孔接种亲本 2.5×10^5、耐药株 5×10^5 细胞，过夜后按上述处理。该接种密度差异是协议的一部分。方法写明 trypsinized cells 收集并离心；没有足够信息证明此前脱落到培养液中的全部细胞也被纳入，因此不添加“涵盖所有已死亡细胞”的主张。

### 2.2 真实测得的 RNA 时间点

2018 §4.2 及所有 GEO 样本 treatment protocol 均写处理持续 24 h，control cells 接受 drug-free medium。因此：

| 表达类别 | 实际角色 | 不能声称 |
|---|---|---|
| A549sens-ctrl、A549res-ctrl | 同状态平行 24 h 无药对照参考表达 | 同一 Annexin 实验给药前的实测 t=0 表达 |
| A549sens-11µM、A549res-11µM、A549res-34µM | 24 h 药后群体表达 | 初始状态；同一个细胞随后是否死亡 |

若将这套参考表达输入“初态条件化”模型，**同状态平行 24 h 对照是否代表挑战起点**是额外需要验证的近似，而不是数据已经保证的事实。血清饥饿／无补充培养的时间效应尤其不能被省略。

## 3. 已导出的可计算表

| 文件 | 内容 | 验证 |
|---|---|---|
| `GSE108214_probe_expression.csv.gz` | 29,729×22 的 GEO deposited VALUE，行是探针、列是 GSM | 每列 29,729 行、行序完全一致、全值可数值解析且有限 |
| `GSE108214_probe_annotation.csv.gz` | 与矩阵完全同序的 GPL17077 注释 | 29,729 个独一探针全部找到源注释 |
| `GSE108214_samples.csv` | 22 个 GSM 的状态、剂量、时长、对照角色 | 4/3/5/5/5 分组计数与源文件一致 |
| `GSE108214_protocol_metadata.json` | 每个 GSM 的原始关键描述和协议 | 保留源字符串，不以摘要替代来源 |
| `Sarin2017_Annexin_source_values.csv` | 10 行原始 Annexin 倍数及 DOCX 坐标 | 不含重复列入的均值、SEM 或 SubG1 数字 |
| `Sarin2017_Annexin_summary_check.csv` | 各组 n、重新计算的均值／SD／SEM、原表汇总 | 原表与复算并排保留，见下节微小差异 |
| `Sarin_condition_correspondence.csv` | 3 个条件与各自 control/treatment GSM 集合、Annexin n 的连接 | 不生成个体或重复级伪配对；`training_eligible=false` |
| `Sarin_validation.json` | 形状、来源计数、有限值与汇总差异记录 | 实际运行提取脚本生成 |
| `extract_sarin_bridge.py` | 仅解析本地来源并写派生表 | 可复算，不下载、不训练、不改源文件 |

来源平台是 Agilent GPL17077 bulk microarray，不是单细胞 RNA-seq。VALUE 的官方描述为 background correction + quantile normalization；本轮不额外变换或声称它是原始 counts。虽然数值形态类似对数表达，源字段没有明确写 log base，因此导出保留原尺度。

29,728 个探针有非空 GENE_SYMBOL，共 21,592 个不同的非空 symbol 字符串；这不是已经完成统一 ID 后的基因数。26,793 个探针有 LOCUSLINK_ID，29,718 个有 REFSEQ。官方 VALUE 说明“probes with REFSEQ IDs only (29,729)”，但所附平台注释实际存在少数空 REFSEQ，故本轮保留全部探针并明确注释缺失，未静默删掉。平台 `ENSEMBL_ID` 列含 **ENST 转录本 ID**，不能当作 ENSG 基因 ID。没有执行探针合并、旧 symbol 更新或对现有模型基因空间的无依据映射。

### 3.1 表达样本与凋亡条件对应

| 状态／处理 | 无药 24 h 参考表达 GSM | 药后表达 GSM | RNA 对照/药后 n | Annexin n |
|---|---|---|---:|---:|
| A549 + 11 µM cisplatin ×24 h | GSM2892607; GSM2892612; GSM2892615; GSM2892622 | GSM2892608; GSM2892613; GSM2892616 | 4 / 3 | 3 |
| A549rCDDP2000 + 11 µM ×24 h | GSM2892609; GSM2892614; GSM2892617; GSM2892623; GSM2892626 | GSM2892610; GSM2892618; GSM2892620; GSM2892624; GSM2892627 | 5 / 5 | 4 |
| A549rCDDP2000 + 34 µM ×24 h | 同上一行 5 个耐药株对照 | GSM2892611; GSM2892619; GSM2892621; GSM2892625; GSM2892628 | 5 / 5 | 3 |

耐药两个药物剂量共享同一组参考表达；不能把共享对照计算成两份独立数据。不同源的重复编号只代表各自文件中的顺序，未证明跨剂量甚至跨研究配对。

### 3.2 Annexin 实测值与分母

| 状态／剂量 | S8 原值 | 复算均值 | 原表均值（SEM） |
|---|---|---:|---:|
| A549, 11 µM | 6.922; 4.994; 5.649 | 5.855000 | 5.856 (0.566) |
| A549rCDDP2000, 11 µM | 2.141; 2.114; 1.027; 2.516 | 1.949500 | 1.950 (0.321) |
| A549rCDDP2000, 34 µM | 4.956; 2.811; 3.374 | 3.713667 | 3.714 (0.642) |

第一组原表均值与已印出的原值复算相差 **0.001**。保持原值，不擅自归因于笔误或隐藏精度，不“修复”源文件。三组 SEM 按样本 SD/√n 复算均与原表保留三位小数一致。

这些数字的单位全部是 **fold change relative to respective untreated controls**。概念上为每种背景自己的药后 Annexin 读数 / 自己的无药读数；S8 Fig5 部分未提供这些无药分母。由此：

1. 不能写成 6.922% 或 69.22% 的凋亡率；也不能用 1−viability 替换。
2. 不能仅凭亲本倍数大于耐药株倍数，就计算两种背景绝对凋亡百分比差；各自分母可能不同。
3. 可以忠实考察**固定协议下相对各自基线的 Annexin 诱导倍数**在状态之间及耐药剂量之间的差异。
4. FITC-Annexin/PI 检测是凋亡实验；当前已读方法、图注和 S8 未明确给出导出值是否为“早期+晚期之和”的门控定义。本轮 endpoint 名称保留原文，不自行重定义为 early、late 或 summed fraction。
5. SubG1 是另一检测终点，S8 中有对应值，但本轮没有混入 Annexin 标签或当作新增独立生物样本。

## 4. 能与不能支撑的研究论断

**可支撑的有限用途**：具有固定来源的两种群体状态；有各自平行无药参考表达和同协议药后表达；有同来源另一研究相同顺铂方案的真实 Annexin 诱导倍数。它可作为一个带来源/协议边界的状态对照案例，检验模型是否能对固定顺铂条件区分已知状态的相对诱导反应。

**尚不能支撑**：实测 t=0 单细胞表达对应未来凋亡；同一实验 RNA—功能联合监督；绝对凋亡概率回归；跨多药物的排序评估；真实多步序贯治疗策略或 RL 优势。只有一个药物、两个状态，无法通过随机划分 10 个源观测创造独立跨状态／跨药评估。

因此本轮数据状态应写为“**原始数据已读取；跨研究同来源同协议条件连接已建立；初态实测、同实验连接及多药验证仍未闭合**”，不得将 Sarin 单条链标为课题完整可行性已通过。

## 5. 其他已实际检查的第二轮线索

| 来源与本地文件 | 实际取得／核验 | 本轮用途或排除原因 |
|---|---|---|
| [GSE222187](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE222187), DOI [10.1007/s12032-022-01923-5](https://doi.org/10.1007/s12032-022-01923-5), `PMC9797463.xml`、`GSE222187_family.soft.gz`、`GSE222187_PROCESSED_DATA.txt.gz` | 同一论文的 12 个 A549/DDP 样本（CON/DDP/TRF/TRF+DDP，各 3），真实 TPM 文件已下载。论文有 Annexin/PI ×48 h；亲本 0.25 µg/mL、耐药株 2 µg/mL DDP | 论文旧的“未上传 GEO”不能再作为现在不可得的依据。但 GEO 全部是 A549/DDP，没有亲本 RNA；TRF 是与 DDP 同期的 6 h 正常/18 h 无糖无血清周期，不能把末端 TRF RNA 说成预处理后给药前状态。本轮尚未取到各生物重复的 Annexin 数表，未作为状态闭环通过 |
| *Non-canonical NOTCH1 signaling regulates ferroptosis vulnerability in dormant lung cancer cells with stable resistance*, DOI [10.1038/s41419-025-08355-9](https://doi.org/10.1038/s41419-025-08355-9), `PMC12780219.xml` | A549/A549CR、休眠耐药状态；RNA bulk 测序；Annexin 与 NucView caspase 等实测。数据声明 HRA007891、作者 [SRCC](https://github.com/wei-R45/SRCC) | GSA human 链接访问本轮失败，不能等同于数据不存在或必然受控。研究同时涉及铁死亡等多种死亡，需严格保留端点。未读取 RNA 数值，未完成对应，保留候选 |
| *Non-canonical dihydrolipoyl transacetylase promotes chemotherapy resistance via mitochondrial tetrahydrofolate signaling*, DOI [10.1038/s41467-025-63892-3](https://doi.org/10.1038/s41467-025-63892-3), `PMC12508156.xml` | A549cisR/DLAT 操作、cisplatin 2 µg/mL ×48 h，Annexin/PI；数据声明含 PXD064757 蛋白组 | 文中 RNA-seq 搜索对应 TCGA 分析；本轮未找到本实验 A549cisR 初始转录组。不能把患者 TCGA 或蛋白组当作所缺 A549 同状态 RNA |
| Z-ligustilide, DOI [10.3390/ijms242317046](https://doi.org/10.3390/ijms242317046), `PMC10706864.xml` | A549/A549DDP 与 RNA-seq、Annexin/7-AAD 24 h 功能线索 | 原文数据声明为 upon request，本轮未取得表达矩阵；未联系作者。不可标为公共数据已读取 |
| [GSE43493](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE43493), `GSE43493_family.soft.gz` | A549/顺铂耐药派生，敏感 3、耐药 3，lncRNA/mRNA profiling | 可读取源表达但尚未找到同一来源的可对应 Annexin 源表，不能和任意另文 A549/DDP 功能混配 |
| [GSE110495](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110495), `GSE110495_family.soft.gz` | 真实标题为 *Therapeutic targeting of the pre-metastatic stage in human brain metastasis*，PMID 29986997 | 不是 A549 亲本/耐药状态比较，排除该用途 |
| [GSE161116](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161116), `GSE161116_family.soft.gz` | NSCLC 原发及脑转移患者组织、免疫基因面板，PMID 33420630 | 不是 A549 同源派生株，排除该用途 |
| [GSE73158](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73158), `GSE73158_family.soft.gz` | A549 的 siATG5/ULK1/RELB 等操作，21 个样本、7 条件 | 不是顺铂耐药亲本/派生株；不能依据后续文献的“resistance”标签改写原始元数据 |

以上“未完成”均指本轮实际覆盖，不能解释为不存在可用研究。上一轮 cFLIP、RNF113A、Triana、EphB1 的限制不因本轮新增线索被撤销，上一轮文件未改。

追加的 YAP/TEAD–paclitaxel、Nobiletin–A549/ADR、early cisplatin persister 三条指定线索已分别查到原文，详见 `Followup_leads_review.md`（含版本、页码／章节、真实端点、数据获取缺口和追加查询日志）。没有把这些未取得完整数值的实验描述计入 Sarin 已导出数据。

## 6. 复算与下一步针对性缺口

执行（普通 CPU 解析，无训练）：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round2/extract_sarin_bridge.py'
```

本轮该命令实际完成；从官方 SOFT 读取全部样本、原探针精度导出、注释行校验及 S8 原值坐标提取通过。原表第一组均值 0.001 的差异是源表核验发现，不是提取程序失败，已记录在 validation JSON。

下一步仍针对**可实测的初始状态及足够多的干预**补齐，而不是把 RNA 末端对照悄悄替代为初态：优先同一研究的预处理后、再挑战前 RNA，以及该群体在多个小分子处理后可读取的 Annexin/Caspase 终点源值。任何可接受的群体参考近似须作为待验证假设写入模型和评测设计。
