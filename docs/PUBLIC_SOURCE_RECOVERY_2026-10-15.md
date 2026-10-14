# 公开来源恢复、实际数据连接与全架构验收（2026-10-15）

## 1. 本轮结论与版本

**取得了可下载的真实表达、缺失掩码、样本/条件连接和实际诊断结果；完整科研可行性仍未通过。** 本轮不是只生成接口：在云端读取 TIDYI CSV/H5AD、Tahoe 词表、HM97662 原始估计计数、Sarin GEO 全部微阵列数值及原始 Annexin 补充表，完成新公共基因轴上的投影，并独立逐值复核。没有将这些来源合并成虚构的同细胞、同培养孔或逐重复 RNA—功能配对。

初始交接为 `4a78560bdeb2e0d98332d5a9520764a120abf612`。本轮开始时 main 已为 `3218e7ce65ba7184d0ea506b46a8ecd6e4ebc5c4`；在其上建立 `codex/source-contracts-20261015`，保留已经合并的公开表达恢复实现。PR #1 的待合并分支及其成果未被覆盖，也未作为本轮新增代码成果。本轮没有修改冻结划分、奖励或本机历史数据，没有恢复正式模型训练。主架构原文件仍未访问，以用户给出的六段研究链和仓库规则为约束。

### 证据层级

| 层级 | 本轮对应证据 | 不能替代什么 |
|---|---|---|
| 文档记载 | Round22 的 Bothe 4/12、历史两板 TIDYI 743 细胞、Martin 双域、原 native/final 范围 | 本轮重新读取原始文件或重新运行历史模型 |
| 源码核实 | 原恢复/分类实现、sci-Plex `_source_rows` 至联合训练数据消费路径；本轮实际改动 | 真实数据数值、模型效果 |
| 云端实际执行 | 下面的矩阵恢复、H5AD/CSV 交叉读回、原 Annexin 表、9 个名义化学图、CPU 诊断、修改前后完整测试 | 实验材料鉴定、跨来源生物可比性、真实多步治疗效果 |
| 尚待原资产/新实验验证 | 原两板数据与冻结角色、Bothe 全量生产、共享表示、P→G→策略同链、独立功能及序贯验证 | 不标记为已闭合 |

环境实测为 Python 3.13.5、NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、scikit-learn 1.8.0、pytest 9.0.2、PyTorch 2.10.0+cpu、RDKit 2025.09.4、h5py 3.15.1。补装了官方离线 PyArrow 22 和 TensorBoard 2.20 等依赖。没有 CUDA；不把当前运行成本外推为 RTX 3080 的正式训练成本。终端直接访问 GitHub 的 DNS 路径失败，使用连接器及本人编写的只读 GitHub Actions 获取固定源码和公开文件，随后在当前云端执行；不是交给模型代理代做。

## 2. 已实际完成的来源级数据连接

### 2.1 TIDYI/Tahoe：公共身份轴、原始计数与原分选标签

固定 Tahoe 快照 `2dc57900b7981cfcf5e211527169a0b006546a95` 的 62,710 个基因，与 TIDYI 导出表的 32,858 个基因按 Ensembl 身份求交并排序，得到 **32,461 个共同基因**。没有使用留出表达或标签挑选这个身份轴，也没有拟合跨来源归一化。

TIDYI CSV 为 369 行：368 行明确 A549，1 行细胞背景缺失。缺失身份行 `ATAGGAGGCGCACAGATGGA`（E20）保存在排除表，不补成 A549。严格入组数据为 **368 × 32,461**，含 **11,945,648 个计数值**，其中 **9,656,961 个为实测零**；这些共同基因的 mask 都为 true。保留原细胞 ID、板、孔和原始 `PI-APC- / PI-APC+ / PI+APC+` 标签，不把它们重新命名为已核验的其他染色门。

进一步实际下载并完整读取官方 `SS3_Human_A549_adata.h5ad`（200,429,210 字节）。其中 `X` 和 `layers/counts` 都是 369 × 32,858，两者与 CSV **全部 12,124,602 个计数逐值相同**，细胞与基因身份及顺序也相同。该 H5AD 仍只有板 8，没有恢复板 1。

源 metadata 的总计数/检出基因数与导出矩阵重算值存在原有差异；H5AD 也不能消除此差异。本轮保留 source QC 和 export QC 两组字段，用固定源文件哈希及逐细胞差异签名限定接受范围，**不是增加任意容差、修改原 QC 或把差异静默忽略**。此现象此前仓库已经记录，本轮是重新验证，不冒充首次发现。

产物 `reference-v1/` 包含 `genes.parquet`、`tidyi_cells.parquet`、`identity_exclusions.csv`、`tidyi_common_counts.npz`、`manifest.json`。基因轴换行文本 SHA256 为：

`6e495561730981f64aca5bafe1b5ab17728c8de0ebde3efd2c9ca838ac727ec5`

**它是新建的、来源绑定的公共坐标，不是已认证的历史冻结训练资产。** 新 Parquet SHA 为 `f16276b49d3f06c982446e5405f24b3b3c73d713f2f4680e19edfc2cacaa8d16`，不能用“同为 32,461 列”替代与历史资产的逐 ID、顺序、数据角色及预处理核对。

仓库 TIDYI README 已说明第二板来自作者另行发布的 `classifier_data.tar.gz` 中的 `human_ss3.h5ad`，而非此公开 A549 导出文件。历史压缩包约 2.94 GB，本轮没有重新下载；743 细胞原数据/冻结角色未在云端恢复。不能把“本轮仅读到板 8”扩大为“作者没有公开第二板”。

### 2.2 HM97662：全部公开源计数、细胞背景与共同轴

实际读取 GSE333828 完整计数表及 SOFT，恢复 **57,905 基因 × 18 样本，共 1,042,290 个源值**。18 个样本为 A549、H841、SBC5 各 6 个；A549 不是全部 18 个。每个背景是 3 个未处理、3 个 HM97662 处理样本，GSM 与表列名、背景、重复标签及作者协议逐项核对。

原数值是 **Salmon estimated counts**，其中 **129,069 个为非整数**，不四舍五入为 raw integer counts。原协议为 1 μM、6 天，标准时间 144 h；未处理剂量 0 是条件编码，不表示已测溶剂摩尔浓度。

投影得到每个样本 **30,268 个观测共同基因，2,193 个未观测基因**；后者保留 NaN 与 false mask。完整非 A549 数据仍保留明确背景，不当作 A549 增样。原始估计计数、样本顺序、映射和每个投影值均已读回。

产物 `hm97662-v2/` 包含原生/共同轴 NPZ、`samples.csv`、`source_metadata.json`、`gene_mapping.parquet` 和 manifest。角色为 `unassigned_external_source`。本轮没有重新读出 HM 的原始 Annexin 重复，也没有获取经认证的 HM97662 化学结构；历史近似读图均值不能被 RNA 三重复扩成三份功能监督。因此 **表达恢复已完成，完整 RNA—凋亡—结构条件模型链尚未闭合**。

### 2.3 Sarin：完整 RNA、原始 Annexin 倍数及合法条件级连接

实际读取 GSE108214 的 **29,729 探针 × 22 RNA 样本，共 654,038 个官方 VALUE**。源值最小值为 -0.816772203，保留负值；单位是作者背景校正、分位数归一化的 microarray VALUE，**不能当作 UMI/counts，也不能在 log base 未确认时再强行 log1p**。

用实际读取的 HGNC 与 GRCh37.75 transcript 注释核对探针—基因，保留完整候选、冲突和排除台账；不依靠 symbol 猜配，不把多个探针任意平均。共同轴上 **12,070 个基因可观测，20,391 个为 NaN/mask=false**。原探针值全部另存，不因共同轴过滤而丢失原始证据。

原始 PLOS ONE 2017 S8 DOCX 的 Fig.5 表提供 **10 个实测 Annexin 诱导倍数**，不是百分比：

| 背景、处理条件 | RNA 处理样本 | 平行 24h 未处理 RNA | 原始 Y 数 | Annexin fold 均值 | 原数值 SEM |
|---|---:|---:|---:|---:|---:|
| 亲本 A549，顺铂 11 μM、24 h | 3 | 4 | 3 | 5.855000 | 0.566016 |
| A549rCDDP2000，顺铂 11 μM、24 h | 5 | 5 | 4 | 1.949500 | 0.320892 |
| A549rCDDP2000，顺铂 34 μM、24 h | 5 | 共享上述 5 | 3 | 3.713667 | 0.642076 |

SEM 仅是原始表数值的描述性复算，**不是表达模型效果的置信区间，也不证明跨实验重复独立性**。本轮另用独立 CSV/XML 读取实现，把所有 RNA 源值、10 个 Y 的原表单元格以及 3 个连接重算；RNA 数值与原生/共同轴输出一致，没有创建任何 RNA—功能逐重复配对。

合法连接是：同一来源明确的背景与名义处理条件之间，2018 RNA 与 2017 Annexin 的 **cross-study condition reference**。没有共同培养孔、逐重复对应、药前同细胞 RNA。Annexin 实验亲本/耐药的铺板数量还不同（250,000/500,000 cells/well），保留为混杂。两个剂量的耐药条件共享对照，不当作独立对照实验。

这确实提供了可计算的有限功能挑战：共同 11 μM 下两背景的响应差异、耐药背景内 11/34 μM 的两候选比较。它不能识别高维 RNA 相对 state-ID 的普遍增量；只有一个药物、一个时长，并且缺少亲本 34 μM，不构成完整多状态×多药×剂量×时间网格。

### 2.4 sci-Plex 原始剂量和名义化学图

实际读取原始 hash-well metadata：4,992 个计划记录中 A549 有 **1,920 孔，包括 1,880 处理孔和 40 vehicle 孔**，188 个活性目录身份。原始剂量为 10/100/1,000/10,000 nM，转换为 0.01/0.1/1/10 μM；观测时间 24/72 h，重复标签 rep1/rep2。

在生产路径 `sciplex_source._source_rows → extend_with_sciplex → TrainingData`，既有代码输出标准 `dose_um`，却丢掉可取得的原始剂量与单位。本轮最小修复保留 `source_dose_value/source_dose_unit/dose_semantics`；vehicle 的零标为来源占位，不称溶剂浓度。真实元数据转换已复算，合成集成测试覆盖实际准备/读取路径。**计划孔不等于 QC 通过的 1,871 个历史处理条件；未重新下载全部表达分片或改写冻结联合数据。**

实际读取 9 份 PubChem JSON，并以 RDKit 重建图、计算 InChIKey 和 2,048 位 Morgan 指纹：顺铂、地塞米松、藤黄酸、I-BET726、Lestaurtinib、Midostaurin、奥沙利铂、Resveratrol、Trametinib，9/9 InChIKey 与原响应一致。顺铂/奥沙利铂的多片段表达全部保留，没有只留最大片段。此处是**名义化学身份与特征的独立核查**，不是认证每篇实验的盐型、溶剂、供应商批次、实际配制剂量，也没有把新指纹替换冻结化学资产。HM97662 缺图仍未解决。

## 3. 已实际运行的 TIDYI 状态诊断

使用仓库原有 `prepare_tidyi_a549.py` 和 `tidyi_a549_baseline.py` 的固定流程，只把输入和输出重定向到隔离的公共数据目录；未改分类模型参数、种子或原始标签。单板 368 个 A549，外层四折按孔列分组，内部三折校准；每折只在其训练部分做表达特征选择和标准化。源内 log1p(library-size 10,000)、500 个训练方差基因、LR C=0.1，seed=20260918。完整折身份、逐细胞 out-of-fold 概率、校准表已保存。

| 固定方法 | 平衡准确率 | Macro F1 | Log loss↓ |
|---|---:|---:|---:|
| 类别先验 | 0.333333 | 0.208100 | 1.066268 |
| 表达，未校准 | 0.743712 | 0.741545 | 0.622161 |
| 表达，sigmoid 内层校准 | 0.709772 | 0.701771 | 0.562452 |
| 仅孔位诊断 | **0.878946** | **0.856642** | 1.919872 |
| 仅 QC 诊断 | 0.544262 | 0.496713 | 0.796499 |

实际 CPU 运行约 11.25 s，仅为该小型诊断测量。它是**复跑已有固定探索方案**，不是新盲测或原冻结两板 C 的完整评价。孔位在平衡准确率上高于表达，且 log loss 较差，说明类别判别与概率可靠性需分别考察；不能只挑一个好指标。

该结果与已知分选布局相容，提示标签可由技术/布局信息预测；不能据此证明表达没有生物信息，也不能证明更强方法必然失败。它直接反对把“来源内分选分类分数”未经独立验证就当作“给药后自然群体实测凋亡率”或 RL 奖励。历史跨板诊断的不同数字不在本轮重新验证范围内。

## 4. 整个研究架构：现在闭合到哪里

| 主问题/数据连接 | 本轮真正闭合的部分 | 仍缺的监督、合法比较或评价 | 当前结论边界 |
|---|---|---|---|
| 状态表示及测量时状态分类 | 公共 TIDYI 精确身份/计数/标签；来源内固定诊断 | 原两板资产、跨平台验证、原染色门语义和技术混杂控制 | 可以研究源内原标签；不能推出自然凋亡率 |
| 状态条件化响应 P | 新公共轴；HM/Sarin 真实表达、背景、平行对照、mask；sci-Plex 原剂量 | 全部 joint 表达与冻结角色、新域可用 E/D/P、独立状态增量实验 | 源值能对应，不等于已训练共享空间或预测可靠 |
| 未见药物/剂量/时间/完整协议 | 真实元数据明确各轴的存在与未知 | 各自独立新版本范围、fit-only 预处理及真实留出效果；原跨药不能替代其他轴 | 不授纯剂量/纯时间因果作用或完整协议外推资格 |
| 表达→独立实测凋亡 G | Sarin 10 Y 与 3 条合法条件级连接；HM RNA 完整 | 足够状态/药物/条件覆盖、独立分头 G、材料/完整协议、校准/final；HM 图与真实 Y 仍缺 | 有限定功能挑战，完整高维表达—凋亡增量未闭合 |
| 药物/剂量/时长联合选择 | 药物名义图审计、实际剂量元数据；耐药 Sarin 两剂量挑战 | 多药/时间完整候选域及实测 Y，直接排序基线、选择偏差控制、独立 regret | 不能以两个候选或内部 G 奖励验证联合选择/RL |
| 多步滚动与真实序贯 | 定位 Bothe 历史与 SequenTx/早期 A549 顺序效应来源；保留全部目标 | Bothe 全 12 个生产计数/历史，真实步骤转移与独立序贯终点；目前未新恢复 A549 序贯数值 | 模型 rollout 与真实顺序药效分别未验收，不移到“展望”后宣称完成 |

字段级必须追踪：基因 ID/版本/顺序、原始测量单位与归一化、观测 mask、药物原名/结构/材料、细胞背景、剂量原值/原单位/标准值、暴露/恢复/测量时间、有序事件、对照/vehicle、板/批次/重复、观测类型/分母/单位、训练与评价角色、源 URL/版本/文件 SHA。未知字段不填成 0；相同 unknown 也不证明两个协议相同。

## 5. 可识别性与方法取舍

### 5.1 明确随机变量、单位与目标

设 W 为背景、来源、培养/实验块及已知协议；动作 a=(药物、剂量、暴露/恢复事件)；R 为合法参考群体表达样本，X_t 为观测时刻的表达，Y_k 为 assay k 的功能终点。破坏性未配对快照首先支持估计 P(X_t | R,a,W) 的群体分布或矩，而不是一条被实测的单细胞轨迹。预测单位可以是细胞分布、条件均值或实验功能终点；统计独立单位应按实验设计选孔、培养重复或研究块，不等于细胞数。

最简单的不可识别反例：X_0、X_t 各以 1/2 概率为 0 或 1，映射 X_t=X_0 和 X_t=1-X_0 都给相同边际。只观察未配对边际不能区分两个 coupling。OT、flow matching 或 cycle consistency 是加入建模假设，不能把选中的 coupling 重新称为实测配对。

但**不能反过来要求所有实验必须同细胞药前药后配对**：独立随机培养群体可以检验条件分布预测；平行对照可以定义群体响应；独立无 RNA 的功能实验可以检验候选排序外推。结论限于相应层级。当前同期 vehicle 不是 t0 RNA；用它作输入时，应把任务命名为“给定同期参考的条件预测”。

### 5.2 来源内表示、共享表示和条件分布映射

本轮推荐先建立同信息条件下的**来源特异基线**，同时保留共享表示与条件分布映射为待比较路线，而不是擅自替换已确认研究主线。

| 路线 | 所需假设/数据 | 区分它的实验 | 失败时处理 |
|---|---|---|---|
| source-specific E_s/D_s/P_s，终点各自分头 | 源内单位、背景、协议、mask 一致；只迁移已定义的效应量 | 源内留出、无迁移基线、合法共同条件的外部排序 | 保留源内结论，不合并数值 |
| 共享潜在表示+来源观测模型 | 有生物条件重叠或可验证迁移锚点，技术校正不抹去真效应 | 相同角色下跨源条件检索/响应迁移、来源探针与生物信号保留联合检查 | 仅重建好不授共享资格；必要时退回来源特异 |
| 条件群体映射/分布预测 | 可比较的独立参考与处理群体，明确 coupling 假设 | 均值误差、delta 指标、分布误差与简单条件均值/线性基线 | 分布好不推导个体命运；ODE 内部积分时间不当真实给药时钟 |

独立自编码器存在不可识别性：对任意可逆 T_s，用 T_s E_s 和 D_s T_s^{-1} 不改源内重建，却可任意改变各源潜空间相对坐标。因此来源内重建成功不足以证明共享表示。source probe 也不是单独的成败判据：真实生物背景与来源混杂时，不应仅以“来源不可预测”强制去除信息。

### 5.3 功能终点、选择与 RL

Annexin fraction、相对未处理 fold、caspase 活性/时间积分、一般死亡和生长/存活分别定义 Y_k，保留分母、标度、观察时钟及对照。TIDYI 富集分选类频率不等于自然培养群体的 Annexin 率。单个细胞损伤表达与群体累计死亡率也不是同一随机变量。

单次有限候选的必备基线是 a_hat=argmax_a G_hat(R,a,W)。同一 G 的完整枚举已求得其最大评分，另拟合 Q 不能用“自己的 G 评分更高”证明更好。若对所有候选都有 |G_hat(a)-E[Y(a)]|≤ε，则真实候选 regret 至多 2ε：在真实最优 a* 和所选 a_hat 各插入 G_hat，并用 G_hat(a_hat)≥G_hat(a*)。**当前数据没有证明这样的全候选误差界，不能把推导当 OOD 保障。**

RL/规划应检验真正的额外价值：历史改变后续状态/可用动作、多步反馈、长期目标或相同预算下的搜索效率。必须给出观测 R/W/历史、动作协议、真实/代理奖励来源、转移监督、终止规则及独立真实 Y 的评价。用户既定目标不含新加成本惩罚，本轮没有引入任何药物/剂量/时长/步数/风险惩罚。单步选择、多步模型误差与真实顺序效应分别验收；不能用 G 或模拟轨迹验证自身。

## 6. 定向文献查证与尚待检验的创新

检索入口包含 web/PubMed/官方期刊、Consensus 的关键词检索与记录读取、官方 GEO/BioStudies/PubChem 和作者 GitHub。它们不是同一关键词搜索结果的简单复述。可读取的原 XML/补充及实际源文件用于判断；不可访问的主文或矩阵保持未核验。以下并非穷尽式系统综述，也不以“未检索到”证明不存在。

| 最近/最接近来源 | 已核查的具体部分与可用事实 | 对本项目的限制或实验启示 |
|---|---|---|
| SequenTx，Nature Machine Intelligence 2026，DOI 10.1038/s42256-026-01192-1 | 官方摘要、Data/Code availability、Extended Data 图注、作者 repo 和公开补充表。L1000 转移与 CTRPv2 viability 奖励；Extended Data Fig.6f 有 A549 Entinostat→Daunorubicin 生长曲线 | 不是“没有 A549 顺序研究”。但生长/viability 不是本项目 Annexin；预测 reward 表也不是实验 Y。订阅主文全文未取得，不宣称逐页读完。其公开 RNA 机制 accession 不自动给 A549 RNA—Annexin 完整链 |
| Sarin 2017/2018，PLOS ONE / IJMS，DOI 10.1371/journal.pone.0181081；10.3390/ijms19030767 | 原 XML、S8 表、GSE108214 所有 RNA/metadata 已实际读 | 提供 2 背景、1 药、3 条件有限挑战，允许条件级连接；不能分离高维 RNA 与 state-ID 的一般增量 |
| HM97662 2026，GSE333828 | 完整公开 RNA 实读；保留 1 μM/6 天、3 细胞系 | 可扩表达输入，但结构、功能重复和多动作覆盖尚缺，不能补成完整 G 监督 |
| 单细胞扰动泛化 benchmark，Nature Methods，DOI 10.1038/s41592-025-02980-0 | 官方摘要/扩展图注及数据/代码链接核查；主文全文未取得 | 提醒分开细胞背景、扰动、剂量/时间和指标，提供比较路线；不直接搬用其数值或把遗传扰动基线结论概括到本课题 |
| GSK6853/BRPF1，2025，DOI 10.1007/s10637-025-01585-w | 可读全文包含 RNA-seq、Annexin；Data availability 却写未生成/分析数据 | 本轮未得到可下载表达矩阵 accession。记录正文与可用性声明不一致，不猜测数据或作者动机，不把摘要有 RNA 当可训练数据 |
| Mortenson 2004，DOI 10.1007/s00280-004-0811-4，PMID15197486 | 原 PubMed 摘要：A549，bortezomib 与 gemcitabine/carboplatin 先后顺序改变凋亡；FACS/DAPI/caspase 测量 | 保留为真正 A549 序贯凋亡外部验证线索，而非否定该领域有证据。本轮未取得完整剂量/时钟/数值表，不伪造指标或纳入训练 |
| CellCage，DOI 10.64898/2026.05.05.723030 | 本轮读仓库完整来源报告；其原文/行级矩阵未重新恢复 | 历史基线是影像、终末 RNA 可为多后代 CCE，不能当药前单细胞 RNA；矩阵与 CCE 映射缺口仍在 |

尚待验证的创新应是：在明确参考输入权限和板/来源/历史混杂边界后，**状态信息是否在独立响应及功能筛选中产生可复现增量**；以及真实独立序贯数据中是否优于同信息直接预测/枚举。增加 E/P/G/Q 模块、套用 RL、共同基因数量或没有同名论文，都不构成创新证据。

## 7. 预先明确的实验设计（未擅自创建新冻结划分）

1. **状态增量**：同一目标和合法参考权限，独立训练 context/drug-dose-time、state-ID、参考群体均值、高维参考、合法组内 shuffled-reference；按培养块/板与化学组件分组评价。置换须在设计允许的交换单元内，不能伪造原配对。若没有足够独立状态变化，明确该因子不可分离，而不是以“shuffle 不显著”证明状态无用。主指标为条件均值/delta 误差和来源适用的分布指标；报告低回收/死亡细胞 QC 选择偏差。只在随机 cell split 有增益，不能通过独立生物状态假设。
2. **独立轴泛化**：原 cross-drug final 不动；剂量、时间、完整协议另建版本与 fit/selection/final 范围，所有特征选择、归一化、AE/P/校准均限定新训练域。Round22 已指出 Tahoe 中剂量与板域、sci-Plex 时间与板域同时变化，主口径应明确联合域挑战与同板敏感性，不能偷偷换名称。现有 D1/T1 草稿未自动批准或执行。
3. **跨来源表示**：比较源内模型、共享表示、群体映射，在同一信息集/目标上评估；不能用尚未对应的 dose/time 强凑 matched condition。MSE/MAE 必须写单位，MMD 带宽与基因选择只用训练域。来源探针、跨源生物条件检索和独立响应同时报告，检查潜空间是不是仅编码来源。
4. **独立功能**：各 assay 的 G/校准/final 独立登记。共享同 Y 的不同 RNA 或代理视图只算一个结局；重采样以真实实验块/研究为单位。Sarin 可做预先固定的 3 条件一致性/两剂量排序，不做高维模型的有效独立交叉验证或大量调参。无 RNA 的外部药物—Y 可验证结构/条件候选排序迁移，但不能证明 RNA 中介或单细胞命运。失败判据包括排序不优于药物/条件基线、校准失效和收益仅来自 assay/source。
5. **联合选择与序贯**：候选域必须保留原真实 drug/dose/time/protocol 和缺测项。与药物均值、简单条件预测、完整枚举/搜索比较，报告 Spearman/top-k/NDCG/真实 regret 等预先选定指标；“oracle”只在已测候选集合内定义。序贯分别需要 A→B、合适单药/同时/反序对照和对应时钟、洗脱、参考状态与独立功能读数；未测中间 RNA 不自动否定终点级策略验证，但限制转移解释和适应性决策。没有真实序贯评价时不得以 rollout 误差低宣布药效成立。

以上是新实验设计，不是已运行成绩。正式统计显著性/置信区间须有足够独立实验块；不能由单板细胞数、重复使用同对照或模型随机种子膨胀样本量。

## 8. 实现改动与实际测试

| 文件/调用路径 | 问题和行为变化 | 真实/测试验证 |
|---|---|---|
| `scripts/recover_public_reference.py` + `configs/public_reference_sources_v1.json` | 无本机冻结资产时重建来源绑定身份轴；严格原身份/计数/差异签名，保留排除与 QC；新输出不可覆盖 | 实际完整 TIDYI/Tahoe 导出读回；输出进入下一行恢复 CLI |
| `scripts/recover_public_expression.py` | 增加 `--gene-axis-manifest`，核实轴 SHA/长度/基因数并记录来源；不认证冻结训练身份；预先拒绝错轴 | 实际运行 HM/Sarin 恢复与全值复核；原生产逻辑及前人恢复代码不冒充本轮新增 |
| `src/lung/data/sciplex_source.py` | 原剂量 nM 和 vehicle 语义不再从联合数据入口丢失；标准值和冻结角色不改 | 实际来源 1,920 A549 计划孔复算；集成测试到 TrainingData |
| 三个相关 test 文件 | 新增 17 个通过的用例，覆盖非法 ID/顺序/计数、精确 QC 合同、不可覆盖、轴来源证明与实际剂量消费 | 先有失败记录再实现；未删除旧断言、弱化标准或静默跳过失败 |
| 三个 branch-scoped Actions | 固定 Git bundle、缺失官方 CPU 依赖、公开 H5AD 获取 | 三项实际成功；无模型训练、无 main 写权限，只归档公开源 |

定向命令：

```bash
python -m pytest tests/test_sciplex_response_integration.py tests/test_sciplex_extraction.py tests/test_external_gene_axis.py tests/test_public_expression_recovery.py tests/test_public_reference_recovery.py -q
```

结果：**44 passed**。随后实际运行 `python -m pytest -q`，并在独立未修改基线 worktree、相同环境下完整复跑。CPU BLAS/OMP 线程均设为 1。

| 运行 | passed | failed | errors | skipped | subtests passed | 实测时长 |
|---|---:|---:|---:|---:|---:|---:|
| main 基线 3218e7c | 1,577 | 107 | 232 | 45 | 39 | 142.73 s |
| 本轮修改工作树 | 1,594 | 107 | 232 | 45 | 39 | 145.22 s |

两次完整日志及 `failure_inventory.csv` 逐测试项保留。339 个失败/错误的 node ID 和原因，在仅归一化仓库路径后完全一致；本轮没有新增失败项，新增 17 项全通过。许多失败是未上传的历史实验资产，另有既有平台/生命周期/缓存断言问题；不能把全部失败都归因于缺文件。**全套测试仍然失败，不宣称仓库测试全绿或完整链可运行。**

精确测试日志、环境、红—绿记录、输入/输出哈希及独立读回放在下载证据包，未作为数据本体提交 Git。文档仅记录结果，不覆盖历史结果文件。

## 9. 可执行接续与验收

### 重建本轮三个数据出口

在项目根目录运行，替换 `SOURCES` 与 `RESULTS`；`RESULTS` 下对应目录必须不存在。来源 URLs、完整 SHA/字节数见配置和随数据包的 manifests。无需本机历史 experiments、权重或训练运行。

```bash
python scripts/recover_public_reference.py \
 --gene-table "$SOURCES/tahoe/gene_metadata.parquet" \
 --counts "$SOURCES/tidyi/SS3_Human_A549_counts.csv" \
 --metadata "$SOURCES/tidyi/SS3_Human_A549_metadata.csv" \
 --output "$RESULTS/reference-v1"

python scripts/recover_public_expression.py hm97662 \
 --soft "$SOURCES/GSE333828/GSE333828_family.soft.gz" \
 --matrix "$SOURCES/GSE333828/GSE333828_raw_counts.tsv.gz" \
 --gene-axis "$RESULTS/reference-v1/genes.parquet" \
 --gene-axis-manifest "$RESULTS/reference-v1/manifest.json" \
 --output "$RESULTS/hm97662-v2"

python scripts/recover_public_expression.py sarin \
 --soft "$SOURCES/GSE108214/GSE108214_family.soft.gz" \
 --annexin-docx "$SOURCES/sarin/pone.0181081.s008.docx" \
 --hgnc "$SOURCES/annotation/hgnc_complete_set.txt" \
 --gtf "$SOURCES/annotation/Homo_sapiens.GRCh37.75.gtf.gz" \
 --gene-axis "$RESULTS/reference-v1/genes.parquet" \
 --gene-axis-manifest "$RESULTS/reference-v1/manifest.json" \
 --output "$RESULTS/sarin-v2"
```

本轮参考基因词表与两个 RNA 源是完整对应的公开导出，不是随机抽样几行的先导；但来源覆盖仍有限。`source_expression.npz` 可直接检查原生表达/ID/尺度，`common_axis_expression.npz` 提供显式 mask。它们**不是现有 AE/P/G/Q 冻结训练 manifest 的替代品**，未经同链域/角色审查不能直接接正式训练。

### 下一阶段最优先的五项及决策后果

1. **恢复原冻结公共输入，而不是重复找同一板。** 输入为已有作者 `classifier_data.tar.gz`/A549 NPZ、原 genes/roles/normalization/source manifests；先只读哈希和 ID/顺序/角色比较，输出冻结资产一致性报告。若与本轮公共轴一致，可在新明确命名空间复用身份；若不一致，保留双版本并修正原因，不覆盖原资产。继续按原跨板方案检验 C，但不得把 C 当 Y。
2. **完成 Bothe 12/12 的生产原件验收。** 输入为 12 组 FASTQ 官方完整性回执、原计数 leaf/执行 receipts、注释/基因桥和受控生产配置。逐样本核完整配对、真实整数计数、17,107 共同可测基因与缺失、对照/历史，输出全 12 样本生产 manifest 和 mask/读回。原模型停止有效；仅原允许的数据阶段若需恢复执行，必须按用户当前授权范围，不凭旧 running 字段启动本机任务。缺一个 leaf 就不通过全量验收。
3. **在原 joint 与新 source-bound 数据上先检验表示/状态增量。** 需要角色/参考池、完整表达及 source E/D/P 输入；先做无迁移/简单线性或群体均值，再比较共享与状态分支，输出同目标真实独立误差及 nuisance 对照。增益若消失，保留原问题为被反证/证据不足，不用更多模块掩盖。
4. **扩大独立功能覆盖，逐 source 验证而非跨库强配。** 在 Sarin 限定连接之外，核既有 Annexin r5、PR1 的外部来源与材料/剂量/时钟；原 Y 各算一次，输出 assay 分头的独立角色及 G 对照结果。HM 图无法取得则保留其 RNA 用途，不伪造指纹；只有少数条件时不用它训练高维 G。外部排序若无增益，表达预测成绩不能替代促凋亡筛选成绩。
5. **推进真正 A549 序贯数值恢复与独立评价。** 优先从已定位 SequenTx A549 生长实验和 Mortenson 原论文核完整数据/协议：前者另设非凋亡验证域，后者若取得数值可提供凋亡顺序挑战；并结合 Bothe 历史检验转移。需要实际原始或可靠发表数值、定义清楚的同一候选/对照和时钟。只有模拟或非凋亡 Y 时不通过主目标；若得到真实顺序域，再决定 RL 相对枚举/搜索的实际收益。

数据恢复与科学效果是两级验收：任何来源能读取、投影、通过局部测试或改善一个指标，都不授予 `whole_project_qualified=true`。本轮所有相关输出仍为 false。

## 10. 可定位来源

- TIDYI 官方 study/files：[E-MTAB-16354](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-16354)；本轮精确 CSV 与 H5AD URLs 及 SHA 见配置/证据包。另一作者训练对象定位见本仓库 `08-datasets/TIDYI/E-MTAB-16354/README.md` 的“作者训练对象”部分，历史获取/结果不等于本轮复跑。
- Tahoe 固定词表：[gene_metadata.parquet](https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/2dc57900b7981cfcf5e211527169a0b006546a95/metadata/gene_metadata.parquet)。
- HM RNA：[GSE333828](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE333828)，各 GSM 的 treatment protocol、sample title 与原计数表。
- Sarin RNA：[GSE108214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108214)；[2018 原文](https://doi.org/10.3390/ijms19030767) §4.2–4.3；[2017 原文](https://doi.org/10.1371/journal.pone.0181081) Fig.5、相关 Methods 与 [S8 原表](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0181081.s008&type=supplementary)。
- sci-Plex 原始表：`A549_MCF7_K562_hashTable_metadata.txt.gz`，源 URL/检索时间/哈希见公开获取 receipt；完整元数据 4,992 行，勿与 sci-Plex2 的相对 EC50 剂量混淆。
- SequenTx：[主文入口](https://doi.org/10.1038/s42256-026-01192-1)，Data availability、Extended Data Fig.1/3/6、Supplementary Table10；[作者代码](https://github.com/DELTA-TJ/SequenTx)。
- 泛化 benchmark：[Nature Methods](https://doi.org/10.1038/s41592-025-02980-0)，Data availability 与 Extended Data Fig.3；未将付费全文标为已读。
- GSK6853：[原文](https://doi.org/10.1007/s10637-025-01585-w)，RNA-seq/Annexin Methods 与 Data availability。
- A549 真正顺序凋亡线索：[Mortenson PMID15197486](https://pubmed.ncbi.nlm.nih.gov/15197486/)，Abstract Methods/Results；仅此层级已核验，完整数值仍待恢复。

所有源文件与推导结果按版本独立保存；本报告是本轮主研究文档，历史报告不被改写。
