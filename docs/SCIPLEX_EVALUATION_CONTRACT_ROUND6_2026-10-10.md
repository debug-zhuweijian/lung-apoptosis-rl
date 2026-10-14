# sciPlex 24/72 h 与 Tahoe 参考分布：Round 6 评价契约

日期：2026-10-10。  
范围：sciPlex3/GSM4150378 的 README、当前提取/回读状态、小型审计表和细胞元数据；[Round5 覆盖报告](FULL_DATA_FEASIBILITY_COVERAGE_ROUND5_2026-10-10.md)；当前 `TrainingData → EngineData → condition_encoder` 的必要时间编码路径。  
本次仅新增本文。未读取或重扫稀疏表达矩阵、未训练或生成模型预测、未修改共享实现和旧冻结数据。以下划分为可执行设计契约；本文没有生成正式划分包，联合实现状态须另以其版本清单核验。

**结论：sciPlex 已提供可读取的 A549 24/72 h 药物—剂量条件表达，可以关闭“只有 24 h 表达、没有真实 72 h 目标”的缺口，并支持明确的条件补全、时间留出和来源内分布预测检验。Tahoe 重复处理跨板覆盖提供参考分布与平均表征比较的设计入口。二者都不能直接关闭表达增量的真实凋亡证据、同细胞未来命运或共同 T* 的暴露时长选择。** Hastings 已有的局部共同终点蛋白信号比较继续保留；sciPlex 不重复承担或扩大那份证据。

## Technical Review

### Errors：必须避免的错误解释

- **把 24/72 h 条件表达当成共同 72 h 的时长选择。** 两个时间点的独立收样没有自动提供“24 h 停药，再于 72 h 检测”的 Y。即使真实持续暴露协议得到核实，Y(持续24 h，24 h读出) 与 Y(持续72 h，72 h读出) 仍是不同观察终点。
- **把同来源 72 h 条件留出叫未见时间外推。** 若开发数据包含其他药物或剂量的 72 h 表达，测试只是新的药物—剂量—时间组合，时间 72 h 已被学习。
- **把 244,281 个细胞当成生物学重复数。** 细胞是孔/实验内观测；条件有 rep1/rep2 源重复标签。重复独立性的证据不能从 cell 数或 barcode 唯一性推出。
- **把物理小时与生成内部 τ 混用。** 当前模型分别编码 `duration_h` 与 [0,1] 生成时间；增加 ODE 步数、把 τ 设为某值，不能制造新的真实处理时长。
- **用当前旧 role 冲突机械删除新来源药物。** 新联合包可以按新的化学连通组整体重分角色；须保留旧版本和旧 role 来源，重新绑定新权重，而非丢掉有效数据以维持历史划分。

### Rigor Issues：仍不能由新增表达自动解决的部分

1. 培养板嵌套于时间，时间和任意板效应不能无假设地分离；RT/ligation/PCR 批次又跨越两个时间，所以也不能把它说成全部技术批次完全分离。
2. 使用匹配时间/板的 vehicle 作为预测输入，是带参考条件的任务；该参考不是同细胞实测 t0，也不是完全不接触目标时间信息的前瞻预测。
3. 72 h 仅覆盖源表 47 个药物目录条目。对这部分的效果不代表其余 141 个药物已有 72 h 真值，不能把作者子集当成未经核实的随机样本。
4. sciPlex 没有实测 Annexin/caspase 标签。作者回收量拟合的 viability 不能替代凋亡率；表达预测更准不等于干预选择在真实凋亡终点上更好。
5. Tahoe 中“对应板参考分布优于固定平均”的结果可能来自批次、QC 或群体组成。必须有相关基线与对照；即便表达预测改善，也不自动识别单细胞状态的凋亡功能增量。

### Notation Issues

- 本文 `t_obs` 指源记录的 24/72 h 观察/处理时间标签；只有原协议核实后才对相应字段赋予严格暴露持续时间含义。
- `Δt` 指暴露持续时间，`T*` 指共同最终观察终点；两者不能仅因数字相同就省去协议。
- `τ` 指 flow matching 的内部插值/积分坐标，范围 [0,1]。
- `condition`、`well/Combo`、`replicate`、`sample+cell` 是不同层级，下面分别给出原始键。

### Citation Gaps

sciPlex 来源为 Srivatsan et al., *Massively multiplex chemical transcriptomics at single-cell resolution*，[DOI 10.1126/science.aax6234](https://doi.org/10.1126/science.aax6234)。本轮核了本地原 XML 的标题/DOI、现有作者代码审计口径和数据元信息，并亲读本轮[协议专项报告](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/README.md)、[time_control_contract.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/time_control_contract.json)与[QC 审计](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/qc_protocol_audit.json)。专项审查确认作者报告的是从给药到破坏性终点采样的 24/72 h；连续暴露、换液、补药及洗脱细节仍为 unknown。本文未重复其原协议检索，不能把未知项当成“未换液”或“无撤药”。

本轮数据完整性复用此前完整提取及回读记录，不冒称重新读取了 3 GB 源文件或 404 个矩阵分片。下面的科学推导和划分建议均标为本课题设计，不是原论文已证明的结论。S4246 的新增化学发布限制单列于第5节，v1 哈希通过不能替代该限制。

## 1. 当前确实可获得的数据及统计单位

| 项目 | 当前核验结果 | 使用含义 |
|---|---|---|
| 官方 accession | GSE139944 / GSM4150378 | 一个来源注册入口，不是一个生物重复 |
| A549 筛选前细胞 | 244,281；24 h 162,171，72 h 82,110 | 原记录完整保留 |
| 指定 hash-QC 标记 | 24 h 143,015；72 h 67,443；合计 210,458 | 本轮专项报告与作者固定 commit 的 valid_cells 名单逐 A549 ID 完全一致；不冒称正文其他处理已全部复现 |
| 来源条件×重复摘要表 | 1884 行 | 包含聚合多个孔的 vehicle 行，不等于 1884 个物理孔 |
| 细胞元数据中的 Combo | 1920 个：24 h 1536，72 h 384 | 本轮重算的孔键计数，其中 vehicle 40 个 |
| hash-QC 后非空 Combo | 1919 个 | 一个孔为 QC 后空观测，不能造出零表达目标 |
| 时间/浓度覆盖 | 24 h 188 个非 vehicle 条目，72 h 47 个；四档 10/100/1000/10000 nM | 0.01/0.1/1/10 µM；不与 Tahoe 0.05/0.5/5 µM 强行配成同浓度 |
| 两时间共有药物—剂量 | 188 个；187 个在两时间×两源重复均非空 | 非空仅表示可观测，不是功效保证或人为保留阈值 |
| 基因视图 | 58,347 人基因原始轴；32,461 共同轴；202 个分片各两视图 | 原始整数计数，尚无本来源训练划分或训练内归一化 |
| 当前状态 | `complete_expression_extraction_not_training_ready`；既有全分片 readback 为 passed | 提取/读回通过与模型接入、训练有效是不同验收 |

唯一零 QC 条件是 `S7207`、Bisindolylmaleimide IX (Ro 31-8220 Mesylate)、10000 nM、72 h、rep2、plate52，实际孔键 `plate10_D10plate52`：源细胞 10，当前 hash-QC 后 0。必须保留预期条件和缺失原因；不能把“没有可用表达细胞”编码为凋亡率 100%、表达全零或成功清除肿瘤。

README 曾列每重复至少 1/2/20/50 细胞时的覆盖敏感性，这不是本契约的门槛。本契约不新增任意 cell n 下限。零细胞使该经验分布未观测；小而非零的群体保留并报告噪声与估计器适用性。若某统计量有数学上的最小 n 要求，应报告该指标不可计算或预先选定适用估计器，不在看结果后删除条件。

源论文 72 h 64,440 与当前 hash-QC 67,443 的差异仍未闭合。新增专项审计已取得作者固定 commit 的 `valid_cells.tsv`：A549 两时间的保留集合与当前 hash-QC 完全相同，差异 ID 均为 0。因此不能再将这 3,003 个细胞差额解释为本地漏做了已公开 hash 规则；正文的具体历史处理仍未知。当前研究可按已验证名单开展，不能删除细胞凑齐论文数字。全细胞系另有 1 个非 A549 差异，本结论仅对 A549 成立。

## 2. 从源键到可审核条件，禁止隐式逐细胞配对

本来源尚无新训练包的正式 condition_id；以下使用**实际原始复合键**，不将人为命名冒充作者观测 ID：

- 来源：`GSM4150378`。
- 科学条件键：完整化学身份/目录条目＋剂量＋时间＋A549＋适用协议。
- 实验/孔键：`(source, sample, Combo)`；Combo 同时定位源 plate_oligo 与 well_oligo。本轮核验每个 Combo 只映射一个药物—剂量—时间—源重复—培养板组合。
- 源重复标签：`rep1`、`rep2`；同名标签在不同药物/时间间不等于同一孔、同一细胞或已证明的成对纵向样本。
- 细胞键：`(source, sample, cell)`；sample 为 `Chem3_Screen2`。本轮核了 sample+cell 唯一。
- 文件行定位：`source_cell_index`、`shard_id`、`row_in_shard` 及冻结的基因顺序。分片边界与 source/condition/replicate 无关，不能按 shard 随机切分科研数据。

例如原细胞键 `(Chem3_Screen2, A01_E09_RT_BC_196_Lig_BC_160)` 对应 `source_cell_index=1223`、`Combo=plate2_B7plate49`、`shard_id=0`、`row_in_shard=366`。这是一个 72 h 目标细胞的身份定位，不是与 24 h 某细胞一一对应的证据。

任何新 condition_id 必须保存到这些源键的完整映射。所有来自同一目标孔的细胞一同进入科研角色；禁止把同孔细胞随机分成训练和测试，之后宣称留出条件泛化。细胞抽样只可用于同一已分配角色内部的数值训练或指标估计。

## 3. 最小可执行协议 M1：来源内药物—剂量—时间组合补全

**目标：** 检验已见药物、已见时间下，一个未用于拟合的组合能否被预测。不是跨药物冷启动，不是训练未见 72 h。这两个条目按元数据和可对照的来源锚点选定，没有读取表达结果挑选容易成功的案例；它们只构成小型入口检验，不代表全部面板。

### 3.1 固定角色与实际孔键

1000 nM = 1 µM。下表原记录均属于 A549 / `Chem3_Screen2`：

| 目录条目/时间 | 源重复 | 原始 Combo | 培养板 | hash-QC 细胞 | M1 用途 |
|---|---|---|---|---:|---|
| S1007 Roxadustat，1 µM，24 h | rep1 | `plate3_B7plate9` | plate9 | 53 | fit |
| S1007 Roxadustat，1 µM，24 h | rep2 | `plate3_B1plate33` | plate33 | 158 | fit |
| S1007 Roxadustat，1 µM，72 h | rep1 | `plate2_B7plate49` | plate49 | 235 | final evaluation |
| S1007 Roxadustat，1 µM，72 h | rep2 | `plate9_B1plate51` | plate51 | 258 | final evaluation |
| S1098 Rucaparib phosphate，1 µM，24 h | rep1 | `plate3_G8plate9` | plate9 | 40 | fit |
| S1098 Rucaparib phosphate，1 µM，24 h | rep2 | `plate3_F2plate33` | plate33 | 106 | fit |
| S1098 Rucaparib phosphate，1 µM，72 h | rep1 | `plate2_F8plate49` | plate49 | 229 | model selection |
| S1098 Rucaparib phosphate，1 µM，72 h | rep2 | `plate9_F2plate51` | plate51 | 221 | model selection |

其余来源条件可以用于 fit，但须满足相同 QC、化学身份与源协议声明；不得以 test/selection 孔的细胞参与表示拟合、基因选择、归一化参数估计、超参数优化或训练损失。预训练来源及其已见药物/时间另列；表中药物也在 Tahoe 的历史结构编码交集中，因此这个协议尤其不能称全链未见药物。

两个 72 h 科学条件分别完整保留两份源重复，不把一份用来拟合另一份当作条件泛化测试。这里使用 fit、selection、final evaluation 三种用途，不强制额外造第四分区。若完全冻结模型和超参数，可省略 selection 学习，但不得随后在 evaluation 标签上选方案。

### 3.2 M1 合法的匹配 vehicle 输入

允许以同来源、同时间、同源重复、同培养板的 vehicle 作为参考；以下是对应源孔：

| 培养板/时间/重复 | Vehicle Combo | 当前 hash-QC 细胞 |
|---|---|---:|
| plate9 / 24 h / rep1 | `plate3_C6plate9`；`plate3_C12plate9` | 50；86 |
| plate33 / 24 h / rep2 | `plate3_C6plate33`；`plate3_C12plate33` | 140；147 |
| plate49 / 72 h / rep1 | `plate2_C6plate49`；`plate2_C12plate49` | 267；186 |
| plate51 / 72 h / rep2 | `plate9_C6plate51`；`plate9_C12plate51` | 248；151 |

必须明确选择以下一种控制使用政策，不能混写：

- **角色隔离政策**：先将完整控制孔或预先登记的控制资源分配给用途，模型拟合和最终推理使用的控制资源按契约隔离；不因物理上每板两个控制孔而强制把它们硬拆成四份。
- **允许评价时参考政策（M1 推荐声明）**：最终推理可读取上述同板同时间 vehicle。该参考可与同板开发条件共享，因为它本身就是本任务允许的上下文；不能用测试处理细胞代替它。双方模型得到相同的参考信息，报告共享控制导致的相关性，不把这项设置称无目标时间/目标板信息的完全外推。

无论选哪种政策，预处理和可学习参数仅在允许 fit 的数据拟合。仅允许推理时读取控制，不等于允许拿这些新控制重新训练 AE、更新全局尺度或选模型。若确实进行无监督适配，必须将其登记为另一个允许目标域适配的实验，双方使用同样信息。

Vehicle 条件摘要在 `a549_condition_coverage.csv` 会聚合多个孔；不能拿“72 h rep1 vehicle 总池”默默替代同 plate49 控制，然后宣称同板匹配。可以预定义跨板 pooled reference 作为一个对照，但必须保留其真实组成。

### 3.3 M1 最简单基线与指标

先采用足以解释结果的有限比较，不因网络复杂就省略基线：

1. **匹配 vehicle 经验分布**：预测处理后仍与允许的 vehicle 相同，提供无药效基线。
2. **相同药物/剂量的 24 h 分布保持**：使用相应 fit 孔的 24 h 表达作为 72 h 预测；只是假定响应保持，不是假装同细胞追踪。
3. **条件均值/简单条件模型**：使用药物信息、剂量和时间，但不读取细胞级参考变化；训练规则与复杂模型同样冻结。
4. **显式时间编码消融**：模型骨干、预处理及其余允许输入相同，只比较显式物理时间编码的增量。若仍保留时间特异 vehicle、板信息或 protocol，名称须为“给定这些上下文后的显式时间编码增量”，不能叫完全无时间信息。
5. **参考均值/分布比较**：固定 pooled reference、匹配参考的均值、匹配参考完整分布。明确均值计算在原始/规范化表达空间还是 latent 空间，不能把 E(mean X) 与 mean E(X) 的差异都归为细胞异质性。数值接口重复同一均值不产生新的实测样本。

至少分别报告每个目标孔的群体均值误差和预先固定的分布差异；可再按条件/源重复等权汇总。模型抽样细胞数、参考抽样规则、随机种子及核带宽等在评价前固定；核带宽或特征筛选不由测试表达拟合。不能让 258 个细胞的孔凭 cell n 自动压倒较小孔，也不能把大量随机生成细胞当成新增生物证据。

药后减 vehicle 的群体变化可作为辅助表达指标，但其参考和尺度须相同且先定义。没有同细胞真值，不计算“该输入细胞预测到正确后继细胞”的准确率。不能用本协议的表达 MMD/均值误差直接替代 Annexin、凋亡排序或药物选择收益。

## 4. 独立协议 M2：24→72 h 时间留出

M2 与 M1 是不同科研问题，必须使用独立划分、预处理/权重身份与报告。不能在 M1 已用 72 h 调过参后，换一张标签列表称为 M2 未见时间。

### M2-S：本来源严格 72 h 表达留出

- Fit 与 model selection 仅用本来源 24 h；具体 validation 可在 24 h 内按完整化学/条件组留出。
- 72 h 的所有处理表达**以及 vehicle 表达**不得进入 AE 预训练、归一化/基因筛选、模型选择、特征拟合和测试时参考。参考使用预先允许的 24 h 或外部固定来源，并声明这种输入条件。
- 外部预训练如果已用其他来源的 72 h 数据，应明确称“未见本来源的 72 h”，不能称整个系统未见该时间。若要求全系统未见 72 h，则相应外部训练暴露也要审查。
- 最终评价可覆盖现有 47 药×4 剂量的 72 h 条件，按完整孔/源重复保留；零 QC 孔登记缺失，不补造目标。
- 最简单对照是相同药物/剂量的 24 h 分布保持；比较的是在这组输入条件下预测更晚时点的能力，不是对任意时长的一般证明。

只有 24 h 监督时，存在许多在 24 h 相同、在 72 h 不同的函数。例如任意 `f(d,c,t)=f24(d,c)+(t−24)·h(d,c)` 都可匹配同一 24 h 数据，而 h 未被那些数据决定。这是本课题说明欠约束性的例子：允许用明确先验的模型接受 72 h 真实挑战，但不应因为模型接收 time 参数就宣称 24 h 数据已识别了时间规律。

### M2-R：未见 72 h 处理表达，但允许 72 h vehicle

这是一种合法且更具参考信息的任务：72 h 处理表达完整留出，允许读取明示的 72 h vehicle 作推理参考。应命名为**带目标时间参考的响应外推**，不能与 M2-S 共用“完全未见 72 h”标题。

可增加一个简单均值基线：在事先固定的表达尺度上，将已见 24 h 的药物相对 vehicle 变化加到 72 h vehicle 均值。该基线仅用于其定义的群体均值指标；不把加减得到的向量冒充实测细胞或原始合法 count 分布，也不据此推断细胞命运。

仅有 24/72 h 两点不能验证 48 h 插值、任意时间的非线性轨迹或多步滚动。若训练已用其他条件 72 h，则改称 M1 式组合补全，而不是 M2。

## 5. 独立协议 M3：新联合包的化学组留出

M3 检验跨药物，不能与源内剂量/时间补全或严格 72 h 留出混称。

- 新联合包应保留完整化学身份作为“是什么物质”的字段，再建立独立 `split_group_id` 作为防泄漏分组。完整结构、非立体连接及最大有机片段的保守候选链接可形成连通组；分组保守不等于断言这些化合物生物等同，也不意味着合并其响应 Y。
- 不能因共有无机盐/对离子把所有药物连成一组；“最大有机片段”的具体规则和不确定立体/混合物处理必须有版本记录。
- 同一保守组的相关处理在各来源、各剂量、各时间、各重复的目标表达随组分配。若只让 G 不见某药 Y、P/AE 已见该药表达，声明其实际泛化层级，不能称整个链从未见该药响应。
- 旧 v3 资产保持不动；新包可整体重分角色，并保留 `legacy_v3_role`。不为维护新包中的旧 role 而机械排除有效新药。新包具有新的数据/化学/划分/上下文/预处理身份，相应训练权重须匹配；不自动复用旧权重作为严格留出的模型。
- 各协议可以独立定义训练/选择/评价，按任务建立不同 manifest；不要求 M1/M2/M3 同时共用一套固定四分区。跨协议比较需要记录模型实际接触了哪些目标。

现有 `audit/tahoe_chemical_encoding_overlap.csv` 的 15 个完整编码相同、18 个最大片段候选、25 个忽略立体候选是**2026-09-24 审计所绑定旧 training-v1 化学表的历史数字**，不是本轮新联合包冲突数。本轮先亲读[历史 v1 结构审计](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v1/README.md)及实际表：188 个源结构字符串可由工具解析、v1 登记 188 个完整 ID，[v3_full_matches.csv](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v1/v3_full_matches.csv)为 18 个完整记录匹配；41 条未指定立体、29 条多片段、1 条无 InChIKey，全部明确保留，不按名称补造结构。10 项输入/输出 hash 一致；这证明来源结构编码和关联，不证明历史物料身份。

**当前消费者应使用已生成并亲读核验的 [chemical_inputs_v2](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v2/manifest.json)，v1 保留为历史版本。** v2 仍有 188 个源目录条目，但只有 187 条合格的单结构主输入，主指纹矩阵为 187×2048；S4246 的 `fingerprint_row=-1`、`eligible_for_structural_features=false`，登记为相对立体关系已核、绝对组成未定的来源实体。本文亲核 187 个 eligible 行与主指纹的 0–186 行一一对应，v2 的 18 项输入/输出 hash 均匹配；原始化学物料/SDF 解释由主代理亲读，本文没有重做该化学鉴定。

S4246 的两个镜像候选在[独立解释表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v2/relative_stereo_candidates.csv)及 2×2048 指纹中另存，两个 `mixture_weight` 均未赋值，`observed_material_identity=false`。它们不是两个新增实验，也不能擅自平均为 0.5/0.5，或据测试 Y 选择较好候选。不得删除孤立 `*` 后擅定单一绝对构型，也不得用主表行号 -1 误读最后一行指纹或以占位指纹冒充合格输入。需要结构不确定性感知表示的消费者尚需另行验收。

亲读的来源条件表显示 S4246 为 Tranylcypromine (2-PCPA) HCl，只有 24 h，8 个剂量×重复行、954 个 QC 细胞；源元数据和表达全部保留。其余 187 条不因单条争议整体失效，但最终消费者的 eligible 清单和结构身份必须绑定 v2，不能从 RDKit 可解析直接推出全部 188 条正式可用。M1 指定的 S1007/S1098 孔不受该单条争议直接影响；其他 fit 条目按最终输入资格登记，仍遵守科研角色和训练暴露规则。

另亲读实现代理的[legacy_split_conflicts.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/legacy_split_conflicts.json)：保守连接/母体候选 25 个；S1068/S2673/S5001 三药跨旧 roles，共 2174 个 QC 细胞，全部为 24 h。该表是实际结构/角色元数据运算结果；本文核其输入 hash 和 QC 汇总，不重复运行分组程序。本契约采用已确定的“新包全局保守组重分、保留全部 188 个来源条目”方向；保留来源记录不等于预先批准每条记录在当前结构消费者中都可用。

正式冻结 M3 前应保存实际连通组、冲突角色、每来源/时间的受影响条件数及新角色分配。特别报告 72 h 保留情况。一个“全局药物留出”成功建立不意味着 M2-S 同时成立。

## 6. 培养板/时间混杂与源重复的解释

当前元数据证明 24 h 有 16 块培养板，72 h 有 4 块，任一培养板只属于一个时间；两个时间共享 `Chem3_Screen2`，也在全部 4 个 RT 板、4 个 ligation 板、2 个 PCR 板中都有细胞。

本课题的简单识别性说明：若只写“观测均值 = 时间项 + 任意培养板项”，每块培养板只在一个时间出现，则可以给某时间项加一个常数，再从该时间的所有板项减去相同常数，拟合值不变。因此这些元数据不能无额外假设地分离时间与任意培养板效应。将培养板当随机效应、认为不同时间板可交换或用 vehicle 校正，都须明确其假设和检验范围，不能自动变成已经消除了混杂。

匹配 vehicle 差分可帮助处理共享背景差异，但无法保证消除 drug×plate、差异回收和其他交互效应。建议分别报告：

- 处理表达本身、相对匹配 vehicle 的变化；
- 加/不加时间的同信息预算比较；
- 对应板参考 vs pooled reference；
- plate-only、可得 QC 摘要以及简单条件基线；
- 源重复分层结果与完整条件平均，不仅报一个按细胞数加权的大均值。

rep1/rep2 的实验含义由源协议证据界定；仅凭标签不把同名跨时间记录当成配对追踪。即使确认是两份生物学重复，两个重复内数十至数百细胞也不增加生物学 n。对细胞重采样得到的区间，只能说明该孔经验分布/抽样估计的不稳定性，不能称跨生物重复或跨实验室的总体置信区间。没有所有原始重复文件并不阻止经验分布检验，也不要求任意通用样本数门槛。

## 7. 当前代码的时间语义与接入验收

本轮按主代理确定的现有 `TrainingData → EngineData → condition_encoder` 路线核查，未以旧 TimeData/timecourse 分支代替。

| 源码 | 已亲读的当前行为 | 不能推出 |
|---|---|---|
| [reader.py:290](../src/lung/data/reader.py#L290) `action_features` | 从条件行读取 dose_um、duration_h 与 source/background/protocol 字典索引 | 源协议已证明所有时间标签就是连续暴露时长 |
| [engine/data.py:49](../src/lung/engine/data.py#L49) `flatten` | duration_h 传为 duration，保留上下文 ID | 新来源 registry/矩阵路由已经可运行 |
| [response.py:54](../src/lung/engine/response.py#L54) `condition_kwargs` | elapsed 默认 0；未显式给 horizon 时取 batch horizon 或 duration | 所有训练记录已经共享一个最终 T* |
| [encoders.py:96](../src/lung/models/encoders.py#L96) `DurationEncoder` | 正有限小时数，经 log1p 后输入 MLP | 训练只有一个小时值也能由数据识别时间趋势 |
| [encoders.py:173](../src/lung/models/encoders.py#L173) `ConditionEncoder` | 合并药物、剂量、duration、上下文、历史、elapsed/horizon、duration/horizon | 把任意 horizon 改成 72 就获得撤药后真实结果 |
| [encoders.py:193](../src/lung/models/encoders.py#L193)、[latent_staged.py:117](../src/lung/models/latent_staged.py#L117) | τ∈[0,1]，训练的生成路径在 noise 与 target latent 之间 | τ=0.5 是某个真实治疗小时或实测中间状态 |
| [flow.py:11](../src/lung/models/flow.py#L11) | Heun 以 steps 个数值步从 0 积分到 1 | steps 改变暴露时长或产生新的时间标签 |

当前单记录监督中，当 horizon 默认等于 duration 且 elapsed=0 时，两个比例特征分别为 0 和 1；物理 24/72 h 的区别仍来自 duration 编码，而不是数值积分步数。共同 T* 的撤药方案则必须另有明确动作协议和该 T* 真实 Y，不能伪装成现有单记录的 duration 字段。

接入应完成的小规模验收包括：上述源孔/细胞键到新条件 ID 和分片行的追溯；完整基因轴和来源路由；新 source/protocol vocabulary；训练内预处理；两时间真实记录能以正确单位到达条件编码器；测试目标和允许参考的角色守卫。它们是实现和数据可读性检查，不能通过“给模型一个 72”宣称时间预测已经有效。

历史作者 `Size_Factor` 和原处理代码的拟合范围也应审查。建议从已保留整数计数按新协议拟合必要预处理，避免未经说明复用含评价表达信息的全源统计。不能将 PCA/HVG/带宽/对齐映射先在全部测试表达上拟合，再只隔离最后一层标签。

## 8. 与 Tahoe 参考分布比较如何共同推进原主线

Round5 已核 Tahoe 1336 条总条件、1305 条 eligible 处理、14 板、28 vehicle；1128 个完整化学身份×剂量×24 h 组，其中 154 组跨板、325 条条件。源对照关系是同板群体候选，单细胞前后配对不可用。这里复用该元数据证据，本轮没有重复读取 Tahoe 表达。

两来源可以支持以下分层问题：

1. **表达层 A 的候选检验**：对应来源/板的参考分布，相比固定 pooled reference、表达均值或 latent 均值，是否改善真实药后群体分布预测。控制相同药物/剂量/时间与允许信息，再比较 plate-only/QC；不把板号本身称为生物状态。
2. **表达层 B 的可计算检验**：sciPlex 具有真实 24/72 h 目标，可做 M1/M2；Tahoe 的跨板重复处理可做参考条件化或跨板检验。各自的留出层级必须单列，不能合称同一种泛化。
3. **跨来源联合开发**：共同基因身份和已审核化学关联使新联合包可设计，但剂量档不同、观测平台与参考语义不同；保持来源上下文和单独的来源分层结果。不把没有完全相同剂量的两个条件当成实测对应。
4. **仍缺的功能链**：这些表达来源没有对应的凋亡 Y，不直接关闭 C 或 D 的真实功能收益。可以在正确冻结、校准及转移假设下把生成特征用于独立功能挑战，不要求每个功能 Y 都有实测药后 RNA；但不能仅凭新增表达数量宣布这条连接已验证。
5. **共同 T***：Hastings 的两时长/同72 h蛋白读出继续是局部 E08；sciPlex 的不同时间收样不能充当其多状态 Annexin 扩展。完整状态条件化联合选择与“表达超出群体平均”的功能增量仍需相应实测对应。

若参考分布优于均值，只能按实际评价层级报告收益。分布采样和 OT 先验定义一种可学习的耦合，不把它变成观察到的逐细胞前后映射。匹配终点时刻的对照群体作为公开来源参考是合法建模选择，但须保留“参考条件化响应预测”的语义，不能悄悄变成同细胞给药前治疗决策。

## 9. 本轮交付后的冻结清单与结果边界

执行某个协议前，保存其可审核材料：

- 协议名 M1/M2-S/M2-R/M3、预测时合法输入、预训练暴露和使用目标参考的规则。
- 源键到新 condition_id 的映射、完整化学身份与独立 split_group_id、role、source/plate/replicate/time/dose，以及所有零/缺失观测。
- 训练内预处理/表示/核参数与模型身份；针对新联合包的新角色和权重绑定。旧 v3 与历史 role 保留。
- 实际选择和最终评价的完整目标孔清单、允许控制池、基线、均值/分布指标及各指标分母。
- “输入读取通过”“可以构造实证检验”“模型效果有收益”分别报告。完成前两项不要求先证明方法获胜；负结果也是有效检验。

本轮没有定义任意样本数门槛，没有要求所有方法必须四固定分区，没有扩大或缩小毕业论文主线。它将已有表达明确分配给可证伪的问题，并保留真正未闭合的功能和共同终点连接。

## 10. 来源版本与本次核验范围

直接读取的入口：

- [sciPlex README](../08-datasets/sciPlex3/GSM4150378/README.md) 与[提取 README](../08-datasets/sciPlex3/GSM4150378/extracted/v1/README.md)。
- [extraction_status.json](../08-datasets/sciPlex3/GSM4150378/extracted/v1/extraction_status.json)、[readback_audit.json](../08-datasets/sciPlex3/GSM4150378/extracted/v1/readback_audit.json)、[audit_summary.json](../08-datasets/sciPlex3/GSM4150378/audit/audit_summary.json)。
- [条件×重复覆盖](../08-datasets/sciPlex3/GSM4150378/audit/a549_condition_coverage.csv)、[两时间覆盖](../08-datasets/sciPlex3/GSM4150378/audit/shared_dose_time_rep_counts.csv)、[历史化学交集](../08-datasets/sciPlex3/GSM4150378/audit/tahoe_chemical_encoding_overlap.csv)、[plate×time](../08-datasets/sciPlex3/GSM4150378/audit/plate_time_cross_tab.csv)。
- [细胞元数据](../08-datasets/sciPlex3/GSM4150378/extracted/v1/a549_cells.parquet)：仅读所需列，核对源键、孔归属、QC 计数和文件行定位，未加载表达矩阵。
- [Round5 覆盖报告](FULL_DATA_FEASIBILITY_COVERAGE_ROUND5_2026-10-10.md)，SHA256 `75981afec379241145f282cb32181c25d93453f9a1c730230213081318d1e5d8`。
- [研究架构](G:/研究生科研文件/毕业论文/研究架构.md)：§2、§3、§4.2/4.3、§6；源码定位见第7节。

当前读取版本摘要：

| 文件 | SHA256 |
|---|---|
| sciPlex README | `f5901458cfb896a43399dc2f0c656f53880f43155c4c16d20285b222d8f88c1a` |
| audit_summary.json | `3a95bc335f3ba733bb77c480e83431e37f7fcb5e5f75d6c999b82e521d25b8e6` |
| a549_condition_coverage.csv | `6e1cbb87c71f0b66bea30e2dcf98d110e0a89aad06d1a51b1867e735d7cc3808` |
| 历史 chemical overlap | `595b70c7aec1e0927dd1dba7cce8b712809cfe9ef78691be9be5d1cb86a6bc62` |
| extraction_status.json | `7455e6408089c84c7c68ce9dc8f70908e744ab90438fa1938a6f37f6e10a722e` |
| readback_audit.json | `3be40a615d9954a7d7726a7657efdb6b5c2846e9a46b30ba36f47135b22dae1a` |
| a549_cells.parquet | `95242aca98787f6c961d381673e714eea24751f9669fbac8c2929bd1e2d9d325` |

本轮新增核验：1920 个孔键、1919 个 QC 非空孔、40 个 vehicle 孔，sample+cell 唯一性；M1 八个目标/开发孔与八个 vehicle 孔的实际字段；188 个共享药物—剂量及唯一全组 QC 空孔；当前时间编码静态调用路径；新结构 sidecar 及角色冲突小表。当前 v2 sidecar manifest SHA256 为 `c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05`；历史 v1 为 `18e9b4597824f7698a9e0f81e572b9805d6d03ecff6668deb1e9e4e24731ccbf`，冲突表为 `c135f936fbd02b2758dde88f596bc86c01459d8add8c2767fdc9f37d2a9d7289`；协议机器契约为 `f4400fcb190c2b51e2cff39e258c4a355e84c6ef38800d2f71ae156636d827be`，QC 审计为 `f02561feb3c63c63ae8a88c70234b18057cb52ec04ce8a86701c7ecdf66cbdd6`。既有完整矩阵回读、原协议专项审查、新联合化学冲突重算与实际控制表达审计分别保留各自负责人的证据范围，不冒称由本文重复执行。
