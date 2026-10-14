# 完整目标数据覆盖与下一项最小实证设计：Round 5 独立复审

日期：2026-10-10；本轮修订补入 Hastings 共同终点证据、Nav-Gal E07 读取状态与 Tahoe 元数据窄审计。  
范围：毕业论文研究架构、数据可行性验收契约、Nav-Gal `challenge-v1` 及读取评价入口、Sarin `state_gene_round4`、`coverage-v4` 及其绑定表；本轮新增 Hastings `cleaved_caspase_challenge_v1` 与 Tahoe `tahoe_state_design_round5` 三个小文件。  
操作：只读审查、像素复算及 Nav-Gal readiness 检查；本轮仅修订本文。未训练、未运行科学模型预测、未修改其他数据、代码或文档。

**结论：当前已有可实际执行的有限 A/B/C/D 组件检验，且新增 Hastings 证据已提供单一 A549 背景、固定 cisplatin 剂量下，在共同 72 h 终点比较 2 h 去药与持续 72 h 暴露的相对 cleaved caspase-3 均值。原稿“共同 T* 实测时长比较完全缺失”的概括应撤回。完整主目标仍未覆盖：这个局部时长比较不是 Annexin 比例，也没有多状态的完整联合候选块；“细胞表达信息超出群体平均/类别”的原核心假设尚未被现有条件级功能连接识别。** 群体类别或参考 RNA 整体表示相对于一个固定参考的检验、RNA 超出类别的信息检验、逐细胞命运预测，是三个不同层级。前两者的区别必须保留，不能因后一层缺失就否定前一层可以开展。

下文“可执行”指标签、观测单位和合理留出规则已经能具体列出；不表示当前 P/C/G 已接通、方法会优于基线或统计功效已足够。建议的划分和基线是**本课题本轮设计**，不是作者原始实验设计，也未在本轮执行。

## Technical Review

### Errors：需要纠正的覆盖解释

1. [coverage-v4/coverage.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v4/coverage.json) 中 `observed_initial_expression_conditions=0`、`complete_expression_feature_conditions=0` 只统计其绑定的 **304 条核心条件**。Nav-Gal 的群体类别挑战和 Sarin 的参考表达包没有因此消失；这也不是 Tahoe 全部板位和表达状态变化的盘点。
2. 同一文件中的每种观测定义只有 1 个来源/化学连通块、无法构造四个非空固定分区，是当前隔离规则与这组样本的组合结果。它不能推出所有留组、固定外部模型或回顾性检验都不可能。对“新来源”泛化，它确实揭示当前同定义独立来源有限，不能换成随机细胞切分来掩盖。
3. Sarin 的共同基因轴对齐只解决身份对应，不把 bulk 微阵列变成单细胞计数，不把 24 h 平行无药对照变成给药前实测，也不把不同研究的相同条件变成同样本表达—功能配对。
4. Nav-Gal 的 10/11 µM 是同一实验输入的未决来源冲突，不能作为两档实测剂量。0–36 h 连续暴露曲线也不能充当共同 T* 的暴露时长选择。

### Rigor Issues：完整主目标仍需守住的边界

[研究架构 §3](G:/研究生科研文件/毕业论文/研究架构.md#L32) 的四项分别对应下文 A/B/C/D。§2 允许可获得的群体状态，§3 同时保留细胞表达超出药物信息和群体平均的增量假设。**用群体类别先检验 A 的较粗层级是合法推进，不得将这个结果重命名为细胞表达增量已经验证。**

[研究架构 §4.2](G:/研究生科研文件/毕业论文/研究架构.md#L71) 要求完整动作协议与共同终点，[§6](G:/研究生科研文件/毕业论文/研究架构.md#L188) 要求终点、校准、策略训练与最终评价隔离。后述小型检验不能绕过这些约束，也不要求为每个小数据集机械分四份。

### Notation Issues

本文区分：

- **群体类别**：例如 parental/resistant、nonsenescent/senescent，指有来源定义的背景输入。
- **参考 RNA**：平行或历史群体表达，可以是合法的参考输入；不等于当前细胞实测 t0。
- **目标 Y**：Nav-Gal 为图像 Annexin 阳性百分数；Sarin 为相对于各自无处理组的 Annexin fold；核心表为具体流式协议定义的 fraction。三者不直接合并成同一个数值标签。
- **时点与暴露时长**：连续处理后在 t 读出，不等于处理在 t 停止后于共同 T* 读出。

### Citation Gaps

本轮生物学事实来自亲读的新旧原文、源图及当前源包；新增 Hastings 事实另有原文节号、图号与独立复算记录。本轮已亲核 Tahoe 窄审计的三个小文件及跨板组成员，确认存在可设计参考分布比较的元数据覆盖；没有读取大型表达矩阵，也未验证板位差异是否对应生物状态。因此状态更新为**元数据有候选比较，表达/协议/生物状态效应待验证**，不把核心终点表的零计数外推为全源为零。本文关于可识别性、划分用途与最小对比的推断均为本课题分析。

## 1. 实际数据覆盖，不合并统计单位

| 数据包 | 当前确实有的输入与 Y | 不能由此推出 |
|---|---|---|
| Nav-Gal challenge-v1 | 2 个有诱导依据的 A549 群体状态；vehicle/Navitoclax/Nav-Gal；每条曲线 10 个发表均值，共 60 点；24 h 为 4 个 active 状态×药物均值及 2 个 vehicle 均值 | 60 个独立实验、两个实测剂量、同细胞 t0、已配 RNA、共同 T* 撤药结果 |
| Sarin state_gene_round4 与功能源表 | 22 份 bulk 微阵列；9 份平行 24 h 无药对照参考、13 份药后表达；3 个状态—剂量条件的 10 个原始附表功能读数 | 22 个功能条件、22 对表达—Y 配对、实测 t0、绝对 Annexin fraction |
| coverage-v4 核心表 | 304 条条件；38 个有可用 Annexin 的条件单位；当前单药 24 h 化学输入候选 14 条、13 个结构组；尚无生成表达特征 | 所有 38 条都是当前 P 可接收输入、所有状态来源都没有参考信息、全研究只能有 14 个可评价条件 |
| 核心表中固定时间剂量块 | 6k 的 5/10/20 µM、24 h；AS-E 的 5/7.5/10 µM、48 h；均有已审核的具体 Y | 同时验证了时间外推、跨结构迁移或共同终点时长选择 |
| Hastings cleaved_caspase_challenge_v1 | A549、cisplatin 5 µg/mL；2 h 后去药与持续 72 h；共同起点与共同 72 h 终点；2 个图示相对 cleaved caspase-3 均值 | Annexin 比例、RNA 增量、多状态/多药/多剂量联合选择、已恢复原始重复 |
| Tahoe state_design_round5 | 1336 条条件，1305 条 eligible 处理；14 板、28 vehicle；1128 个完整化学身份×剂量×24 h 组，其中 154 组跨板重现、共 325 条条件 | 14 个生物状态、154 个新增生物学重复、实测同细胞 t0、已验证表达或功能增量 |

Nav-Gal、Sarin、核心表及新增 Hastings 在此分别登记，不把发表均值、源读数、条件数或跨板重现数相加成独立样本量。Sarin 的 10 个读数属于 3 个功能条件；Nav-Gal 的各时点有关联，作者 n=3 生物学重复并未恢复为逐重复表。原始重复缺失不取消发表均值的点预测用途，却限制新的方差和总体可靠性推断。

核心 14 条当前候选按观测定义分为 Macaya 4 条、Mphahlele 9 条、MCT1 1 条。三个来源没有自动构成三份同协议独立验证；不同门控/分母仍要分别评估。Nav-Gal 和 Sarin 作为另列挑战，既不受这个 14 条计数排除，也不能未经接口与语义审查直接塞入同一个终点头。

## 2. A/B/C/D 的可执行性与剩余缺口

| 原子问题 | 已有可执行检验 | 主要实现缺口 | 当前仍属数据/可识别性缺口 | 允许的结论范围 |
|---|---|---|---|---|
| A1 群体类别或参考表示整包 vs 固定参考 | Nav-Gal 两状态共享两药；Sarin 两状态共享 cisplatin 11 µM；可比较状态感知与无状态参考预测 | 读取状态语义、合法生成预测、固定无状态替换规则；Sarin 需尊重原始微阵列尺度与 mask | Nav-Gal 类别与铺板密度/诱导历史共变，不能分离纯状态因果作用；小来源限制外推 | 有限群体背景/参考表示的预测检验 |
| A2 表达超出类别或同群体平均 | Sarin 可作 RNA 整包挑战；Tahoe 元数据已提供重复处理跨板的参考分布比较候选 | 合法读取表达，核对协议；实现 pooled reference、均值、分布及 plate-only/QC 基线 | Sarin 类内 RNA 与功能重复未对应；Tahoe 尚未验证表达差异/板位混杂及功能 Y 连接 | 可以开展相应层级检验，不能由元数据或 A1 收益推出细胞表达功能增量成立 |
| B 未见条件 | Nav-Gal 留一药功能标签；同曲线留未来时点；Sarin 留耐药 34 µM；核心固定时间剂量块留一剂量/药物 | 冻结划分、特征生成/校准边界、适用范围标记 | 极少药物/来源，不支持广域可靠性；同曲线留点不等于新培养或新研究 | 指定协议内的药物标签迁移、有限剂量/时间预测 |
| C 实测凋亡功能一致性 | Nav-Gal 图像比例、Sarin fold、核心流式 fraction；新增 Hastings 相对 cleaved caspase-3 是单列的凋亡相关蛋白读出 | 为各观测定义给出适用输出或预定方向的排序分数；不把蛋白信号接入 Annexin 比例头 | Sarin 缺绝对基线；Hastings 图示 relative 的逐点归一化分母未独立恢复；跨观测定义尺度关系尚未证明 | 各自测量定义下的误差/顺序一致性 |
| D1 有限药物/剂量选择 | Nav-Gal 同状态两药；Sarin 耐药状态 11/34 µM；核心同协议候选块 | 同一冻结选择程序、同一可比候选表、简单策略/直接评分/枚举对照 | 单独这些块没有同时覆盖药物×剂量×时长的完整链；不能拼接不同折模型的分数冒充一个策略 | 已测可比候选内的有限选择质量 |
| D2 局部共同 T* 时长选择 | Hastings 两臂在共同 72 h 有相对 cleaved caspase-3 均值；实际停止/继续时间可定位 | 需要接受 published relative protein 尺度的预测/评价入口及冻结方案；不能复用百分比出口 | 洗液次数/残留清除未量化；图示单位分母细节未独立恢复；仅一个状态、药物和剂量 | 局部 E08 有条件覆盖；来源内发表蛋白信号的两方案挑战 |
| D3 完整状态条件化联合增量 | 药物、剂量及局部时长组件已分别有实测材料 | 建立兼容状态、动作、观测定义与冻结预测的完整链 | 尚无已核实的多状态联合候选功能评价块；蛋白信号不能直接换算 Annexin 比例 | 整体 E09 仍未通过，不以组件拼接宣布完成 |

A 的层级不应被过度解释。以冻结模型分别输入 Sarin 的状态特异参考 RNA 与一个预定义的固定参考 RNA，确实可以比较**整个表示包**的预测；只是即便有收益，也不能分辨收益来自 RNA 中超出类别的细节，还是 RNA 表示恰好识别了 parental/resistant 类别。若没有收益，同样是可报告结果。不得通过为每个参考样本复制同一条件 Y 来制造类内信息。

当前“无生成特征”“无法走 24 h 生成器的 48 h 请求”“单药生成器不接受同时组合”首先是**现有方法适用范围或实现缺口**。扩接口本身不证明时间迁移或组合预测有效；需要在已有真实标签上检验。也不能因某条标签暂不适合当前模型，就说它不是可用数据。

## 3. 下一项最小实证：先做 Nav-Gal 有限留一药参考响应检验

Nav-Gal 整体外部预测评价入口现已接通真实数据读取；本轮实跑返回 `input-ready/prediction-missing`，没有科学预测。下面的两折小检验仍是设计稿，需单独记录各折开发与预测生成范围，不把入口读取成功当作已完成这些比较，也不为启动而先训练新的高维终点头。

### 3.1 四个 active 24 h 观测与两折用途

以下 ID 均在当前 `challenge-v1/primary_24h.csv` 实际存在：

| 简称 | 实际 observation_id | 折 1：留出 Nav-Gal | 折 2：留出 Navitoclax |
|---|---|---|---|
| N0 | `GonzalezGualda2020:Fig4b:navitoclax:nonsenescent:h24` | 开发 | 评价 |
| N1 | `GonzalezGualda2020:Fig4b:navitoclax:cisplatin_induced_senescent:h24` | 开发 | 评价 |
| G0 | `GonzalezGualda2020:Fig4b:nav_gal:nonsenescent:h24` | 评价 | 开发 |
| G1 | `GonzalezGualda2020:Fig4b:nav_gal:cisplatin_induced_senescent:h24` | 评价 | 开发 |

两个 vehicle 观测为 `GonzalezGualda2020:Fig4b:vehicle:nonsenescent:h24` 与 `GonzalezGualda2020:Fig4b:vehicle:cisplatin_induced_senescent:h24`。这个最小方案评价原始发表比例，**不使用它们作待预测条件的隐式校准**。若另研究扣除 vehicle 后的变化，要单列目标、同时向双方提供相同 baseline 信息，并说明这是带已测对照的条件化任务。

每折只使用开发药物的两状态均值，不使用留出药物任一时点的 Y。两个剂量情景都保留；本节最简单参考基线不依赖浓度，但后续化学/剂量模型必须同时报告两情景，不能挑较好者。当前所有标签已被研究人员看过，整个检验始终标为回顾性。

### 3.2 无需调参的最简单合法基线

令开发药物为 d，状态为 s∈{0,1}，开发均值为 y(d,0)、y(d,1)。在留出药物上预先固定：

- **无状态参考 R0**：两个状态均预测为开发药物两状态均值的平均。
- **状态条件参考 R1**：状态 s 的预测等于开发药物在同状态 s 的均值。

写成公式为：

$$
\widehat y_{R0}(s)=\frac{y(d,0)+y(d,1)}{2},
\qquad
\widehat y_{R1}(s)=y(d,s).
$$

R0 与 R1 使用相同开发标签，差别仅是是否保留状态对应；它们不学结构规律、不拟合超参数。R1 明确假定此药对之间的同状态响应具有可迁移性，该假定可以被另一个药物的真实 Y 否定。它是状态类别信息的低成本参考，不是拟议完整模型或 RNA 增量证明。

采用预先声明的等状态权重，逐点报告误差及读图范围敏感性，不按未知细胞数重新加权；不新增基于像素范围的生物学 p 值。若加入完整模型，必须冻结整个学习程序并在每折开发范围内完成任何 endpoint 校准；特征或模型已经见过留出药物的何种数据要另列。

**这项检验能回答的 B 是：同一研究、固定 24 h、这一对有化学关系的药物之间，终点标签留出的状态响应迁移。** 它不等于新来源、新培养、广泛未见化学骨架或 P/C/G 全链未见药物。若预训练 P 曾见该药响应，只能按实际情况声明“终点标签留出”，不能称全链冷药物测试。

**不能将两折各自产出的分数拼成 D 的同一策略排序。** 每折只有一个留出 active 药物；另一个候选的 Y 已用于开发。这样的交叉预测可以报告条件预测误差，却没有形成一个对完整留出候选集合统一决策的冻结策略。

### 3.3 完整两药选择挑战采用另一用途，不能混用标签

如需让完整状态条件化模型接受 A/C/D 的共同检验，应将 N0/N1/G0/G1 **整体保留为该评价的目标**，不在这些值上拟合/校准/选法。模型与比较模型由外部开发材料冻结；四个 Y 仅计算两状态各自两药的点误差、可分辨顺序与已测候选内选择结果。

可固定的选择基线为：总选 Navitoclax、总选 Nav-Gal、等概率随机选两药（报告其精确期望，不必用随机模拟制造样本）、同一预测器的直接评分/完整枚举。固定策略全部报告，不能看 Y 后只留下表现较好的固定策略作为“事先基线”。真实 Y 中最佳候选可作评价参考，但它是事后 oracle，不是可部署基线。

若选择器与枚举使用完全相同的确定分数、完整候选和无额外约束，枚举已经返回该分数的最高者；RL 不能仅靠更高内部评分证明收益。任何额外收益必须在独立 Y、信息/计算约束或明确的真实序贯任务中出现。这是本课题针对当前有限候选任务的分析。

上述留一药组件检验与整体留出挑战可作为两个明确区分的回顾性分析，但若组件结果已用于改模型，就不能再把同一四点的整体评价称为未受影响的新确认集。主代理应在运行前选定用途并保留接触记录，不必为此向用户反复索取普通实现确认。

### 3.4 时间检验应另列为 B，不能冒充原始状态收益

当前每条曲线有 h0/h4/h8/h12/h16/h20/h24/h28/h32/h36。可预定用六条曲线的 h0–h20 开发，保留 h24 为主评价，h28–h36 为相关的后续检查；观测 ID 为上述实际 curve_id 加相应 `:h{小时}`。

最简单基线是每条曲线的最后已观测均值保持不变。它是**带早期药后响应信息的同曲线未来读出预测**。若状态感知方法也使用了早期 Y，收益不能归为仅凭给药前状态的预测收益。使用 h28/h36 去拟合再预测 h24 时只能称补点/插值，不能称向未来外推。时点之间不能随机打散来创造独立 n。

## 4. Sarin：具体输入、留出与不可识别层级

### 4.1 参考表达与真实功能 Y 的实际连接

参考表达是平行 24 h 无药组，不是同一细胞 t0：

| 状态 | 可作有来源群体参考的 RNA sample_id | 药后 RNA |
|---|---|---|
| A549 parental | GSM2892607、GSM2892612、GSM2892615、GSM2892622 | 11 µM：GSM2892608、GSM2892613、GSM2892616 |
| A549rCDDP2000 | GSM2892609、GSM2892614、GSM2892617、GSM2892623、GSM2892626 | 11 µM：GSM2892610、GSM2892618、GSM2892620、GSM2892624、GSM2892627；34 µM：GSM2892611、GSM2892619、GSM2892621、GSM2892625、GSM2892628 |

目标读数位于 `Sarin2017_Annexin_source_values.csv`，不是上述 GSM 的逐样本配对。具体 record_id 为：

- A549 11 µM：`Sarin2017_Fig5a_A549_11uM_sourceRow4`、`Sarin2017_Fig5a_A549_11uM_sourceRow5`、`Sarin2017_Fig5a_A549_11uM_sourceRow6`。
- 耐药 11 µM：`Sarin2017_Fig5a_A549rCDDP2000_11uM_sourceRow4`、`Sarin2017_Fig5a_A549rCDDP2000_11uM_sourceRow5`、`Sarin2017_Fig5a_A549rCDDP2000_11uM_sourceRow6`、`Sarin2017_Fig5a_A549rCDDP2000_11uM_sourceRow7`。
- 耐药 34 µM：`Sarin2017_Fig5a_A549rCDDP2000_34uM_sourceRow4`、`Sarin2017_Fig5a_A549rCDDP2000_34uM_sourceRow5`、`Sarin2017_Fig5a_A549rCDDP2000_34uM_sourceRow6`。

三条件均为 24 h；单位为相对各自 untreated 的 fold。附表行号不证明跨条件重复配对。原附表打印均值与打印逐值的轻微舍入差异应保留已有说明，不为了统一改原值。若本轮使用重算均值，显式命名为由公开逐值重算，并保持同一规则。

基因包实际保存 22×32461 的目标轴及缺失状态；12,070 个严格唯一探针基因可以直接定位，16,852 个轴内基因有至少一条已解析探针关系。多探针未自动平均，未解析基因保留缺失，未进行新归一化。原值是已存储的背景校正、分位数归一化微阵列 VALUE，log base 未明确注释。不能把共同基因 ID 当成共同数值分布。

### 4.2 可执行用途

1. **A/C 的固定条件状态挑战**：将 parental 11 µM 的 3 个值和 resistant 11 µM 的 4 个值整体作为终点评价。输入可为有来源类别或上述冻结群体参考 RNA；比较同一预测器使用状态特异参考与固定参考。两个输入分支的预处理、药物信息、协议和终点出口保持一致。该设计不需要逐细胞 t0，但必须禁止把对应 11 µM 药后 RNA 当成干预前输入。
2. **B 的有限剂量检验**：用已有 11 µM 条件开发，完整保留 resistant 34 µM 的 3 个值作为新剂量评价。最简单基线为 resistant 11 µM 的训练均值保持不变。只有一个留出剂量，因此仅能报告这个剂量上的迁移；不证明任意浓度关系。已用该折开发数据的 resistant 11 µM 不能再充当独立同折对照成绩。
3. **D 的两剂量选择挑战**：若要评价 resistant 的 11 vs 34 µM 选择，将这两个条件的全部 7 个 Y 一起留给同一外部冻结选择器。基线固定选 11、固定选 34、随机选择、直接评分/枚举。两条件是同状态、同时间的 fold 目标；可检验相对诱导这一目标的选择，不转写为绝对凋亡比例最优。不能把第 2 项的训练点加测试点混成独立策略比较。

同一 10 个值不能既用于三项分析的反复选法，又被称为三份独立验证。可以预先声明互不混淆的回顾性分析；若只选一个后续确认用途，应保留其整组标签不参与开发。

### 4.3 现有缺口的性质

- **实现/方法适配**：参考微阵列的合法聚合、数值尺度适配、缺失 mask、类别与参考输入的接口、fold 出口。完成这些不会自动获得效果证据，但不一定需要新增湿实验。
- **缺少配对信息**：RNA 样本和功能重复之间没有已证实的逐样本连接，不能按后缀 `_1` 或附表同一行号强行拼对，也不能把 3/4/3 个值复制到所有参考细胞/样本。
- **表达超出类别的识别缺口**：只有两种群体参考与类别共同变化；功能结果在状态—剂量层面。能检验整个 RNA 表示包相对固定参考，不能把收益唯一归因于类内 RNA 细节。补充条件是有实际对应 Y 的额外表达变化，而不是简单增加未配功能的 RNA 样本数。
- **绝对比例缺口**：缺相应 untreated 的绝对 Annexin 值，不能把 fold 除以任意常数、或直接送入限定 [0,1] 的比例头充当绝对凋亡概率。
- **来源迁移边界**：原 RNA 与功能来自不同研究记录的同名细胞/协议条件，不是同实验同样本；培养历史、密度与维持药物到挑战的洗脱信息仍有约束。保留这种转移假设并不妨碍有限挑战，但不能伪称精确实测配对。

## 5. coverage-v4 中能落地的条件块

### 5.1 同协议药物选择：两个 Macaya 块分开

| 实验块与候选 | condition_id | 实际可用 observation_id |
|---|---|---|
| S2C：Trametinib 0.5 µM、24 h | `condition:5b939d27f0bf5b86e77d9158` | `observation:d31eb03cae0c0c52e4f746fb`；`observation:618cf0e295d728546d0c8282`；`observation:70c962c25931f7ea3424470d` |
| S2C：Lestaurtinib 0.625 µM、24 h | `condition:25c417181bfc81e4509c2355` | `observation:5eb42156757e9e27a37d603f`；`observation:bd44a4dc068f1a453fccd8d0`；`observation:8a85fafeb69f1784de3670e0` |
| S3DE：Trametinib 0.5 µM、24 h | `condition:dbfe73bf236863f0d10a3046` | `observation:c6b26ad50838c209bed4bf80`；`observation:a8e4b787e3f1afd05d3423f4` |
| S3DE：Midostaurin 0.625 µM、24 h | `condition:d6c4a9bc645c7cedb76a8a11` | `observation:bacc4b1b198c0fb2a7296427`；`observation:4ebd10c129567f80bfa867da`；`observation:46053fca35ac6d35071402a3` |

S3DE 第三个 Trametinib 源位置 `observation:7668b44d1738df2be4c69ecb` 是 missing，不能补值或计为第三个实测。S2C 与 S3DE 虽共享 Trametinib，仍分别报告同实验块的候选选择；跨块药物比较需要处理块差异，不能只因观测定义 ID 相同就把所有均值直接排一张排行榜。

每个块可整体保留为冻结预测器/选择器的 C/D 检验；固定候选、随机选择、直接评分均是合法对照。这些块只有一个来源定义的参考状态，不能独立证明状态增量。若将 S2C 用于开发、S3DE 用于评价，需注明 Trametinib 已在开发出现；这不是新药全部留出。两块同属一研究，也不是来源留出。

### 5.2 固定时间剂量块，不受“14 条”筛选完全代表

| 来源/处理条件 | 真实 observation_id | 可做的最小组件检验 |
|---|---|---|
| 6k，5 µM、24 h | `observation:f1d3829177e790e0a1712de0` | 与同源 10/20 µM 构成固定时间剂量块 |
| 6k，10 µM、24 h | `observation:e865896e5dfd604022ea5e17` | 可留整条件作剂量补全/外推；单图不是逐生物重复 |
| 6k，20 µM、24 h | `observation:8432ad2462f741a98872d8a4` | 化学身份未关闭，精确结构驱动结论仍需限定 |
| AS-E，5 µM、48 h | `observation:fdc3333ae87d60baf2bf0970` | 固定 48 h 的剂量响应/选择 |
| AS-E，7.5 µM、48 h | `observation:8ede5133ab1e4c9107c0a241` | 可用 5/10 µM 开发、此点留作插值；不能称外推 |
| AS-E，10 µM、48 h | `observation:ec69c6ff43bafd9f8a7f4587` | 可用 5/7.5 µM 开发、此点留作剂量外推 |

6k 的未决化学结构不使同一源化合物下的“剂量读数可读取”失效。固定化合物的剂量基线可以不依赖完整 SMILES；若用结构预测器，须先解决结构或采用预先登记的输入不确定性分析。AS-E 48 h 被当前 24 h Tahoe 生成器拒绝，是当前模型覆盖缺口；已有 48 h Y 仍可支持独立的固定时点剂量检验。不能仅改 guard 就说 P 的 48 h 表达预测获得了监督。

剂量预测最简单对照为开发条件的常数/最近剂量均值；插值方案可预先固定线性插值且报告只有一个被留点。剂量选择应把整个候选块作为同一冻结选择器的评价，不把用于拟合的两个点和测试点混称为三个独立留出选择结果。若使用“始终最高剂量”作为对照，也应事先定义；在未定义成本或毒性目标时，不凭空加入惩罚来制造策略优势。

MCT2014 的 `observation:618ffc8355b9fa154f574ee5`（24 h）与 `observation:390969baa3cfbb36b2aad759`（48 h）是同 9a、0.1 µM 的不同读出时点。它们可作为特定条件的时间响应材料，仍没有“24 h 撤药后在 48 h 的 Y”。未决结构和缺少撤药响应是两种不同问题。

## 6. 新增共同终点证据与仍需扩展的实测对比

### 6.1 Hastings 独立复审：局部 E08 已有真实证据

原文为 Hastings et al. 2020，[DOI 10.7554/eLife.53367](https://doi.org/10.7554/eLife.53367)。本轮亲读 [PMC7282820.xml](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5/pulse/PMC7282820.xml) 的 `s2-1`、`s4-2`、`s4-5`、`s4-6`、`fig1s1`、`fig1s2`，并查看两张原图。**未发现需要修改当前冻结包的实质数字或时间对应错误。**

| 字段 | 2 h pulse 后去药 | 持续暴露 |
|---|---|---|
| 共同背景与剂量 | A549；cisplatin 5 µg/mL | 同左 |
| 真实给药时间 | t=0 开始，t=2 h 去药 | t=0 开始，持续至 72 h |
| 共同观察终点 | T*=72 h；去药后名义随访 70 h | T*=72 h |
| 观测 ID | `Hastings2020:Fig1S2:A549:pulse_2h_then_drug_removal:h72:relative_cleaved_caspase3` | `Hastings2020:Fig1S2:A549:continuous_72h:h72:relative_cleaved_caspase3` |
| 图示均值 | 1.2693 | 2.9116 |
| 数字化范围 | 1.2025–1.3361 | 2.8448–2.9784 |
| 作者测量定义 | relative cleaved caspase-3，MAGPIX 蛋白信号 | 同一来源同一图轴 |

时间原点来自 Fig. 1—figure supplement 1A：图上两臂均从 0 到 72 h，pulse 的 0–2 h 为给药、随后明确示意去药。它不是“去药后再数 72 h”，也不只是两个不同观察时间。因此与 Nav-Gal 连续暴露曲线的证据性质不同。去药次数、换液体积及残留药物清除未在所查材料中被定量说明，不能加写成精确残留为零的药代模型；但这不否定已记录的实验去药方案及其共同终点结果。

数字化来源为 Fig. 1—figure supplement 2 最后一个 Relative Cleaved Caspase 3 panel。原图为 3981×5264 像素；脚本取 0/8 刻度中心 y=5071/4352.5，两个 72 h 标记中心分别为 (3839.5, 4957.0)、(3839.5, 4809.5)，按原图线性轴换算。两点完整复算与冻结 CSV 一致，calibration 也一致；8 项来源/输出 hash 均通过。独立目视范围 pulse [1.10,1.40]、continuous [2.75,3.10] 与结果一致。这里是可复现读图加独立粗区间检查，不是恢复了仪器原始观测。

以下尺度与统计约束必须保留：

- `s4-5` 报告按各 analyte 的 0 h median 归一化并作 log，随后描述 PCA；不能把该分析变换直接套在当前线性 relative 图上。该图的逐点归一化分母及两个实验臂的原始归一化数表尚未独立恢复。当前合法目标是**该来源发表图的 relative signal**，不是重新计算的原始浓度或跨研究统一 fold。
- 图示两数可用于相同发表轴上的点预测误差和两方案顺序挑战，不把差值解释为凋亡百分点或绝对 caspase 活性差。即使两条读图范围分离，也不能以它们新作生物学显著性推断。
- Fig. 1—figure supplement 2 图注为 n=3、mean±SD。[作者透明报告原 DOCX](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5/pulse/elife-53367-transrepform-v1.docx) 进一步说明实验一般为 biological triplicates，并列出例外；本 multiplex 比较不在例外列表。该 DOCX 已直接读取，不能把 Fig. 1—figure supplement 1B 的单细胞例外移植到此处。
- `digitization_lower/upper` 来自标记中心 ±4 像素、轴刻度 ±2 像素的分析者假设，不是作者 SD/SE/CI。2 个均值不是 6 条已恢复的生物重复。官方 Figure 1D 源表也不能仅凭同名蛋白列被认作此 supplement 两臂的原始重复。
- cleaved caspase-3 是本图的凋亡相关蛋白读出，不是 Annexin 阳性比例；不能接入现有百分比评价器并乘除 100。没有这些终点的同样本 RNA，也没有多状态、剂量或药物变化来支持相应增量结论。

**可执行的局部检验：** 将上述两点整体作为一个来源内、回顾性的时长挑战，向同一冻结预测器提交两种完整方案；在它支持 published-relative-protein 单位时计算两点误差，否则只使用预先定义方向的排序分数作顺序挑战。最简单选择对照为固定 pulse、固定 continuous、随机二选一，以及同一评分器的直接枚举。不能看过该图后挑一个固定策略称其为新发现；不能由两点拟合又在同两点评价来证明泛化。

这是**局部 E08 有条件覆盖**：共同终点和真实两种暴露方案有证据，精确尺度、洗液细节及泛化范围的限制被显式保留；不是 Annexin G 已训练，也不是 E09 全目标完成。

### 6.2 时长组件的最低信息结构及当前剩余范围

在一个明确状态、同药同剂量、同检测定义下，实际取得以下两臂在共同 T* 的 Y：

1. 暴露至较早时点后按已知协议撤药/换液/停止，再观察到 T*。
2. 持续暴露至 T*，或另一个明确暴露时长，按同一终点检测。

必须同时知道实际停药时间、停药后如何处理、共同终点及分母/门控。一个来源中的发表均值可先承担点预测/顺序挑战，不必先取得逐细胞 t0 或所有原始重复。但从另一篇研究单独找到一个短暴露值，不能自动与当前 Nav-Gal 长暴露均值拼为可比双臂；来源可比性或同研究对照必须有证据。

**Hastings 已在相对 cleaved caspase-3 这一测量定义上提供上述局部信息结构；不能再把时长组件说成完全没有实测对比。它仍不是完整主问题已经覆盖。** 要检验状态条件化的时长选择，应至少在两个有定义的状态上获得共享的这组比较，且决定前可用的状态信息有明确来源。不要求两状态真的选出不同方案；相同最优方案也是合法结果。

后续寻找的重点应转为可比较 Annexin 或可验证观测桥接、多个状态共享时长候选及完整联合评价块，并补充有来源的去药细节。已有蛋白读出应保留使用，不能因其不等于 Annexin 而抹去；也不能将其升级为已找到所有缺失 Y。本文不要求新增湿实验。

### 6.3 连接完整联合目标的一种小型实测设计

一种节省条件、无需完整笛卡尔积的示例为：两个预定义状态共享四个候选，并在同一 T* 检测：

- 一个药物 A、剂量 c1 的短暴露—撤药方案作为共同锚点；
- 同药 A、剂量 c2、相同短暴露/后续协议；
- 药物 B、其合法剂量、相同短暴露/后续协议；
- 药物 A、剂量 c1、持续到 T* 的方案。

这提供药物、剂量、真实暴露时长变化及状态×候选的共同评价块；**2×4 只是信息结构示例，不是所有方法必须满足的样本量、功效保证或当前已有数据。** 可由等价的、有共同锚点且来源可比的公开条件块支持，不要求字面复制这个设计。若拟训练高容量模型，开发数据与确认范围仍需另外匹配，不能把少量块视为充足训练集。

冻结完整预测—选择程序与无状态/固定剂量时长/直接评分对照后，这一类块才允许在同一真实 Y 上检查各环节连接。状态、药物或时长各自出现在彼此不兼容的数据源里，不能替代这种连接。

### 6.4 关闭“表达超出类别/群体平均”需要什么

需要在相同大类背景及可比候选下，有额外、确实变化的状态表达或状态摘要，并有对应的真实响应。对应单位可以是有明确来源的独立细胞群体/批次、匹配的平行培养或已定义亚群，不强迫破坏性 RNA 测量发生在同一真实细胞 t0；也不普遍要求全转录组。

关键是能够比较：

- 药物/剂量/时长＋类别/群体平均；
- 同样信息再加实际变化的表达/状态摘要。

两分支需要共享同一评价条件与允许信息。表达样本更多但所有 Y 都被复制自同一条件均值，不满足这一要求。若仍声称逐细胞命运，而不是群体条件分布，则还需要适合那个更细声明的真实连接。

对 Tahoe，多板相同处理的候选覆盖已由元数据窄审计确认，详见下节。仍需核查实际对照/药后表达分布、协议及板位/QC 混杂；不能把“能找到比较条件”改写为“已验证生物状态功能变化”。

## 7. Tahoe 元数据窄审计：覆盖已确认，表达与生物学效应未验证

本轮亲读 [audit.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/tahoe_state_design_round5/audit.json)、[repeated_treatment_groups.csv](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/tahoe_state_design_round5/repeated_treatment_groups.csv)、[condition_design.csv](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/tahoe_state_design_round5/condition_design.csv)。在三张小文件内重算 1336 条总条件、1305 条 eligible 非 vehicle 处理、14 板与 28 个 vehicle 条件；处理均为 24 h，1305 条均具有 `full_isomeric_smiles_v2` 身份。

按完整化学身份＋dose_um＋duration_h 分组得到 1128 组，其中 154 组在多个板出现，共 325 条条件；154 组的所有成员 ID、浓度、时间与板数均逐组核对。另有 6 个仅在同板重复的组，共 12 条条件，解释总重复差额中不属于跨板的部分，不能将它们混入跨板 154 组。

一个具体比较候选是 Captopril、0.05 µM、24 h：`tahoe:smp_2400`（plate10）和 `tahoe:smp_2662`（plate13），完整化学身份为 `full_isomeric_smiles_v2:00c9a1cecef5d9d06ffd86f6f30f1c98b24d35f5c1b7883135030bbbe7f1c8f9`。这只证明存在相同处理的跨板记录，不证明它们是同一细胞初态的不同响应或已确认的生物重复。

审计登记每板 2 个 vehicle；源 control_links 的 2616 条关系均为 `same_plate_control_candidate`，`individual_cell_pairing=not_available`。这些关系来自主代理对源 metadata 的读取；本轮读取其登记结果，没有另读源 control_links 或大型表达矩阵。对照为研究采样时刻的同板群体，不是同细胞给药前测量；`source_sample_unit` 的独立生物重复资格尚未建立。

下一步可以形成如下**有数据条件的设计候选**：

- 比较训练集 pooled control reference、对应板的 control 均值、对应板的完整 reference distribution；是否保留分布而不仅均值，是一个可明确比较的表示问题。
- 加入来源/plate-only 与可获得 QC 摘要基线，核查收益是否主要来自板位或技术质量。留整板时，不能训练一个已见板的 one-hot 效应后将其当作未见板已知参数；只使用评价时合法可得的 QC/参考信息。
- 同处理跨板留出可以检验参考条件化或跨板迁移，不自动等于未见药物。若要测新药，须按完整化学实体隔离它在所有板和剂量的相关记录；不把当前 role 列直接当作新实验防泄漏已经完成。
- 若允许读取评价板 vehicle 表达，必须将其列为双方共享的可用参考输入，明确属于有目标板参考的任务，不能再称完全没有目标域信息；评价处理的药后表达仍须完整保留。
- 输出首先是药后表达分布的预测检验；只有另有可对应的实测凋亡 Y 才能延伸到功能状态增量。14 板不能重命名为 14 个生物状态，154 组不能重命名为 154 个新的生物学重复。

因此，这项更新修正了“元数据覆盖未核”的状态，但没有关闭 A2 的功能识别缺口，也没有声称任何分布模型已胜过平均表征。实际表达分布、协议差异、QC 与板位混杂仍需检验。

## 8. 执行优先级与保留目标

1. Nav-Gal 读取/评价入口已完成真实 readiness，下一步才是生成并提交适用的冻结科学预测。选择上文有限留一药参考检验或完整四点外部挑战时，明确标签用途；现有入口不会替模型证明其训练历史和终点适用性。
2. Sarin 先允许有来源的类别/参考表示挑战，修正接口与尺度后再评价 RNA 表示包；不为了接口方便生成逐细胞伪标签。
3. coverage-v4 的 Macaya 和固定时间剂量块用于明确的 C/D 或 B 组件；不要求四固定分区，也不跨协议拼假候选。
4. 先保留并使用 Hastings 已有的局部共同 T* 蛋白信号挑战，再补多状态、可比功能定义与联合候选连接；同时在 Tahoe 候选元数据基础上核查实际表达、参考分布/均值及 plate-only/QC 对照。缺失的连接不能由更多模拟、同曲线读数或相同标签复制关闭。

本轮 E07/E08/E09 状态应分别记录：

| 验收项 | 本轮证据 | 当前判断 |
|---|---|---|
| E07 方法输入 | [Nav-Gal 评价脚本](../scripts/evaluate_nav_gal_challenge.py) 的真实 `load_challenge → readiness` 已由本审查运行，返回 `input-ready/prediction-missing`，4 个 active/2 个 vehicle 主观测，`scientific_model_predictions_evaluated=false` | 该外部预测评价入口的真实输入读取已通过；完整科学模型、Sarin 与 Hastings 消费路径不能据此一并判通过 |
| E08 共同终点 | Hastings 同起点、同剂量、2 h 去药 vs 72 h 持续、共同 T*=72 h、2 个相对蛋白均值 | 局部有条件覆盖；Annexin 比例与多状态联合选择仍未覆盖 |
| E09 全目标覆盖 | Nav-Gal/Sarin 的状态功能材料，核心药物/剂量块，Hastings 局部时长，Tahoe 候选参考分布比较分别具备部分依据 | 完整主线仍未通过；缺的是兼容的状态增量和完整链连接，不是“完全没有时长数据” |

[评价入口说明](NAV_GAL_CHALLENGE_EVALUATION_2026-10-10.md)记录主代理复跑 19 项合成测试通过。该成绩验证软件算术与输入守卫，不是实测科学模型结果；本审查未重复运行这 19 项测试，只实际运行真实数据的读取/readiness。E07 通过的范围不能扩写为预测效果通过。

整体状态仍是：**有真实材料支持分层、可证伪的实验推进；完整主线覆盖未通过，实际方法效果尚未得到本轮验证。** 这个判断既不依赖“立项所以一定成功”，也不把目前未具备所有环节等同于研究不能继续。

## 9. 文件、版本与核验范围

直接依据：

- [毕业论文研究架构](G:/研究生科研文件/毕业论文/研究架构.md)：§2、§3、§4.2/4.3、§6。
- [完整主线证据要求](DATA_FEASIBILITY_EVIDENCE_REQUIREMENTS_2026-10-10.md)：A/B/C/D、发表均值、共同 T* 与信息不确定性契约；本文按本轮新证据更新 E08 覆盖状态，未修改该旧文件。
- [Hastings 冻结挑战清单](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5/cleaved_caspase_challenge_v1/manifest.json)、[观测表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5/cleaved_caspase_challenge_v1/observations.csv)、[契约](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/duration_round5/cleaved_caspase_challenge_v1/contract.json)；纯提取逻辑见 [extract_hastings_duration_challenge.py](../scripts/extract_hastings_duration_challenge.py)。
- [Nav-Gal 挑战契约](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal/challenge-v1/challenge_contract.json)、[24 h 实际表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal/challenge-v1/primary_24h.csv)、[独立复审](NAV_GAL_CHALLENGE_REVIEW_2026-10-10.md)。原文为 González-Gualda et al. 2020，[DOI 10.1111/acel.13142](https://doi.org/10.1111/acel.13142)，原 XML §2.3、§4.4、§4.6、Fig. 4。
- [Sarin 基因包清单](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_gene_round4/manifest.json)、[样本元数据](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_gene_round4/GSE108214_aligned_samples.csv)、[条件对应](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round2/Sarin_condition_correspondence.csv)、[功能逐值表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round2/Sarin2017_Annexin_source_values.csv)。功能原文为 Sarin et al. 2017，[DOI 10.1371/journal.pone.0181081](https://doi.org/10.1371/journal.pone.0181081)，Fig. 5a 与 S8；RNA 原文为 Sarin et al. 2018，[DOI 10.3390/ijms19030767](https://doi.org/10.3390/ijms19030767)，GSE108214。
- [coverage-v4 清单](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v4/manifest.json)、[覆盖摘要](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v4/coverage.json)、[条件清单](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v4/condition_inventory.json)。具体剂量和观测 ID 从其绑定的 r4 条件、干预、观测 Parquet 读取；两张干预组成行不会重复计为两个组合 Y。

原 Round5 已重核三个清单所绑定的 **36 项来源/输入/输出 SHA256，0 项不一致**；本次新读 Hastings 又核对 8 项、0 项不一致。Nav-Gal readiness 的来源检查实际通过。清单版本为：

- coverage-v4：`42f249b45263f78345230fe8486b24ceeda24b7ad4d269142a208fde4e8ba2e6`。
- Sarin state_gene_round4：`845fd9d573413e73859dd7c2f1f2b30c4d5e87d2d0e219983485825ae00f21c6`。
- Nav-Gal challenge-v1：`2cf25c43750345816c98f18f3eb7b79ad2f0d19d2a05a4b081ade98e1fb9faa0`。

新增 Hastings manifest 为 `d101ed2f07f71d15888074d340fa60ad5d07baf4d541370203a6a0632c818b27`；透明报告原 DOCX 为 `23868516c7d019c317f346b7142ef3a1177508f06f4d21c513469f46cd0389f8`。Tahoe 小文件本轮哈希为：audit `bcc27e97aaf07c72ff0e89ef4f86400c3b64af76e061dfda42bc7d5e92d611cc`，repeated groups `e2e198e73f6cfcd8e948c13821f17d4be2268b770dc18716c2592bb505e37de8`，condition design `fcf32fd988e8958181afba9e97c8caeec5c9e81fdc9910ae8b33ab2b4cdf7fdc`。原审计所列源 Parquet 哈希由主代理生成，本次不冒称另读大型表达或重新运行其整个构建过程。

这是证据、数值与读取入口核验，不是科学预测/训练运行结果。已有源码和数据更改属于其他工作；本次只修订本文，未覆盖冻结包或其他文件，未提交或推送 Git。
