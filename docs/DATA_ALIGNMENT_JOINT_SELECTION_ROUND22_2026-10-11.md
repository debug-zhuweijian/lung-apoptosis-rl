# Round22：独立的回顾性药物—剂量—完整协议联合选择评价

本轮补齐限定 Viel 来源的 D 评价入口：最终同一块含 2 药×2 剂量×2 协议的 8 个真实发表条件，所有方法在同一完整域比较。旧 `evaluation-plan-v2` 仍是每块同药同剂量的双协议评价，两套角色合同独立保存。实际运行通过的是新元数据构建、源绑定读取、fit/selection按角色读取和反例测试；真实 P/G/Q、奖励资格及最终效果没有执行，整个主问题仍未通过。

## 原件与可比范围

原研究为 Vielhauer et al., *Evaluation of a reductively activated duocarmycin prodrug against murine and human solid cancers*, Cancer Biology & Therapy 14(6):527–536 (2013), DOI [10.4161/cbt.24348](https://doi.org/10.4161/cbt.24348)，[PMC3813569 原全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3813569/)。本轮亲读既有正式全文抓取、图3C原图和图5C左侧官方高分辨率拼接图，版本及定位见[原件定位回执](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/source_locators.json)。没有以搜索摘要或旧报告代替原文。

| 对齐项 | 原件证据 | 本轮处理 |
|---|---|---|
| 背景 | Cell lines；Methods/Annexin V apoptosis；图3C、图5C左 | A549 常规培养群体，T25，名义种子 3×10^5，处理前培养24h；保留 `a549_parental_standard_culture`，没有匹配初始RNA |
| 药物与剂量 | 图3C、图5C左的坐标 | BocNHO-CBI-indole2 与 seco-CBI-indole2；各50pM、500pM、10nM、40nM、1µM；保留原20物理候选ID、来源名义结构、批次未经认证状态 |
| 时钟与完整处理 | 图3图注；图5图注；Washout experiments | 连续24h；或暴露6h→倒液/清洗/新培养基→恢复18h；共同T24，不把6h当观察终点 |
| Y坐标 | Results/Annexin；图3图注；图5C；Annexin方法 | 早期＋晚期Annexin阳性，占终末收获亲本群体的比例；坏死不相加，growth/trypan blue不替代 |
| 图间依赖 | 两图各有自己的vehicle；源条件表 | 保留2个不同控制ID与2个图ID；同图多条件及跨角色共享对照，不声称瓶/批配对或独立重复 |
| 重复与不确定性 | Annexin方法1µM的n=2排除例外；Washout方法n=3 | 连续1µM n=2；洗脱1µM例外适用范围不清仍为缺失。每Y权重1，不按n加权；已有读图区间是像素敏感性范围，不是SEM/CI |

原文 Results 的洗脱段落引用 Fig3B 与 A549 图注明确的 Fig3C 不一致。本轮按图3C的 A549 标签、图注与Methods定位，不引入EMT6数据。图5C只取左侧T24，右侧T72不混入。

跨图可做**发表条件均值、协议特定终末收获群体**层面的描述性选择比较。洗脱6h时浮细胞是否重新保留未报告，两协议终末分母可能因倒液改变，因此不能把它写为初始全细胞群体凋亡/死亡率的因果比较。源控制的洗脱sham过程也不能补造。既往公开均值已经被审阅，新划分不是前瞻盲测。

## 新角色合同与公平比较

按每药五个名义剂量的数值顺序，极端两剂量用于fit、中位剂量用于selection、其余两剂量用于final；规则只读取候选与条件元数据、观测ID。每个药＋剂量的两协议一同分配：

| 新角色 | 剂量 | 条件均值数 | 可用范围 |
|---|---|---:|---|
| fit | 50pM、1µM | 8 | 新实验的G/simple拟合、fit-only变换与模拟Q训练域 |
| selection | 10nM | 4 | 同2药×2协议的校准、固定规则选择或奖励验证；未冻结任何新效果阈值 |
| final_test | 500pM、40nM | 8 | 同2药×2剂量×2协议、一次固定模型的完整候选选择评价 |

这检验的是两种已见药物、同背景、未用于本实验fit的两剂量条件。它不检验未见药物、高维状态增量、任意时长插值、多步滚动或真实序贯。P训练仍须使用对应的既有合法划分；新角色不能被用来覆盖冻结训练数据。

源绑定[view-v1](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/view-v1/view.json)完整保存20候选、20唯一Y ID、完整协议、source figure/published n/status及共享控制。读标签与元数据入口分开；预检只hash监督字节，不解析Y。旧v2 SHA256固定为：plan `c5621b3d4efbaaf46fdfba2a2d8fd23edd6527fdf21e4bab74f1089f4133a6d8`；manifest `87bf7f09cbd913d69faabc386ecbf50fd757a9ce4bac9dbe0913b0fae0c599a4`。

四类策略共同绑定一个信息工件、响应特征工件、终点目标和固定G反馈，每种模型对整个块只允许一个模型工件：

1. **固定条件家族**：固定药物为源compound ID字典序首项，固定剂量为该块最低名义剂量，固定暴露为24h；另含三项同时固定。允许子集在view中预先保存，先有完整8候选G分数再对合法子集枚举；结果仍相对同完整8域评价，不给缺分候选填0。
2. **简单候选排序**：必须提供同8域、相同信息访问权限、同终点坐标的冻结简单模型分数，并明确实际使用特征。只使用药物或均值表征的消融仍需沿同信息权限运行。
3. **单次直接终点评分**：完整域固定G精确argmax；它是给定评分器内的必要有限枚举对照。
4. **学习候选价值**：提供完整域冻结Q分数；训练反馈工件必须与直接G同一。未来RL/Q保留原终点奖励，不新增药物、时长或风险成本。

另报告对完整域均匀选择的精确期望作为数据无关诊断，不通过抽样制造重复。模型相同最高分保留全部并列候选，按固定均匀权重计算期望结局，不能看Y解并列。任一方法缺少完整分数、错误view/role、旧fold identity、声明的不同信息或G反馈、候选逐行模型字段都会拒绝。裸分数及自洽哈希不能检测未如实披露的跨fold拼接，也不独立证明模型确实按声明运行；实际生产路径和模型角色元数据仍须独立审阅，输出始终 `execution_certified=false`。

## 实际指标与有限样本边界

[实现](../src/lung/data/joint_selection_evaluation.py)接通显式冻结工件后才进入role对应Y读取，计算各策略所选候选的期望发表Annexin比例、相对同完整已测域最佳者的regret、模型分数MAE、描述性均方误差`mean_squared_error`和成对排序一致性。MSE与MAE都对同一完整候选域中已冻结的source group mean逐候选平均，不以MAE作为条件期望最优性证明；新增MSE不改变选择目标、旧指标或旧奖励资格规则。最终8候选有28比较对，但这些对不能当28独立样本。固定条件/均匀诊断的预测误差描述的是其共同G评分器，不是约束策略的额外预测能力。

读图区间敏感性给出选择值上下界、regret保守下界及区间盒内最坏上界、不重叠区间的稳健排序比较数，以及仍可能为最佳的候选。区间不假定独立，不用于制造生物置信区间；共享刻度/控制导致相关性。指标在合成fixture上核对并列、精确有限期望和全部区间角点，实际源Y评测尚未调用。

最终只有8个发表条件均值、4个药＋剂量组和2幅来源图。原始重复与瓶/批标识未取得，无法计算可靠生物标准误、显著性或统计功效；不能用细胞数、候选数、28成对比较、旧10 folds或模型模拟回合数补充独立样本。有限样本允许检验该冻结配置是否选错候选，但不能据此建立稳定泛化或预先保证Q/RL收益。主问题的A549高维状态—表达—实测Annexin完整桥和独立来源效果仍是关键缺口。

## 可执行顺序与当前证据

本轮已实际执行`build`与`preflight`；[预检回执](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/metadata_preflight.json)确认8/4/8、20唯一条件均值和最终2×2×2域，Y未解析。[阶段测试回执](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/implementation_receipt.json)区分真实元数据验收和模拟指标fixture；随后[role-API回执](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/implementation_receipt_final_role_api.json)记录新标签接口，最新[public final隔离与MSE回执](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/implementation_receipt_final_mse_v2.json)记录78项定向测试通过（5.31s）。首次CLI缺少项目根sys.path，在写view前失败；已按项目脚本惯例修复并保留[失败记录](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/cli_first_failure.txt)。

公共`measured_units(role)`对冻结源数组逐记录扫描，仅解析明确role的记录；不将整个JSON解码再过滤。fit/selection的源读回实际验证[8条fit、4条selection](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/real_fit_selection_label_readback_v2.json)的候选身份、fraction和读图区间顺序；guard记录final数值解码次数为0，未调用final评价或拟合。未选择记录仍须扫描原字符与ID，不能声称源文件字节没有被读。新helper首次项目层级路径取错，在任何标签读回前失败，修正并保留[记录](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/role_readback_first_failure.txt)。后续标签消费入口可明确请求fit或selection，公共API拒绝final，final只进入显式冻结评价。

取得所需模型运行授权及相容工件后，顺序是：按新角色产生P/完整协议特征→只用fit的G与简单基线→selection按预先固定规则验证→固定相同G反馈、拟合私有Q→冻结所有模型及最终8候选分数→显式读取final并枚举比较。新角色尚没有对应的G/simple/Q生产工件；旧fold模型不能因可读取就成为新实验模型。标准连续24h P描述符不是实测洗脱RNA，协议外推需单列检验。真正匹配的高维RNA—Annexin监督与序贯标签仍不能由此新视图补造。

未来现成冻结分数可通过以下入口评价；本轮没有执行最后两步：

```powershell
python -X utf8 scripts/audit_joint_selection_evaluation.py preflight --output experiments/<new-run>/joint_preflight.json
python -X utf8 scripts/audit_joint_selection_evaluation.py freeze --packet experiments/<new-run>/joint_scores.json --output experiments/<new-run>/joint_frozen_receipt.json
python -X utf8 scripts/audit_joint_selection_evaluation.py evaluate --receipt experiments/<new-run>/joint_frozen_receipt.json --output experiments/<new-run>/joint_metrics.json
```

packet字段、全部角色观测ID与分数所需候选ID见[输入合同](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/frozen_score_packet_contract.json)，其中不含伪预测。输出使用新路径，拒绝覆盖。实现与[毕业论文研究架构](<G:/研究生科研文件/毕业论文/研究架构.md>)§3候选完整协议、§4.3–4.4原终点价值及§6四类比较一致；架构、旧plan-v2、冻结split与reward均未修改。
