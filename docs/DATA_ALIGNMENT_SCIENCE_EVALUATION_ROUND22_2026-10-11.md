# 第二十二轮：科学连接原件与冻结评价入口

日期：2026-10-11。工作范围为定向原件查证、真实数值读回和评价准备。**完整研究目标仍未验收；本轮没有拟合、加载模型参数、真实模型推理或恢复训练。** 四假设的定义及范围逐项对照[研究架构§3、§4、§6](G:/研究生科研文件/毕业论文/研究架构.md)，没有修改架构、奖励、冻结划分或现有资格字段。

## Research Analysis

### Key Strengths

- FOXN3官方补充实际取得并完整读回，补充获取缺口已关闭。原文声明的DOCX为15,644,900字节，实际大小及MD5 `95b7487d67d4981ab86da41b5b245314`全部一致，ZIP CRC通过；7张图及4张表的文字已读。[补充文件](../experiments/20261011-data-alignment-round22/science_evaluation/sources/FOXN3_supplement.docx)、[读回](../experiments/20261011-data-alignment-round22/science_evaluation/sources/FOXN3_supplement_readback.json)、[全部文字](../experiments/20261011-data-alignment-round22/science_evaluation/sources/FOXN3_supplement_docx_text.md)。
- Sarin来源可保留为限定的A549群体状态功能挑战。2017原XML/原Annexin表及2018原XML/GEO字段均亲读；全部654,038个官方微阵列VALUE与既有原探针NPZ逐值精确相同，22×32,461共同轴矩阵实际读回，12,070观测列及20,391缺失列保留。[实际原件读回](../experiments/20261011-data-alignment-round22/science_evaluation/sarin_original_source_readback.json)。
- 新增[冻结结果比较模块](../src/lung/data/science_evaluation_comparison.py)和[CLI](../scripts/audit_science_evaluation.py)，补齐不同方法在同信息、候选和目标下配对比较的入口。77项相关检查通过，包括错误终点域、负误差、Y重复扩增、候选缺失、角色/折/坐标变化等反例。[测试回执](../experiments/20261011-data-alignment-round22/science_evaluation/pytest.txt)。这些是工程证据，尚无新的模型效果。

### Potential Weaknesses / Reviewer Concerns

| 核心连接 | 本轮原件判定 | 未完成条件 |
|---|---|---|
| FOXN3 A549 RNA—Annexin | 主文RNA为siFOXN3/siNC；主文Annexin为KO、OE或回补。新补充S4E的siKD凋亡实验明确是NIH3T3；S4A–C A549为体内肿瘤/蛋白，不是A549 siKD Annexin。 | 补充没有把A549 siKD变成KO/OE同状态，也未补RNA取样时钟；不能生成匹配高维功能Y。 |
| GATOR1 | 主文Annexin方法为50µM cisplatin24h；既有官方24 RNA条目均为BEAS-2B，正文出现A549不改变RNA归属。 | 本轮Springer SSL失败、EuropePMC包读取超时、NCBI小附件403，失败均保存。补充未取得，不能推定不存在更多数据。 |
| HM97662 | 保留6个A549 RNA与2个近似Annexin组均值，只有1个活性drug/dose/duration组合。 | 角色未分配、完整结构仍不可认证、flow/RNA完整协议等价未验证；RNA三重复不扩成6个Y。 |
| Sarin2017/2018 | 2个明确A549来源状态、1药、3条件、10个原Annexin fold数字；同来源、协议、剂量、24h可作跨研究条件对应。 | 未共享培养孔/重复ID，无t0实测，独立批次身份未知；微阵列观测出口及fold头尚缺。限定挑战不能替代完整高维状态、药物—剂量—时间选择。 |

FOXN3来源版本为PMC13339424.2、正文§4.1/4.7/4.14与Fig2A/3H/4J/6I；官方补充S4E是决定性排除证据。[原正文XML](../experiments/20261011-data-alignment-round21/a549_state_annexin_targeted_gap/PMC13339424.xml)。GATOR1正文§Cell lines、Apoptosis analysis、RNA-seq及官方PRJEB98686仅支撑所述范围。[原正文XML](../experiments/20261011-data-alignment-round21/a549_state_annexin_targeted_gap/PMC12824275.xml)。可定位原段落另见[locator JSON](../experiments/20261011-data-alignment-round22/science_evaluation/primary_source_locators.json)。不将获取失败表述为“数据不存在”。

#### Sarin如何合法检验，以及仍无法识别什么

2017 [DOI10.1371/journal.pone.0181081](https://doi.org/10.1371/journal.pone.0181081)，`sec004`/`sec009`/Fig5a及S8 DOCX第6表；2018 [DOI10.3390/ijms19030767](https://doi.org/10.3390/ijms19030767)，§4.2–4.3与[GSE108214](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE108214)。两文均明确ATCC A549及RCCL A549rCDDP2000、过夜贴壁、4h血清饥饿、无补充IMDM与顺铂24h。耐药维持培养含2µg/mL顺铂，维持到挑战的额外洗脱时长未知；不能另补无药历史。

| 实际条件 | RNA参考/药后 | Annexin原数字数 | 可开展的限定比较 |
|---|---:|---:|---|
| 亲本11µM×24h | 4/3 | 3 | 与耐药11µM共同条件检验状态信息 |
| 耐药11µM×24h | 5/5 | 4 | 状态比较；耐药内两剂量选择 |
| 耐药34µM×24h | 共享上述5参考/5药后 | 3 | 与耐药11µM同背景两候选 |

原终点是相对各自无药对照的Annexin诱导倍数，分母绝对数未提供。它是独立观测域，具有可检验用途；不能换成fraction，不能因非fraction或非同细胞t0排除整个来源。共同11µM条件下，冻结高维参考模型与药物信息模型对两个群体状态的预测一致性可以被该有限源数据否定。但只有2个状态、1个共同处理条件，不能把高维表示收益与state-ID效应分离，不能获得同状态内逐RNA重复的功能监督，也不能据此验证未见状态泛化。

“各自EC10”比较是亲本11µM与耐药34µM；应与共同11µM比较分开，否则状态与剂量同时改变。parental34µM缺失，因此不是完整2状态×2剂量网格。没有其他药物或时长，不能验证多药/时间联合选择。

[新限定范围manifest](../experiments/20261011-data-alignment-round22/science_evaluation/sarin_diagnostic_scope_manifest.json)保留3实际条件、原GSM/10个Y身份和两种可检验子范围：共同11µM状态比较、耐药11/34µM选择。两范围共享部分Y，不当独立实验。现有`training_eligible=false`和角色未分配保持；manifest未指派fit/selection/final，也未开启G。

接续[v2 manifest](../experiments/20261011-data-alignment-round22/science_evaluation/sarin_diagnostic_scope_manifest_v2.json)进一步绑定两原XML、原S8 DOCX、官方SOFT及RNA/Annexin/共同轴共11份原件和派生资产，明示原float64 microarray VALUE、log_base未知、NaN/mask与原样本顺序。显式源字段验证和5个篡改语义反例通过，v1保留；见[v2回执](../experiments/20261011-data-alignment-round22/science_evaluation/sarin_scope_v2_receipt.md)。新comparison允许`state_ID_only`作为声明的基线特征使用，便于区分高维信息与背景类别，尚无模型执行。

### Missing Related Work

本轮补的是理论原件，而非再做广域文献推荐。Gretton2012的MMD定义及定理、MOPO2020的模型误差—回报分析、P-DQN2018v1的动作价值定义已下载并亲读。[原件定位与本课题推导](../experiments/20261011-data-alignment-round22/science_evaluation/theory_locators.md)明确页码、假设和不能移植的保证。

- MMD比较群体分布，不能恢复实测个体coupling；板内细胞不能当生物重复。
- MOPO原定理有折扣无限时域、价值函数类与可容许误差上界，并使用惩罚奖励。当前项目不继承其保证、不改变奖励；借其分析分别验证P转移错误和G终点错误。
- 本课题的有限候选误差推导仅在全候选统一预测误差有界时得regret≤2epsilon；单次有限全集枚举是同一评分器内的精确argmax，Q增益要落到独立实测Y。

### Suggested Improvements

#### 四假设与完整目标的可执行关系

| 假设 | 必要性/监督对象 | 当前可证伪实验 | 尚缺关键环节与边界 |
|---|---|---|---|
| A状态增量 | 状态差异是否帮助条件分布预测/功能预测；比较同来源同条件，控制信息和混杂 | native请求可对full_reference/same_reference_mean与drug-only比较真实留出表达；Sarin共同11µM可作相对Annexin的有限状态挑战 | 匹配P尚无；Sarin微阵列E/D/P与原尺度G未运行，2状态不隔离高维相对state-ID稳健增量；高维A549功能增量仍未关闭 |
| B未见条件 | 模型在未见药、剂量、协议、来源上可能偏移 | 以冻结条件/药物/背景划分分别报告表达MMD-V、mean MSE；不同外推任务单列 | 相容新P、留出结果及迁移证据未齐；单源24h不提供任意时间/真正序贯标签 |
| C功能一致性 | RNA代理不自动等于实验定义的凋亡结局 | 冻结预测后分别比较Annexin fraction MAE、Sarin Annexin fold MAE、Martin原caspase AUC MAE | scope-bound G及独立校准未跑；来源连接及域相容性须验证，三尺度不能合并 |
| D联合选择 | 评分误差和选择偏差可能影响所选候选 | 同完整候选域、输入权限、原奖励，比较直接评分/枚举、适用固定候选和Q；在实测Y计算regret；新增Viel单模型2药×2剂量×2协议最终块 | 旧endpoint每final块仍2个协议候选；新joint view元数据验收通过但真实P/G/Q与功能选择尚未运行，高维状态及真实序贯的主目标覆盖仍未齐 |

原[endpoint评价器](../src/lung/data/endpoint_protocol_evaluation.py)能算单块冻结分数的MAE、顺序及选择差距；原[Martin评价器](../src/lung/data/state_function_evaluation.py)能在独立域内评估冻结caller分数。它们不认证P/G/Q执行。新comparison也保留`execution_certified=false`、`caller_metrics_only=true`、无生物CI/显著性及`whole_project_qualified=false`。

新入口实际预检[metadata_preflight.json](../experiments/20261011-data-alignment-round22/science_evaluation/metadata_preflight.json)：3240个角色/请求视图，其中3176为单暴露原生表达请求、64为bulk pending；不能把请求数称独立实验。endpoint10fold，24唯一源Y身份，生物重复ID未恢复，每final块2候选；Martin4背景/12平行RNA参考/20发表功能观测/2域/8块，属于非A549辅助。原始Y、表达值、模型参数均不由此预检解析或执行。

新比较packet须绑定实际metadata与各方法的冻结评价JSON及源文件SHA。A/B只接受表达`population_mmd_v`或`mean_mse`；C严格校验Annexin fraction、Sarin fold和真实Martin domain/scale/readout；D使用实测候选regret。逐项检查fold/role/block/endpoint/坐标/信息集合、完整候选/单位覆盖及每Y仅一次统计单位，拒绝负误差和使用final选模型。特征使用/决策规则需明确；A的状态消融是在相同可用数据权限下有意改变特征使用。此入口比较已有结果，不计算新P/G/Q或伪造预测；尚无真实模型结果packet可验收科学增益。

接续新增[独立joint-selection报告](DATA_ALIGNMENT_JOINT_SELECTION_ROUND22_2026-10-11.md)与[评价入口](../scripts/audit_joint_selection_evaluation.py)：旧v2不改，20物理候选按纯剂量元数据分为8fit/4selection/8final，最终完整2药×2剂量×2协议；实际build/preflight不解析Y。四类策略及均匀诊断使用同完整域、同终点和G反馈，固定条件子集规则提前冻结；只有显式冻结分数后才读对应role Y。原图各自控制及像素读图区间保留，跨图瓶/批配对未知，统计独立单位数未知。新元数据与模拟指标测试通过不认证真实模型、科学效果或整个D/主问题。

#### 已准备的执行顺序

1. 继续数据和来源消费工作：Bothe生产计数/共同轴，Martin及Sarin分别保持native观测域、mask、角色和完整协议。真实域准备通过后再逐一验收E/D/P输入消费，不把接口成立当模型成立。
2. 在停止训练约束下可运行当前[metadata CLI](../scripts/audit_science_evaluation.py)：`preflight --output NEW_JSON`；现有源保持只读。Sarin raw readback脚本[verify_sarin_original.py](../experiments/20261011-data-alignment-round22/science_evaluation/verify_sarin_original.py)只能写新回执，不重划分或重新拟合。
3. 取得额外模型运行授权后，先产生源/角色/坐标严格绑定的真实P留出描述符；A/B表达对照共用原候选、随机流、参考及评价样本，并保持所有final标签隔离。
4. C按域训练/校准对应G，仅从声明fit/selection取得相应标签；匹配功能连接不足的域保留pending。Sarin若用于外部诊断，整个源不得反过来用于G校准；其fold头不能由fraction头裁剪/换名。
5. 只有G符合预先声明的独立selection判据，才按固定奖励构造单次决策/训练私有Q。冻结G/Q与所有简单基线后，用同candidate/block/fold/role/information_set的现有实测评价结果执行`compare --packet FILE --output NEW_JSON`。
6. 主目标整体验证要求同链E/D/P/C/G/Q既有真实输入又有相容独立功能与选择结局，联合药物—剂量—时长覆盖由实际标签决定。各模块本地指标改善不足以通过整体验证；当前仍缺关键覆盖和实际模型产物。

### Open Questions / Future Directions

| 层级 | 当前条件 | 必须另行验收 |
|---|---|---|
| 单次选择 | 限定源内可比候选/实测Y已存在，预测和G/Q执行待授权及资产 | 固定完整候选、校准和独立终点；不把代理分数当凋亡率 |
| 多步滚动 | 有物理协议/历史数据准备；缺匹配源模型与独立滚动结果 | 每段输入/状态/历史/clock及误差累积，最终共同终点另验 |
| 真实序贯 | 辅助背景/路径保留，未取得相应完整A549序贯表达—Annexin验证链 | 实测顺序/间隔/撤药/恢复及最终功能标签，不能由单步通过或多次模拟推定通过 |

本轮CPU实际完成654,038源值和22×32,461矩阵读回，说明此项数据验证可在现机完成；完整模型训练/正式实验的资源资格由主代理当前硬件及资产快照判断，本轮不作未经运行的耗时/显存保证。

模型运行授权可以解除实际P/G/Q实验的执行限制，不能补出缺失的高维功能矩阵或真实序贯标签。保持完整目标未完成；若定向公开资料仍无法补齐，需用户决定是否扩大数据取得边界或调整验证承诺，本轮没有替用户改题、改奖励、移核心目标到展望。

## 修改、验证与接续

新增3个源码/测试文件与本报告，实验回执和原件仅写`experiments/20261011-data-alignment-round22/science_evaluation/`。保留旧源、旧资格、冻结实验及其他并行改动；未提交、推送、联系作者或写Zotero/Memory。

验证命令：`F:/anaconda/envs/study/python.exe -m pytest tests/test_science_evaluation_comparison.py tests/test_endpoint_protocol_evaluation.py tests/test_state_function_evaluation.py -q`，77 passed；CLI真实metadata预检、Sarin官方SOFT/原DOCX/NPZ原值精确读回、FOXN3补充大小/MD5/CRC核对分别有实际回执。新代码自审后C错误终点域和负误差两项已修复，相关反例保留。结果与当前研究架构§3/§4.3–4.4/§6一致：分层评价、有限合法候选、不混观测域、不以模拟或caller文件认证科学结果。

其后`state_ID_only` fixture加入，原三组命令实际为78 passed（15.63s），旧77回执保留；新joint-view及comparison定向验证另见新专项回执，不把不同执行的数量叠为一次测试结果。
