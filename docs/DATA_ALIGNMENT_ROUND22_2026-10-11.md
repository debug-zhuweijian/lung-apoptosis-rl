# 完整数据对齐与科研可行性：第二十二轮

日期：2026-10-11（北京时间）。当前状态：**完整目标未完成；当前仅将已启动的 Bothe 数据阶段做完，其余新科研工作暂停，模型保持停止。**

本轮用户先明确选择“保持模型停止，继续数据与证据工作”，[决定已记录](../experiments/20261011-data-alignment-round22/root_model_execution_decision.json)；22:18 后要求“先做到这里”，并对后台数据作业明确回复“把这个阶段做完吧”。当前授权仅继续已启动 Bothe 的剩余计数、全量共同轴投影、观察包及实际验收，阶段完成后停止继续推进。所有真实参数加载、forward、拟合与训练仍停止。其他新检索、实验划分或方法实施停在当前版本，见[本轮接续与停止条件](../experiments/20261011-data-alignment-round22/CONTINUATION_HANDOFF_2026-10-11.md)。

本轮继续用户确认的肺腺癌促凋亡治疗、先以 A549 验证的细胞状态条件化响应预测与干预筛选主线。研究范围仍以[毕业论文研究架构](G:/研究生科研文件/毕业论文/研究架构.md)和本轮用户要求为准。下文区分数据已经读取、实现存在、实际运行通过和科学效果验证；前一层的通过不自动转移到后一层。

## 1. 当前状态的重新核实

引用聊天“训练速度评估与曲线参数讲解”最近两轮发生平台流中断。旧下载和计数进程在本轮首次真实进程检查时均已不存在，`driver.json` 的 `running` 是失效的生命周期快照。保存旧快照，恢复时核查实际文件、回执与进程，不以旧 session ID、mtime 或该字段判断任务仍在运行。

- Bothe 首次核对为 8/12 个双端样本已有官方校验回执，3 个真实 RNA 计数样本完成。随后0295、0302整对官方MD5/SHA发布，变为10/12；0302坏R2及旧0299 partial保留，0303继续续传，0306排队。新0299双端全文CRC/配对检查已通过，17:44:37实际启动Subjunc比对；这仍不是第四个计数完成。
- 初始启动两个独占下载支路，进程及新旧来源绑定见[下载启动回执](../experiments/20261011-data-alignment-round22/bothe_recovery/download_launch.json)。0302支路已整对完成退出。18:52剩余支路因0303 R2分段声明8MiB、实际仅191,680字节及curl56退出，未将坏段append；已接受2,457,862,144字节前缀和失败正文/日志保留。19:01按当前accepted检查点仅重绑定恢复声明，独占续传0303→0306，见[实际恢复启动](../experiments/20261011-data-alignment-round22/bothe_recovery/remaining-network-retry_launch.json)。计数保留已完成三个leaf与旧0299 partial，不重复写同一样本。
- 19:58再次核对五个数据作业的PID、CreationDate、ExecutablePath与完整命令：下载、计数、全量发布/投影、关闭中间文件存储检查和bulk观察包等待均仍是本轮对应实例。0303已整对发布，当前11/12官方校验、4/12完整FASTQ、3/12计数；0306R1继续分段续传，0299的Subjunc已结束，其22,745,102,306字节SAM之后的严格配对过滤尚未完成。见[五作业身份回执](../experiments/20261011-data-alignment-round22/bothe_recovery/execution_continuation-20261011T1140340255390Z.json)。该回执与此次工具核查是时点证据，后续仍应重新核对，不能单看`running`。
- 20:10后0299完整原生执行回执闭合，真实计数变为4/12。主代理对[前三个闭合样本](../experiments/20261011-data-alignment-round22/first3_gene_projection_v1/receipt.json)与[第四个样本](../experiments/20261011-data-alignment-round22/fourth_gene_projection_v1/receipt.json)实际parse原207,555 feature坐标/整数计数，映射各17,107共同可测基因、15,354缺失基因并逐值读回TSV。0299原49,161,137对reads、4,828,143 assigned intron fragments；映射后有1,773观察零，不能与缺失混合。这是4个真实leaf的局部数据投影，尚未运行全量production consumer或发布12样本包。
- 21:10[最新数据实读](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_data_readback-20261011T131038215166Z.json)为11/12 raw、5/12完整FASTQ、4/12 count，最后0306仍待整对验收。21:14[五作业独立CIM核查](../experiments/20261011-data-alignment-round22/root_late_process_readback-20261011T131440662251Z.json)逐一核PID、创建时刻、实际python路径和命令中的批准脚本，五实例均存活，源字节与启动审查一致；没有重启或重复写入。原始数据/SAM正文此快照未重新hash，已闭合全字节校验只在原来源绑定仍一致的范围内复用。
- 21:41[最新闭合及实际进程回执](../experiments/20261011-data-alignment-round22/bothe_recovery/readonly-continuation-20261011T134154911245Z/snapshot.json)更新为**12/12 raw官方整对闭合、5/12完整FASTQ、4/12 count**。0306由固定whole validator exit0发布canonical，两端当前大小与回执匹配；复用原MD5/SHA，不重hash正文。下载67148已不存在，以闭合回执证明该支路完成；count/finalizer/storage/observation四实例均匹配原创建时刻、exe和完整命令，未发现重复writer。0300的原Subjunc16504/parent53960仍执行`-T 1`，live SAM未hash或压缩；57%只是当时比对阶段日志，不是全量完成度。旧11raw与五活跃实例保留为原时点证据。
- [主代理资产读回](../experiments/20261011-data-alignment-round22/root_existing_assets_readback.json)已实际核对核心终点来源/表字节、联合表达资产的全部直接文件哈希、基因轴、归一化与分组隔离。旧 Martin 两份读回只在源绑定、输出和消费者身份仍一致的范围内复用。
- 联合表达索引为 2,762,092 个细胞，完整基因轴为 32,461。该计数来自当前索引，不代表独立生物重复或新增终点实验。TIDYI `1_to_8` 角色登记可用；`8_to_1` 仍为 `STATE_SPLIT_INSUFFICIENT`，不改划分消除该缺口。
- 核心 r5 的 308 个条件、846 个源位置中 844 个非缺失实值，复算 42 个可用 Annexin 条件汇总。不同来源、分母与检测定义分别保留；这些数字不是独立研究数。
- 已核对的旧 AE 注册状态为 `interrupted`；旧登记实现身份与当前源树不一致。未加载或重哈希 PT 正文，不赋予旧权重新的 bulk、完整协议或终点资格。

上述真实数据检查未执行模型forward、拟合、参数更新或功能final评价。后续已读取原C final的类别和gate元数据，用于冻结既有58个疑点的敏感性计划；没有读取final RNA或计算分类表现。核心r5属于既有公开源表审计，不能将其误写为从未接触结局的前瞻性盲测。

## 2. 整个项目的验收表

下表中的“待模型实验”意味着数据或结构允许安排该检验，不表示有科学增益。涉及源内单元、代理迁移或缺失字段的结论仍受所列边界限制。

| 核心连接 | 所需数据及当前实际字段/单位 | 对齐与实现入口 | 实际证据与当前边界 | 剩余工作 |
|---|---|---|---|---|
| 基因身份 | 联合 `genes.parquet:ensembl_id` 32,461；来源基因轴/映射、observed mask | 联合轴、Martin source/joint 双坐标、Bothe GeneID 桥 | 本轮联合轴唯一且与 normalization 的有序 ID 一致；Martin 已有源值读回，缺失不补成观察零 | 完成 Bothe 全量真实 count 投影；分别验证各来源模型出口 |
| 药物身份 | 来源材料、完整结构、`drug_id`/`fingerprint_row`、名义结构和未知项 | identity-v3、来源化学附表及合法候选登记 | 联合化学资产字节仍匹配冻结 manifest；名义产品不等于实验批次鉴定 | 缺结构来源保持相应消费者不可用；不得用假 FP 补齐 |
| 细胞背景与来源 | `source_id/background_id/protocol_id`，原始 cell line/sample | source-specific 输入和角色 | A549 主问题与 Martin 非 A549 辅助分开；不通过换 source 字符串迁移模型 ID | 验证新来源模型与独立留出 |
| 表达观测域 | 单细胞原计数；Martin `log2(TPM+0.01)` 与非整数 estimated counts；Bothe intron fragments | E、D、来源预处理及观测损失 | 单细胞 fit-only normalization 轴/scale 已读；Martin来源双域E/D已实现与合成复审，负值、非整数及mask保留；真实模型未运行 | 匹配权重、独立final表达验证、共享latent与Bothe来源出口 |
| 剂量 | 原值/原单位、统一 `dose_um`，材料与暴露记录 | 条件编码及候选输入 | 联合 metadata/候选数据版本已绑定；不把不同来源同名浓度直接配对 | 逐字段消费核验及未见剂量评测 |
| 暴露/恢复/观察 | `exposure_duration_h`、withdrawal、recovery、horizon 与 elapsed | 完整物理协议、请求身份 | sci-Plex 24/72h 是来源内快照；Vielhauer 6h+18h恢复与24h暴露为共同24h候选 | 完整协议 P/D 连接和匹配参数；不将快照当真实轨迹 |
| 处理历史 | 有顺序的药物、洗脱、换液、恢复事件和 unknown mask | Martin observed protocol；Bothe 四历史；physical P | Martin 16组/40名义事件已读，RNA 与四药72h功能时钟不同 | Bothe 完整表达恢复；历史增量与滚动验证 |
| 实验协议 | 检测分母/门控、培养/收获、撤药/残留等原始约束 | 完整候选 ID、source protocol，未知字段保持未知 | 同 drug-dose-duration 不合并不同协议；未知维护日程不写成已知0 | protocol 消费路径实现、留协议与敏感性评价 |
| 原始状态 C | TIDYI 同细胞表达与原分选字符串，板/孔位/QC | 原始三分类及跨板对照 | 本轮实际读回743唯一原始细胞与248 fit细胞的32,461列输入，原始计数/ID/板/孔/三字符串及消费者数值一致；标签频率不等于自然群体 Annexin 率 | 冻结 C 的实际评测与跨来源迁移，混杂诊断和失败结果分别报告 |
| 实测终点 G | Annexin fraction、caspase 源尺度、growth/SYTOX 各自定义 | 核心 r5、完整协议监督、Martin 分头及独立挑战；新joint G入口 | 源值确实存在；新G消费预测mean/variance与完整条件的代码及合成链已复审，真实预测/G仍0。24个协议Y身份不等于24个独立实验 | 匹配预测到真实Y拟合、source/状态迁移和独立功能验证 |
| 重复及依赖 | cell/GSM、experiment block、biological replicate、共享对照、读图范围 | 来源身份和观测 ID 台账 | 同一 Nav Y 的10/11μM视图不扩样；RNA三重复不变成三份功能Y | 维持实验块比较，公开均值可作有限检验，不能伪造重复或显著性 |
| 数据划分 | fit、selection、calibration、final 等登记角色，chemical connectivity group | 冻结联合划分和 endpoint role plan | 本轮 group role 隔离通过；十个 protocol fold 双分支观察身份相同、fold内角色不交叉；未调用保留标签 API | 不同 fold 不拼接成单一冻结全候选排名；实际选模在允许角色内进行 |
| 候选价值 Q | 同信息、同目标的候选分数、实测留出结局及模型身份 | 同一G枚举、独立terminal ridge Q诊断与适用规划/RL对照 | 新joint入口已连接G的冻结模拟反馈→单次Q→冻结同域评价，工程反例关闭；真实Q0，无正式Actor/RL资格 | 本链真实产物、独立选择评价及既定策略连接 |

## 3. 科研主问题与可证伪条件

| 研究问题 | 必要性、理论和识别假设 | 数据可以监督什么 | 实现与验证安排 | 当前判定 |
|---|---|---|---|---|
| A：状态增量 | 条件分布差异是否可由可用参考表达解释，且超出药物/群体均值；参考上下文可能含测量混杂 | 来源内对照/药后群体表达与原始状态标签；不识别同一细胞未来命运 | P 的无状态/同参考均值/完整参考分布比较；C 另检板位/QC；匹配划分和输入预算 | 表达层有可执行检验条件，实际增量待模型实验；高维状态到实测 Annexin 的增量仍未完整覆盖 |
| B：未见条件可靠性 | 预测目标与留出单位一致；剂量、时间补全与跨药/跨协议外推分别定义 | 冻结联合 chemical group 角色和来源内24/72h观测，完整协议有限挑战 | 留药、留完整条件、留来源/协议；预处理只在 fit；报告对应误差/校准/排序 | 已具部分数据与划分条件，模型效果待运行；任意连续时间与广域协议泛化未验收 |
| C：实测功能一致性 | 预测特征可以作为 G 输入，但表达域与分选 C 后验的迁移需检验；同名条件不制造配对 | Annexin 来源块及独立挑战，caspase等保持不同目标 | 固定 P/C 与 G 校准角色，按检测定义在独立结局检验全部正负案例 | 有真实条件结局，完整表达—终点连接及主问题同链验证未完成 |
| D：联合选择收益 | 单次选择相同候选域内比较；模型误差与代理奖励不能由高模拟回报掩盖 | Vielhauer 有共同T*=24h的有限剂量—暴露候选；其它来源按各自实验块 | 枚举/固定剂量时长/无状态/Q在同信息与目标下比较；只对已测完整候选域算选择差距 | 限定联合选择数据、G/Q学习和冻结评价已实现及合成复审；真实评分待运行，不宣称RL优势 |
| 主问题整体验证 | A/B/C/D 局部通过仍须检查状态输入→响应→实测功能→选择是否对同一可解释检验成立 | 现有来源可以分项研究，尚未补齐完整 A549 高维状态—相容候选—Annexin 验证矩阵 | 显式记录各连接的跨来源假设，查证数据、固定全链后独立检验 | **完整科研可行性未通过；不得改成局部先导完成** |

上表是本课题针对当前观测范围的研究设计和待检验假设，不宣称文献已为本模型证明收益。以下原始理论本轮已下载并由主代理亲读相应内容，文件版本/哈希见[原件回执](../experiments/20261011-data-alignment-round22/science_evaluation/sources/theory_download_receipt.json)。

**必要性的具体证据。** Sarin2017原补充S8、对应Fig.5的第6表中，同为cisplatin11μM/24h，parental A549与A549rCDDP2000的FITC-Annexin相对各自未处理对照的作者汇总分别为5.856与1.950 fold；resistant34μM为3.714 fold。主代理实际读原DOCX并从原表解析3/4/3个数值，见[原表读回v2](../experiments/20261011-data-alignment-round22/root_sarin_primary_annexin_readback_v2.json)。parental三项已发表舍入数复均值为5.855、原作者摘要5.856，两者分别保留，不改源值凑一致。这支持“背景差异与条件选择需要独立检查”的有限场景依据，不证明高维状态是差异原因，不把fold变为绝对阳性比例，亦不由三个条件推导未见背景有效。Vielhauer原文Fig.3C/5C另提供相同24h终点下连续24h与6h撤药+18h恢复的A549协议，因而联合时长选择应比较完整协议而不只读取采样24h；实际来源角色和公开均值范围分别保留。

- **分布评测**：Gretton 等 *A Kernel Two-Sample Test*（JMLR 2012），§2.2、Lemma 4/6，期刊页727–728（PDF第5–6页），定义分布均值嵌入间的 MMD，并给出含独立副本的总体核期望和 U-statistic。A/B 的 MMD衡量分布差异，不恢复单细胞前后配对。本项目 `population_mmd_v` 明确为 V-statistic 口径；不能将其当成论文无偏 U-statistic或将同板细胞都当独立实验重复。[原文](https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf#page=5)
- **候选动作价值**：Xiong 等 P-DQN，arXiv:1810.06394v1，§2、式(2.1)，PDF第2–3页，定义策略下累积回报的条件动作价值与 Bellman 目标。该论文研究混合连续动作，本项目有限离散候选没有因此成为 P-DQN 实现。单次终点决策没有后续动作时，Q目标按本课题定义退化为给定状态和完整候选的预期终点，无需人为增加bootstrap或成本惩罚。[原文](https://arxiv.org/pdf/1810.06394v1#page=3)
- **滚动与模型误差**：MOPO（NeurIPS2020），§3–4、Assumptions4.2/4.3与Theorem4.4，PDF第3、5–6页，要求离线转移样本、适当函数类和可接受的模型误差上界。当前表达快照不自动提供真实 `(s,a,r,s')` 轨迹，也未证明上界成立；该定理不能作为本项目的策略收益保证。这里只据其问题与假设审查滚动误差/分布偏移，不采用论文的奖励惩罚来改变已确认目标。[原文](https://proceedings.nips.cc/paper_files/paper/2020/file/a322852ce0df73e204b7e67cbbef0d0a-Paper.pdf#page=5)

**本课题的有限候选误差推导。** 固定同一背景、信息、检测定义和观察终点，令 `R(a)` 为可比候选的预期实测终点，`Rhat(a)`为冻结预测；假设在该有限域中对全部候选有 `|Rhat(a)-R(a)|≤ε`。若 `astar`最大化`R`、`ahat`最大化`Rhat`，则

`R(astar)-R(ahat) = [R(astar)-Rhat(astar)] + [Rhat(astar)-Rhat(ahat)] + [Rhat(ahat)-R(ahat)] ≤ 2ε`。

中间项因最大化定义非正，两端分别不超过ε。这一推导说明选择应结合全候选误差核验；它不是 MOPO 的定理，也不证明本数据中已有统一ε。公开均值/读图界与真实预期终点之间仍有采样和测量不确定性，不能把有限测量候选的差距称为全条件最优性。

**状态增量的可检验含义。** Gneiting原arXiv:0912.0902v1的§2.2（PDF第12页）明确：有限二阶矩下平方误差对均值泛函严格一致。已取得并亲读[原PDF](../experiments/20261011-data-alignment-round22/science_evaluation/sources/Gneiting_arxiv0912.0902v1.pdf#page=12)，版本/读取更正见[原件回执v2](../experiments/20261011-data-alignment-round22/science_evaluation/sources/Gneiting_arxiv0912.0902v1_readback_v2.json)。不把均值预测和其它损失下的理想点预测混为同一目标。[公开原件](https://arxiv.org/pdf/0912.0902v1#page=12)

以下是本课题对当前问题的自行推导，使用平方损失而非新增奖励。令`W`包含相同药物、剂量、完整协议及允许给基线的信息，`X`为实际可用状态/参考表达，`Y`为同一定义的表达均值或终点，假设二阶矩有限。令`m0=E[Y|W]`、`m1=E[Y|X,W]`。因为`E[Y-m1|X,W]=0`，展开`Y-m0=(Y-m1)+(m1-m0)`后交叉项期望为0，得

`E[(Y-m0)^2]-E[(Y-m1)^2]=E[(m1-m0)^2]≥0`。

非零增量要求`m1`不能几乎处处等于`m0`；不是“多加表达维数”就必然有可用增益。这是未知总体条件均值的信息上限，不保证有限样本学得的P或G优于基线。实际验证必须同角色、同`W`、独立留出，并对状态ID、同参考均值及完整参考分布分别比较。平行24h对照`X`或代理来源`X`需要另行检验其与目标状态的关系；两种背景、一个药物不足以把高维信息和状态ID效应区分为广泛规律。该推导不涉及机制因果证明，也不替代分布MMD和候选选择误差评价。

**共享表示的额外识别条件。** 以下同样是本课题的推导，不能将来源自重建通过写成共享表示已经建立。若某来源只由`D(E(x),c)`重建监督，且函数类对可逆坐标变换`A`封闭，则`E'(x)=A E(x)`、`D'(z,c)=D(A^-1 z,c)`给出完全相同重建与观测损失。自重建本身因此不能唯一确定该来源latent与另一来源的坐标对应。当前[Martin来源模型](../src/lung/models/martin_source.py)的`encode/decode/observation_loss`是来源内双域重建；它还没有与joint AE的锚定、共享参数或响应迁移损失。共同gene ID和mask只确定观察轴，不消除计数/TPM等观测机制及latent坐标差异。后续共享E/来源D须明确可识别锚点或参数共享假设，并在来源留出及下游响应上独立验证；这项核心缺口继续保留，不以维度相同接入旧P。

**未配对细胞的识别边界。** 当前[原响应取样](../src/lung/data/reader.py)的`sample_response/_response_rows`分别抽control和target；[原耦合](../src/lung/engine/latent_pipeline.py)的`_couple`在表达或latent空间求entropic OT并抽coupling索引。这是已核源码行为。只观察两个边际分布时，许多联合分布具有相同边际；单独拟合边际不能确定哪个control细胞对应哪个真实药后细胞。OT引入本方案的计算耦合假设，A/B的heldout分布误差可以检验其预测价值，但不据此识别单个真实细胞未来命运或因果反应。该数学边界不是新增训练目标，也未修改原耦合。

## 3.1 本轮新可执行路径及证据层级

- **Bothe完整观察消费**：新增bulk-only消费者与等待CLI，固定12来源样本、32,461列、17,107可测基因及NaN/observed mask。它要求本轮全量计数闭合回执、production manifest身份与真实joint TSV，再发布和重新读回观察包，不以manifest存在视为完成。主代理亲读实现并核16项实际来源pin；独立复审关闭进程误认和完成退出竞态后43项合成检查通过，真实全12包仍pending。vehicle是T32平行参考，RNA是bulk total RNA intron fragments，不授共享latent、bulk P或Annexin资格。见[实际消费者复审](../experiments/20261011-data-alignment-round22/independent_bothe_observation_review/review_receipt.json)。
- **状态输入的独立训练比较**：observed、zero_h、mean_h三份P合同及模板已闭合，复用同一AE，各自独立训练；原OT、目标、角色与早停不改。CFM mean来自原OT后选出的h[ii]，selection独立control抽样，production/C为全2431代理参考，native final为本条件允许draw；不将不同阶段样本均值说成同一数值总体。source_state_id没有数值编码，zero_h不冒充state-ID基线。owner78项与独立17项是分别运行；真实P参数/效果0。见[合同与真实预检](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/variant_plan_and_guard_v2.json)、[独立阶段取样复审](../experiments/20261011-data-alignment-round22/independent_reference_variant_review/technical_review.md)及[实施说明](DATA_ALIGNMENT_REFERENCE_VARIANTS_ROUND22_2026-10-11.md)。
- **原native final表达评价**：主代理亲读当前实现并核[原367条件/200,682行E-role元数据](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/primary_metadata_readback_v4.json)，保留全部参考/目标ID、角色、坐标、32,461有序基因和fit-only统计。52个是原化学连接组件，不是独立生物重复；367均为未见chemical，独立未见dose/time及同chemical剂量/时间留出均0，full exposure/history holdout未识别。未来评价按共同标准化坐标算mean MSE/MMD_v，三份闭合结果须同AE/配置/抽样/noise/原T baseline才能配对比较。owner39项、独立12项分别通过；当前真实RNA/模型评价与三份结果比较未运行。见[工程交付](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/implementation_handoff_v2.json)、[独立复审](../experiments/20261011-data-alignment-round22/independent_full_protocol_native_evaluation_review/technical_review.md)和[范围说明](FULL_PROTOCOL_NATIVE_FINAL_EVALUATION_ROUND22_2026-10-11.md)。
- **状态可识别性与剂量单位**：[原小型元数据诊断v2](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/summary.json)及[独立逐ID复算](../experiments/20261011-data-alignment-round22/independent_state_context_review/technical_review.md)复现34context、1089组/2195成员、1212相关组合对；进一步固定W与板后多context支持为0。它支持合法同期参考表达的预测效用检验，不能单凭效用证明独立生物初态作用。188组/751条件有同drug-dose的24/72h，但所有axis条件留在同一角色；R/G/C没有补出独立剂量/时间/协议留出。[1871孔单位读回](../experiments/20261011-data-alignment-round22/state_context_identifiability/dose-units-v1/readback.json)恢复sci-Plex原nM÷1000到µM的来源证据；冻结原空字段未填改，模型消费未改变。实际RNA/模型/收益0，见[完整解释](DATA_ALIGNMENT_STATE_CONTEXT_IDENTIFIABILITY_ROUND22_2026-10-11.md)。
- **已确认辅助序贯来源**：GSE306207官方A375原TPM的4历史×2作者报告生物重复已取得；主代理亲读原SOFT第14/67及8样本字段、原Reporting Summary第3/4页，并从原XLSX独立复算[501680源值/259688共同轴值](../experiments/20261011-data-alignment-round22/root_sequence_source_readback.json)。32461基因全部exact匹配、float64原TPM和测得零保留，不拟合归一化。I-BET726和Oxaliplatin的完整官方名义立体图、2048 FP另[亲读并逐位复算](../experiments/20261011-data-alignment-round22/root_sequence_chemical_readback.json)，Oxa与旧v3行25完全相同，全部组分保留。该来源支持已测A375末端历史表达，role仍unassigned；没有反向/单Oxa/更广剂量RNA或Annexin，DMSO时钟/洗脱/前药保留仍unknown。见[来源与证据](DATA_ALIGNMENT_SEQUENCE_SOURCE_ROUND22_2026-10-11.md)。
- **CellCage定向数据恢复**：[25个真实HTTP metadata回执](../experiments/20261011-data-alignment-round22/cellcage_source_recovery/actual_readback_v1.json)与原文/补充已核。主代理亲读原全文转换的Methods“Sequencing data analysis”“Profiling A549 cells following treatment with Olmutinib”：RNA在最后32h影像后取得并按CCE汇总，不能替初态RNA。未取得本源gene×CCE矩阵或CCE/时间/处理/门控对应表，实际源RNA/row join为0；不生成零占位。近似GEO辐射数据与未证明版本对应的restricted Zenodo不接成A549资料；检索空结果不证明资料不存在。见[定向尝试、原件定位与必要依赖](DATA_ALIGNMENT_CELLCAGE_SOURCE_ROUND22_2026-10-11.md)。
- **Martin来源E/D与学习入口**：双域源值、16组/40名义事件、60唯一GSM和NaN/mask已在[修后v3真实读回](../experiments/20261011-data-alignment-round22/martin_source/data-readback-v3/real_input_readback.json)通过；已实现独立`train_martin_source.py --fit/--resume`。当前cycle预检查30唯一fit GSM、15唯一selection GSM；双域不是双倍重复。43项合成更新/恢复/身份检查及独立复审通过，真实forward/PT/fit为0。模型仍是独立来源候选，没有建立共享latent或bulk P。详见[来源报告](DATA_ALIGNMENT_MARTIN_SOURCE_ROUND22_2026-10-11.md)。
- **Martin独立final**：新final-only工厂保持15唯一GSM（3参考、12target）与4个响应请求，不把final伪装成selection。完成run的best/last、角色、fit统计、source/基因轴及选择epoch均须匹配，然后只评价冻结双域重建。当前工程35项检查及独立22+4项反例通过；这些回执版本分别登记，真实final RNA/forward/误差尚未运行。见[final报告](DATA_ALIGNMENT_MARTIN_FINAL_ROUND22_2026-10-11.md)和[独立复审](../experiments/20261011-data-alignment-round22/independent_martin_final_review/results.json)。
- **完整协议P**：已实际读取2431个固定fit代理参考细胞×32,461基因，20个Viel候选的numeric/physical输入及mask准备通过；32个总身份中的Nav8/Bothe4保留pending依赖。独立入口实际连接E—fullP—D，模型与完整物理字段通过合成工程检验；实际匹配P、C/G/Q与实测效果均未运行。新增native训练准备明确dose/观测endpoint已知，exposure/washout/recovery/维护历史未知；Viel pulse/恢复协议仍OOD，没有用schema能表达来假称已有监督。详见[协议报告](DATA_ALIGNMENT_PROTOCOL_CONSUMPTION_ROUND22_2026-10-11.md)。
- **D的联合选择评价**：旧十fold每块只比较同drug/dose的两时长，继续原样保留。独立新增[单模型joint view](DATA_ALIGNMENT_JOINT_SELECTION_ROUND22_2026-10-11.md)：两药最低/最高剂量组8fit、中位组4selection、另外两剂量组8final，双协议同角色；按剂量元数据建立，没有按Y选划分。最终块为两药×两剂量×两协议，基线、G与独立Q在相同完整域及信息下比较，禁止拼跨fold分数。实际metadata、fit/selection源值与角色消费已读回，final数值与真实模型效果未评价；MSE/MAE/选择差距及读图区间敏感性入口已实现，读图界不是生物CI。
- **真实生产及G/Q学习入口**：[串行生产CLI](../scripts/produce_full_protocol_predictions.py)默认只预检同一新view的20候选；主代理[实际守卫预检](../experiments/20261011-data-alignment-round22/root_fullP_batch_preflight.json)拒绝PT、模型构造、solve、Y和produce，仍成功读到20个真实元数据身份。未来显式执行才调用真实producer，独占写入、精确计划恢复、已完成预测认证复用，partial保留而拒绝覆盖。首次启动导入路径失败及最小修复单独留证，没有把CI能导入冒称CLI可运行。
- **G/Q与同信息比较**：[独立学习入口](DATA_ALIGNMENT_JOINT_SELECTION_LEARNING_ROUND22_2026-10-11.md)要求20个实际source-bound P产物、同模型/参考/采样；G和条件子集基线只用fit8真实均值，旧lambda=1、fit-only统计与每Y一次权重；独立Q用同一冻结G的fit候选终止模拟反馈，不用measured Y直接训练Q、不增加bootstrap或成本。原DA-AC/Actor未被替换，终端ridge Q只是诊断候选。冻结后从保存参数重算scores，再读指定角色Y。信息声明、全域统计冒充fit-only与有限参数溢出被clip掩盖的三项P2已修复并独立反例关闭。独立24项CI发生上游b3→b8漂移，报告明确不能将其计为当前b8的完整联测；以最后稳定源码综合CI另记。
- **原状态C与预测功能特征**：[新C入口](DATA_ALIGNMENT_RAW_STATE_C_ROUND22_2026-10-11.md)保持原weighted CE、三类及温度校准，C训练独立于P。sidecar复演同2431参考细胞的同P采样，通过固定表达坐标及同AE的h后逐细胞算原三类概率，要求float64 P均值/方差精确复现；没有采用`C(expression_mean)`或合类proxy。C资产/角色/计数及final入口的3项P2已独立复现修复，最新32项工程检查通过。C→G可选索引加入三类概率作为预测特征，root6项和peer9项调用层检查明确mock了真实reader，不能充当实际C参数/输出证据。当前真实C/P/G均为0，跨来源校准和Annexin解释继续未通过。见[原C独立回执](../experiments/20261011-data-alignment-round22/independent_raw_state_scoring_review/postfix_results.json)及[可选连接复审](../experiments/20261011-data-alignment-round22/independent_joint_selection_learning_review/optional_C_delta/review_receipt.json)。
- **公平比较与实际执行证据分开**：[比较入口](../src/lung/data/science_evaluation_comparison.py)检查同来源、终点、角色、信息、完整候选及绑定字节，并保留`execution_certified=false`。它不能从caller的声明证明生产器运行或未见信息；真实生产、拟合和留出回执须另核，不把caller JSON当作整链完成。全P和新AE已完成断点恢复现在验证完成/selected/best/last证明后返回，不再追加训练；缺proof明确拒绝。

## 3.2 定向原件查证与合法的有限来源

- Bothe2021原版PMC8403771，Methods `s10/s15/s27` 和Fig.3图注已由主代理亲读XML。四历史来自A549的Dex/EtOH 4h、PBS两洗、含5%FBS无额外激素培养24h、再次Dex/EtOH 4h，终点为32h；“无激素”不意味着血清没有皮质醇。其监督是bulk total RNA的唯一intron计数，不是Annexin终点或已测每一段转移。本项目冻结Subjunc计数与论文STAR2.7.0a为两个明确不同处理实现，不能称为作者数值的精确复现。见[原件](../experiments/20261011-data-alignment-round17/bothe_protocol_source/PMC8403771.xml)与[源定位回执](../experiments/20261011-data-alignment-round22/bothe_recovery/primary_bothe_protocol_readback.json)。
- 原始状态C的[实际读回](../experiments/20261011-data-alignment-round22/root_state_C_readback.json)核对248个fit细胞：91个`PI-APC-`、83个`PI-APC+`、74个`PI+APC+`。官方E-MTAB-16354 study原Description明确BOBO3/Annexin分选并保留上清；原标签PI字符串的通道推导仍未取得。已核源码中的C按每细胞类别1+2概率聚合，此数据验收不产生概率或天然群体凋亡率；QC/孔列摘要是数据混杂检查准备，不是状态增益结果。未调用selection/calibration/final状态API或真实模型。
- 原C final gate的[后续实际元数据读回](../experiments/20261011-data-alignment-round22/root_state_gate_followup.json)精确对上plate8的368个A549原cell ID、标签与四门事件字段；50个作者负类样本所有子门均0、另8个singlet也0，现有导出字段不能直接确认其负类门，而不能据此证明作者错标。58原ID已冻结，主分析保留368原标签；预定义310样本敏感性单独报告，分母不同不能用更高accuracy宣称模型改善。原标签/划分未改；PI/BOBO3通道解释及plate1同门字段继续查证。
- plate1同门字段的缺口随后解决：找到G盘保留的作者`human_ss3.h5ad`历史原件，主代理[直接只读obs](../experiments/20261011-data-alignment-round22/root_author_C_gate_readback.json)核743个A549 ID/原类/四事件字段，并逐值核对子代理CSV导出；没有访问X/layers/var、复制或下载整H5。plate1为375细胞、原137负/136正/102dead，其中35负类有APC−门、74负类singlet而无子门、28负类无singlet/子门；plate8原58疑点完全一致。102个plate1导出疑点不证明错标，不改fit/selection/calibration标签或训练目标。旧receipt的`plate1...not_recovered=true`是该时点事实，由本新原件读回更新；PI/BOBO3标签生成链仍待原作者步骤证据。
- FOXN3 的官方补充 DOCX 已实际取得，15,644,900字节、官方MD5相符。主代理亲读原DOCX `word/document.xml` Fig.S4E：该 knockdown 凋亡实验是 NIH3T3、Dox2μM/24h；不是 A549 siKD 凋亡标签。其RNA siKD、主要KO/OE/回补与上述对象/条件不能自动连接，未向核心G新增标签。见[补充原件](../experiments/20261011-data-alignment-round22/science_evaluation/sources/FOXN3_supplement.docx)与[读取证据](../experiments/20261011-data-alignment-round22/science_evaluation/sources/FOXN3_supplement_readback.json)。
- Sarin2017 原文 `sec009` 和 Fig.5a（原件PMC5528889.1）分别规定24h FITC Annexin检测及相对各自未处理对照的fold；主代理亲读这些段落。已取得共同11μM下两种A549背景的功能值，可作限定相对Annexin挑战，不因不是fraction或非同细胞t0而整源排除。parental34μM功能格缺失，仍不能形成完整两状态×两剂量矩阵；两个背景不足以单独区分高维表达与状态ID贡献，也没有广域未见状态验证。[方法与Fig.5原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5528889/#sec009)、[原件与实际数据读回](../experiments/20261011-data-alignment-round22/science_evaluation/sarin_original_source_readback.json)。
- GATOR1 等补充取得失败和未核实范围继续保留；网络失败不是无相关数据的证明。定向补充只解决具体证据缺口，没有把辅助 caspase/growth 或非A549背景改为核心 Annexin 数据。

## 4. 单次、多步滚动、真实序贯分别验收

| 层级 | 所需证据 | 当前真实状态 | 不可转移的资格 |
|---|---|---|---|
| 单次候选选择 | 完整合法候选/参考、匹配 P/D、G终点校准与独立候选结局 | 输入与有限真实结局可用，完整链与模型产物未运行 | 单次通过不能证明任意历史或多步 |
| 多步滚动 | 合法历史、撤药/恢复、每段转移域、累计clock、滚动误差及共同终点 | 有辅助协议与历史数据，完整滚动 P/D及实测核验待补 | 模型内部轨迹不能变成观察轨迹 |
| 真实序贯 | 顺序/间隔/剂量/次数、相同背景真实结局、覆盖与独立评价 | Bothe为A549四种真实激素重诱导历史，32h表达待全量恢复，没有本来源促凋亡结局；主A549序贯功能连接继续核查 | 历史表达不授序贯Annexin资格；辅助非A549、growth/caspase也不自动获得该资格 |

## 5. 资源与后续运行顺序

本轮初始读取到RTX3080 10GiB、约32GiB主存，I盘约754GiB可用。初始数据作业为最多两个下载与一个冻结单线程计数；初始空闲主存约13GiB。资源快照只反映观察时刻，当前进程、空间与训练/完整推演峰值需对应实际观察核验。

[只读模型资源实读](../experiments/20261011-data-alignment-round22/root_model_resource_preflight.json)检查3,590个实际源CSR的header/indptr及角色索引，fit+selection为1,769,627个缓存行，源全轴非零项为2,289,003,246。当前`AEDataSupply→NormalizedExpressionCache._build`存稀疏log1p值，不是完整稠密表达；以float64值、最大int64索引/indptr及valid计，共同轴缓存数组保守上界36,640,016,915字节，另计cell-ID索引、manifest、原CSR缓存、h/response/feature、权重与文件系统开销。共同轴筛选不增加非零项；源计数值正文未重读，大小/mtime前后匹配已登记源身份。此为实际索引加源码公式的容量估算，未建全缓存或测训练峰值。

[当前配置与缓冲实读](../experiments/20261011-data-alignment-round22/root_resource_schedule_readback.json)重新核对上述数据/源码绑定，并亲读`AEDataSupply/AEGroupPrefetch`、`NormalizedExpressionCache._part/read`及`SparseRows._matrix`：norm部分是磁盘mmap、最多16部分，源CSR没有常驻heap-copy缓存，AE最多64源mapping；36.64GB是缓存数组的磁盘上界，不是一次全部读入heap的要求。当前batch128、每update16条件的control+target float32输出为531,841,024字节；当前组加一个预取组为1,063,682,048字节。这个公式仅覆盖两组输出，另有映射驻留页、稀疏切片、float64临时量、pinned副本、模型、optimizer与GPU激活，不能将1.06GB作为总峰值。GPU拟合按一个模型串行；三种状态输入对照需要三次独立P拟合并复用同一AE，尚无实际吞吐和整链峰值数据。

18:09的实际资源快照为I盘约790.95GB空闲，Subjunc working set约8.84GB、availableRAM约5.50GiB。计数进行中不能沿用初始空闲主存判断并行真实模型可行；正式训练仍停止。源E/D参数规模、稀疏缓存和原单机训练记录支持分阶段安排，整项目RAM/显存/存储及多轮耗时仍须对应实际模型运行核验，尚不登记资源全部通过。

Bothe前三个真实作业的 Subjunc 约60–83分钟、过滤约39–53分钟，另有FASTQ完整检查；据此剩余九个串行作业约20–30小时是实测外推，下载、失败恢复和样本大小会改变完成时刻。主代理独立7项恢复边界测试、真实旧leaf读回和只读preflight均通过，[审查回执](../experiments/20261011-data-alignment-round22/bothe_recovery/root_count_recovery_review.json)对应原wrapper哈希。计数已在北京时间17:12恢复，旧driver备份、0299partial原卷改名保留，新任务先完整核FASTQ；见[count启动回执](../experiments/20261011-data-alignment-round22/bothe_recovery/count_launch.json)。没有新增墙钟暂停，不改变原计数协议。

完整保留九个新leaf的未压缩容量约744–750GB，另有约30GB新raw/stage，接近当时约808GB空闲空间；这是线性外推，实际峰值未验收。NTFS压缩最初只有fixture证据，随后取得下一段真实0299结果。微软文档对普通NTFS压缩文件列出最大未压缩30GB限制，因此不能让大SQLite或整个新目录继承压缩；只评估已关闭、本轮新建且在支持大小内的SAM文件，既有三leaf保持。未删除或迁移现有文件。[官方限制](https://learn.microsoft.com/windows/win32/api/winioctl/ni-winioctl-fsctl_set_compression)

20:10启动的0299限定关闭SAM存储处理已[实际完成](../experiments/20261011-data-alignment-round22/bothe_recovery/closed-sam-ERR5000299-20261011T121014238803Z/result.json)：aligned allocation由22,745,104,384降为10,853,658,624字节，filtered由18,901,504,000降为8,886,030,336；总计释放21,906,919,424字节。两文件逻辑长度、SHA、mtime前后相同，独占run锁覆盖压缩与核验；旧三个leaf、SQLite和live文件未动。实际FSCTL合计约393.57秒，不能把单个样本比例推广为全项目压缩率。该真实样本缓解当前存储缺口，全量峰值仍持续核对。

后续顺序保留：全量读段/计数/投影与bulk观察消费 → 新E/D与完整协议P/C消费及合成工程复审 → 审查匹配资产和真实运行方案 → 获授权后真实来源重建/响应验证与A/B消融 → C独立诊断及迁移 → G受控fit/selection/冻结final → 同候选域基线与Q比较 → 多步滚动和真实序贯分别取得证据。

数据准备不自动恢复正式训练。需要真实模型 forward、参数加载、拟合或新增训练时，先把命令、输入版本、角色、资产兼容性、输出和评价范围准备成可审阅方案，再处理该额外授权。
当前方案已保存为[独立模型运行与正式评测顺序](MODEL_EXECUTION_PLAN_ROUND22_2026-10-11.md)。完整state输入对照、native final/三结果比较、原C final和G/Q两角色冻结的工程路径及专项复审已补齐；未运行模型与核心数据缺口明确保留。21:14资源快照为I盘716,218,871,808字节空闲、available RAM 7,056,468KiB；它不是全模型峰值。真实模型运行须额外授权，当前没有启动。

21:41资源更新为I盘709,049,917,440字节空闲、available RAM 3,491,016KiB（约3.33GiB），Subjunc working set约8.86GB。实际单机数据处理仍在进行；源E/D规模和mmap支持分阶段安排，但整链GPU/RAM峰值、吞吐与多轮模型成本尚未实测。用户已明确保持模型停止，当前不因数据完成恢复训练。

## 6. 本轮接续

四个专项责任分别为 Bothe恢复、Martin来源 E/D、完整协议消费、科研原件与公平评价。各分支只修改当前责任文件；原dirty tree、冻结划分/奖励与旧失败成果保留。

[架构对应检查v3](../experiments/20261011-data-alignment-round22/root_architecture_correspondence_v3.json)将当前输出逐项对应到研究架构§2、3、4.1–4.4、5–7；架构SHA256为`34e08ddb0f48e035fc6b3ba4f6db4f9a172bb5e159a30bc9ca8c8ffff5fb188e`且未修改。Martin新增独立source E/D是待检验候选，尚非已建立的共享latent；单细胞完整协议生产只在matching资产下开放，Nav/Bothe缺失依赖仍拒绝；没有把这些实现写成整套架构已完成。

综合报告与验收表继续追加，真实证据放入[round22目录](../experiments/20261011-data-alignment-round22/)。[最新全项目验收v3](../experiments/20261011-data-alignment-round22/whole_project_acceptance_v3.json)记录7个核心问题的必要性、理论、所需/实际数据、对齐、入口、验证、证据、边界与缺口；[13维数据验收v3](../experiments/20261011-data-alignment-round22/data_axis_acceptance_v3.json)和[源字节证据索引v3](../experiments/20261011-data-alignment-round22/root_acceptance_evidence_index_v3.json)加入12raw闭合、状态context/单位、新辅助序贯源及用户模型停止决定。whole_project_qualified仍false，旧v1/v2保留。没有提交、推送、外部覆盖、Zotero写回或全局Memory更新。

最终稳定源码联测的[第一份回执](../experiments/20261011-data-alignment-round22/root_stable_core_ci_v1/receipt.json)保留103通过/25 setup错误，错误均为旧G fixture缺新增mandatory reference合同；真实producer正确拒绝，未放宽实现。只为该合成样本补同一observed合同后，[G与相关C补验](../experiments/20261011-data-alignment-round22/root_stable_G_C_ci_v2/receipt.json)37项通过，源前后SHA一致；两次范围重叠，不相加成一次通过数。另[真实C reader到G的受控复审](../experiments/20261011-data-alignment-round22/independent_joint_selection_learning_review/real_reader_bridge/technical_review.md)4项通过：原reader/原state声明核验确实执行，20行三类mean确实进入G，P、数值来源和文本asset为fixture，真实模型/Y/solve为0。该测试不冒充真实表达—终点验收。

早期部分合成pytest没有显式basetemp，fixture落入C盘默认临时目录，偏离用户I盘运行产物边界；这是执行路径偏差，不能隐去或把mtime作为唯一运行归属。旧C临时文件没有删除。后续运行均先确认新的短I盘basetemp不存在，避免pytest清除已有目录及Windows深路径失败；最终稳定源码联测、路径与结果另记。真实研究数据、PT和生产运行产物没有因这次CI落点偏差被搬迁。
