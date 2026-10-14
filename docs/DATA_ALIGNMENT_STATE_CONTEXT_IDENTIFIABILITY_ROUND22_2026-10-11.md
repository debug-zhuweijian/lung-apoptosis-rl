# 原冻结参考上下文的状态增量可识别性诊断，Round22

日期：2026-10-11。结论：**现有 native/joint 数据能够检验在记录条件下，加入合法同期参考表达是否提高留出药物的表达分布预测效用；当前元数据不能把这一效用唯一归因于独立生物初态，也没有独立的剂量、时间或完整处理协议留出。** 本轮实际读取小型元数据并保存 source-bound 表，没有读取 RNA 值、final RNA、功能 Y、参数或模型，没有修改角色、划分、目标、奖励和训练语义，也没有建立或批准新划分。

任务对应毕业论文[研究架构](G:/研究生科研文件/毕业论文/研究架构.md) §3 状态增量与未见条件假设、§4.3 表示/响应职责、§6 验证安排。完整原目标继续保留；当前缺口没有改写为通过或移到展望。

## 实际读取、来源与可复算入口

[诊断脚本](../experiments/20261011-data-alignment-round22/state_context_identifiability/diagnose_metadata.py) 从自身位置解析 I 项目根目录，仅写新版本子目录。实际执行成功的 [v2/summary.json](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/summary.json) 保存输入的前后 SHA256、实际字段清单、输出 SHA256、时间和零模型读取范围；v2进一步逐行核sci-Plex重复、完整非空同角色参考池及34上下文的角色计数，原主要数字与v1一致。

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/experiments/20261011-data-alignment-round22/state_context_identifiability/diagnose_metadata.py' --output 'I:/科研/肺癌细胞凋亡-强化学习/experiments/20261011-data-alignment-round22/state_context_identifiability/v3'
```

此命令只复算诊断；新输出目录必须不存在。`v1/v2` 已完成，保留原件。读取范围为 joint 的 `conditions.parquet`、`control_links.parquet`、`role_condition_counts.parquet`、`split_groups.parquet`，round12 的三张 CSV，native 的 manifest/requests，以及已完成 native final 的 metadata_v4/preflight/final_requests。没有重新打开 train/val/test 细胞索引或源表达分片。joint manifest 仍为 `513c9650…ba05`，round12 manifest 为 `829d6ce8…1f34`，native manifest 为 `7c04dcd7…bb71`，均与原登记版本严格匹配。

初次脚本把 enriched final request 与旧 native 整行比较而拒绝，发生在新输出写入前。实际差别为9项 final 元数据扩展；最小修正后逐项核所有原字段完全不变，并枚举允许的扩展、核 final role 和参考细胞数。保留[初次失败说明](../experiments/20261011-data-alignment-round22/state_context_identifiability/initial_native_final_contract_failure.json)，没有为此改动生产源码。

| 原字段或资产 | 实际意义与读取结果 | 不能推导的内容 |
|---|---|---|
| `chemical_or_experiment_group_id`、`drug_id`、`split_group_id` | 完整化学实体/目录身份、防泄漏连通组；545实体、512连通组 | 545个独立生物状态；名义结构等于已核实实验试剂 |
| `dose_um` | Tahoe为0.05/0.5/5 μM；sci-Plex为0.01/0.1/1/10 μM | 两来源相同剂量网格 |
| `duration_h` | 当前 native 的观察条件24/72h；必须结合 source 时间语义 | 已核实连续 active exposure；生成积分时间 |
| `source_time_h`、`reported_treatment_hours`、`time_semantics` | sci-Plex1871条非空，Tahoe1305条为空；后者24h来自已有全局协议 | 空字段等于0或不同时间来源可以忽略 |
| `exposure_duration_h`、`withdrawal_protocol` | 全3176条均为 null；native 的 prior/washout/withdrawal/exposure unknown均为 true | 所有处理的完整历史已相同、没有撤药或换液 |
| `source_id`、`background_id`、`protocol_id` | 2来源、1背景 `CVCL_0023`、2协议；source与protocol一对一 | 独立于来源的 protocol 效应、跨背景外推 |
| `plate`、`culture_plate`、`replicate_id/scope` | 对照与处理严格匹配同来源/背景/时间/板；sci-Plex还核相同replicate | 板差异是已知生物状态；Tahoe sample ID均是生物重复 |
| `control_condition_ids_json`、`available_control_*`、native `reference_sampling_weights` | 每处理登记2物理对照；同角色非空孔等概率，再孔内细胞等概率 | 合并全部参考细胞后等权与原抽样权重完全相同 |

上述原字段、连接与数值均在脚本中与原 parquet 和 native request 逐行比较；各输入 before/after 哈希一致。`protocol_id` 中的文字不能覆盖 explicit unknown mask。

剂量的原单位还做了独立[读取核查](../experiments/20261011-data-alignment-round22/state_context_identifiability/dose-units-v1/readback.json)：Tahoe1305条 `source_dose_value/unit` 已直接保留µM；joint中sci-Plex1871条的这两个原值字段为空，但固定计划孔表保留 `well_oligo/plate_oligo/catalog_number/dose/time_point/replicate`。实际逐孔连接1871条并核目录身份、时点、重复和 \(\mathrm{dose}_{\mu M}=\mathrm{dose}_{nM}/1000\)，完全符合当前归一化µM值。[单位连接侧表](../experiments/20261011-data-alignment-round22/state_context_identifiability/dose-units-v1/sciplex_nominal_dose_unit_join.csv) 保存原来源nM与规范µM，未填改冻结joint空字段，也未自动接入模型。可复算脚本为 [verify_nominal_dose_units.py](../experiments/20261011-data-alignment-round22/state_context_identifiability/verify_nominal_dose_units.py)。这解决了本诊断中名义剂量的单位定位证据，不验证实验材料形态、实际剂量基准或完整暴露协议。

## 34个上下文与1089组究竟提供什么

重读得到34个上下文：Tahoe14板，sci-Plex24h的16个和72h的4个；68个物理 vehicle 条件，六角色控制细胞总数65055。它们为对应时点的独立同期 vehicle 群体，**没有同细胞处理前 t0 测量**。34是参考上下文数量，不能作为34种已证实生物初态或34个独立干预状态。

定义记录条件 \(W_{\mathrm{recorded}}\) 为 source、background、protocol、完整化学实体、dose和观察时刻，并附加原时间语义与所有 unknown mask。本轮跨上下文匹配要求这些记录字段相同；**相同 missing/unknown 状态只表示缺失模式一致，不证明未观测协议相同**。

| 冻结角色 | eligible处理 | 同记录W跨上下文组 | 成员条件 | 与fit共享完整化学实体 | 相对fit新增上下文 |
|---|---:|---:|---:|---:|---:|
| fit | 1498 | 493 | 989 | 277（本集合自身） | 0 |
| selection | 374 | 134 | 282 | 0 | 0 |
| residual_fit | 299 | 99 | 198 | 0 | 0 |
| aux_selection | 310 | 109 | 218 | 0 | 0 |
| calibration | 328 | 108 | 216 | 0 | 0 |
| final_test | 367 | 146 | 292 | 0 | 0 |
| 全部 | 3176 | **1089** | **2195** | — | — |

来源内分解为 Tahoe154组325成员、sci-Plex935组1870成员。原 round12 overlap 的成员ID集合完全复现。[逐组表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/same_recorded_W_groups.csv) 保留原字段、成员ID和协议 unknown；[逐对表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/same_recorded_W_condition_pairs.csv) 列出1212个不同上下文条件对。1212为组合对数，重复使用成员与参考池，不能作为1212个独立重复。

大多数组有两个上下文。Tahoe的3组 Afatinib分别跨3个上下文，均属fit；selection中的 Adagrasib 0.05μM有16条件、14上下文，产生大量相关组合对。另6组12个Tahoe成员只是同一上下文内的多条件记录，未纳入跨上下文状态比较。正式统计须按原化学防泄漏组/条件组织与预定权重汇总，不能由组合对数量扩大有效样本量或由少数组的大量重复主导结果。

这些比较可以承担的职责是：在相同记录W下，用各自合法参考输入，比较冻结算法的药后群体表达误差；同一目标条件上比较分别拟合的 observed/zero_h/mean_h，其目标、生成能力与其他信息相同。它支持参考信息的预测效用检验，并可被无增益结果否定。换一个旧模型的参考推演只是敏感性分析，不能替代独立重训对照。

其混杂边界也可以从元数据直接看见：所有跨上下文组同时改变实验板；sci-Plex还改变作者重复。source、plate、time/replicate决定当前 `reference_context_id`。在记录W之外再固定 source-qualified plate/culture_plate，所有分组的上下文数最大为1，具有多个上下文的组为0。因此本包没有“同一板与相同记录条件下、独立改变参考生物状态”的比较。这个结果针对上下文身份的支持范围，不宣称已测出参考 h 的方差或互信息；本轮未读RNA，尚不知道34个参考表达分布在预定坐标中有多大差别。

背景与协议混杂也不能靠1089组消除。两个source恰好对应两个protocol；18个完整化学实体跨来源出现，但同实体、同数值剂量、同观测时点的跨来源组为0。跨来源 h 差异不能用于声称状态作用，放宽结构、剂量或时间后也不再是相同W比较。

10个仅有一个非空同角色对照的处理全部属于 `aux_selection`；当前T、V、E条件均有2个可用孔。ID和数量见[单孔表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/single_available_reference_well_conditions.csv)。不能补造空孔或照搬2孔权重。

## 剂量、时点、协议留出：现有支持和缺口

实际[剂量系列表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/dose_series_within_role.csv) 有 Tahoe376个化学实体的24h三剂量系列，以及sci-Plex24h的187个、72h的47个四剂量系列。这里统计剂量值是否出现，并不表示每个重复孔都有充分细胞或达到了统计效力要求。原缺孔/低回收记录继续保留。

sci-Plex的188个同药同剂量组合同时有24/72h，共751条件，见[时点成员表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/same_drug_dose_two_time_members.csv)。角色分布为fit68组、selection28、residual_fit20、aux_selection20、calibration20、final_test32。所有同药的剂量/时点始终留在同一冻结角色：没有训练中见过某药、在另一角色单独留出其剂量或时点的条件。

| 所需验证 | 原数据实际支持 | 当前冻结设计不具备什么 | 合法后续边界 |
|---|---|---|---|
| 未见药物表达预测与参考增量 | E中367条件、52原防泄漏组；完整化学实体相对T/V均未见；已知source/dose网格/观察时刻 | 完全未见培养上下文或单细胞未来命运 | 固定各模型/指标后用同一E评价；真实运行仍停止 |
| 独立剂量留出 | 来源内多剂量系列真实存在 | V/R/G/C/E的各来源剂量值都在T出现；同药没有跨角色剂量留出 | 可设计另一个明确的剂量实验版本，但本轮未创建或批准；不能把原跨药成绩当纯剂量泛化 |
| 独立时间留出 | 188个同药同剂量的24/72h组合 | T本身已有两时点；其他角色只是新药；72h培养板与24h不同 | 另声明限定时点留出与参考可获得性；若要求整个链未见时点，AE/归一化等不能提前使用被留时点目标 |
| 与板分离的时间作用 | 每时点均有作者rep1/rep2；同药同剂量可做来源内分布比较 | culture_plate嵌套于时点；不是同一细胞纵向观测 | 保留time/plate关联，不能据两点声称连续外推或共同T*治疗时长收益 |
| 完整协议、暴露与恢复迁移 | 观察时间及部分名义单药记录可核；unknown真实保留 | joint全部exposure/withdrawal缺失；protocol与source绑定，没有已知洗脱/恢复RNA监督 | 保持Viel pulse/recovery协议OOD与可证伪功能挑战；不以相同unknown或命名full授予监督资格 |

若未来只从现行T内部撤掉部分目标来训练P，而共同AE与fit预处理仍曾消费它们，只能在明确“共享表示已见该表达域”的条件下解释P层留出；不能称完整链严格未见。严格独立的整链版本需按新研究协议固定训练资产、统计与全部消费者身份。本报告是缺口与设计边界，未生成新角色、shadow split或训练输入。

时间/重复的原始依据来自已核SHA的[作者论文XML](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/Srivatsan2020_sciPlex_PMC7289078.xml)：large screen段及Fig3A规定四剂量、两重复；72h段与GEO/元数据给出另一组破坏性终点群体。原主文所述培养板位置随机化不等于当前34上下文的生物状态操纵。已读[时间协议契约](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/time_control_contract.json) 明确保留连续暴露、换液、补药和洗脱未知；没有把补充材料尚未取得的细节补为“无操作”。

## 辅助角色能提供哪些合法补充

[原角色支持表](../experiments/20261011-data-alignment-round22/state_context_identifiability/v2/frozen_role_support.csv) 从当前 [EngineData](../src/lung/engine/data.py:82) 的 `open_data/permitted` 和 [TrainingData](../src/lung/data/reader.py:162) 权限读取职责，未修改这些代码。它描述 stage 对数据角色的权限，不表示所有任务都具有相同的响应条件消费者；state stage另按TIDYI fold消费标签。

- `residual_fit` 299条件、99个跨上下文组可供原 backward/correction/uncertainty 的拟合职责；数据含20个同药同剂量双时点组。它与主P的T化学组不相交，但一旦辅助模块使用其目标，就不是完整链的独立测试。
- `aux_selection` 310条件、109组可供原 correction/uncertainty 的辅助选择，含20个双时点组及上述10个单孔条件；不能同时作为该辅助模块的最后无偏评价。
- `calibration` 328条件、108组可供既定校准职责，含20个双时点组；校准完成后不得把已用目标再标成独立final。
- 上述角色均覆盖原34个上下文、原来源剂量/时间边际，没有提供相对T的新剂量、新时点或新已知协议，也没有同药跨T的axis-only留出。可提供各自原用途内的新药诊断与已有条件的校正/校准，不自动解决独立剂量、时间或协议评价缺口。

新独立 full-P learner 当前只消费T/V，native final消费者只消费显式E。R/G/C原角色的元数据可读取，不等于已存在同资格的 full-P 评价命令；若以后需要其限定冻结诊断，应按实际角色建立相应评价消费者和版本，不能把R/G/C伪装成E，也不能为方便评测加入主P训练。

native中另有64条Martin bulk LOBO请求，实际只对应16个目标状态组、4个背景及4个parental参考上下文，均为0.5μM osimertinib；LOBO副本不增加独立样本。它们的背景、完整历史或终点改变W，且不属于joint的34单细胞上下文；source E/D重建通过也不能变成状态条件化P转移或A549剂量留出的证据。这里只解析原请求元数据，没有读取bulk表达或功能终点。

## 理论、可识别目标与实验解释

**本课题的统计定义**：对每个原目标条件c，\(\mathcal R_c\) 为同角色合法同期vehicle参考群体，\(W_c\) 为记录处理条件，\(B_c\) 为板/重复等测量上下文，\(\mathcal Y_c\) 为该条件的药后群体表达。状态输入取 \(S_c=T_v(E(\mathcal R_c))\)，其中v为observed/zero_h/mean_h。当前可检验目标是相同留出分布、信息和权重下独立拟合预测器的风险差，\(\Delta_{0\to v}=R(\widehat P_0)-R(\widehat P_v)\)。正值、零或负值均是合法实验结果，未运行前没有收益结论。

理论动机可用条件分布的proper loss表述。Gneiting与Raftery（2007）§1–2定义严格proper评分，§4.1、式(19)，期刊p365，给出log score的严格proper性质。[作者原PDF](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf#page=7)。在真实条件密度存在、可积且支持正确的假设下，本课题由负log loss定义推导的oracle风险差为

\[
R^*(W)-R^*(W,S)=\mathbb E\!\left[\log\frac{p(Y\mid W,S)}{p(Y\mid W)}\right]=I(Y;S\mid W).
\]

这是本课题的代数动机，不是已估计的互信息或本文模型收益保证。原full-P没有据此新增log-density评价；现有有限样本、非oracle模型、latent压缩及优化误差均可令独立训练结果无增益。若S携带板、来源或观测时点相关信息，条件预测增量也不等于生物状态的因果作用。

实际表达评价保持既有MMD和均值误差。Gretton等（2012）§2 Definition2、式(1)，期刊p726，定义MMD；§2.2 Theorem5、p727，说明其在指定空间和universal RKHS假设下区分分布的性质。[原JMLR论文](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf#page=4)。这里引用分布比较依据，不把该文的独立样本假设或渐近检验保证直接套用到重复孔、共享参考与本项目有限biased估计，也没有新增指标、损失或显著性阈值。

当前可识别的是原实验采样机制下的群体 \(p(Y\mid W,\mathcal R)\) 预测效用。它不识别同一个真实细胞的 \(p(Y_{t>0}\mid X_{t0},a)\)：参考与目标为不同破坏性观测群体，边际分布可以有多个相容的前后联合分布。原OT指定的计算coupling不能把这个缺失联合观测补为真实命运；本轮没有新增配对。

未来已授权方法的A-expression验证应继续使用三条独立训练及同一E目标、同参考可获得性、同原权重/随机能力，并按source、记录时点和上述同W组分别报告。化学组件等权、相关成员的分组误差及预定义板/重复敏感性用于解释，不能随机拆细胞扩充独立重复。若总体增益只来自source/time或少数板位，或组内效果不稳定，应否定或限制原预测增量解释；不能用C分类、功能代理奖励或OT内部loss来替代此评价。模型拟合、heldout风险、RNA状态差异和上述敏感性实际结果全部仍未运行。
