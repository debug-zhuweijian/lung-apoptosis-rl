# 独立剂量与观测时点评测提案，Round22

日期：2026-10-11。**未批准、未实施。** 本提案补充[现行axis留出为0的诊断](DATA_ALIGNMENT_STATE_CONTEXT_IDENTIFIABILITY_ROUND22_2026-10-11.md)，不修改原global roles、split或实验，不替代原跨药final。仅读取冻结T的支持元数据，尚未创建shadow训练/评测视图、消费者、配置、归一化、参数或模型运行。采用研究设计也不自动恢复真实模型操作。

建议准备两个独立实验版本：Tahoe名义中间剂量预测，以及sci-Plex从24h训练域到72h已报告观测条件的转移。两者检验限定轴的表达预测可靠性，不预设模型优于基线，不支持单细胞命运、完整洗脱协议、共同终点治疗收益或序贯效果。

## 1. 实际支持范围和缺口

[计数脚本](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/count_T_support.py) 只读joint的4张小型parquet：conditions、control_links、role_condition_counts、split_groups。原manifest SHA为`513c9650…ba05`，四表逐项校验并核读取前后哈希。[v2实际回执](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v2/support_counts.json) 与7张CSV保留原角色`fit`、条件/化学/板/重复/参考ID、剂量单位、target/reference计数及完整已记录非轴字段；没有给任何行赋新的训练或测试用途。

| 原T范围 | 化学实体 / 原防泄漏组件 | 条件 | 参考上下文 | target细胞（非独立重复） |
|---|---:|---:|---:|---:|
| Tahoe 0.05μM | 201 / 同下185组件集合 | 218 | 5 | 388262 |
| Tahoe 0.5μM | 201 / 185 | 218 | 5 | 446149 |
| Tahoe 5μM | 201 / 同上185组件集合 | 270 | 6 | 501923 |
| sci-Plex 24h全T | 82 / 81 | 656 | 16 | 61537 |
| sci-Plex 24h中与72h同药的部分 | 17 / 17 | 136 | 4 | 10282 |
| sci-Plex 72h全T | 17 / 17 | 136 | 4 | 23226 |

全部条件仍属原T，不使用E的RNA或功能Y。每个条件有正target计数和2个正计数同角色参考孔，来源、A549背景、protocol、观测时刻、板严格匹配；sci-Plex还匹配原replicate。协议比较登记全部提供的非轴字段，包括时间来源/语义、exposure、withdrawal、is_sequential；**exposure/withdrawal仍为null，历史未验证，相同未知不能视为完整物理协议相同**。

**剂量支持：**[201个实体系列](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v2/Tahoe_nominal_three_value_series.csv)均有0.05/0.5/5μM。两端共488条件、10个上下文；中间值218条件、5个上下文，其中只有plate13也在两端上下文中。plate2/5/8/11为中间值独有，主要比较因此同时有板/参考域变化。进一步固定同一参考上下文，[同板支持表](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v2/Tahoe_middle_value_context_support.csv)仅有14个实体的完整三剂量系列，全部plate13，每值14条件。其余204个中间值条件缺少同板两端支持，这是匹配范围不足，不能写为缺失RNA或零响应。

**时点支持：**[68个同药同剂量系列](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v2/sciPlex_two_reported_time_series.csv)覆盖17药×4剂量，24/72h各两个原replicate、各136条件，匹配背景和已记录非轴protocol。剂量网格均为0.01/0.1/1/10μM。两时间的culture plate没有交集。其余65个24h原T化学实体没有72h观测，不补造该部分测试条件。

**缺孔与低回收：**在上述eligible T实体内，三剂量缺值和72h缺同药同剂量24h支持均为0。[原不合格条件表](../experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v2/registered_ineligible_conditions_not_added_to_T.csv)单列12个登记记录：sci-Plex S7207的一个72h孔无QC细胞，其已登记药物角色是calibration，原condition角色仍空；S4246的8孔为结构不能编码，有细胞而不是空孔；另3个Tahoe不eligible记录原缺失原因为空，未补造原因。它们均未加入T。真实计数支持不等于效力充足：Tahoe低剂量5条件target少于20、最少2；sci-Plex72h的7条件少于20、最少1。保留原记录和后续固定的低回收敏感性说明，不因预期难预测而删除。

## 2. 两项拟采用的实验定义

**D1：Tahoe有限名义剂量预测。** 若获批准，以原T、Tahoe同一来源/背景/24h的0.05和5μM条件作为新实验表达训练范围（488条件）；0.5μM的218条件只作为该版本预定评价目标。原T记录身份保留，新实验用途须另登记，当前没有执行这种分配。0.5是两端的log-dose中点，这里只检验一个实测离散中间值，不声称连续剂量曲线已辨识。

建议全218条件作为“剂量＋板域变化”的主要可用域，预先单列同板plate13的14条件作为配对敏感性结果。若用户要求主结论必须控制同板，则主要支持只能限于这14个条件，证据覆盖仅一板，不能保留201药的纯剂量主张。两个口径须在模型运行前选定，不能看到结果后切换。同期24hvehicle可作为查询参考，它是独立群体，非同细胞t0；训练阶段只能拟合其已声明训练范围的合法参考，不能从中间值target获取状态。

**T1：已报告观测时点转移。** 若获批准，以sci-Plex原T的24h全656条件/82药训练新版本，在72h的136条件/17药上评价；它们的同药同剂量24h记录均存在。新版本的拟合统计、AE、带宽和P不得消费72h target或72h参考RNA。评价时允许72h同期vehicle作为该任务声明的查询输入，必须记录该信息暴露并仅用新24h-fit统计变换；因此本任务是“给定72h同期参考的条件分布预测”，不是给药前t0预测72h的真实细胞命运。

24h与72h为作者报告的破坏性观察群体，不能把24h解释为给药24h后恢复48h，也不能将72h当已核实连续active exposure。时间与culture plate变化同时发生；训练只有24h这一时点，未提供多时点训练来识别时间函数。此挑战能否定或支持限定72h转移的实际性能，不能由通过推导连续时间动力学或纯时间因果作用。17药来自实际72h测量面板，结果不扩展到没有72h标签的65个T药物。

## 3. 资产、选模与验证安排

两项均须建立**各自的新独立实验数据契约与namespace**，原全T资产和实验原封保留。旧joint归一化、graph、带宽、AE及h缓存曾消费全部T，已经包含0.5μM和72h表达；它们不得通过换名、改metadata或只重训P来冒充整链未见。若采用提案，必须执行以下次序：

1. 先批准用途与主评价口径，再登记新训练/选模/评价范围，保留所有原condition/control/role ID与源哈希。当前CSV为支持库存，不能直接当新的dataset或消费者输入。
2. 只在相应新表达训练范围拟合归一化、graph/带宽、E/D或AE，以及对应冻结h/cache。沿用固定32461公共基因身份和静态化学特征时也记录来源；不通过留出RNA进行基因筛选或统计调参。
3. 分别拟合P，所有参数、resume、数据消费与评估必须绑定本版本统计和范围。若使用C、表达校正、不确定性或后续G/Q，同链资产必须重建/重新拟合并绑定新表示及监督范围；不能接入原全T环境后称整链未见。功能G/Q的额外监督域另登记，不因RNA轴评价获得Annexin或联合选择资格。
4. 选模建议保留原V身份，但限定同来源、仅非留出轴值，须在批准后另核具体条件与参考可用性；本轮只实算T，不宣称该选模子范围已完成。不得用T的0.5μM或72h目标选超参、早停、评分方向或阈值，也不在T中暗建shadow选模划分。若该选模安排不足，先提交明确修订而不是使用评价目标。
5. 冻结同一指标、目标、条件权重、参考可获得性和随机能力后评价。原历史QC与全T预处理是已发生的数据暴露；新结论只能声明新实验资产的训练隔离，不能追称人类未见盲测。

现有learner/normalization仍按原冻结joint全T执行，**尚无这两个axis范围的受审核数据消费者**。采用提案后还需要相应独立实现与data-only核验；不能现在向旧入口传一个新文件名就认为隔离生效。本轮没有实现该消费者或创建训练命令。

表达验证保留既有MMD、均值误差及来源适配坐标，按原化学防泄漏组件汇总，并报告板/重复与低回收限制，不将细胞数当独立重复。适用对照包括分别重训的observed/zero_h/mean_h，及D1的两端简单插值/混合分布、T1的无时间增量或24h响应沿用基准；不使用留出表达选其中最好的对照。三状态分支可共用本axis版本的新AE，却不能共用旧全T AE。若两项均采用，至少增加2套独立预处理/AE域与相应P实验；资源、时间和效果尚未实测，正式训练保持停止。

当前可供用户决定的是：是否采用D1和/或T1，以及D1的主要范围是广义218条件还是仅同板14条件。批准前它们仍为研究提案；原研究主线、原跨药final、完整协议/功能/联合选择与多步序贯目标均不改变，也不由本提案计数或将来两项局部通过而宣告完成。

复算支持库存的命令（新目录必须不存在）：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/count_T_support.py' --output 'I:/科研/肺癌细胞凋亡-强化学习/experiments/20261011-data-alignment-round22/independent_axis_evaluation_proposal/v3'
```
