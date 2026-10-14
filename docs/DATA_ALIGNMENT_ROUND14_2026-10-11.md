# 第十四轮：标准响应特征生产、完整协议消费与真实监督视图

日期：2026-10-11。接续[第十三轮](DATA_ALIGNMENT_ROUND13_2026-10-11.md)，目标仍为完整数据对齐与全课题可行性核验。本轮采用 GPT-6.1-sol 专项实现及独立复审，主代理核对真实调用、实际文件与[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)。没有恢复正式训练，也没有执行真实模型推理。

## 1. 本轮实际完成到哪里

原来只有候选输入，标准 P 缓存始终拒绝；现在实现了实际生产程序、固定缓存格式和严格读取程序，并把完整实验条件传入 G/Q 评分接口。真实 Annexin 监督另建独立视图，模型特征生产不读取这些标签。

完成的是程序与数据接口，不是已经生成或验证有效的模型：**真实 P 描述符仍为 0**。当前登记的联合 pipeline 未提供 response、pool、state、representation 四类匹配权重；正式 G/Q 学习、权重加载、角色冻结及真实整链效果在本轮结束时尚未完成。

| 项目 | 实际状态 | 证据 |
|---|---|---|
| 完整候选输入 | 28：Vielhauer 20，Nav 两种剂量解释 8 | [冻结输入包](../08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1/manifest.json) |
| 独立活性观测身份 | 24：Vielhauer 20＋Nav 4 | [监督包](../08-datasets/LungCancer_A549_Integration/endpoint-protocol-supervision-v1/manifest.json) |
| 标准连续 24h P 请求 | 14：Vielhauer 10＋Nav 4 | [生产程序](../src/lung/engine/endpoint_protocol_producer.py:33) |
| 实际指纹输入 | 28×2048，读取现有真实文件 | [V2 输入桥](../src/lung/engine/endpoint_protocol_features_v2.py) |
| 真实参考控制细胞 | Tahoe fit 参考池 2,431 个 | [主代理真实预检](../experiments/20261011-data-alignment-round14/producer-preflight-main-v1/preflight.json) |
| 真实 P 缓存/模型评分 | 0/未执行 | 同上，model_execution_performed=false |
| 聚焦回归 | 55 项通过，纯 mock、格式和实际元数据检查 | 19 producer＋14 adapter＋12 supervision＋10 旧桥 |

这 24 条是来源发表的条件均值，不是 24 个独立研究或 24 个原始生物重复。原核心 r5/v6 的 42 个 Annexin 条件汇总仍单独保留，没有与本轮数量机械相加。

## 2. 标准 P 实际生产程序

新入口为 [generate_endpoint_protocol_features.py](../scripts/generate_endpoint_protocol_features.py)，核心为 [endpoint_protocol_producer.py](../src/lung/engine/endpoint_protocol_producer.py:185)。它复用现有真实表达响应实现，执行路线为：

```text
标签隔离输入桥 → 28 候选 → 去重为 14 个标准请求
→ 固定 fit 参考与新来源映射 → T-only EngineData
→ 训练词表内的 context → request_id 为键的连续 24h 动作
→ ExternalConditionWorld → 参考 root → generate_feature_rows
→ 固定特征、实际 reference draw、生产回执 → 严格 cache loader
```

标准请求按 request_id 生成一次。同药同剂量的 6h＋18h 候选与持续 24h 候选共享这一标准参考描述符，但保持不同真实协议。该描述符不是洗脱终点 RNA，也不是 Nav 衰老状态的实测初始 RNA；来源到同一 Tahoe 参考的映射登记为 **explicit_assumption_unvalidated**。

固定描述符包括 population[d_model]、当前 missing-masked bio[27]、background/source/protocol context[3×context_dim]、单段历史 history[1, history_input_dim]，以及 elapsed=horizon=24。历史宽度由实际 ModelConfig 定义为 d_model＋3×context_dim＋1。现有生物特征的基因集合仍缺失并带 mask，不能把 27 维当作已证实的机制特征。C 分类概率保留为诊断，不进入描述符或替代 Annexin。

主代理亲读了 [BiologicalWorld._features](../src/lung/engine/world.py:403)：每次动作形成一行历史，最终 torch.stack 得到二维矩阵。本轮纠正了初版 producer 将它误当一维向量的错误，并新增 flatten history 拒绝测试；mock 的成功不能代替这项原函数核查。

生成与加载检查真实输入及指纹、参考与控制细胞索引、来源映射、基因轴、fit 预处理、生产代码与 checkpoint。权重要求 stage、data identity、epoch 和 optimizer_step 对应且非 synthetic；历史 latent_ae/latent_cfm 元数据缺 synthetic 字段时，只允许这两类在当前真实 data identity 与 config.synthetic=false 的约束下接受缺字段，不泛化到其他阶段。

回执绑定实际 root、抽样 seed、控制 cell IDs、population 形状与字节摘要，生成前后复核依赖。loader 校验固定 14 请求覆盖、维度、有限数值、特征内容 identity 和实物 hash。**这些检查证明工件一致性，不能作为密码学执行证明**；协调重写所有自洽文件仍不能由 hash 独立判定是否真实执行过。当前没有执行生产，不能宣称每个特征已有实际计算证据。

## 3. 完整实际协议怎样进入 G/Q

[ProtocolConditionAdapter](../src/lung/engine/endpoint_protocol_model.py:49) 将以下真实字段拼成具名数值向量：药物指纹、实际剂量、暴露 6/24h、洗脱时钟、无药恢复 18/0h、共同终点 24h、换液事件、前药保留/浮游细胞回收类别、作者状态及其协变量。

未知数值同时保留 known mask，未知类别保留 unknown 位；名义接种数量保留原单位，不换算为测量时密度。Nav 明确存在且值为 None 的 induction_compound 表示已知无诱导；Viel 未提供该字段表示未知，本轮已区分。标准 P 的 history 与时钟只用于溯源，不能取代真实候选历史。

[enumerate_with_g / rank_with_candidate_q](../src/lung/engine/endpoint_protocol_model.py:187) 将实际向量传入分别注册的评分器。同一排行要求 source、细胞背景、state、剂量解释和完整 endpoint_definition 可比。scope 不锁定剂量或 protocol arm，因而允许 Viel 同状态下两臂比较；Nav 需先固定状态和 10/11μM 解释。

评分器必须有非空 model、algorithm、evidence 身份、唯一 feature_names 和明确 G/Q 类型。返回分数检查 fraction/percent 范围，保留 ties，不加剂量、时长或其他成本偏好。本轮输出仍是 **scored_supplied_scorer / scorer_training_verified=false**；任意 caller 函数被注册不等于已有合格训练模型。这正是下一轮要补的正式学习与文件加载接口。

## 4. 真实监督怎样保存

[EndpointProtocolSupervision](../src/lung/data/endpoint_protocol_supervision.py:49) 是显式读标签入口，与生产 P 的入口隔离。[已实际构建的监督包](../08-datasets/LungCancer_A549_Integration/endpoint-protocol-supervision-v1/README.md) 包含三个视图：

| 视图 | 条件均值数 | 用途 |
|---|---:|---|
| vielhauer_source_nominal | 20 | 同共同 24h 终点的完整暴露方案 |
| nav_figure_caption_10uM | 4 | 图注剂量解释 |
| nav_methods_11uM | 4 | 方法剂量解释，同一组 Y 的敏感性分支 |

两个 Nav 视图不能合并为 8 个独立观测；combine_views 会拒绝重复 observation_id。percent→fraction 只转换单位，保留原值、assay、分母、读图上下界和 source_record。一条来源观测权重为 1，不按生成细胞数增加样本权重；读图上下界不是生物 SD、SEM 或置信区间。

当前角色全部 unassigned，roles_frozen=false。API 可接受明确 fit/final_test 两类角色，没有强制四个非空分区。全部公开结局已经读取，后续协议只能称回顾性划分，不能称未看过结果的盲测。loader 从冻结来源重新构造每条记录并逐项比较，实际 readback 已通过。

## 5. 实际验证与当前阻断

主代理聚焦验证：

```powershell
F:/anaconda/envs/study/python.exe -m pytest tests/test_endpoint_protocol_producer.py tests/test_endpoint_protocol_model.py tests/test_endpoint_protocol_supervision.py tests/test_endpoint_protocol_feature_binding.py -q
```

结果 55 passed in 5.71s。producer 的成功生成/读取路径使用纯 mock data/world/checkpoint reader；这些检查没有构造真实网络、优化实际模型或运行真实推理。[独立复审](../experiments/20261011-data-alignment-round14/independent_review.md) 核查真实原函数和语义，并复核两项 adapter 修复。

实际 CLI metadata preflight 结果为 pending，缺 response/pool/state/representation，checkpoints={}。这仅说明**当前登记的 pipeline 没有匹配权重**，本轮未扫描全部磁盘，不据此声称全机无可用权重。监督 CLI --verify 实际读回通过。整轮文件验证见 [validation.json](../experiments/20261011-data-alignment-round14/validation.json)。

## 6. 与原架构及完整目标的对应

| 原要求 | 本轮已实现 | 尚需验证 |
|---|---|---|
| §2/4.2 状态与完整候选条件 | 真实协议、结构、剂量、时钟、状态和未知值输入 | 信息是否带来真实预测增量 |
| §4.1/4.3 条件响应 P | request-keyed 实际生产程序、标准参考定义、cache loader | 匹配权重、真实 cache、跨源参考迁移及表达留出效果 |
| §4.3 实测 Annexin G | 真实均值视图与条件消费接口 | 正式拟合/加载、角色与校准、同定义留出效果 |
| §4.4 私有候选 Q | 显式完整候选评分路由与 G 枚举对照 | 正式 Q 参数学习/加载、冻结回报资格、独立选择收益 |
| §6 整体增量与功能验证 | 单位、可比 scope、测试隔离要求已落实到接口 | 同一冻结 P→G/Q 程序的真实整链评价 |

细胞表达预测的状态增量可在第十二轮真实表达设计中检验，不要求每条功能 Y 都配同样本初始 RNA。另一方面，细粒度初态表达超出类别/均值的功能增量、状态×实际暴露时长及同一细胞未来命运仍没有直接验证。Viel 有真实时长选择块，Nav 有真实两状态×两药块，但不能把分开取得的两块当作完整状态×时长联合数据。

下一步实现正式 G/Q 学习和加载，以及按 observation/完整条件冻结的回顾性评价协议；在明确训练授权与匹配权重之前只运行元数据预检。架构、旧消费者、v1 输入与监督包、核心 r5/v6/coverage-v5 保持原身份。Cloud/Dot 本轮没有新派单，沿用已返回意见；未提交或推送。完整目标继续 active，不能以 55 项接口检查验收完整科学可行性。
