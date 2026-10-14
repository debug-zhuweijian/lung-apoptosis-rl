# 状态增量的三条独立 full-P 对照，Round22

本轮补上状态增量对照的真实可调用训练与消费路径。新增 `observed`、`zero_h`、`mean_h` 三条独立 full-P 实验模板，训练、selection、保存参数、来源声明、未来推演和 native final 都必须使用同一变体契约。当前三条仅完成真实数据元信息预检和合成工程验证，没有真实权重、正式拟合或状态收益结果。

研究必要性与目标对应毕业论文 `研究架构.md` 的状态信息增量主假设和 §4.3 表示/响应职责。三条对照控制同一数据、角色、来源/背景/协议、候选条件、训练目标、OT 耦合、优化器与奖励，只改变 P 的参考 h 输入。本课题的检验设计是分别拟合 \(P_{\theta_v}(y\mid T_v(h),c)\)，在相同留出表达与度量下比较；结果可以支持或否定增量假设，不保证 observed 优于基线，也不将来源状态 ID 字符串当作已经编码的状态。

| 变体 | P 实际参考输入 | 独立拟合/selection 与消费 | 可检验边界 |
|---|---|---|---|
| observed，默认 | 原逐细胞冻结 E(h) | 原参考 population 的逐细胞 h | 基准/当前可用表达状态，仍受可获得参考条件限制 |
| zero_h | 全零 h，预测阶段不调用 E(reference RNA) | 同条件编码、模型维度与目标；拟合/selection/P 都 h=0 | 无 RNA 状态的 P 输入；共享 AE、目标表达和原 OT 监督仍使用 RNA，不称整个过程无 RNA |
| mean_h | 同一允许参考 population 的冻结 h 均值，重复到每个样本 | 整份 population 先求 h 均值，再做 P microbatch | latent 状态摘要对逐细胞状态信息；不是 E(expression_mean)，不是每块不同均值 |

CFM 用原 `_coupled` 返回的允许参考 h（原 OT 后 `h_pool[ii]` 的有限抽样，等 selected-row 权重，可能重复）变换后才分块，原 source-target coupling/target z 不改。B4/selection 使用原 uncoupled control-condition→cell 抽样的完整有限 reference draw；serving 与同 P C sidecar 使用固定全部 2431 fit proxy reference 的 equal-cell h，native final 使用其当前允许的完整 final reference draw。相同算子的有限集合、权重和 n 不同，不声称同一数值总体均值或无限总体均值。阶段人口已写入 mandatory `population_definition_by_stage`，`same_numerical_population_mean_across_stages_claimed=false`，旧/改人口合同严格拒。抽样范围和表达坐标按原职责保留，未虚构同细胞 t0/target 配对。

## 最小实现与严格版本契约

新增逻辑位于已有 [engine producer](../src/lung/engine/full_protocol_response.py)、[独立 learner](../src/lung/engine/full_protocol_response_learning.py) 和 [CLI](../scripts/train_full_protocol_response.py)。`models/full_protocol.py` 原字节 SHA `c6bdce150ed8a0cc53e3e26d3f438597805f918c6a4a4e1fa4f6304e1aa64e27` 未变；Bothe live 数据 PIN、旧 latent/model、旧实验与现行 YAML 未修改。

配置字段为 `research.full_protocol_response.reference_variant`，默认 observed。settings 增加 `settings_version=fullP_learning_reference_variant_settings_v2` 与 `reference_variant`。实际 P checkpoint metadata、模型 declaration、输出和执行审计均保存 `reference_contract`，schema 为 `fullP_reference_h_input_v1`，包括 variant、允许 population 口径、RNA_state_used_by_P、mean_coordinate 和 source_state_id_numerically_encoded=false。actual metadata 必须与 declaration 一致，缺失字段或把 zero_h/mean_h 权重声明成 observed 均拒绝，不能推断默认变体后静默载入旧权重。

engine 共用 API：

```python
reference_contract(variant) -> dict
declared_reference_variant(metadata) -> str
apply_reference_variant(h, variant) -> Tensor[N,D]
encode_reference_h(shared, reference_tensor, variant) -> Tensor[N,D]
prepare_reference_h(shared, reference_array_or_tensor, variant,
                    *, device="cpu", microbatch=64) -> Tensor[N,D]
```

Adapter 的 `.reference_contract` 来自核验过的 actual metadata；serving 用 `.predict_h(h_slice, physical, numeric, noise=...,steps=...)`。其 physical 合法域保持 Viel 已知 active exposure，native 的未知暴露不能借此改写为 known；native final worker使用 `prepare_reference_h` 后再原 `full.sample_latent` 并保持真实 unknown mask。

P-only 设置通过 `joint_ae_config` 从 shared AE 的训练配置中剥离，三条 P 可共用同一完成 AE、同一冻结 cache 和同一个原 C。C `state_config` 也剥离该 P-only 设置。AE/原 C 目标、数据与角色不因 ablation 变化。

当前 C 源从 f39ed74f 到 6e613e1 的最小改动只有：调用上述 AE-only config；同 P sidecar先对完整 reference 准备 h，再 `predict_h` 切片。原 C reader/final/三类概率/17项指标/58 gate 规则未改。该调整防止 mean_h 在 P 和 C 复演中变为每块不同参考；同 P float64 gene mean/variance 精确复演口径保持。

## 数据准备和实际检查

[变体计划与 guard](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/variant_plan_and_guard_v2.json) 实际逐条检查三变体 settings 和 1498 fit/374 selection native 请求；同 protocol identity、未知暴露/历史 mask、目标、奖励及 AE/C 配置成立。PT load、NN 初始化、真实拟合/forward、真实 RNA samples reopening guard 全为 0。

[来源只变更读回复用](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/source_only_data_reuse_v2.json) 核 gene、normalization、data identity 和 reference identity 与前已读取的 2431 reference/24 native vectors 一致；更新 producer source binding，未重新打开 75 原始 reference shards。旧 RNA 回执按原版本保留，当前 source readiness 单独记录，不能把旧接口 CI 直接覆盖到新源码。

P/learning/C 当前 **78 passed/30.95s**（I 新 `experiments/a22t4`，[日志](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/pytest_v4.txt)），其中 producer32、learner19、C27。验证 zero_h 预测不调用 E 且参考改变不改结果；mean_h 同整群 h 均值并独立于切块、不同于 E(mean x)；三条 tiny 独立拟合在 CFM/B4/selection 消费同变体；跨变体恢复在 PT 前拒；mean_h 中断恢复的 optimizer、AE、metadata、checkpoint 完全一致；默认既有消费和 C final 仍通过。初次新增测试将 toy 的 selection 调用数错算为4（实际2），3项断言失败，75项其他通过；修正本测试为共8调用，未据此改变模型语义，失败简报保留在计划回执中。

## 模型授权后的可执行顺序

三份完整独立模板已保存：

- [fullP_observed.yaml](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml)
- [fullP_zero_h.yaml](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_zero_h.yaml)
- [fullP_mean_h.yaml](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_mean_h.yaml)

先使用原完整 B1 独立 namespace `experiments/full-protocol-joint-AE-v1` 拟合 matching joint AE，完成 proof 后共用同 frozen representation。再按计划内三条完整 `train_full_protocol_response.py --config <对应模板> --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/full-protocol-native-P-v2-<variant>` 分别拟合和 selection；各自完整模型声明、参数、cache origin、actual registered tensor audit与角色身份独立保存。

native final 使用每条已完成 namespace 的同 `reference_contract`，固定源/角色/样本/协议和目标指标，分别产生真实 heldout 评价后比较。新 worker已同步共用 helper，不允许零 h 训练后改用 observed 来评估。一次换 reference 的旧权重推演最多是对同模型输入的敏感性分析，不能替代三条独立重训 baseline。

若后续比较联合选择，三条变体各自建立完整20候选 P/index、同 P C sidecars、G/Q训练与冻结评价 namespace；同一个 view 的评分不能混用不同变体模型或不同信息。共享原候选域、角色、目标与评价规则，保留 P-only/P+C 消融；状态增量首先由表达 heldout 判断，其迁移到独立功能终点与联合选择的收益另需实验。

native 部分协议监督没有新增 washout/recovery RNA，Viel pulse 等仍 OOD/未验证。参考 h 控制对照并未接入数值 source_state_id，不证明真实纵向状态轨迹、多步滚动或序贯治疗；原完整研究范围保持未完成而未降级。当前缺 matching AE/P 参数、真实三独立模型训练与 final 评价、条件增量效果及跨来源功能验证，正式模型运行仍禁止。

## 独立增量复审与当前接续

独立复审已亲读当前 producer `af7aa029…`、learner `103e933f…`、CLI `91f53b8a…` 与 C `6e613e1b…`，并核对原 `models/full_protocol.py` PIN 不变；结论为 **CLOSED_ENGINEERING_ONLY**。I 新 `experiments/a22rv3` 的 17 项定向检查通过（12 项独立检查和 5 项 owner 检查，10.25s），覆盖原 OT 的均匀 marginal、有限参考均值、强制阶段人口合同、三变体 C 的完整 reference 复演、训练/B4/selection 同变体及恢复身份。[独立结果](../experiments/20261011-data-alignment-round22/independent_reference_variant_review/results.json) 与 [审查说明](../experiments/20261011-data-alignment-round22/independent_reference_variant_review/technical_review.md) 保留执行范围、源码 SHA 和旧窗口回执。

独立四行合成 OT 检查实际显示 pool 均值与 post-OT 有限抽样均值可以不同，row/column marginal 仍满足原 L1 ≤ 1e-6 检查；该结果支持阶段集合边界的登记，不能仅凭 coupling 的目标调用依赖证明统计目标泄漏。来源与数值见 [合成 trace](../experiments/20261011-data-alignment-round22/independent_reference_variant_review/stage_population_trace.json)。本轮独审没有真实 RNA 重开、真实 PT、模型 forward、拟合或功能标签 Y 读取，保留工程测试与科学证据的区别。

native final 的三项函数检查发生在该 owner 修改中的源码窗口，独立审查明确排除其最终版本，不能计为本 A 复审对 native final 的闭合；最终 native 文件由对应 owner 与独立 reviewer 验收。当前接续绑定保存于 [current_readiness_v3.json](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/current_readiness_v3.json)，旧 v2 与原真实读回均保留。下一步仍是获得模型运行授权后，按三份独立模板分别拟合、在相同留出目标下评价，再另行检验功能与联合选择收益。
