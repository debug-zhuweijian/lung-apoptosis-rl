# 条件输入 v6：SJB 24 h 剂量条件追加

构建和核验日期：2026-10-10。由 `scripts/extend_endpoint_alignment_round9.py` 调用现有 `EndpointBuilder`、`write_alignment` 与 `build_endpoint_condition_inputs.main` 生成；原 r4/v5 和通用 builder 未修改。

**实际新增 4 条 SJB 发表均值观测、4 个条件，其中 3 个非零给药条件具有已核官方名义结构、剂量和 24 h 输入。没有生成表达特征，没有训练模型。**

| 实际指标 | r4/v5 基线 | r5/v6 |
|---|---:|---:|
| 条件 | 304 | 308 |
| 观测记录 | 842 | 846 |
| 其中 observed | 840 | 844 |
| 带排除原因记录 | 16 | 16 |
| 材料/化合物记录 | 282 | 283 |
| 结构可用材料 | 51 | 52 |
| 有化学输入的条件 | 56 | 59 |
| A549、单药、24 h、有合格 Annexin 记录且无输入阻塞的条件 | 14 | 17 |

最后一行只是条件输入候选口径。它们仍有来源和实验协议差异；**17 不代表统一 G 已校准、17 个可直接互换的终点、盲测完成或状态迁移通过**。

## SJB 的数据语义

- 条件来自 DOI 10.3390/cimb47030155 的 Fig4A，剂量 0/0.625/2.5/10 µM，图注明确 24 h。0 µM 为来源对照，不生成药物结构输入。
- 每个图均值只登记一条观测，`replicate_kind=published_mean_not_raw_replicate`。作者 n=9 的重复类型未知；汇总 `available_numeric_records=1`、`measured_replicates=0`，生物重复 SD 保持缺失。
- 图示四象限值没有进入这四条均值观测。百分比按除以 100 转为 fraction，像素读图范围独立保存，不解释为 SD、SE 或 CI。
- SJB 使用独立 `Wu2025_SJB2_043:Fig4A_Annexin_parent_gate` 协议，分母保持作者分析母门；上游 gate、事件数及漂浮细胞回收细节未报告，不解释为最初接种群体的自然凋亡率。
- 新来源的表达读取、同样本 RNA—终点配对和表达特征数均为 0。状态只登记未实测表达的 A549 参考背景。

## 材料与旧审查链

新 `source_reviews/state_function_round9/sjb/chemical-review-v1` 完整继承 v5 使用的 `chemical_round3/revision-2` 三个候选，保留 6k 与 8h 的暂缓状态，另新增 SJB。

SJB 的官方名义产品为 MCE HY-15757、CAS 63388-44-3；论文明确厂家，当前官方产品/数据表明确结构。原论文未报告历史 catalog/lot/该批纯度，相关声明保留。结构检查通过不等于原试剂批次已核实。

新审查 manifest 递归绑定旧 revision-2 和其祖先 manifest，并绑定官方材料原件。v6 的输入 manifest 保存完整审查依赖哈希。

## 验证与研究架构对应

`round9_validation.json` 保存实际读回结果；`prior_packages_before_after.json` 保存原 r4、v5 和旧材料审查 85 个文件的前后哈希，全部一致。所有 283 行、每行 2048 位的指纹按当前冻结 SMILES 逐位重算；全部旧指纹按化合物 ID 比较不变，未审核材料仍为零向量并保留不可用标记。

15 项定向测试通过，包含完整临时构建、剂量/时长/分母/单位错误拒绝、重复接入拒绝、无伪 RNA、原材料保留和递归证据检查。`execution_record.json` 记录命令与结果。

已对照毕业论文《研究架构》§4.1–4.3、§4.5：本次完成的是数据与候选定义层；保留真实浓度/时长、协议 ID、来源及标签状态，没有把药后 RNA 当初态、没有用模拟结果替代终点、没有启用终点头或策略训练。`architecture_conformance.json` 保存此次核对的架构文件哈希及边界。`formal_training_ready` 仍为 false。
