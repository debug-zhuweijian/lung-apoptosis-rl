# Tahoe跨板同条件覆盖：元数据窄审计

日期：2026-10-10。**实际元数据存在可进一步核验的跨板比较，不应从外部终点核心表的“0个实测初始表达条件”推断整个Tahoe不存在参考群体或表达评测机会。** 本轮没有读取表达矩阵、训练模型或确认板间差异属于生物状态。

输入为`training-identity-v3/conditions.parquet`、`control_links.parquet`及其complete manifest；对应哈希见[audit.json](audit.json)。完整条件身份摘录见[condition_design.csv](condition_design.csv)，跨板组与成员ID见[repeated_treatment_groups.csv](repeated_treatment_groups.csv)。

| 计数 | 已实际复算的结果 |
|---|---:|
| 总条件 | 1336 |
| eligible处理条件 | 1305 |
| 处理实验板 | 14 |
| vehicle条件 | 28，每板2个 |
| 完整化学身份×浓度×时长的不同组 | 1128 |
| 同处理出现在至少两板的组 | 154 |
| 这些跨板组包含的条件 | 325 |
| 处理时长 | 全部24h |

分组使用`chemical_or_experiment_group_id`、`dose_um`、`duration_h`，不按药名近似匹配。另有6个只在板内重复的组、共12个条件；不能与154个跨板组相混。原始control_links共2616条，包括后来排除的处理记录；其关系均为`same_plate_control_candidate`，逐细胞配对均为`not_available`。处理记录的replicate_scope均为源样本单位，尚未核实为独立生物重复。

## 对原课题的意义

这是本课题的待验证设计：在相同处理的跨板条件及同板对照基础上，比较对应参考群体分布、只用其均值、统一参考群体，以及plate-only/QC基线。先核对实际表达、板的实验协议及目标划分，再判断能否以留出表达分布评价参考信息的预测效用。这个设计不是本文已经获得的模型效果。

对照是相应采样时点的平行群体，不是该处理细胞在给药前的实测t0。14块板不等于14种已定义生物状态，154组跨板重现不等于154个独立生物重复。即使匹配参考改善药后表达预测，也还要排查批次/质量因素；它不直接证明RNA对凋亡功能结局的增量价值，更不能恢复同细胞未来命运。

独立复审已逐组核对154个组的成员ID、剂量和时间。对应完整目标说明见[Round5覆盖复审](../../../../../docs/FULL_DATA_FEASIBILITY_COVERAGE_ROUND5_2026-10-10.md)。
