# Tahoe + sci-Plex 联合响应数据 v1

构建日期：2026-10-10。状态：**complete，真实新包 fit-only 预处理及实际读取验收通过；尚未正式训练**。这里的 complete 仅指响应数据资产，不表示真实凋亡预测或完整课题可行性已通过。

本目录由项目根目录的 `configs/data_joint_sciplex_v1.json` 显式选择，与旧 `training-identity-v3` 分开保存。保存联合训练所需的实际 metadata/index/fingerprint/control-link/分片路由资产；表达计数继续只读引用 Tahoe 与 sci-Plex 已登记源分片，不复制全量矩阵。

## 数据身份

- config SHA256：`ccaff6906a6ebabb1d443edbeffc271af0fec92fae194cdac89aa090b921a92a`
- 完成预处理后的 manifest SHA256：`513c9650acec6613d61c6f32be26d68ec477d7fb9c3c38f43742829443cbba05`
- 完成后的 EngineData identity：`29092acd61b8d5d0f5d59642f43bf1df7c55e780e350d3e2f24fa4e55aa355d9`
- training signature：`ce85e8447b6ec447407aa9f67bd6e9044d9f0475f0cc3c0c28633921f980dc86`
- chemical_inputs_v2 manifest SHA256：`c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05`

原 metadata-only manifest SHA256 为 `1944f2880180b711ba8ddb196fedf0b4386794140ca17ea663673ed169c27bd9`；合法预处理新增归一化、图与带宽资产后改变 manifest 内容 hash，training signature 保持一致。README 是治理说明，不在当前 manifest 的数据文件 hash 列表内。

## 实际规模

| 范围 | 数量 |
|---|---:|
| 联合表达索引细胞 | 2,762,092 |
| sci-Plex 原始 A549 / hash QC 通过细胞 | 244,281 / 210,458 |
| sci-Plex 可用细胞 | 209,504 |
| sci-Plex 可用 24 h / 72 h 细胞 | 142,061 / 67,443 |
| 完整计划 A549 孔 | 1,920 = 1,880 处理 + 40 vehicle |
| 可用 sci-Plex 处理条件 | 1,871 |
| 暂缓 S4246 条件 / QC 细胞 | 8 / 954，全部 24 h |
| 无 QC 细胞的计划处理孔 | 1 |
| 新增严格匹配对照链接 | 3,742 |
| 完整化学实体组 / 防泄漏连通组 | 545 / 512 |

无 QC 细胞的孔是 S7207、72 h、rep2、plate52、`plate10_D10`、10,000 nM。条件元数据保留 `planned_well_no_qc_cells`，不将其解释为零响应。

## 使用契约

- `source_shards.parquet` 明确 `gene_axis`。Tahoe 通过原始 Tahoe 列号投影；sci-Plex common NPZ 按已有 32,461 公共基因顺序读取。`RoutedSparseRows` 已进入普通响应、AE、预处理和 MMD 路径。
- 细胞以源 cell ID 和源矩阵行精确对应。对照按同来源、A549、同作者报告时间、replicate、culture plate 匹配；它们是同期独立 vehicle 细胞群，不是逐细胞处理前观测。
- 来源内 24/72 h 写入模型时间输入；撤药/换液/补药/连续暴露窗口不确定性保留。新增条件全部 `decision_eligible=False`，不能自动成为共同 T* 决策候选。
- 模型上下文容量至少 `n_sources=2,n_backgrounds=1,n_protocols=2`。新角色划分、协议词表和数据身份要求新拟合/训练，旧权重不自动适配。
- 保留 188 个非 vehicle 源目录项，其中 187 个可用单结构主指纹。S4246 是相对立体实体，`fingerprint_row=-1`；镜像候选单独保存，不擅选一个绝对构型，也不放入零指纹。结构可编码不等于历史实验材料已独立验证。

## 新全局划分

policy 为 `joint_full_and_parent_connectivity_components_v1`。真实非空完整连接图/最大有机片段连接图构成保守连通组；去除立体信息但保留同位素。仅 `split_group_id` 使用这种防泄漏关系，`chemical_or_experiment_group_id` 和完整输入指纹保持原实体差别。

18 个 sci-Plex 目录项与当前 v3 完整结构一致；25 个具有保守连接图重叠关系。旧 role 冲突有 3 个：S1068、S2673、S5001，涉及 2,174 个 QC 细胞，全部 24 h，72 h 为 0。新联合包整体重分角色并保留 `legacy_v3_role`，没有因冲突删除这些药物。旧 v3 文件保持原样。

该全局跨药划分不等于 M1 状态条件化评测或 strict72h-out；这些结论需要独立协议和实验。

## 验证记录

项目根目录执行的 metadata-only 构建成功；`source_reviews/sciplex_round6/code_probe/joint_reader_validation.json` 位于同级 `endpoint-alignment-v1` 审查目录，记录真实 sci-Plex 24 h、72 h、Tahoe 24 h 小读取及 `EngineData → ConditionEncoder` 有限输出。验证全索引无重复细胞、防泄漏组不跨角色、全部新增对照上下文一致。

新接入/AE/旧训练数据/identity-v2/v3 五组回归：60 passed。元数据快照另经独立工程复审，15 项资产哈希、4 项关键测试及真实入口限制通过。

2026-10-10，实际预处理进程正常退出（exit 0）：仅在 fit 的 1,453,616 个细胞、1,566 个条件拟合，按来源等权、来源内条件等权、条件内细胞等权计算。10,151 个缓存分块、32,461 基因归一化、完整 32,461×33 图和 4,096 对细胞的五档核带宽均已完成。没有训练模型。

主代理完成后复验 19 个登记资产的大小和 SHA256、基因轴、图自身首列及邻居唯一性、核带宽与距离中位数。实际 `EngineData(require_preprocessing=True)` 仅开放 T/fit，分别读取 Tahoe 24 h、sci-Plex 24 h 和 72 h，每组 2 个处理细胞及 2 个对照；独立从原计数重算标准化，与接口输出一致。Tahoe protocol index 为 1，sci-Plex 为 0，旧权重和旧 reference 的编码不可直接照搬。完整证据与复验脚本见项目根目录 `experiments/20261010-joint-sciplex-preprocessing/complete_validation.json` 和 `verify_complete.py`。

完整工程说明见项目根目录 `docs/SCIPLEX_INTEGRATION_AUDIT_ROUND6_2026-10-10.md`。
