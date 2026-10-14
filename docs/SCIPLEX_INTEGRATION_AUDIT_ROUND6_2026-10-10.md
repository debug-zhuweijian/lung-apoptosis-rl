# sci-Plex 实际响应模型接入审查与验证（Round6）

日期：2026-10-10。执行目录：`I:/科研/肺癌细胞凋亡-强化学习`。

## 当前结论与边界

已完成新的 Tahoe + sci-Plex 联合元数据包，并以真实原始计数验证了 `TrainingData → EngineData → condition_kwargs → ConditionEncoder`。sci-Plex 已能进入现有响应模型的实际数据入口，不再只是旁路审计表。新包状态为 **`metadata_ready`**；本轮没有对真实新包拟合归一化、邻接图或 MMD 带宽，没有启动正式训练，不能称为训练就绪或预测性能已验证。

这次接入解决的是来源、细胞行、基因列、药物输入、剂量、来源内终点时间、对照和数据划分的工程对应关系。它没有建立同一细胞处理前后的实验配对，也没有把表达响应与外部凋亡测量伪造为逐样本配对。与毕业论文研究架构第 4.2 节候选条件表示、第 4.3 节条件响应预测的职责相容；凋亡功能评估、状态条件化增益和干预筛选的科学有效性仍须各自验证。

## 1. 输入来源与版本

| 输入 | 实际文件及作用 |
|---|---|
| 原始表达行和细胞元数据 | `08-datasets/sciPlex3/GSM4150378/extracted/v1/a549_cells.parquet`、`extraction_status.json` 中登记的 common CSR NPZ 分片 |
| 基因轴 | 同目录 `common_genes.parquet`；与旧 v3 的 32,461 个 Ensembl ID 顺序逐项相等 |
| 药物输入 | `endpoint-alignment-v1/source_reviews/sciplex_round6/chemical_inputs_v2/`；188 个目录记录、187 个可用的 2,048 维主指纹 |
| 时间和对照协议 | 同一审查目录 `protocol/time_control_contract.json`、`qc_protocol_audit.json` |
| 完整计划孔 | `protocol/a549_planned_wells_with_hash_counts.csv`；1,920 个 A549 孔，包括没有 QC 细胞的孔 |
| 原有基础版本 | `identity-v3` 和 `training-identity-v3`；只读依赖，没有覆盖 |
| 显式新配置 | [configs/data_joint_sciplex_v1.json](../configs/data_joint_sciplex_v1.json) |
| 实际新输出 | [training-joint-sciplex-v1](../08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/README.md) |

上表的 `endpoint-alignment-v1` 位于 `08-datasets/LungCancer_A549_Integration/`。化学侧表输入路径明确相对项目根目录，输出路径相对侧表目录。加载器按该契约验证 manifest、依赖表和输出内容 hash，不能把文件路径字符串当成内容身份。

主要固定身份：

- 新配置 SHA256：`ccaff6906a6ebabb1d443edbeffc271af0fec92fae194cdac89aa090b921a92a`。
- 新 manifest SHA256：`1944f2880180b711ba8ddb196fedf0b4386794140ca17ea663673ed169c27bd9`。
- 新 training signature：`ce85e8447b6ec447407aa9f67bd6e9044d9f0475f0cc3c0c28633921f980dc86`。
- 化学侧表 v2 manifest SHA256：`c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05`。
- 本轮所读旧 training-v3 manifest SHA256：`95e1c0cbc60bf9410ae699442d8d28f542af5de11f6bb6e68fc3545fba5df18a`。

## 2. 每一层怎样实际对应

| 层次 | 来源字段 | 新包字段及使用方式 |
|---|---|---|
| 单个细胞 | `cell`、`shard_id`、`row_in_shard` | 来源前缀 `cell_id`、`matrix_path`、`matrix_row`，精确定位表达行；不依靠药名定位细胞 |
| 实验孔 | `catalog_number,time_point,dose_uM,replicate,culture_plate,well_oligo` | 这六项生成稳定 `condition_id`；保留源 `plate_oligo`、replicate、well 信息 |
| 药物 | `catalog_number` | `drug_id=sciplex:<catalog_number>`，与 v2 侧表一对一连接，再读取 `fingerprint_row` 指向的完整结构指纹 |
| 剂量 | 官方计划孔 `dose` 为 nM，提取表为 uM | 计划值除以 1,000 后与提取表分组和 QC 数逐项一致；模型读取 `dose_um` |
| 时间 | 作者报告 24/72 h | `source_time_h`、`reported_treatment_hours` 保留其语义；兼容现有模型的 `duration_h` 取同一数字 |
| 来源/协议 | sci-Plex 的来源与处理协议 | `source_id=sciPlex3:GSM4150378`、独立 `protocol_id`；模型使用冻结词表中的整数编码 |
| 对照 | 相同来源、细胞系、时间、replicate、culture plate | `control_links.parquet`；每个可用处理孔连到该上下文的两个 vehicle 孔，实际取同一 role 的对照细胞 |
| 化学实体 | 完整源结构 | `chemical_or_experiment_group_id` 保留完整结构身份，供模型与动作身份使用 |
| 防泄漏划分 | 完整连接图/最大有机片段连接图 | 独立 `split_group_id`，仅决定新包 role，不替代化学身份或输入指纹 |

`sciplex_source.py:_source_rows` 从完整计划孔出发，因此没有 QC 细胞的孔依然存在于 conditions 中，并标注原因。QC 未通过的细胞不进入训练索引；结构不满足当前单结构输入契约的处理孔也保留元数据而暂不进入该训练索引。

对照是同一实验上下文的**同期终点 vehicle 细胞群**，不是每个处理细胞已经观测到的处理前状态。链接中明确保存 `individual_cell_pairing=not_available`。模型的分布匹配/耦合不能升级为真实细胞配对。

同一个vehicle孔的不同细胞可分配到不同role，作为明确声明的共享上下文参考；细胞不重复并不意味着对照孔或培养批次独立留出。本包不能据此声称“新对照孔／新批次完全独立泛化”。

## 3. 基因列与实际调用链

旧 Tahoe 原始计数的列数和顺序不同于公共基因轴，必须使用 `genes.tahoe_column` 投影。sci-Plex 的 common NPZ 已经处于 32,461 公共基因顺序，只能使用 `arange(32461)`。将旧 Tahoe 列号机械应用于 sci-Plex 会越界或错误投影。

本次保留原有 `SparseRows` 方法，新增 [RoutedSparseRows](../src/lung/data/reader.py)，依据冻结 `source_shards.parquet` 的 `gene_axis` 选择读取器。它保持请求行顺序、重复行和稀疏格式，并继续使用已有 CSR 缓存验证机制。直接输入是压缩 NPZ；CSR 目录属于既有缓存实现，不把二者混称为同一种源格式。

实际调用链：

```text
configs/data_joint_sciplex_v1.json
  → prepare_training / extend_with_sciplex
  → conditions + fingerprints + control_links + train/val/test + source_shards
  → TrainingData / RoutedSparseRows
  → sample_response / sample_response_group / sample_ae_group
  → EngineData.flatten
  → engine.response.condition_kwargs
  → ConditionEncoder（完整结构指纹、剂量、24/72 h、来源、背景、协议）
```

不能只改普通 reader 而遗漏训练旁路，因此下列入口同时接入同一读取工厂：

- [preprocess.py](../src/lung/data/preprocess.py)：新包的 fit 原始计数读取使用路由；统计仍只在 fit 数据内拟合，保留来源等权、条件等权、细胞等权的既有规则。
- [bandwidth.py](../src/lung/data/bandwidth.py)：MMD 采样表达使用同一路由。
- [engine/ae_data.py](../src/lung/engine/ae_data.py)：AE 专用 reader 的复制保留双轴路由，不退回只认 Tahoe 的构造。
- 既有表达缓存、潜变量缓存通过 `reader.rows.read` 进入路由；数据签名和分片注册内容进入缓存身份。

新词表需要 `n_sources=2,n_backgrounds=1,n_protocols=2`。`EngineData` 对显式响应模型配置校验容量，不足则报 `MODEL_CONTEXT_CARDINALITY_MISMATCH`。新包的划分、协议编码和数据身份均改变，旧权重或旧缓存不能被自动宣称适配。正式模块配置未被本次修改。

现有独立 `timecourse/TimeData` 路线也已检查：它服务另一套已知 24 h 到目标 72 h 的协议，当前本机 `timecourse-v1/manifest.json` 未生成，且不提供模块训练所需的全部 role/AE 缓存接口。因此本轮扩展实际 `TrainingData`，没有把两套评测协议合并或修改旧时间路线。

## 4. 化学输入与新划分

187 个可编码目录条目保留作者报告的完整结构，包括盐/多片段和未指定立体信息；`eligible_for_structural_features=True` 只表示现有编码器能够读取。它不表示历史实验试剂、批次或跨来源实验配对已经独立核验。18 个源目录项与当前 v3 的完整结构相同；保守连接图关系得到 25 个重叠候选，这两个数字不可互换。

S4246 的官方结构中虚拟原子关联相对立体信息，不能删去后选一个绝对对映体。v2 侧表保留这一记录，`fingerprint_row=-1`、结构资格为 false；两个镜像候选及指纹单独存放，没有组成比例，也没有伪造为两个实验。加载器拒绝把带虚拟原子的结构标为可用，拒绝为这种情况塞入零指纹。其 8 个条件、954 个 QC 细胞均为 24 h，保留元数据但暂缓现有单结构响应训练。

原先若强行保留新包内的旧 Tahoe role，会产生 3 个跨旧角色冲突：S1068、S2673、S5001，涉及 2,174 个 QC 细胞，全部为 24 h，72 h 为 0。记录见 [legacy_split_conflicts.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/legacy_split_conflicts.json)。

采用已锁定的 `joint_full_and_parent_connectivity_components_v1`：实际非空完整连接图或最大有机片段连接图连边，去除立体信息但保留同位素，整体连通组重新分配六个 role。空值不连边。保留 `legacy_v3_role` 审计旧划分，不以冲突为由丢弃这三种源药。生成 512 个防泄漏连通组；可用条件仍具有 545 个完整化学实体组，二者保持独立。

## 5. 实际规模及缺口

| 范围 | 数量 |
|---|---:|
| 提取的原始 A549 细胞 | 244,281 |
| 作者 hash QC 通过 | 210,458 |
| QC 未通过 | 33,823 |
| 新包可用 sci-Plex 细胞 | 209,504 |
| 其中 24 h / 72 h | 142,061 / 67,443 |
| 完整计划孔：处理 / vehicle | 1,880 / 40 |
| 可用处理条件 | 1,871 |
| 相对立体结构暂缓条件 | 8 |
| 无 QC 细胞的计划处理孔 | 1 |
| 新增严格上下文对照链接 | 3,742 |
| 联合表达索引总细胞 | 2,762,092 |
| train / val / test 细胞 | 1,705,349 / 856,061 / 200,682 |

无 QC 细胞的孔为 S7207、72 h、rep2、plate52、well `plate10_D10`、10,000 nM；记为 `planned_well_no_qc_cells`，不能赋予零响应。作者 valid cell ID 与当前 QC 集合的一致性由协议审查包记录；它不自动解释正文中另一 72 h 总数与源文件计数的差异。

本包六个 role 的细胞数为 fit 1,453,616；selection 316,011；residual_fit 251,733；aux_selection 259,150；calibration 280,900；final_test 200,682。数量是细胞数，不能作为独立药物或实验重复数使用。

## 6. 时间协议与决策隔离

模型可编码作者报告的 24/72 h 来源内终点时间。`time_semantics` 从审查契约读取，换液、补药、撤药、洗脱未确认的信息保持未知：`exposure_duration_h=null,withdrawal_protocol=null`。时间点之间是破坏性采样的不同群体。

所有新增 sci-Plex 条件均 `decision_eligible=False`。`BiologicalWorld._action_metadata`、`roots`、`root` 按显式字段隔离这些条件，旧无该列的行为兼容。该设置防止新增终点监督自动变成共同 T* 策略候选。24 h 处理加上未观测的 48 h 恢复不能当作实测 72 h 条件；本包也不证明真实连续暴露或逐细胞 24→72 h 轨迹。

## 7. 已执行的验证

真实构建命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/prepare_training.py --data configs/data_joint_sciplex_v1.json --metadata-only
```

实际小读命令和机器可读报告：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/verify_joint_reader.py'
```

[joint_reader_validation.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/joint_reader_validation.json) 记录：全索引元数据无重复细胞，防泄漏组不跨 role，3,742 个新对照链接上下文正确，S4246 和缺失孔未被错误纳入训练。旧 v3 签名仍可验证。

真实数据小读使用 `require_preprocessing=False`、关闭磁盘缓存写入。选取 sci-Plex 24 h、sci-Plex 72 h、Tahoe 24 h 各一个条件，每个读取 6 行 × 32,461 列，检查被读取 NPZ 的 SHA256；三次均真实调用 `EngineData` 和条件编码器，得到有限的 `3×256` 输出。这只是随机初始化编码器的输入链路核验，未训练、未归一化，不构成预测性能结果。

定向回归命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 -m pytest tests/test_sciplex_response_integration.py tests/test_ae_data_supply.py tests/test_training_data_v1.py tests/test_identity_prepare_v2.py tests/test_reviewed_identity_v3.py -q
```

结果：**60 passed（40.11 s）**。新小型 fixture 覆盖不同原始基因顺序的双源读取、重复行顺序、实际小型 fit 预处理/MMD/AE 路由、条件编码器 72 h 输入、协议容量不足拒绝、完整计划缺孔、虚拟原子拒绝、`-1` 结构资格边界、依赖改动拒绝和 world 隔离；旧 v1/v2/v3 接口回归同时通过。

## 8. 未完成事项

1. 真实新联合包的 fit-only 归一化、基因图、MMD 带宽和必要缓存尚未建立；接着运行必须输出到新资产目录，不能续用旧数据身份的已拟合参数。
2. 联合响应模型尚未正式训练或验证跨药泛化；现有编码器前向成功不能替代这些实验。
3. 来源内药物—剂量—时间组合补全（M1）、严格 72 h 外推（M2）及共同终点决策属于独立评价契约；当前全局药物留出包没有自动证明它们。M1不是状态条件化收益检验，准确范围见对应评价契约。
4. S4246 相对立体实体需要适配不确定化学表示或另外取得充分物料证据；原有 Quinidine 混合物缺口也继续保留。
5. 表达与凋亡功能终点的跨来源桥接、外部校准和有效干预筛选仍需按主研究的证据要求验证，不能从本次行列对齐推导为已解决。

源码变更保持可选新配置入口，原 `prepared/v1`、旧 identity/training 版本及正式训练配置未被本次覆盖；没有正式训练、提交或推送。独立复审由另一代理另行记录在同目录 Round6 review 文档。
