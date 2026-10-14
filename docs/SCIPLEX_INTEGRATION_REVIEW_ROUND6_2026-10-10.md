# sci-Plex 实际接入独立工程复审（Round6）

日期：2026-10-10。执行项目：`I:/科研/肺癌细胞凋亡-强化学习`。审查者与实现者分工独立；本审查未修改实现源码或冻结数据。

**结论：本次“联合元数据成包、读取与关键接口”的工程验收通过，未发现必须先修复才能继续新包预处理的错误。** 双源列轴、化学资格、防泄漏组、完整计划孔、对照上下文和决策隔离均取得了相应代码与实际资产证据。该结论绑定下述 `metadata_ready` 快照，不代表真实联合包已经完成预处理、训练或科学验证。

主代理在本次复审末段另行启动了真实 fit-only 预处理。本报告没有等待或验收该运行；其产物和更新后的 manifest 需要另作运行验收。不得把本报告的 4 项小型 fixture 测试写成真实全量预处理已经完成。

## Technical Review

### Errors (incorrect content)

- 在本次已核查范围内，**未发现新的阻断性代码或资产错误**。没有提出通过修改真实 Y、删除缺失孔、合并实验重复或放宽化学身份来获得“通过”的做法。
- 本结论没有覆盖所有训练阶段与所有下游消费者；覆盖范围及保留问题见下文。实现代理报告的 60 项测试结果和真实 18 行读取结果，与审查者本次亲自执行的验证分开列示。

### Rigor Issues (needs strengthening)

以下是必须维持的解释和评测边界，**不是要求停止当前预处理的代码缺陷**。

1. [src/lung/data/sciplex_source.py:245](../src/lung/data/sciplex_source.py:245) 的 `control_map = control_roles(...)` 按对照细胞分配 role。本次真实表核实，40 个 sci-Plex vehicle 孔均有细胞进入多个 role；[reader.py:393](../src/lung/data/reader.py:393) 抽样时只使用所请求 role 内的细胞。这支持“同一细胞未跨 role、按相同实验上下文提供参考”，**不支持“对照孔、培养批次或生物实验完全独立留出”**。当前全局药物留出研究可以声明使用同实验上下文参考；若改称新来源/新批次泛化，须另行划分并重新拟合。误差条的独立单位也不能改成细胞数。

2. [src/lung/data/sciplex_source.py:127](../src/lung/data/sciplex_source.py:127) 附近把作者终点时间写入 `duration_h`，同时保留 `exposure_duration_h=None`、`withdrawal_protocol=None` 和来源时间语义；[world.py:189](../src/lung/engine/world.py:189) 及 [world.py:274](../src/lung/engine/world.py:274) 阻止这些条件自动变成决策动作/初态。**字段可被编码不等于暴露或撤药协议已明确。** 24/72 h 响应监督可以进入模型；它不能单独验证“暴露 24 h 后撤药至共同终点”与“持续暴露 72 h”的时长选择，也没有同细胞 24→72 h 配对。

3. [src/lung/data/reader.py:184](../src/lung/data/reader.py:184) 的 `PREPROCESSING_REQUIRED` 守卫与实际快照一致。新包的来源词表、划分和数据身份已改变；[engine/data.py:27](../src/lung/engine/data.py:27) 的容量检查要求显式响应模型容纳 2 个 source 和 2 个 protocol。**旧权重、旧归一化与旧缓存的适配未由本轮证明。** 本轮没有给联合模型的预测性能、状态增益、凋亡功能一致性或筛选收益任何正向效果判定。

### Notation Issues

- 未发现会导致当前工程对应关系错误的符号问题。
- 必须持续区分 `chemical_or_experiment_group_id`（当前完整化学实体身份）与 `split_group_id`（保守防泄漏连通组）。实际可用处理条件有 **545 个前者、512 个后者**，不是两份相互矛盾的药物统计。
- `duration_h` 是现有模型接口字段，sci-Plex 来源中仍须连同 `time_semantics`、`protocol_id`、未知暴露/撤药字段解释；不能只读名字就视为已核验的干预持续时间。

### Citation Gaps

- 本次为代码与本地冻结资产审查，没有新增外部方法定理或重新检索原始文献。代码行为按下文实际函数定位，实验协议解释沿用已缓存并完成审查的 [Round6 评价契约](SCIPLEX_EVALUATION_CONTRACT_ROUND6_2026-10-10.md)。
- [实现审查说明](SCIPLEX_INTEGRATION_AUDIT_ROUND6_2026-10-10.md) 与真实 probe 是实现方证据；本次独立复算能证明的内容如下。原始实验报告的含义、模型随机前向成功和研究假设成立，是三种不同的证据。

## 1. 快照与本轮责任边界

| 项目 | 本次复审所绑定的值 |
|---|---|
| 新包 | `08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1` |
| manifest SHA256 | `1944f2880180b711ba8ddb196fedf0b4386794140ca17ea663673ed169c27bd9` |
| training signature | `ce85e8447b6ec447407aa9f67bd6e9044d9f0475f0cc3c0c28633921f980dc86` |
| 当时状态 | `metadata_ready` |
| 化学来源 | `chemical_inputs_v2`，manifest SHA256 `c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05` |
| 独立重算的资产 hash | manifest 登记的 15 个文件，共 28,481,549 字节，0 项不匹配 |
| 本次独立检查记录 | [independent_checks.json](../experiments/formal-review-sciplex-round6/independent_checks.json) |
| 本次检查记录 SHA256 | `179b3408948b9a6d5e544bcdb431b0f6fe887b0051e9ee55625b080a4dce085d` |

实际审阅源码：`data/common.py`、`prepare.py`、`sciplex_source.py`、`reader.py`、`preprocess.py`、`bandwidth.py`、`ae_expression_cache.py`，以及 `engine/ae_data.py`、`engine/data.py`、`engine/world.py` 和定向测试文件。

审查者没有重读全量表达矩阵，也没有重新哈希全部 Tahoe/sci-Plex 原始表达分片。源原件核验由已冻结的来源登记、其内容身份和真实读取 probe 提供；本次复算的是新包的 15 个登记资产及必要小表。也没有重复运行实现代理报告的全部 60 项测试。

## 2. 逐项验收

| 检查项 | 代码对应与独立证据 | 判定 |
|---|---|---|
| Tahoe 与 sci-Plex 的基因列轴 | [reader.py:107](../src/lung/data/reader.py:107) 的 `RoutedSparseRows` 按登记的 `gene_axis` 选择读取器：Tahoe 用 `tahoe_column`，common 用 `arange(n_genes)`。真实注册表含 3,388 个 Tahoe 分片、202 个 common 分片。小型测试故意反转 Tahoe 列顺序，同时核对交错来源及重复行的返回顺序。 | 通过 |
| 普通响应、MMD、AE、预处理共用路由 | [reader.py:337](../src/lung/data/reader.py:337) 的分组响应和 [reader.py:365](../src/lung/data/reader.py:365) 的 AE 都通过 `self.rows.read`；预处理、带宽使用 `training_rows`；AE 克隆保留双源 reader，表达缓存以相同 reader 建立。微型 6 基因 fixture 实际经过预处理、带宽和 AE 路径。 | 通过；不外推真实全量运行完成 |
| 化学 v2 的资格与 -1 | [sciplex_source.py:62](../src/lung/data/sciplex_source.py:62) 验证可用指纹的形状、二值、非零、连续索引和非虚拟原子，未定项要求 `fingerprint_row=-1`；[reader.py:290](../src/lung/data/reader.py:290) 先检查 eligible，再读取指纹。真实 S4246 的 8 个条件全部被该入口拒绝，避免 Python 的 -1 误取最后一行。 | 通过 |
| 真实身份与防泄漏组分离 | [sciplex_source.py:25](../src/lung/data/sciplex_source.py:25) 单独生成组，[sciplex_source.py:234](../src/lung/data/sciplex_source.py:234) 用组角色重新分配联合包。独立去除立体信息、重算完整/母体 SMILES 连接图后，同图均未跨 role 或 split group；真实 chemical ID 与化学表逐项一致。 | 通过 |
| 旧版角色保存 | 真实 Tahoe `legacy_v3_role` 与旧 `training-identity-v3/conditions.parquet` 的 role 逐项一致。该字段是历史记录，当前监督读取使用新 role。 | 通过 |
| 计划孔完整保留 | [sciplex_source.py:92](../src/lung/data/sciplex_source.py:92) 从 planned wells 构建条件。真实 1,920 个孔全部存在，包含 8 个结构暂缓孔与 1 个零 QC 孔；暂缓和缺失孔均未出现在 role count 的可用索引中。 | 通过 |
| 对照对应 | 3,742 条新增链接逐条检查 source/background/protocol/time/replicate/culture plate 相同，链接目标为 control；每个可用处理条件都有同 role 可用对照。具体抽取为未配对群体参考。 | 通过；仍非同细胞前后配对 |
| world 动作与初态隔离 | 真实元数据经 `_action_metadata` 产生 1,128 个已有 Tahoe 候选动作，sci-Plex 动作 0。fit roots 706 个，sci-Plex roots 0。直接对真实 sci-Plex fit 条件传显式 expression，仍先拒绝 `RESPONSE_ONLY_CONDITION_NOT_DECISION_ELIGIBLE`。 | 通过 |
| 元数据状态与训练入口 | 对真实新包默认构造 `EngineData` 拒绝 `PREPROCESSING_REQUIRED`；显式 `require_preprocessing=False` 才能进行本次元数据检查。 | 通过 |

“通过”表示该项所列工程契约和对应检查通过，不表示本课题全部科学问题已通过。

## 3. 独立复算的真实统计

| 统计单位 | 实际数量 |
|---|---:|
| 联合条件记录 | 3,256 |
| sci-Plex 完整计划孔 | 1,920 |
| 其中处理孔 / vehicle 孔 | 1,880 / 40 |
| 可用 sci-Plex 处理条件 | 1,871 |
| sci-Plex 可用细胞（含其 vehicle） | 209,504 |
| S4246 暂缓条件 / QC 细胞 | 8 / 954 |
| 无 QC 细胞的计划处理孔 | 1 |
| 新增严格上下文对照链接 | 3,742 |
| 可用处理的完整化学实体组 / 防泄漏组 | 545 / 512 |
| 全联合索引细胞 | 2,762,092 |
| 细胞分入多个 role 的 sci-Plex vehicle 孔 | 40 |

无 QC 孔是 `sciplex:S7207`、72 h、rep2、plate52、well `plate10_D10`；它在计划表存在，不能被赋予零表达或零效应。S4246 的 954 个细胞通过作者 QC，但不满足本版单结构主指纹输入契约，因此与 QC 未通过的 33,823 个来源细胞分开计数。

上述数量分别是条件、孔、细胞、链接和结构分组，不能互相代替独立生物重复数。实现方 probe 报告的全索引重复细胞数为 0；本次检查冻结索引文件 hash 与该快照一致，并未重新扫描全部 2,762,092 行来重复同一检查。

## 4. 实际运行证据及其归属

### 4.1 审查者本次执行

1. [metadata_checks.py](../experiments/formal-review-sciplex-round6/metadata_checks.py) 只读取小型元数据表及 15 个新包资产，独立复算统计、角色、化学组、对照上下文和 hash。
2. [entry_checks.py](../experiments/formal-review-sciplex-round6/entry_checks.py) 使用真实 `EngineData` 与元数据，调用 action/root 入口。未读取表达、未归一化、未加载或训练新模型权重。
3. 定向小型测试：

```powershell
& 'F:/anaconda/envs/study/python.exe' -B -X utf8 -m pytest tests/test_sciplex_response_integration.py -q -k "metadata_routes or small_fit_preprocessing or unresolved_relative or cannot_enter_existing_world" --basetemp experiments/formal-review-sciplex-round6/pytest -p no:cacheprovider
```

结果为 **4 passed, 4 deselected in 4.15s**。第一次命令因审查者临时目录的父目录尚未建立而在 fixture setup 失败；建立自己的临时父目录后原命令通过，没有修改实现代码来消除失败。

这是可复现的小规模验证，含小 fixture 的预处理；不得与当前另行启动的真实联合预处理混称。

### 4.2 已读且绑定快照的实现方证据

- [verify_joint_reader.py](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/verify_joint_reader.py) 与 [joint_reader_validation.json](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/code_probe/joint_reader_validation.json)：真实 sci-Plex 24 h、72 h、Tahoe 24 h 三个条件，各读取 6 行×32,461 列；核对所读取分片 hash，并通过随机初始化 ConditionEncoder 的 CPU 前向。
- probe 的实际条件 ID 为 `sciplex:84dcfada8600e338e118d614`、`sciplex:733c3ea17f91fe0f51594549`、`tahoe:smp_1495`。三个前向输出均为有限的 `3×256`，协议词表容量实际为 source 2、background 1、protocol 2。
- 实现方回归记录 **60 passed in 40.11s**，范围含旧数据读取、AE 与 identity v1/v2/v3 兼容。本次只复跑上述最相关的 4 项，不冒称独立重跑全部 60 项。

随机初始化条件编码器的成功前向只证明输入张量和接口可用，不证明药物响应预测正确。真实原始行验证也没有产生凋亡功能标签。

## 5. 完整课题仍须分别验收什么

当前工作使 sci-Plex 实测表达可以通过主模型数据入口参与未来实验，解决了这部分源行、基因列和处理条件的工程对应。完整课题仍不能据此标记通过：

- **状态增益：**可以按已有独立评价契约检验群体参考表达或分布相对固定参考/平均表示的价值；当前接入未证明这种价值，更没有把同期对照重称为逐细胞 t0。是否存在超出群体类别/批次的表达增量，需要相应对照和观测变化。
- **未见条件：**当前全局化学连通组划分支撑相应的跨药评测准备。源内剂量×时间组合补全、严格 72 h 留出、允许 72 h vehicle 的参考设置、新来源泛化是不同任务，不能用同一个“test”名称混称。
- **真实凋亡功能一致性：**本包没有新增同条件的表达—Annexin 原始实验配对。外部功能挑战和桥接模型须沿用自己的测量定义、冻结开发/留出用途和独立评价要求。
- **药物、剂量、时长选择：**sci-Plex 的 24/72 h 时间信息不自动给出共同 T* 的撤药/持续暴露比较；本次 world 隔离正是为了防止此处发生语义越界。既有 Hastings 局部共同终点证据的范围也没有因本包接入而扩大。
- **运行与资源：**真实联合预处理、正式训练、冻结预测和外部评价需要后续运行证据；本报告不预判方法一定成功，也不因尚未得到结果而判定所有研究方法不可开展。

本次仅新增本报告及独立临时验证目录；未修改、回退或覆盖他人源码、冻结表、旧 identity/training 包或训练配置，未提交或推送 Git。
