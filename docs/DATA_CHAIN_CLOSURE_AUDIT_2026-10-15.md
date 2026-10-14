# 全架构数据链闭合审计（2026-10-15）

基线：`4a78560bdeb2e0d98332d5a9520764a120abf612`。本报告是本轮云端读取仓库 Round22 文档后的**证据限定审计**，不是新增真实数据处理或完整可行性验收。既有冻结划分、训练停止状态、目标与奖励均不变。原始实验数据未上传 GitHub；本轮未访问本机原件、模型权重或完整毕业论文架构原文件。

## 证据等级与实际操作

- **本轮云端实际执行**：GitHub API 读取 main、AGENTS.md、README、08-datasets/index.md、Round22 综合验收、状态可识别性、独立轴方案、模型计划、科学评价文档；在云端 Python 环境实际核验 Python 3.13.5、pytest 9.0.2、torch 2.10.0+cpu、NumPy 2.3.5、pandas 2.2.3、RDKit 2025.09.4，临时文件读写通过；终端 `git clone` 因 DNS 失败。
- **Round22 历史记录，尚未本轮重验原始数据**：joint 2,762,092 细胞、32,461 基因；Bothe 12/12 原始双端验收、4/12 计数；Martin 60 RNA 与原观测域；HM97662、Sarin、A375 序贯数据的有限证据。
- **尚未执行**：完整原始矩阵读回、source-to-joint 全量验证、真实模型 forward/fit、功能模型训练、独立 final 评价、真实序贯药效测试。因此科研完整可行性**未通过**。

## 不可缩减的主问题及闭合判据

| 数据链 | 已记载的可用证据 | 当前无法授予的资格 | 闭合必须补齐 |
|---|---|---|---|
| 状态表示 C/E | Tahoe、sci-Plex、TIDYI 原始状态与同期对照；共同基因轴、来源内 mask | 同细胞 t0、状态因果增量、跨来源共享 latent | 原件哈希和表达逐值读回；训练内拟合 normalization；plate/source/replicate 控制；真实状态 vs context-only vs within-context shuffle 的独立 grouped 评价 |
| 响应预测 P | 同来源药物、剂量、24/72h 终点群体；Martin 原来源 RNA | 单细胞未来轨迹、未见完整协议、独立剂量/时间泛化 | source-specific/共享/条件分布路线的同信息比较；药物、剂量、时间、完整协议各自留出；源值/单位/mask 检查 |
| 实测功能 G | 42 个可用 Annexin 条件汇总（Round22 记录）；HM97662 同来源有限条件桥；Sarin 相对 Annexin fold | 将 Annexin fold、fraction、caspase、SYTOX、生长抑制混为一个 Y；未测 RNA–Y 配对 | 每个原 Y 唯一身份、实验块与分母、时钟、背景、药物结构、剂量/单位、对照、重复；训练/校准/final 隔离；外部真实功能指标及不确定性 |
| 联合选择 Q/策略 | source-bound 候选与完整协议元数据；终点 ridge 诊断 | 真实选择收益或 RL 增益 | 候选全集、结构、dose/time、缺失动作约束；同 Y 同信息直接功能预测排序/枚举 vs Q/RL；独立真实终点 regret/NDCG |
| 多步模型滚动 | 单步响应及完整历史元数据的部分记录 | 生物学真实多步转移 | 真实多时点/历史状态监督或明确受限的模拟滚动检验；逐步误差及覆盖，不能用自生成轨迹作真实效果 |
| 真实序贯 | GSE306207 A375 8 RNA（Round22 记载），Bothe 撤药历史 | A549 序贯促凋亡、顺序优于反序或同时处理 | A549 或经严格论证的迁移域中 A→B、B→A、单药、同时、vehicle 的真实独立功能/状态比较；保留暴露、洗脱、采样时间及重复 |

### 统一对齐约束

每个观测必须能追溯 `source/version/source_sample_id/assay/expression_unit/normalization/gene_id_namespace/gene_order/gene_observed_mask/background_id/drug_id/structure_identity/dose_value/dose_unit/exposure_duration/measurement_time/ordered_protocol/vehicle/control_id/plate/batch/replicate/endpoint_type/endpoint_scale/statistical_unit/role`。字段 unknown 不能等价于零或默认协议；同名药物、同数值剂量、共同基因数量、文件格式一致均不足以构成生物配对。实测零要求 mask=1；assay 未测要求 mask=0。源内分组和独立来源之间的匹配只能建立在可追溯的实验条件层级，不强求所有任务同细胞 t0/t1。

### 可证伪实验与决策

1. 状态增量：预注册在相同来源、药物、剂量、观察时刻、合法参考权限下，比较独立训练的 observed、zero_h、mean_h、within-context shuffled；按原化学组件/板与实验块汇总。若增益在板/来源隔离后消失，不得主张生物状态增量。
2. 未见条件：原冻结 cross-drug final 保留；另建新 namespace 的 dose-only、time-only 和完整协议 OOD 设计，所有预处理/AE/P/G/Q 在新训练范围重建。Round22 记载 Tahoe 同板仅 14 个三剂量完整实体；sci-Plex 24/72h 板不同，不能称纯时间因果效应。**不得擅自覆盖旧角色**。
3. 功能连接：对 Annexin fraction、Annexin fold、caspase 和 growth/death 各设独立终点域。无配套 RNA 的外部 Y 允许评估候选排序外推，但不能证明单细胞 RNA 导致实测 Y。
4. 候选选择：必须与直接 `argmax_a E[Y|state,a]` 和枚举、药物均值及简单排序比较；真实 Y 的 top-k/regret 为独立评估，不能用 G 自身模拟奖励证明 RL 优越。
5. 序贯：分开验收单步选择、模型 rollout 与真实顺序效应。A375 序贯 RNA 不能直接授予 A549 Annexin 序贯效力。

### 当前最高优先级阻断

**第一**，Bothe 全量计数及来源整数轴、missing mask、对照、历史、回执尚无 12/12 生产发布证据；继续核验旧下载与 count 原件，不能以 README 的 4/12 充当新完成。

**第二**，高维表达与独立功能 Y 的条件连接覆盖不足。HM97662 仅有限活性条件且原功能读图是近似均值；Sarin 仅两背景、一药、有限剂量与 Annexin fold；二者均不能形成多药×剂量×时间×状态的充分监督。应继续原始论文、补充、GEO/ENA/ChEMBL 等逐来源核对，恢复可公开的完整矩阵；不得虚构缺失功能重复或药物结构。

**第三**，跨来源共享 latent、真实 fullP→G→Q 同链及独立效果尚未执行。单来源重建成功不能证明共享空间。

**第四**，缺少独立的 A549 真实序贯药效/顺序对照证据；必须保留该目标为未闭合，不能把 A375 结果直接转移或把 rollout 当实测。

## 接续执行及验收产物

在有原始数据的受控环境先只读核 `manifest/source hashes`，生成按 source+sample+condition+assay 唯一键的 alignment ledger（逐字段含原值、标准值、转换公式、未知理由、来源定位）；核验 Bothe 全部 12 对的官方完整性、计数回执、共同轴整数与 mask；再核 RNA/功能/候选联接是否仅在合法层级成立。输出必须包含：source receipts、row-level exclusions、gene/chemical mapping collisions、unit/protocol mismatches、group-disjoint split manifest、real-data tests、frozen evaluation results。任一关键缺口未通过则总标志 `whole_project_qualified=false`。

本轮没有取得上述真实数值产物，不宣称完成对齐。未恢复正式训练，未改变既定奖励、研究主线或冻结划分。
