# Full-P 原生 final 表达评价入口

新增 `src/lung/engine/full_protocol_native_evaluation.py`、`scripts/evaluate_full_protocol_native.py` 与定向测试，将新的 full-P namespace 接到原 joint 的 final 表达角色。默认入口只核对和保存 metadata；本轮未读 final RNA、未载入真实 PT、未执行真实模型或优化器。真实表达评价是后续明确授权的分支，当前不能报告方法收益。

## 实际来源与划分

已实际读取原 `native-response-units-v1/requests.json`、joint `conditions`、`role_condition_counts`、`control_links`、`split_groups`、`genes`、`chemicals`，以及冻结 fit-only normalization/bandwidth。原请求包含 fit 1,498、selection 374 和 final 367 个 single-cell 条件；Martin bulk LOBO 另有 final 16 个，不进入本单细胞 full-P 出口。

[metadata_v4](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/metadata_v4/preflight.json)实际核 final 为 **367 条件、52 个原 split components、32,461 个共同基因**：Tahoe 24h 为 87；sciPlex 24h 为 216，72h 为 64。同角色参考的 metadata 均可用，没有新增划分或角色。完整 source/hash、context vocabulary、名义药物/浓度/basis、观测时钟与未知处理历史见[final requests](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/metadata_v4/final_requests.json)。这些数字证明数据允许相应检验，不是模型效果结果。

原 [test.parquet](../08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/test.parquet)包含 **200,682 个 E-role cell metadata rows**；默认入口读取的列仅为 cell ID、condition ID、role、source/background/protocol 和 matrix 坐标，没有表达数值。逐条件核 target/control ID 集合、当前角色数量及不重叠，保留每个允许 ID 集合的 hash；[final_cell_index.parquet](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/metadata_v4/final_cell_index.parquet)保存实际原件的 E-role 允许坐标。control 的 `conditions.role=null` 不代表 control 无角色；角色在原 cell-index/role-count 层分配。同一 physical well 可有分到不同角色的细胞，不据此称独立生物重复或同细胞 t0。

实际 holdout metadata：全部 367 条件的 chemical/split component 均未出现在 fit/selection。源内 dose 值和 observed hours 均已在 fit 出现，`dose_holdout_at_seen_chemical=0`、`time_holdout_at_seen_chemical=0`。因此当前原 final 可评价未见化学组下的条件表达预测，不能冒称独立的未见剂量、时间外推验证。完整 exposure/history protocol 未被源数据识别，Viel 的功能角色和 20 候选不建立 P 的表达留药、时长或脉冲监督。

## 消费与完成证明

评价不调用仅包含 T/V 的 `NativeProtocolData` 构造器，也不构建限定 T/V 的 representation cache；默认 metadata 不构造 E 权限的 `TrainingData`。未来明确 `--evaluate` 时，先核独立 response run、`-assets`、原 completed AE run 与各声明：原 joint/data/axis/normalization/context、native partial coverage、训练执行依赖、正数训练 counter、selected serving SHA 和模型 stage 必须一致；plan、manifest 与实际配置的学习设置也须相等。原 AE config 使用既定 `joint_ae_config`，P-only reference 变体不改变 shared E/D 的拟合。仅此未来分支会读取受信本地 `last.pt` 的训练完成 proof、selected safe weights 和两个 matching 模型；旧 namespace 权重不能改 metadata 后直接进入。

实际模型加载完成后才构造 `EngineData(roles=('E',), allow_test=True)`。每次 reference/target 抽样沿用原 reader 的同角色控制 well→cell 和 target cell 分布，保存确切 ID、seed 和设置，并要求抽到的 ID 属于已核原 E 允许集合。观察到的 target 仅进入评价，不进入模型输入、模型选择或 baseline 拟合。

Native physical batch直接复用原训练的 `NativeProtocolData.batch`：只有 nominal dose known，原 observed hours 保持 observation feature；active exposure、washout、prior retention/recovery 等 mask 保持未知。未知区间不能通过 Viel 的严格 `validate_batch` 或 serving `predict_h` 填为已知。本入口手工执行 E→full.sample_latent→source decoder→physical_projection；无效生成质量直接失败，保留部分输出，不丢行或填假零。

训练的 reference contract 来自真实 run/serving/declaration，并与 loader 实际 `.reference_contract` 严格一致。`observed`、`zero_h`、`mean_h` 分别使用已登记 input 语义；每个条件完整允许 reference draw 先调用 `prepare_reference_h`，再切 P microbatch。`zero_h` 不运行 E(reference)，`mean_h` 是完整 E-h 的均值，不是分块均值，也不是 E(expression mean)。三种独立拟合及真实评价尚未运行。

## 基线与原始指标

每个条件在同一抽样域、相同 target 和冻结指标下报告 full-P、原 reference 分布，以及同 fit source/background/protocol/observed-hour target 条件的均值。fit mean 只读 T-role target：每条件固定 32 个 uniform-with-replacement cells，condition 等权平均，记录所有 sampled fit IDs 和 array SHA；实际生成与未来比较都按冻结 manifest 认证原 `train.parquet`，读取 fit-role cell metadata 并核 IDs 属于其原 target condition，拒绝把 E/任意 ID 写成 fit。该 T metadata 消费为 lazy，默认 final metadata 和当前缺产物比较均不读取 T 细胞表或表达。它是声明过的有限抽样估计及退化点分布 baseline，不是把 normalization mean 或 E target mean 冒充成 fit-source mean。源 context 无 T target 时明确 pending，没有 reference 替代。

沿用原 fit-only preprocessing，未在 E 重估。manifest 的 `preprocessing.fitted_role=fit`，`bandwidth.role=fit`；4096 个原 fit pair 的 median bandwidth 为 **1.0658622903778048**。所有方法报告未经阈值、奖励或 clipping 重写的 `standardized_mean_mse` 与 `standardized_population_mmd_v`，名称明确使用原 standardized 表达域，MMD 为 `unbiased=False` 的 empirical population V discrepancy；逐条件原值和按原 split component 等权平均分别保存，不以更多细胞充当独立样本量。

未来输出保存 P/reference/fit-mean/observed 的全基因 mean/variance、准确有序 gene IDs、源 raw-count 单位和 transform 身份。指标 mean 与 MMD 使用原 standardized 共同轴域；另存 mean 的 library10000 log1p 坐标。这些量不称为 source raw-count 均值，也不换成 Annexin、死亡或凋亡概率。

可选 `--reference-mean-sensitivity` 仅把同条件允许 expression reference 取均值后送入原已训练模型，保留同 noise、候选和采样配置。它登记为 inference sensitivity，明确不属于独立 trained stateless baseline或状态增量验证。状态收益比较需使用已独立拟合、同原 E draw 与候选域的 observed/zero_h/mean_h 模型，按实际结果判断。

## 三条独立训练结果的实际比较出口

v2 schema 的 `--compare` 读取三份已完成的 native final `result.json`、实际 model-verified `plan.json`、全部逐条件 JSON、full-gene moment 文件和各 SHA。评价先写 request；只有真实 completed matching AE/P 核验成功才保存 actual trained contract 的 plan，zero_h 标志来自真实 declaration，不预写所有 run 均无 independently-trained baseline。每份结果最后闭合该目录全部 report/array 文件的 SHA，部分输出不能被当成 complete。

比较逐项核对原 E 条件和 52 分组、数据与轴/normalization/bandwidth/context、原 source units/nominal observed time、完整有序 reference/target IDs、采样和噪声 seed、population/steps/device/microbatch/设置版本、原 T baseline IDs/moments、原 completed AE origin/hash，以及同 fit/selection config/learning settings（仅 reference_variant 不同）。同源导出 AE 的容器 bytes 可以有各自记录，比较依据各评价已核的原 AE source 身份与未改变的张量来源，不要求独立导出文件路径相同。比较不重新读取或 hash PT、不读取 RNA values、不拟合/选参/构建 cache。

比较保存逐条件 `observed_minus_zero_h`、`observed_minus_mean_h` 的 standardized MSE/MMD 差，各原 split component 内条件等权、再 component 等权的结果，以及三变体和共同 baseline 的原值；差值为 observed−other，负值表示较低 discrepancy。mean MSE 从保存的全基因 moment 实际复算；MMD 引用文件闭合的原 population V 数值，mean/variance 不能重构其 population MMD。未添加统计阈值、显著性检验或生物重复 CI，不自动宣布状态收益。

[实际缺产物比较回执](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/compare_pending_v1/result.json)为 `pending_three_actual_completed_native_final_results`，六个 result/plan 仍缺；这三个目录是明确登记的缺产物探针，没有声称模型已运行。[v4 主读回](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/primary_metadata_readback_v4.json)保存当前 source/hash、原 E metadata 各字段核对与该 pending 回执。三模型 A 增量和 B 未见 chemical 效果仍待独立训练与授权 final 评价。

## 已运行与未来入口

实际 default metadata v1/v2/v3/v4 均通过，未启动模型；旧三版本按其 source 保留。v4 绑定最终 reference API producer `af7aa029b98ca1e9e0b42d9001d6b49160884db04441727237d7c6eda50f13bf`、learner `103e933fc989e3b379436a5f164fff41705a8a712a64f9a0cdf65fd9434cafed`、evaluator `935ace7769270bb7c5e5d495d8b3f8e22195da5ed2ef9e12bc4d896d89e5866e`、CLI `30a3359d03c26d2c000ede2f79f1f465ddff03483eefe987dc21e1a519fa9e78`。定向 tests 初轮 toy facade/regex 失败原日志保留；比较首轮 v5 因新负例 fixture 错调不存在的 producer.atomic_json 而 9 failed/28 passed，实际回执保留，随后直接导入原 checkpoint.atomic_json。最终 [v8](../experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/pytest_v8.txt) **39 passed / 4.77s**，新 I basetemp `experiments/fpne8`。覆盖三 toy packs、重哈希 ID/seed/unit/角色/学习配置/AE/moment 改写、E ID 冒 fit、PT 字节拒绝和原 component 等权。工程 fixtures 不能替代真实 final 表达评价。

[独立复审](../experiments/20261011-data-alignment-round22/independent_full_protocol_native_evaluation_review/technical_review.md)已关闭本版本工程与实际 metadata 范围，独立 **12 passed / 5.56s** 位于新 I basetemp `experiments/fpnir1`；owner 39 项是另一次运行，不冒称单次 51 项。三源与 producer/learner 在复审前后 SHA 保持一致，原 seed 和原 T ID 归属两项实质边界已关闭，重新构造的 E metadata 与 v4 requests/preflight/cell-index 各字段、值和顺序一致。实际缺6结果仍为 pending、所有真实模型/RNA/功能/优化器 guard 为0；此复审未取得真实模型效果。研究中的 state 指可获得 contemporaneous reference 表达分布，不能据本层证明同细胞响应、对应 Annexin 初始状态、机制因果或生物获益。

当前可运行的 metadata-only 命令：

```powershell
& F:/anaconda/envs/study/python.exe scripts/evaluate_full_protocol_native.py
```

未来明确获模型/final RNA授权且对应 namespace 完成后，分别对三种已训练版本使用新输出目录：

```powershell
& F:/anaconda/envs/study/python.exe scripts/evaluate_full_protocol_native.py --config <matching_joint_AE.yaml> --evaluate --response-run <completed_new_fullP_run> --output experiments/<new_native_final_evaluation_version>
```

代码只接受项目内新 output，拒绝覆盖；缺匹配模型在 final RNA 访问之前失败，保留 request/failure；model-verified plan 只在完成匹配核验后保存。每个 response run 自己完成匹配验证，不能拼调用方提供的预测或把其他 namespace 的结果当同一冻结模型。

三份真实评价闭合后，无模型/RNA读取的比较命令：

```powershell
& F:/anaconda/envs/study/python.exe scripts/evaluate_full_protocol_native.py --compare experiments/<actual_observed_final_result> experiments/<actual_zero_h_final_result> experiments/<actual_mean_h_final_result> --output experiments/<new_three_variant_comparison_version>
```

`--compare` 和 `--evaluate` 互斥。缺已完成结果保持 pending；不一致或未闭合结果拒绝并保留 failure。当前正式训练停止安排继续有效，数据准备、39 个工程测试和缺产物比较没有恢复模型授权。

与[毕业论文研究架构 §4.2–4.3、§4.5 和 §6](G:/研究生科研文件/毕业论文/研究架构.md)一致：dose/observation/exposure 三种语义分别保留，P 输出表达、C 原状态、G 实测终点和 Q 候选价值分开；final 不参与模型选择，单次/滚动/真实序贯分别检验。当前数据允许未见 chemical 的表达检验，未来效果、独立 dose/time 外推、真实 functional 一致性与完整主问题仍未验证，目标没有被降低。
