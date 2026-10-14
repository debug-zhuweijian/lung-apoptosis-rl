# 第二十二轮 fullP → G → 联合选择学习入口

本轮新增独立、可调用的生产/评价入口，真实状态仍为 **0 个可消费 fullP 预测，0 个真实 G/Q 参数资产，科研效果及正式奖励资格未取得**。默认入口仅预检，未恢复训练、未加载真实 PT、未执行真实模型 forward、未做真实 ridge 拟合，也未解析 fit/selection/final Y。

## 对应的研究问题及范围

沿用[毕业论文研究架构](<G:/研究生科研文件/毕业论文/研究架构.md>)§4.3–4.4、§6，在 Vielhauer 的同一 A549 来源、同一名义化学身份和共同 24 h 观察终点下，检验完整候选的表达预测特征能否改善实测条件均值预测与药物—剂量—暴露/恢复协议选择。该来源只有固定的 parental 状态描述，没有匹配初始 RNA，因此本试验不能单独证明原始细胞状态增量、未见药物预测或患者获益。真实序贯、多步滚动和其他来源仍分别需要证据。

评价使用[新 joint view](../experiments/20261011-data-alignment-round22/science_evaluation/joint_selection/view-v1/view.json)：20 个唯一已发表 Annexin 条件均值，fit 8（50 pM、1 μM）、selection 4（10 nM）、final 8（500 pM、40 nM）；每药物—剂量的连续/脉冲双协议在同一角色。最终域为 2 药 × 2 剂量 × 2 协议。来源的公开 Y 已经在此前材料审阅中查看，本设计是回顾性评价，不宣称前瞻盲测。来源控制与两张图的关系、protocol-specific terminal harvested-parent 分母、像素读取区间及跨图重复未配对边界不变。

## 实际数据消费与实现

[学习模块](../src/lung/engine/joint_selection_learning.py)和[CLI](../scripts/train_joint_selection_models.py)新增独立 namespace，旧 `fit_g`、标准 P 缓存、旧 plan/split、冻结 reward、World、DA-AC、DLPA 与 Actor 入口均未修改。

1. `build_prediction_index` 从每个候选的真实 `prediction.json` 构建索引，逐项使用 fullP owner 的 `read_prediction` 核内容 digest、当前源请求/绑定、独立登记 registry、匹配 AE/P 文件 SHA、固定参考细胞与 32,461 基因出口。仅能消费工程上登记的 fullP 产物，接口 mock 不具有生产登记资格。
2. `prepare_inputs` 拒绝少于或多于 20、重复或非法候选、文件字节/digest 变化、源协议/缺失身份变化以及混用 P 模型、参考、基因轴、seed/ODE steps。旧 fullP 请求中的角色仅作为原源身份保留；新角色是独立的功能 Y 划分，不自动形成 P 的药物或处理条件留出。
3. full-feature G 消费每候选的 32,461 维预测 common-gene log1p mean 与 variance、2,048 位 Morgan FP、名义剂量、原单位物理事件与有序槽位、暴露/洗脱/恢复/共同终点、来源协变量及缺失指标。未知物理量使用数值 placeholder 加 `known=false`，没有变成实测零。初始 source RNA、历史化学批次、stock form、来源迁移与协议验证的缺口保留；没有把 Tahoe proxy 状态当成 Viel 实测状态。
4. condition-only baseline 获得同一可用信息工件，但只使用条件特征子集，明确为表达特征消融基线。它并非在额外实验信息或不同候选域上比较。
5. `fit_models` 只通过 role-only API 读取 fit 8 个唯一 source means。复用现有 `ridge_parameters` 的固定 λ=1、fit-only population mean/std、未惩罚截距和 8×8 dual solve；每均值 weight=1，不将生成细胞数或来源 reported n 当作独立功能标签。参数归一化在加载时从 fit 8 输入重新核验，selection/final 从不参与拟合或参数选择。
6. 独立 terminal ridge Q 使用同一冻结 G 在 fit 8 合法候选上的模拟 terminal feedback，另做求解并保存独立参数与 8 个明确 simulated 单元；不使用 measured Y 直接训练 Q、不 bootstrap、不增加成本项。该 Q 是单次终点近似/诊断候选，不代表既定 DA-AC 路线或 RL 收益。G 尚无独立功能资格时，该 Q 不能授予正式 World/Actor 或奖励资格。
7. `freeze_scores` 从实际保存的 G/simple/Q JSON 参数计算完整 selection 4 或 final 8 分数，并绑定共同信息、response 特征和反馈 G；`evaluate_frozen` 先从当前绑定工件重算三模型分数，拒绝 caller 自写分数或字节/语义漂移，然后调用 JointSelectionEvaluation。只有此显式阶段才读取该角色的 Y。最终三方法、四种固定条件策略与均匀诊断共八方法均在同一完整域报告；按来源均值给 MSE/MAE、排序与选择 regret，比较对不扩充生物重复数。

`information.json` 与源 prepared 纯元数据重建值精确匹配；所有模型的 mean/scale 从 fit8 重算核对，JSON 参数/schema/维数/finite/数值类型严格检查，bool 不充当 λ/权重/参数。ridge 运算在 clip 前检查 overflow/invalid/divide，不能将无穷值 clip 成有限零分。完整工件 digest 与源码依赖检查是可复现一致性，不等于独立证实拟合执行或功能效果；这些资格仍保持 false。

## 当前实际证据

[默认真实 preflight](../experiments/20261011-data-alignment-round22/joint_selection_learning/preflight_v1.json)只读取候选、角色、基因、参考与绑定元数据，20 fullP 缺失保持 pending。[带模型/Y/solve 守卫的最终 v4 回执](../experiments/20261011-data-alignment-round22/joint_selection_learning/real_preflight_guard_receipt_v4.json)绑定最终 0dac172f 源码，明确 real PT、nn module init、forward、ridge solves 和三角色 Y 解析皆 0。

[合成验证历史](../experiments/20261011-data-alignment-round22/joint_selection_learning/synthetic_ci_v2.json)记录 source50e619版本 I 树 15 项通过，并保留首轮误用 C pytest 临时目录的实际路径及更正；没有删除现存旧 C 临时文件。独立复审实际发现并修复 information 自报语义、全域 normalizer 冒充 fit-only、有限参数算术溢出等反例；后续回执与最终版本以同实验目录及[独立复审目录](../experiments/20261011-data-alignment-round22/independent_joint_selection_learning_review/)为准。全部模型效果测试数据是 synthetic，不能替代真实 fullP/G/Q 验收。

[P-only基线完工回执](../experiments/20261011-data-alignment-round22/joint_selection_learning/implementation_receipt_P_only_v1.json)绑定source0dac172f／CLIee55034e／基线22项tests，独立24passed为当时收集的20 owner项＋4反例；后加2项算术／有限clip检查在I新j22arith4单独2passed，不冒称一次26项命令。独立CI的上游producer快照包含b3与b8，不能据此说完整当前b8消费已经在同一次CI中验收；当前b8零ML guard与独立dtype delta证据分别记录。后继可选C特征由主代理负责，P-only source快照和本回执不自动覆盖它。

收尾时磁盘已出现root进行中的optional-C源版本；回执单列该磁盘快照及`disk_snapshot_is_covered_by_prior_CI_or_preflight=false`。上文工程通过仅指已验P-only基线；新C数据读取、特征消费、参数绑定与复审需要后继证据，不因引用本报告自动验收。

## 单机与未来实际运行顺序

G 的 dual solve 维数只有 fit8；20×(64,922+条件维数) float64 输入约十余 MiB，数值求解本身无需 GPU。这是实现维数估算，不是实数据峰值内存或耗时实测。fullP 的匹配 AE/P 训练和 20×2,431 个代理细胞推演占主要资源，沿 owner 的既定配置/早停/最大轮次、更新级可恢复断点执行，不新增墙钟硬预算。

以下命令是**未来取得模型运行授权、匹配参数及独立登记后**的执行方案，本轮只执行默认 preflight 与合成 CI。

1. 按 fullP owner 的独立 `--fit-ae`、`--fit`、actual matching-model audit 和独立 registry 顺序取得匹配权重。完整 nominal protocol 仍是 OOD，实际 washout/recovery RNA 监督缺失，工程登记不会改变此科学身份。
2. 对新 view 的 20 个 `metadata_units()`，逐候选显式调用 `lung.engine.full_protocol_response.produce(candidate_id, metadata, output/candidate_id, seed=config['seed'], steps=config['response']['inference_ode_steps'], microbatch=config['response']['microbatch'], device=config['device'])`。所有候选使用同一 declaration、参考和采样设置；旧结果不覆盖，失败保留，未齐全不拟合 G。
3. 在生产输出齐全后构建索引并拟合新模型：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py build-index --prediction-directory experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_predictions_v1 --output experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py fit --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/models-v1
```

4. 先显式冻结 selection 4，再评价；不根据 final 结果改参数或奖励资格：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-v1 --role selection --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection_scores-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/selection_scores-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection_metrics-v1.json
```

5. selection 审阅与方法固定后，用相同 models/信息/反馈冻结 final 8 再评价：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-v1 --role final_test --output experiments/20261011-data-alignment-round22/joint_selection_learning/final_scores-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/final_scores-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/final_metrics-v1.json
```

selection 上的独立功能表现、迁移适用性和预先确定的奖励验证标准仍需另行复核；本入口没有发明新的通过阈值，也不会自动将诊断 Q 或 measured metrics 升格为正式环境。没有匹配的 P 产物时，程序保持 pending，不用随机预测、接口 scores 或 synthetic 标签填补实际缺口。
