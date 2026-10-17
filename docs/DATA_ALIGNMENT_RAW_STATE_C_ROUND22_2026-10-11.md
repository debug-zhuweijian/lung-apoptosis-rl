# 原始状态 C 与同 P 候选三类概率，Round22

当前入口说明（2026-10-15）：C 预检直接核对已登记的 joint 数据 manifest、当前配置、源数据 manifest、基因轴、归一化及状态划分，不再递归预检完整 P 候选体系，也不依赖历史 RNA 读回回执、固定 248 个 fit 细胞或旧 `auxiliary.py` 哈希。原标签映射与角色隔离仍核对；预检不重新读取 RNA 矩阵，实际训练仍由 reader 核对来源计数与细胞映射。每次训练从头开始，输出最佳模型和 `final.pt`；下文的断点与历史核验数量保留为当时记录。

本轮新增独立的 C 拟合入口及候选侧工件，当前通过真实数据元信息预检和纯合成工程测试。真实 C 拟合、真实 AE/P/C 权重加载与模型推演均为 0；没有产生实际候选 C 概率，也没有验证跨来源状态迁移或凋亡预测效果。正式训练停止安排继续有效。

## 数据、科学职责与架构对应

本入口对应毕业论文 `研究架构.md` 中状态条件化响应与干预评价的 E→P→D→C/G/Q 消费链。C 的原始任务是识别 TIDYI 原始分选三类 `PI-APC-`、`PI-APC+`、`PI+APC+`，使用同一共同基因轴、冻结拟合归一化和同一 AE 的 h 表示。C 是辅助状态特征；实测 Annexin、caspase、生长、死亡等实验终点仍归 G 的来源定义和独立评价。本轮不改奖励、冻结入口、实验划分、三类定义或多步/真实序贯研究范围。

| 层面 | 所需数据及实现 | 实际证据 | 当前边界 |
|---|---|---|---|
| 原始 C 数据消费 | 共同 32,461 基因、原始计数的 cell_id join、既有 fit 归一化、原三类和 1_to_8 划分 | [根代理实读](../experiments/20261011-data-alignment-round22/root_state_C_readback.json) 已读取 743 来源细胞，fit 248 的 x/label 与直接公式一致；本轮逐 hash 复核并读取角色元信息 | 本轮不重读 RNA 矩阵值；该 QC 不能证明分类效果 |
| AE→C | 新独立 matching joint AE 的完成 AE_result、best/last SHA、源实现与数据身份；原 `auxiliary.train_state` | [C 模块](../src/lung/engine/raw_state_scoring.py)、20 合成测试 | 真实 matching AE 尚未完成；旧中断 AE 未被加载或恢复 |
| C 拟合/校准 | fit 248/selection 64/calibration 63，三类各存在；原 weighted_ce、lambda=0、原早停及温度校准 | [当前真实预检](../experiments/20261011-data-alignment-round22/raw_state_scoring/metadata-preflight-v2/preflight.json) 与 [零模型 guard](../experiments/20261011-data-alignment-round22/raw_state_scoring/real_metadata_guard_v2.json) | final 368 的表达和标签没有被用于拟合、选择或校准；其他 fold 原状态不足 |
| 原始 C final 评价 | 完成冻结 C、同 AE、原 final_test 368、原三类、原 fit 坐标与预冻结角色/样本/标签/源/设备计划 | [当前 final 元信息计划](../experiments/20261011-data-alignment-round22/raw_state_scoring/final-metadata-preflight-v1/final_plan.json)、显式 `--evaluate-final` 及合成 tiny checkpoint 评价 | final 标签与 gate 元信息已实际读取；真实 final RNA、真实 C 分类和功能 final Y 均未读取/运行 |
| 同 P 候选 C 特征 | 同源 AE 原始 best.pt、P safe 导出 metadata 中的 source_checkpoint_sha256、同 AE tensor identity；同 reference/seed/steps 的逐细胞 physical P→标准化→同 E(h)→冻结 calibrated C | 合成实际 tiny safe tensor 加载、源 SHA/数据身份/tensor 漂移拒绝；精确 P mean/variance 复演 | 真实 P/C 尚未运行；表达均值上分类不能替代逐细胞分类后平均 |
| C→G 可选特征 | 三个单独概率均值、方差与采样细胞数，完整 P/C/AE 来源及域身份 | `read_state_sidecar(path)` 只读契约；根代理在 G 独立版本接可选 sidecar-index | 迁移假设、校准稳定性及在实测功能终点上的增量贡献均待独立实验 |

可用分选角色和三类计数保持为：

| 角色 | 总细胞 | PI-APC- | PI-APC+ | PI+APC+ |
|---|---:|---:|---:|---:|
| fit | 248 | 91 | 83 | 74 |
| selection | 64 | 15 | 33 | 16 |
| calibration | 63 | 31 | 20 | 12 |

原数据的 PI 标签原样保留。来源协议提及 BOBO3/Annexin，但原始通道推导尚未恢复；因此没有把这三类自动改称 Annexin 阴性/早晚期凋亡。旧 `train_state` 中已有的合类诊断不变，新侧工件不调用 `population_proxy`，不输出 class1+2 合值或奖励。

## 新独立拟合入口和完成证明

入口为 [run_raw_state_scoring.py](../scripts/run_raw_state_scoring.py)，默认为 metadata preflight；另有显式 `--fit`、`--score`、`--final-preflight` 和 `--evaluate-final`。`EngineData(..., roles=('T','V','C'))` 在构造时授权原 named fit/selection/calibration；`state_arrays(fold, role)` 直接传递 named role，原 data/auxiliary 未修改。

`--fit` 先核 matching AE 的本 namespace 完成 JSON、best/last 字节、源实现和数据身份，再调用原 `auxiliary.train_state`。保持原 fit 类逆频权重、weighted CE、selection 非加权三类 NLL 和 calibration 非加权三类 NLL 温度拟合。新 monitor 只启用原更新级恢复与校准断点，不增加墙钟限制。新文件锁和新输出目录约束避免并发重复写同一 C 实验。

`state_declaration.json` 绑定原 calibrated.pt/best.pt/last.pt/temperature-last.pt、新 safe tensor `state_weights.pt`、AE 原始来源 SHA、AE tensor identity、manifest/源码依赖和真正 last 的训练完成/校准完成。selected best 可以早于最后完成轮次，二者分别记录。已完成恢复只读 JSON/字节，不进入原训练函数或 PT；calibrated 文件存在而本 namespace 声明缺失时明确拒绝，保留成果，不假称完成或重跑。中断恢复仍走原训练与温度断点。

声明核验只提供完成与来源完整性；不构成独立无泄漏审计、状态迁移证据或科学收益认证。合成声明禁止取得真实生产资格。

独立复审原版本的三个反例已修复并回归通过：修改 body.state_fold、将完成实际更新改为 0 后重摘要不能取得完成；实际 safe C 的 train_role 改为 final_test 后即使更新文件 hash 也会拒绝。声明现在联核 run manifest 的原 fold、完成轮次/更新和实际 selected epoch/counter、温度断点及 calibration 特征缓存；实际 serving metadata 在构造 C 前核原三角色、weighted CE/0 lambda、h 坐标、AE/hash/依赖和原三类。

## 独立原始 C final 与预定 gate 元信息敏感性

`final_preflight` 只读标签/gate/角色/源/模型声明元信息，提前冻结原 final cell_id、class_id 顺序、split/轴/归一化 hash、C/AE 来源字节和参数身份、设备及消费源码。当前没有 C 声明，状态为 completed_matching_C_pending；模型授权后必须针对已完成 C 重新产生冻结计划。`--evaluate-final` 要求该计划完整匹配当前完成 C 后才构造 `EngineData(...,roles=('E',),allow_test=True)`，只调用原 `state_arrays('1_to_8','final_test')`。加载对应冻结原始 AE 和 safe calibrated C，同 fit 坐标 encode_h 后计算原三类 NLL、Brier、macro-F1、accuracy、balanced accuracy、三类 recall 和 3×3 confusion；没有 optimizer、校准重拟合或 checkpoint 选择。`auxiliary.state_metrics` 原函数的合类诊断不改，新 final 出口只保留以上 17 个原三类字段。

[根 gate 补证](../experiments/20261011-data-alignment-round22/root_state_gate_followup.json) 已实际读取源 `SS3_Human_A549_metadata.csv`（物理 253KB）369 行中的明确 A549 368 行，并精确 join 原 final 368 的源细胞、作者原标签和 plate8。export gate bins 为 109/93/108，另有 50 个原负类 singlet=1/全部子门0，8 个原负类 singlet=0/全部子门0，共 58 个原负类的 export fields 不直接确认负 gate。源字节前后匹配，已在看新模型前[冻结 58 IDs](../experiments/20261011-data-alignment-round22/root_state_gate_followup_ids.csv)。这没有证明错标，也没有恢复 PI/BOBO3 通道生成链；plate1 等价 gate fields 尚未恢复。

primary 保持全部 368 个作者原始标签（167/93/108），不删除、重标或移动细胞。预定 secondary sensitivity 在同一冻结模型的已预测概率上，单独报告按上述固定元信息规则保留 310 个细胞的同原三类指标（109/93/108），不按模型结果选子集或模型。当前仅规则与样本身份已冻结，没有任何真实 gate 敏感性或分类结果。

## 同 P 数值消费与 sidecar

未来 `--score` 首先调用 full-P `read_prediction` 核实际 P 字节、绑定参数和固定执行注册，再核冻结 C 声明。加载的实际 P safe AE metadata 必须把 `source_checkpoint_sha256` 绑定到 C 使用的原始 AE best.pt；P/C 的 AE tensor digest、n_genes、d_model、latent_dim 必须相同。旧 AE shape 的 protocol 容量与新 P 的显式容量扩展不改变 AE tensor，允许按已有导出证明连接，不能只凭相同模型名字连接。

每个候选复用 P 的同一完整 fit proxy reference、seed、steps 和当前模型文件；noise 在整个 reference 集上一次生成，然后按既定 microbatch 分块。每个细胞的 physical_projection 共同基因 library10000 log1p `y` 通过 `(y - joint_fit_mean) / joint_fit_scale` 回到 C 训练的同 AE x 坐标，再经过同 `encode_h` 和冻结 calibrated C 的三类概率。最后平均每个细胞的三类概率并记录三类方差。

数值口径预先固定为 `exact_float64_P_moments_replay`：复演 y 的 float64 按细胞加权 gene mean/variance 必须与该 P 文件逐元素完全相同，才写 C sidecar。若设备、microbatch 或数值实现使结果不同则拒绝并保留失败目录，需选择匹配原 P 的设备/分块或另立声明口径；本实现不设置经验误差阈值。P 原文件只保存均值/方差，精确摘要复演不等于原逐细胞输出的独立留档或功能迁移证据。

侧工件 `state_sidecar.json` 的实际公共 schema 为 `raw-sorting-state-sidecar-v1`，只读 API 为 `lung.engine.raw_state_scoring.read_state_sidecar(path)`。字段如下：

| 字段 | 内容 |
|---|---|
| candidate_id | 原完整协议候选 ID |
| prediction | `{path, sha256, prediction_identity}`，绑定实际 P 文件 |
| state_asset | `{declaration_path, declaration_sha256, serving_path, serving_sha256, state_identity, reference_AE_origin_sha256, representation_identity}` |
| P_AE_bridge | `{export_checkpoint_sha256, source_checkpoint_sha256, representation_identity}`，在未来评分时从实际 safe metadata/tensor 核验后保存 |
| class_order | 严格原三类顺序 |
| probability_mean / probability_variance | float[3]；不合类 |
| sample_count | 严格 int，匹配 P reference shape，真实契约为 2431 |
| predict_domain | fullP.OUTLET，共同基因 library10000 log1p，非来源原始计数 |
| sampling | `{seed, steps, microbatch, device, policy}`，绑定实际 P 的 seed/steps |
| P_replay_moments_exact / C_input_coordinate | 精确复演声明及共同表达→同 AE h 的输入坐标 |
| scope flags | source_C_transfer_validated、cross_source_C_calibration_verified、Annexin_interpretation、scientific_qualified、functional_Y_read、reward_generated 均为 false |

reader 不读取 PT 或推演模型；核实际 P 字节/identity、C 工件字节/identity、同数据与 AE 来源桥、三类顺序、采样数与三类概率数学边界。JSON 的自摘要及来源桥声明不能单独证明执行或无泄漏；完整性通过不产生迁移、凋亡或整项目科学资格。

## 已运行的检查与尚未运行的实验

当前 27 项本模块检查与独立复审 3 项 counterexamples 共 30 项通过（I 盘新 `experiments/c22t6`，5.35 秒，[实际日志](../experiments/20261011-data-alignment-round22/raw_state_scoring/pytest_scope_final_v6.txt)）。模型操作全部为合成 tiny checkpoint；其中真实元信息预检只读源/角色/hash。覆盖原三 named roles、冻结 AE、tiny checkpoint 的 C 构造、同 tensor/source/data 漂移拒绝、更新级 exact resume、已完成 0 PT/0 重拟合、完成声明缺失拒绝、源码漂移前置拒绝、逐细胞三类概率、共同坐标恢复、P moments 精确比较、失败目录保留、浮点/三类/采样/来源/科学 flag 反例、reader 0 PT、显式 only-E final、原三类出口、冻结计划漂移前置拒绝、final 标签顺序失败保留、primary 标签不变及固定元信息 secondary 敏感性。

[最终独立复审](../experiments/20261011-data-alignment-round22/independent_raw_state_scoring_review/technical_review.md) 为 CLOSED_ENGINEERING_ONLY；复审在 I 新 `experiments/c22rv3` 对当前同源码亲跑 27 项本模块与 5 项独立 counterexamples，共 32 passed/5.81 秒，原三个 P2 及新增 final 混 fit reader 在 PT 前拒绝、final AE identity 异常在 C 构造前拒绝均通过。未改 owner 源码或运行真实 RNA/PT/模型，原始失败记录保留。该 closure 不授予原 gate 通道链、真实 C 分类、P→C 或跨来源功能迁移科学资格。

[当前 guard](../experiments/20261011-data-alignment-round22/raw_state_scoring/guard_default_and_final_cli_v2.py) 实际调用 default 与 final-preflight CLI，guard 的 PT load、NN initialization、real fit、real forward、RNA matrix read、EngineData open 全为 0；只加载共同坐标/协议输入数值资产，重用根 RNA 实读并核 source hash、划分元信息及 58 gate 来源。final 标签/gate 元信息已读，未读真实 final RNA 或做分类/功能 final Y。

尚未运行：真实 matching AE、真实 C 拟合/温度校准、真实候选 P→C 推演、原 final 状态分类及 gate 敏感性评价、跨来源 C 迁移评价、C 特征对独立 G/Q 的增量效果。当前 30 项检查仅验工程行为及元信息，不替代这些实验。

## 未来模型授权后的具体顺序

1. 先按 full-P 学习报告在独立 `experiments/full-protocol-joint-AE-v1` 完成 matching joint AE，原 B1/fit/selection 不变，并产生完成 AE_result。
2. 在同原始 AE 上拟合 C，并在选择 C 的过程完全结束后冻结并评价原 final：

```powershell
F:/anaconda/envs/study/python.exe scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/raw-sorting-C-joint-AE-v1
F:/anaconda/envs/study/python.exe scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --final-preflight --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --output experiments/raw-sorting-C-final-plan-v1 --device cuda:0
F:/anaconda/envs/study/python.exe scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --evaluate-final --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --final-plan experiments/raw-sorting-C-final-plan-v1/final_plan.json --output experiments/raw-sorting-C-final-v1 --device cuda:0
```

3. 独立部分协议 full-P 按原 B3/B4 实验顺序完成、审计实际 tensor 执行并登记。用同候选域产生 Viel 20 份 P；native 训练覆盖仍不提供 Viel pulse/washout/recovery RNA 监督，协议 OOD/未验证边界保持。
4. 每个 P 对应一份 sidecar：`--score --prediction <该份已生成的prediction.json> --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --output <新的对应候选目录> --microbatch <原P分块> --device <原P设备>`。候选目录和 sampling 从保存的 full-P batch plan/result 读取，不能以未匹配 P 或 C(expression_mean)替代。
5. G 保留 P-only 消融，并可在相同 frozen role plan、目标、候选及信息下比较 P+三类 C 特征。先验证原 C 分类/校准与跨来源迁移假设，再以真实独立终点和冻结 Q 评价功能一致性与选择收益。状态判别改善不能替代 Annexin 等功能实测。

当前仅上述数据、契约和可调用方案已准备；模型执行仍需沿用原授权安排，未启动任何正式训练。

当前源码身份：engine `f39ed74f657d69869cf104b1cb4f31f003b501fb1b6f3c35c5952d8e4761a115`，CLI `3378a0ca1f53666cfeb38518a8791cd57e78890d48ca9adbc65988943a97a3d1`，tests `db26b9ea70fb31ec3309ac10c4b187a97fe4ad7782bbc11c1a88b3549b91aca2`。旧 v1/6653 源码 hash 的元信息回执和独立失败反例保留为当时证据；当前 v2 guard 与以上源码绑定。


## 本轮后续独立参考变体的当前版本

状态增量的 observed/zero_h/mean_h 三条独立训练、selection、来源声明、推演及 heldout API 已接通，见 [当前变体报告](DATA_ALIGNMENT_REFERENCE_VARIANTS_ROUND22_2026-10-11.md)。其 phase population 合同分别保留 CFM post-OT有限selected rows、B4/selection原uncoupled有限draw、production/C全部2431fit proxy、native final全允许draw，算子相同但 n/权重/有限均值不同；原OT、RNG、轴、归一化、角色、奖励及 Bothe model PIN未变。最新producer `af7aa029b98ca1e9e0b42d9001d6b49160884db04441727237d7c6eda50f13bf`、learner `103e933fc989e3b379436a5f164fff41705a8a712a64f9a0cdf65fd9434cafed`、CLI `91f53b8a8e90dd293c88d07db96b6451dbdb6b9aff9293b83fe971ba7a0e0bff`、C `6e613e1b50817f602df10b8eff4411c7664d8e72be8050cef3aea2efcbf756d5`。前文旧hash/CI是当时回执，保留，不据此覆盖最新版本。当前78工程项在I新a22t4通过，真实3metadata guard0ML；原真实RNA读回按gene/norm/data/ref身份复用，没有重新打开75shards。C从f39到6e仅AE-only配置与全pop h准备/predict_h精确P复演两个改动，原reader/final/label/gate语义不改。真实模型/状态收益/跨来源功能资格仍pending。
