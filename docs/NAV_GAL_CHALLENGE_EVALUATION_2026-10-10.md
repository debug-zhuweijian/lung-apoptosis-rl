# Nav-Gal 外部冻结预测评价入口

日期：2026-10-10。**当前真实运行状态为 `input-ready/prediction-missing`**：已实际读取并连接冻结数据和化学结构，尚未提交或评价真实科学模型预测。本轮完成的是一个独立的回顾性挑战读取与评价工具，没有启用旧 G 终点头、训练模型或生成科学预测值。

入口：[scripts/evaluate_nav_gal_challenge.py](../scripts/evaluate_nav_gal_challenge.py)。测试：[tests/test_nav_gal_challenge_evaluation.py](../tests/test_nav_gal_challenge_evaluation.py)。原数据入口：[challenge-v1](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal/challenge-v1/README.md) 和 [chemical_review](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal/chemical_review/README.md)。

## 1. 本次已经验证的真实输入

| 项目 | 实际结果 |
|---|---|
| 来源完整性 | 共绑定 44 个唯一文件：42 个依赖/输出按已登记 SHA256 比较通过（含上游 digitized manifest），另记录两个顶层 manifest 自身的本次内容 hash；不把自记录 hash 当作外部认证 |
| 群体条件 | 2 种状态 × vehicle / navitoclax / nav_gal，共 6 条实验臂 |
| 全部已发表时间位置 | 60 行，6 条曲线各 0–36 h、每 4 h 一点；读取全表，不补造 2 h 或 48 h 数值 |
| 主读出 | 原冻结契约的 24 h，共 6 行；4 个活性药物×状态点和 2 个 vehicle 点分开 |
| 化学连接 | 按 `challenge_treatment` 精确连接；navitoclax → 指纹第 0 行，nav_gal → 第 1 行；vehicle 的化学结构为 `None` |
| 化学特征 | 两种独立完整结构、两个不同的 2048 位 uint8 手性 Morgan 指纹；读取和核对已冻结定义，不重建或合并分子 |
| 状态输入 | 原类别、铺板密度、诱导历史、处理时长、基线观测 ID 等条件字段原样保留；没有添加 RNA 向量或同样本配对 |

`load_challenge()` 返回 `conditions`、完整 `observations`、`primary`、`contract`、实际 `fingerprints` 数组、指纹元数据和来源哈希。每个条件的 `chemical` 字段保存其完整化学记录；vehicle 不虚构零向量作为活性药物。

源论文 Fig. 4、§4.4 与 §4.6 已亲读核对：[原 XML](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round3/PMC7189993.xml)。24 h 后先扫描再挑战、两种状态的 4000/6000 名义铺板密度以及图注 10 µM / 方法 11 µM 的冲突均保留，参见[独立审查](NAV_GAL_CHALLENGE_REVIEW_2026-10-10.md)。没有把该冲突通过读数大小或模型效果裁决掉。

## 2. 使用

在项目根目录只检查真实输入：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/evaluate_nav_gal_challenge.py
```

默认输出 JSON。当前实际得到：

```json
{
  "status": "input-ready/prediction-missing",
  "condition_count": 6,
  "published_observation_positions": 60,
  "primary_24h_rows": 6,
  "primary_active_unique_observations": 4,
  "primary_vehicle_unique_observations": 2,
  "scientific_model_predictions_evaluated": false,
  "formal_training_ready": false
}
```

提交外部已冻结的预测后，使用 `--predictions <CSV路径>`。可加 `--output <新JSON路径>` 保存报告；目标目录需要存在，已有文件不会被覆盖。`--challenge` 和 `--chemistry` 可指向另一个保持同一已支持契约及完整来源链的包，默认是本轮已经审查的版本。

例如下面命令只是调用说明，**本轮未执行真实预测评价**：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/evaluate_nav_gal_challenge.py --predictions '<外部冻结预测CSV>' --output '<新的评价报告JSON>'
```

此入口专门评价 24 h 主读出。虽然读取完整 60 行以核验条件、时间和来源，但预测 CSV 不接受 4 h、36 h 等非主读出 ID，不把它们加入主指标或样本量。

## 3. 外部预测 CSV 契约

一个文件只评价一个明确版本的模型与预处理。下列字段均须存在：

| 字段 | 含义与要求 |
|---|---|
| `observation_id` | 必须是 `primary_24h.csv` 中的精确 ID，不能按行号、近似名称或作者重复编号配对 |
| `dose_scenario_uM` | 活性药物用文字 `10` 或 `11`；vehicle 用 `vehicle`，不复制成两个剂量 |
| `prediction_status` | `available` 或 `missing` |
| `prediction_percent` | available 时为有限的 0–100 数字；missing 时留空。70 表示 70%，不会自动把 0.70 乘以 100 |
| `missing_reason` | available 时留空；missing 时填写无法预测该情景的明确原因 |
| `prediction_unit` | 整个文件必须明确为 `percent`；不静默猜测 fraction / percent |
| `endpoint_definition` | 与冻结 contract 中 `endpoint_definition` 完全一致，即 `Annexin_V_positive_cells_divided_by_total_image_counted_cells` |
| `model_id`、`model_version` | 非空模型及版本标识，全文件一致 |
| `model_sha256`、`preprocessing_sha256` | 模型与预处理版本的 64 位小写 SHA256 声明，全文件一致 |
| `state_input_provenance` | 明确模型实际用到的状态信息及来源；该挑战包本身只有群体类别和记录的背景，没有配对 RNA |
| `unvalidated_transfer_flags` | 明确尚未验证的迁移/输入假设；无需迁移时也应说明理由，不能省略字段 |
| `evaluation_is_retrospective` | 必须为 `true`，承认已公开、已阅读的数据性质 |
| `data_exposure_statement` | 非空说明模型/分析者接触这些目标、图和数据的情况，以及是否据此选法、调参或选择读出 |

4 个活性观测必须分别提供 10、11 两情景，共 **8 条预测记录**。无法支持某一情景时仍保留该行，填写 `missing` 和原因。8 条是预测输入情景数量，**真实目标始终只有 4 个不同观测**。同一观测、同一情景不得重复。

vehicle 预测可不提供，或各提供一条独立 `vehicle` 行；其误差单独报告，不进入 4 个活性点的 MAE/RMSE。

这些模型、预处理和数据暴露字段是可追溯声明。入口检查字段、格式和一致性，但没有读取外部模型训练历史来独立证明这些声明真实；报告会明确这一验证边界。它不添加训练权重检查点、审计套件或旧 G 的运行门禁。

## 4. 输出如何解释

每个剂量情景分别评价相同的 4 个活性观测，令预测为 (p_i)、公开均值为 (y_i)，均按百分数存储：

\[
\mathrm{MAE}=\frac{1}{4}\sum_{i=1}^{4}|p_i-y_i|,\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{4}\sum_{i=1}^{4}(p_i-y_i)^2}.
\]

单位是百分点。一个情景只有 4 点全部 available 时才输出该情景完整主指标；缺项则报告可用数和 `prediction-incomplete`，主指标为 `null`，不把剩余易预测点冒充完整挑战结果。

报告同时保留：

- 10、11 情景各自的逐观测预测、误差和完整情景主指标。
- 每个观测在两个输入情景下的预测范围、绝对误差范围；两情景完整时给出两个总体 MAE/RMSE 的范围。缺一个情景则范围为空，不回退为“可用情景即最佳情景”。
- 对每个固定预测，真实图像读数落在 `[L,U]` 时的可能绝对误差范围：最小为预测到该区间的距离，最大为到两端的最大距离。它只传播数字化读图区间，**不是作者 SD、置信区间或生物学不确定性**。
- 同状态内两药的公开均值次序挑战：只有两读图区间完全分离才给出已分辨的次序；否则 `published_mean_order="undetermined"`，预测次序不计为对错。

真实衰老组两药 24 h 的读图区间重叠；因此即使点估计 99.65 与 99.31 不完全相同，不能设置严格的真值优胜者。模型若输出不同预测，可报告其预测大小关系，但无法由当前读图精度确定它是否复现了真实药物次序。

这里没有跨情景选最好结果，没有将两个情景平均成真实剂量，没有把不同时点当独立重复，也没有计算 p 值、bootstrap 生物学置信区间或新恢复的原始重复。

## 5. 测试与研究架构边界

实际执行：

```powershell
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_nav_gal_challenge_evaluation.py -q
```

**19 项测试通过（1.70 s）**。全部性能算术测试均使用明确标注的合成数据、合成结构和受控预测，不是 Nav-Gal 科学模型结果。覆盖内容为：四个唯一目标/两剂量情景的算术；区间重叠时未分辨；缺失情景；vehicle 分离；不允许非 24 h 增加样本；上游 hash 变化、化学行映射、单位、非有限/越界数、版本混用、端点不符、回顾性/暴露声明缺失以及重复记录的拒绝。

真实数据仅运行 readiness：44 个唯一文件完成绑定，其中 42 个已有登记 hash 的依赖/输出比较通过；完整 CSV 和化学映射成功，没有向真实 Y 填入合成预测。来源化学包仍保留其历史材料和谱图核验边界；读取成功不会将名义结构证明扩大为历史 lot 鉴定。

对应[研究架构](G:/研究生科研文件/毕业论文/研究架构.md) §2、§4.2、§4.3 和 §6：该入口承接可观测群体背景与实验定义的功能读出，保留候选剂量冲突和来源信息。它没有配对 RNA，不能证明表达表示或单细胞信息增益；状态与铺板密度共同变化，不能隔离衰老状态的因果作用；连续暴露读数不提供提前撤药到共同终点的结果。因此本轮没有补齐完整时长选择或序贯决策证据，也不把 input-ready 改称全课题可行或 `formal_training_ready`。

本轮仅新增脚本、对应测试和本文。没有修改任何冻结数据、现有训练配置或旧 G；没有训练、生成真实科学模型预测、写 Zotero/Memory、提交或推送 Git。
