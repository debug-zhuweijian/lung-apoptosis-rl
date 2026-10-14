# 外部终点特征生成器：第三轮独立只读复审

审查日期：2026-10-10。审查者：endpoint_pairing_review。审查范围为 `src/lung/engine/endpoint_features.py`、`scripts/generate_endpoint_features.py`、`tests/test_endpoint_features.py`、`experiments/20261010-endpoint-feature-preflight-v3/reference_inputs-candidate/`，并追踪实际调用到 world、endpoint model、data reader 和 schema2 contract。

结论：**当前特征接口、真实调用路径和修补后的参考绑定通过本轮工程复审；真实外部特征生成仍pending，跨来源生物学转移与G效果均未验证。** 本轮发现的旧映射参考ID/plate未绑定问题已由实现代理修补，本审查者独立复验通过。当前冻结候选本身始终具有正确参考ID/plate；此次修补解决的是合同允许错误/旧映射通过的问题。

本复审者只创建本报告。未修改源码、测试、候选、训练资产或核心终点表；未重复准备training-identity-v3，未启动训练，未运行会执行auxiliary训练的完整pytest fixture。数值检查仅用内存中随机初始化的小模型执行推理，不使用优化器，不写权重或真实特征文件。

## Technical Review

### Errors (incorrect content)

当前审查范围内没有仍未修复的确定错误。下面的问题在修补前已复现，修补后已关闭：

- [endpoint_features.py:335](../../../../../src/lung/engine/endpoint_features.py:335) 与 [endpoint_features.py:337](../../../../../src/lung/engine/endpoint_features.py:337)：原实现仅核对target source/background/protocol，未核对映射的 `reference_id`、`target_reference_plate`。内存中将这两项改为 `WRONG-REFERENCE`/`WRONG-PLATE`、保持所有源文件hash不变，原函数仍得到14actions。修补后错误值和缺字段分别报 `SOURCE_CONTEXT_REFERENCE_ID_MISMATCH` / `SOURCE_CONTEXT_REFERENCE_PLATE_MISMATCH`。当前候选映射未经改动，仍通过14actions。
- [endpoint_features.py:324](../../../../../src/lung/engine/endpoint_features.py:324)：修补后还将参考descriptor的 `reference_plate` 与实际训练condition的plate核对。即使把descriptor和mapping同时改为WRONG，也被 `REFERENCE_PLATE_MISMATCH` 拒绝。
- [endpoint_features.py:42](../../../../../src/lung/engine/endpoint_features.py:42)、[endpoint_features.py:430](../../../../../src/lung/engine/endpoint_features.py:430)：新 `descriptor_record` 将传入JSON对象与指定源文件解析内容比较，并记录这次读到的字节hash。仅在内存中修改reference或mapping的evidence、磁盘JSON保持不变，实际 `generate_to_directory` 均在创建输出目录前报 `FEATURE_DESCRIPTOR_CONTENT_MISMATCH`。
- [endpoint_features.py:469](../../../../../src/lung/engine/endpoint_features.py:469)：开始捕获的descriptor记录纳入before，生成后重新checked；feature_assets继续使用开始记录而非重新采信新文件。控制流已亲读。实现代理新增了“生成期间替换mapping，拒绝发布features.json/manifest.json”用例；本审查者未实跑这一文件交换用例，也未修改冻结候选来制造并发。

### Rigor Issues (needs strengthening)

- [endpoint_features.py:207](../../../../../src/lung/engine/endpoint_features.py:207)：原始来源背景/协议映射到Tahoe参考域仍是 `explicit_assumption_unvalidated`。这是模型上下文选择，不是相同培养体系、相同实验或表达—终点配对的证明。输出保留 `background_transfer_validated=false` 与原始source/protocol。
- [reference.json:3](../../../../../experiments/20261010-endpoint-feature-preflight-v3/reference_inputs-candidate/reference.json:3)：候选参考绑定 `metadata_ready` 版本。实际preflight仍有7个缺口：未完成预处理、缺normalization/neighbors/bandwidths、缺response/pool/state权重。现阶段不能称真实外部特征已生成；预处理完成后manifest/data_identity会改变，需重建参考绑定。
- [endpoint_features.py:415](../../../../../src/lung/engine/endpoint_features.py:415)：生物学基因组特征沿用当前world的missing-masked实现。代码可运行不证明缺失生物特征已经获得真实观测值。
- [endpoint_features.py:483](../../../../../src/lung/engine/endpoint_features.py:483)：生成manifest明确写 `generated_frozen_features_not_endpoint_training_manifest`、`formal_training_ready=false`。后续仍需真实条件/观测关联、独立T/V/C/E分组及schema2验证，不能把14个化学输入条件当作14个已训练成功或独立药物测试结果。

### Notation Issues

目前序列化布局与G输入一致，未发现槽位或小时/tick转换错误。world原向量顺序是：

`pool[d], bio[27], zero_history_slot[d], background/source/protocol[3c], elapsed_fraction/remaining_fraction[2]`

因此state_dim=`2d+3c+29`，history_input_dim=`d+3c+1`。序列化只保存population、bio、context及原始ordered history；G使用自己的HistoryEncoder填充history slot。外部elapsed/horizon以小时保存，G再计算分数，与world的tick比例一致。

源码定位：[world.py:388](../../../../../src/lung/engine/world.py:388)、[endpoints.py:100](../../../../../src/lung/engine/endpoints.py:100)、[endpoints.py:124](../../../../../src/lung/engine/endpoints.py:124)、[endpoint_features.py:392](../../../../../src/lung/engine/endpoint_features.py:392)。

### Citation Gaps

本报告依据实际源码、候选元数据及只读执行结果，不依赖外部摘要。当前没有新增论文机制/药效主张，也没有新增表达—终点配对声明。真正训练checkpoint及真实外部特征尚未获得，因此不能提供它们已经实跑或生物学有效的证据。

## 实际执行的检查与结果

| 检查 | 结果与范围 |
|---|---|
| P→pool→G路径 | 内存8基因、d=8、c=2、population=3、ODE1step；实际调用BiologicalWorld.propagate→ResponseModel.sample→physical_projection→BiologicalWorld._features→PopulationPool→MatchedEndpointModel。无优化器、无训练。 |
| G字典输入与world张量输入 | state_dim51，history[1,15]，population/bio/context长度8/27/6；两种G输出均0.5675584673881531。该数值仅是接口检查的随机模型输出，不是药物凋亡预测结果。 |
| C不进入G | 将随机C参数清零，诊断概率改变；population/bio/context/history/elapsed/horizon全部相同，G结果相同。`state_probabilities_diagnostic`与 `state_probabilities_used_by_endpoint_head=false` 的代码语义一致。 |
| 真实候选输入 | 304个条件请求，14个eligible；14actions在修补前/后均通过正确候选。这里只读condition metadata、化学输入和协议，未读取external endpoint Y用于选参考。 |
| 范围排除 | 当前10个48h请求全部排除；261个SYTOX终点排除；10个control请求、12个含control或组合/不支持exposure的请求排除；化学未定保持unreviewed_structure阻断。 |
| fit参考索引 | 2431行；全部role=fit、Tahoe-100M、CVCL_0023、plate1、Tahoe24h协议。smp1589=1144，smp1590=1287。逐cell_id/condition_id/CSR path/CSR row/role/context/plate与实际train.parquet的相同fit-control筛选一致。 |
| 源hash | 实际validate_reference核对全部82个已列来源，通过。其中75CSR共165,954,599字节，仅流式hash，不解析表达数组；不是磁盘大范围扫描或预处理。 |
| 当前data_identity | 候选与当前metadata_ready manifest均为5e5155050151de64618412aed0038812fc6e3dc84813c58b408354869e898dd7。 |
| 缺权重防伪 | 实际ExternalConditionWorld在缺response/pool/state时直接报PIPELINE_INCOMPLETE。未生成占位特征。 |
| reference ID/plate合同反例 | 错误和缺字段、descriptor/mapping共同伪造plate、descriptor缺ID/plate均拒绝；全部修改只存在于内存。 |
| JSON对象与源路径 | 正确对象descriptor_record通过；错对象helper拒绝；实际generate_to_directory在mkdir前拒绝reference和mapping两类不一致。 |

## 模型/测量与状态边界

参考cell index记录的是Tahoe同板fit contemporaneous controls；既不是外部实验测得t0，也不是新增生物重复。真实抽样由 [reader.py:357](../../../../../src/lung/data/reader.py:357) 和 [world.py:307](../../../../../src/lung/engine/world.py:307) 完成，使用fit role control rows。候选descriptor的matched_initial_state=false与代码一致。

生成器的C概率是诊断项。当前G实际使用pool统计编码、生物特征、自己的history编码、映射域上下文和时间比例；未使用C类别概率。不能在算法汇报中把P→C→G讲成已实现的G数值输入链。当前代码的实际链是P→表达编码/pool→world_features→G，C单独诊断。

阶段、权重和数据hash在preflight与world加载时检查；feature_dependencies_v2绑定实际checkpoint、gene axis、preprocessing、chemical identity、generator、condition inputs和initial state资产。生成与schema2监督关联仍是两个阶段。候选正确、合同修补通过，不等于真实模型权重/外部Y闭环已经完成。

## 本轮快照与冻结候选未改动证明

审查前endpoint_features.py SHA256：`d5b6cf37baad07830264d119516c914bc4f64e9d05ac17fad8ebbc80dd6b942a`。

修补后审查的endpoint_features.py SHA256：`262415e7b185c2faca44983dadd8234ca3c29b29eb5daa037d4f07fcd241847f`。

脚本SHA256：`721a76ff3920bcac72863104da1ecca86982abec8629914802002a21a31b1de9`。

含最后并发交换用例的测试源码SHA256：`9c89a0b71378c1040e910eb36f5d7dab4a8aa6d45266ea9d5862f91c86eeec9e`。

候选三文件在复验前后hash未变：

- reference.json：`4607e5a0d5530baa59ecc8349f243066e43663da8bd3e6c43d39b7450770b563`
- context_map.json：`01b05dd98889fa177485d230b044b9d45fd0e313b21977ae44916706dce6d572`
- reference_fit_control_cells.parquet：`94e1c2d2d34446f847fc54b79fea4497ca1107905e0e82bd4cb87a31a020061d`

后续代码修改或预处理完成导致manifest变化时，这些hash是本次审查快照，不能直接宣称审查已覆盖新版本。本轮认可接口与证据合同的工程实现；真实外部特征、G训练、跨来源可迁移性和药物筛选效果仍待实际实验。
