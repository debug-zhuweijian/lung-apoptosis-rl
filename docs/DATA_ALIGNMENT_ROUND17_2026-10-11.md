# 第十七轮：原生状态响应入口、完整物理协议与功能评价

日期：2026-10-11。接续[第十六轮](DATA_ALIGNMENT_ROUND16_2026-10-11.md)。专项实现及独立复审按用户要求采用 GPT-6.1-sol；主代理亲读关键源码、Bothe 原文及官方样本字段，重读真实表达并综合验收。本报告以[毕业论文研究架构](G:/研究生科研文件/毕业论文/研究架构.md)为合同，主线和 A549 优先验证范围没有修改。正式训练保持停止。

## 1. 直接结论：哪些对不上问题已解决，哪些还没有

**已有表达数据的条件、参考群体、基因轴、时点和用途角色，已接入可实际读取的新响应单元；完整研究的数据可行性仍未通过。** 本轮还实现了完整物理协议进入响应网络的接口、多状态功能 G 的最小程序，以及冻结完整候选预测后再读取实测功能 Y 的评价程序。这些改动关闭了若干实现缺口，没有生成新的 RNA—Annexin 配对或真实模型结果。

需要纠正一个关键表述：**原有潜空间 P 已经具有参考表达状态输入机制。** 状态在每一次向量场计算中作为条件输入；Gaussian noise 是生成起点。此前标准 P 生产器固定一个参考群体，才没有覆盖逐目标参考。完整撤药、恢复、前药保留等物理协议则确实需要新增条件接口。两种缺口分别处理，不能统称为“模型完全没有状态输入”。出处见 §3。

| 环节 | 本轮实际结果 | 当前仍缺什么 |
|---|---|---|
| 表达监督单元 | 3,176 单细胞条件和 16 个 bulk 背景—历史组已进入新索引，来源量纲分开；真实数值重读通过 | 模型效果、参考状态语义与跨来源迁移的验证 |
| 原生单暴露 P 编排 | 按每个请求读取完整合法参考，保留源 24/72h，执行身份和缓存边界已实现 | 新联合身份匹配的 AE/response 权重；真实 P 描述符仍为 0 |
| 完整物理协议 P | 有顺序的处理事件、撤药/恢复等字段与未知 mask 已进入新可训练条件层及 CFM/VF 调用 | source→tensor 映射、qualified full-protocol producer、新参数训练与相容 RNA 监督 |
| 多状态功能 G | 两 fit 背景、一个 selection、一个 final 的新角色计划和分测量域 ridge 程序已实现 | 合格 native bulk72 P、实际 G 拟合、受控 selection/final 适配器 |
| 功能评价 | 旧 LOBO 的完整预测一致性冻结与实测标签评价入口已实现 | 可信模型执行绑定和真实预测；一致性散列不授予奖励资格 |
| Q/联合选择 | 新多状态 Q 所需资格已经列清；旧有限标准 P/G/Q 程序保留原范围 | 新多状态私有 Q 实现、冻结 G 的独立资格、同一整链的实测评价 |

核心仍为 **42 个 Annexin 条件汇总、17 个 24h 化学输入**。Martin 的 20 个活性单药相对 caspase 均值和本轮表达请求不并入这个分母。课题立项、工程检查通过或新增数据来源，都不代替主假设的验证；也没有把“所有 Y 必须有同细胞 t0”或“模型必须赢”增设为普遍门槛。

## 2. 新响应单元具体对应什么

资产：[native-response-units-v1](../experiments/20261011-data-alignment-round17/native-response-units-v1/manifest.json)。实现：[NativeResponseUnits](../src/lung/data/native_response_units.py)。这是条件分布监督与合法参考读取，不是同细胞前后轨迹。

| 来源和单位 | 唯一数据规模 | 索引中的职责记录 |
|---|---:|---:|
| Tahoe 单细胞，24h | 1,305 处理条件 | 1,305 请求，原六角色保持 |
| sci-Plex 单细胞，24h | 1,496 处理条件 | 1,496 请求，原六角色保持 |
| sci-Plex 单细胞，72h | 375 处理条件 | 375 请求，原六角色保持 |
| Martin/GSE193258 bulk | 四背景 × 四药后/历史组；48 唯一处理 RNA 列、12 平行亲本参考列 | 四 LOBO 折 × 16 组 = 64 职责请求 |
| 合计 | 3,176 单细胞条件 + 16 bulk 组 | 3,240 请求；64 条 bulk 职责记录不等于 64 次独立实验 |

单细胞使用 32,461 个有序共同 Ensembl 基因的原始 counts。bulk 使用原始 19,712 symbol 轴的 `log2(TPM+0.01)`，保留负值；相应 counts、TPM 和单细胞出口没有混用。共同轴映射只用于对照，缺失保持 mask。

每条请求保存处理条件、完整已知协议、参考身份、source/background、用途角色和基因轴。没有原始字段的暴露/撤药信息保持未知。**源 24/72h 时点不自动等于已验证的持续暴露时长**，特别是 sci-Plex 的 `sciplex_author_reported_endpoint_v1`。

`reference()` 只选参考表达作为输入；`read()` 是显式的表达监督读取。原 CSR 文件可能同时含处理细胞，打开该文件不等于把处理表达用于预测。实际守卫核对被选行、角色、ID、轴和参考来源。`max_rows=None` 读取全部合法参考：每个对照孔等权，孔内每细胞等权；若为读回检查截取前缀，保留原概率质量并明确不重新归一，生产器使用完整参考。

主代理[重新读回](../experiments/20261011-data-alignment-round17/actual_readback_main.json)四个实际条件的参考/处理各一行，共 **8 行单细胞 counts**，逐项与原 NPZ 一致；另 **6 列 bulk** 与官方 gzip TSV round-trip 浮点精确一致。它关闭选定数据路径的坐标/单位问题，不是全数据模型性能报告。Round12 的 34 参考上下文、68 对照条件和 1,089 共享候选组保持原定义，不将板位上下文重新解释成 34 个纯生物初态。

## 3. 状态实际怎样进入 P，新生产器改了什么

主代理亲读以下实际调用，不只依据模块名字：

- [LatentMember.velocity / cfm_loss / sample_latent](../src/lung/models/latent_staged.py:113)：训练与 Heun 生成均调用 `velocity(state, tau, reference_h, condition)`。`reference_h` 随每次向量场计算输入，生成内部 τ 位于生成过程，不是生物小时数。
- [LatentMemberAdapter.sample](../src/lung/engine/latent_pipeline.py:662)：参考表达 → `shared.encode_h` → 条件 latent 生成 → member decoder。
- [ConditionEncoder](../src/lung/models/encoders.py:137)：读取真实药物指纹、剂量、duration、source/background/protocol、历史及 elapsed/horizon。原 protocol 是登记词表 embedding，没有完整物理区间语义。

新增 [native_response_producer.py](../src/lung/engine/native_response_producer.py:259) 的公开 `produce` 按请求重新核对冻结源与角色，从完整参考群体生成表达，固定汇总为有序基因的加权均值和方差。该汇总在投影后的 joint 标准化表达坐标中计算，不报作原 counts 或 Annexin。`full_reference` 与 `same_reference_mean` 对照使用相同参考行、权重、随机种子及每行 noise；可以检验完整参考分布相对同参考均值的表达增量，实际实验尚未执行。

原生生产必须绑定数据 identity、基因轴、normalization、图、上下文词表、AE→response 权重身份和代码版本。单纯表达均值/方差生产只要求其合法 AE/response，不额外要求 pool/C 已训练。未来明确生成时使用 `weights_only=True` 和严格参数核对；本轮只调用 metadata 预检，没有打开 checkpoint 张量、构造真实模型或生成 P。

当前预检把 **3,176 请求**登记为有限单暴露 endpoint 编排支持，**64 bulk 职责请求**为 decoder/匹配权重 pending。前者是输入路由支持，不是已产生 3,176 个预测。新联合身份匹配的 AE/response 尚未登记，`actual_descriptors=0`、`execution_qualified=false`。旧权重的进度与 identity 仍以[第十六轮实际 metadata 核查](DATA_ALIGNMENT_ROUND16_2026-10-11.md)为准，标准化坐标桥不能授予旧权重新模型资格。

独立复审发现并修复：初版内部 backend 可由 caller 注入 mock 并自报 native 资格；重开 reader 可把 fit-only 权限扩为 final。现在公开 `produce` 自己调用内部安全工厂，mock 入口永远为 `audit_mock`；生成前继承原 reader 的角色权限，结束前重查源/权重散列。报告为[独立 native 审查](../experiments/20261011-data-alignment-round17/independent_native_review/technical_review.md)。这些文件散列是可重复的一致性守卫，不是密码学来源认证。

## 4. 完整物理协议怎样进入响应网络

新增 [full_protocol.py](../src/lung/models/full_protocol.py:18)，不修改旧 encoders、latent_staged 或 compact。实际接口为 `FullPhysicalProtocolBatch`：

| 输入 | 含义和处理 |
|---|---|
| drug/vehicle/wash/recovery/media_change 事件 | 保留作者记录的顺序；允许真实组合或前药保留造成同时/重叠区间 |
| 药物指纹、vehicle 与已知 mask | Morgan bits，未知与实测零分开 |
| dose 与显式 unit | 保留声明单位，不隐式混换 nM/µM/percent |
| start/end/exposure/recovery | 物理时钟；已知值必须与共同 T 和适用区间一致 |
| wash_count、prior_retention、media_change | 洗涤次数、前药保留和换液；未知不填成确定零 |
| endpoint_hours/known | 共同观察终点；与生成内部 τ 分开 |

实际网络路径是 **DrugEncoder + event/unit embedding + 带 mask 的数值 → 有序 GRU HistoryEncoder → 与原条件及共同 T 融合 → LatentMember.cfm_loss / sample_latent → 每次 VF**。训练与采样均经过新条件层；它不是只把协议写进 manifest。详见[接口回执/schema](../experiments/20261011-data-alignment-round17/full_protocol_conditioner/README.md)。

新增参数身份为 `full_physical_protocol_latent_v1`。旧 checkpoint 缺这组参数时拒绝直接加载，也拒绝 `strict=False` 伪装匹配。固定 tiny 模型/mock 检查覆盖梯度和调用路径；**没有实际训练这组参数**。从真实 source 请求组装张量、旧条件 kwargs 的科学语义以及合格 full-protocol producer 尚未接通，有限单暴露生产器不能因此宣称支持 pulse。

两轮复审修复了 bool/complex 数值、非法 Morgan bit/flag/count、已知暴露或恢复与区间冲突、end 未知时仍超出 T 等边界。未知字段保留，不为区间空窗推断无药恢复。最终 **27 接口测试及 18 独立固定探针通过**，[复审已关闭](../experiments/20261011-data-alignment-round17/independent_conditioner_review/review.md)。

## 5. 多状态 G、测量域和完整决策块

新增[独立角色计划](../experiments/20261011-data-alignment-round17/multistate-role-plan-v1/manifest.json)；没有覆盖旧 Martin LOBO 或 source bridge。四个背景按 metadata 固定循环：一个 final、下一个 selection、其余两个 fit。角色生成不解析 Y。

| 新计划每 fold | fit | selection | final |
|---|---:|---:|---:|
| 全部活性单药 | 10 | 5 | 5 |
| S2F 头 | 6 | 3 | 3 |
| S2M 头 | 4 | 2 | 2 |

共四 fold × 两 head = 八个角色/测量域组合。四折复用同 20 个发表观测身份，不增加独立功能样本。旧 LOBO 每 fold 15 fit/5 final、没有 selection，继续单独保留。

[G 程序](../src/lung/engine/multistate_endpoint_learning.py:31)使用固定 λ=1 的小样本 dual ridge。设 X 为合法 fit native P 特征；只用 fit 计算总体 mean/std，零 std 置 1，标准化得 Z；截距为 fit 的 Y 均值，系数计算为 `Z.T @ solve(Z @ Z.T + I, y - intercept)`。相对 caspase 输出为 `max(0, raw)`，不裁成 [0,1]。每个来源发表均值权重为 1，三 RNA 重复不复制 Y。属于本轮实现的最小基线，尚无真实拟合或效果证据。

**训练测量域与决策块是不同对象。** 同 S2F 或 S2M 测量域可用两个 fit 背景训练一个头；排序只在当前同背景、同 panel、同 72h 的完整候选块进行。拒绝跨域混排、只送子集或重复候选。held-out 背景是预声明外推对象，并非已验证可用状态。

[NativeMultiStateAdapter](../src/lung/engine/multistate_endpoint_model.py:113)内部解析 typed producer 描述符，绑定 state reference、实际协议、候选、角色计划、source 和 cache，禁止标准24h P、原 bulk RNA 或 caller 向量替代真实 P。当前 `_native_bindings` 明确返回 `NATIVE_BULK72_BACKEND_PENDING`：Martin 缺合格 bulk72 decoder、对应状态/协议响应和权重。实际 G fit 在解析标签或写 checkpoint **之前**停止；这不是一条已经接通的生产链。

数值输入先检查 bool/complex，再转 float64；修前吞掉复数虚部的问题已独立复验关闭。最终 **46 测试通过**，八个真实预检全部 pending。[独立多状态复审](../experiments/20261011-data-alignment-round17/independent_multistate_review/technical_review.md)还记录：显式 fit API 会解析完整 observations 文件后仅取 fit Y，属于用途隔离，不是磁盘级逐角色文件隔离；新计划的冻结只有一致性快照，受控 selection/final 评价尚需实现。多状态 Q 仍为 `Q_fit_implemented=false`，不能把直接 G 排名当作已完成 RL。

## 6. 冻结预测后怎样读取实测功能标签

新 [state_function_evaluation.py](../src/lung/data/state_function_evaluation.py:93)和 [CLI](../scripts/evaluate_state_function.py)对 **旧 Martin LOBO** 实现 `preflight → freeze → evaluate`。这一评价范围与上节新角色计划不混用。

freeze 要求四 fold 的全部 final 候选共 20 个身份，绑定 source、模型 artifact、caller 预测文件、测量域、状态、候选块、尺度和评价代码；不读取 Y。evaluate 先重核所有散列与字段，随后才调用显式 final observation API，分别评价八个完整候选块。

当前指标为发表均值 MAE、完整块内 regret 和可判定排序对。regret 是“该已测候选块的最大 Y − 预测最高分并列候选的平均 Y”；没有暗中加入低剂量或短时长偏好。读图界重叠时排序对为 uncertain，读图范围不是 SD/CI，也没有计算生物显著性。

复审发现并关闭重复 JSON key 与大有限数求平均溢出问题；异常或旧代码身份发生改变时，在 Y API 之前拒绝。对应 **27 新测试 + 5 既有 source 测试 = 32**。见[评价复审](../experiments/20261011-data-alignment-round17/independent_evaluation_review/technical_review.md)。

**这个接口只认证 caller/source 一致性。** 任意模型 artifact 的散列不证明 native P 执行或模型已经训练；`execution_certified=false`、`reward_qualified=false`。公开 Y 此前已可见，计划属于 retrospective；程序阻止当前执行链先读 final Y，不证明研究者历史上没有见过标签。本轮没有真实预测文件的生产冻结或真实模型评价。

## 7. 新 A549 历史 RNA 来源：Bothe2021

Bothe et al., *Glucocorticoid signaling induces transcriptional memory and universally reversible chromatin changes*，Life Science Alliance 4(10):e202101080，DOI [10.26508/lsa.202101080](https://doi.org/10.26508/lsa.202101080)。主代理亲读[原文 XML](../experiments/20261011-data-alignment-round17/bothe_protocol_source/PMC8403771.xml) Fig3A/E、Results s5、Methods s10/s13/s15/s27 和 Data Availability s31，并核对 SDRF。详细[来源报告](../experiments/20261011-data-alignment-round17/bothe_protocol_source/bothe_protocol_source.md)保留原字段与各 accession 的数据类型。

RNA 为 **E-MTAB-9923 / ERP125910**：12 A549 bulk 样本，四种处理历史各三重复。Fig3 方案为 Dex100nM 或 EtOH0.1% **4h → PBS 洗两次 → 无新增激素培养24h → Dex/EtOH4h → T32 RNA**。培养基仍有5% FBS，不能写成绝对无激素。平行 vehicle T32 不是同细胞 t0 或再次处理前 t28。

公开 processed 文件只有三个 DESeq2 对比表，每个24,163行，合计5,030,739 bytes；`baseMean/log2FC/SE/stat/p/padj` 不是逐样本 counts。原始12 paired runs对应24 FASTQ，合计 **60,056,725,984 bytes，约60.06GB**。24项 HEAD 在本轮核查时200且长度匹配，MD5只登记、未下载正文校验。source/sample/ERS/ERR/历史和双端文件对应已核实。

原文方法为 hg19/STAR2.7.0a、MAPQ≥10，以及 RefSeq 最长转录本的唯一 intron 区域计数；需要按源观测域恢复。I盘有空间，但 Linux/toolchain 与确切参考注释/参数尚未准备。没有下载60GB、安装工具或执行量化。详见[raw前置可行性](../experiments/20261011-data-alignment-round17/bothe_protocol_source/raw_preprocessing_feasibility.md)。

这提供有限 A549 历史/撤药表达的公开恢复路径，不能当成 Viel6+18h/T24 RNA、自然凋亡率或同条件功能桥。Zotero全库请求连接失败，个人库是否已有未知；没有表述为未找到。旧 GSE161800 等按 accession 去重，不重复计算新发现。

## 8. 对原架构四方面的验收

| 架构待检验方面 | 现在能做的合法验证安排 | 尚不能报告的结果 |
|---|---|---|
| A 状态表达增量 | 真实同条件/跨上下文索引、full reference 与同参考 mean 的一致权重/随机对照；source 内限定范围 | 状态已经提供稳定增益；对照上下文等于纯生物 t0 |
| B 未见条件可靠性 | 冻结原六角色、完整条件分组与真实 24/72h；Martin 新计划独立分组 | 时间外推/跨背景已经有效；72h采样就是共同终点暴露选择 |
| C 功能一致性 | 真 Annexin 与 caspase 分域；完整块冻结预测/实测均值评价路径 | 高维RNA额外功能价值、跨来源桥、A549完整终点校准已经成立 |
| D 联合选择增量 | 保留真实药物—剂量—协议候选；分别与直接 G、枚举等比较，未来固定模型环境与奖励 | 新多状态 Q 已训练，或内部高分代表实测收益/全局最优 |

多个互补子实验可以支持同一研究问题，但仍要用同一算法职责链完成主问题的功能验证，不能把 A 的表达实验、另一个模型的 C 排名和未执行的 D 拼成“整链通过”。没有要求每个功能条件都有追踪式 t0；无法消除的桥假设必须通过独立限定评价检验并保留失败结果。

最具体的剩余数据缺口是：**相容状态/真实协议下的 P 监督与功能连接**。例如 Martin acute500nM/24h RNA 不是其四单药160/300/100nM、72h功能后表达；Viel真实6+18/T24 Annexin没有匹配RNA。Bothe新增的是另一种T32历史RNA，不能直接填这些缺口。它们可以提供方法辅助监督，迁移到功能条件须单独验证。仅补权重或仅补接口都不会自动解决这项科学问题。

## 9. 本轮验收和接续顺序

主代理验收 **138 个不同定向测试**：native units17、producer16、full protocol27、LOBO评价27、既有Martin读取5、多状态46。分两次92/46运行均exit0；[测试记录](../experiments/20261011-data-alignment-round17/pytest-main.txt)记录命令、范围和耗时。检查是数据/interface、纯 NumPy、tiny fixed/mock tensor及显式来源读取，不是生产科研模型效果。

[主代理验证](../experiments/20261011-data-alignment-round17/validation.json)重核真实8行/6列、3,240请求分母、八个新角色组合、Bothe表和raw元数据、四项独立复审最终代码身份，以及冻结joint/旧协议/Martin bridge/研究架构的原散列。metadata CLI也实际运行：native与LOBO preflight成功且不解析Y；多状态CLI的Python exit2明确表示pending，不是训练成功。实际 native P/G/Q 为 **0/0/0**。

后续按依赖推进：先完成 source→完整协议张量和来源观测出口；恢复 Bothe counts 或取得其他相容公开表达并明确其有限监督范围；解决 Martin bulk72及A549功能候选的native P资格/迁移检验；再完成受控 selection/final 与私有Q。正式训练、权重执行和整链效果评价仍遵守用户停止训练的当前边界。本轮不自动恢复训练。

当前Git保留原有脏工作树。新增与更新范围为本轮数据接口、源码/测试、实验回执和导航文档；没有改研究架构、冻结数据或旧模型，也没有 commit/push、Memory/Zotero写入或新Cloud/Dot派单。Cloud/Dot此前结果仍作历史参考，本轮交付来自本地6.1-sol代理和主代理。

接手可先读[本轮交接](../experiments/20261011-data-alignment-round17/CONTINUATION_HANDOFF_2026-10-11.md)，再读本报告及第十六轮。给 DeepSeek 教学时请保留“原文事实、源码行为、工程验收、待验证假设”四种证据身份，不把测试数量或散列一致性讲成完整研究可行性结论。
