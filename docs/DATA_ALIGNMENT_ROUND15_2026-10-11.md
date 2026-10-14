# 第十五轮：正式 G/Q 程序、评价角色与奖励资格闭环

日期：2026-10-11。接续[第十四轮](DATA_ALIGNMENT_ROUND14_2026-10-11.md)。专项实现和独立复审使用 GPT-6.1-sol；主代理亲读新增函数、真实包和相关理论原文，核对[研究架构](G:/研究生科研文件/毕业论文/研究架构.md) §2、§4.2–4.5、§6。没有恢复正式训练或执行真实模型推理。

## 1. 对数据对不上的担忧，现在可以怎样回答

**已经实现可审查的分层接入方案，尚未完成真实模型整体验证。** 表达响应按真实对照/处理群体监督；状态识别保留 TIDYI 的同细胞原始分选标签；实验 Annexin 均值按真实药物、剂量和完整协议监督条件终点头；候选决策按同一终点和可比范围评价。各层不是同一种配对，不需要把不同来源强拼成同一细胞的前后命运。

本轮补到了此前缺少的实际 G 拟合/文件加载、独立私有终止 Q 学习/加载、观测角色计划和 selection 奖励验证程序。它们不再依赖任意 caller 评分器充当“已训练模型”。**实际 P 描述符仍为 0，G/Q 没有实际拟合，真实 selection 资格验证和最终候选效果都未执行。**

因此，不能仅以“没有同样本 RNA—Annexin”否定所有分层研究，也不能以这些接口测试通过保证课题将取得预定正结果。强的细粒度初始 RNA 功能收益、状态×实际暴露时长联合泛化及任意序贯治疗收益仍缺相容验证。

## 2. 已真实冻结的评价单位

[evaluation-plan-v2](../experiments/20261011-data-alignment-round15/evaluation-plan-v2/manifest.json) 按冻结输入元数据划分，契约 SHA256 为 `521d184cac76e4e032f3be4c968e2b301f5acb5511fb77af87fa5a6874a1fa6f`。修订前[计划 v1](../experiments/20261011-data-alignment-round15/evaluation-plan-v1/manifest.json) 保留原字节，不继续作为默认资格协议。

Viel 两药各五剂量，每个药物＋剂量有持续 24h 和 6h 给药＋18h 恢复两种完整协议，形成十组。每个 fold 保留一整组作为 final_test，并将按固定元数据顺序的下一组作为 selection；其余八组用于 fit。该规则不读取 Y 挑选条件。

| 当前 v2 每 fold、每分支 | 观测均值数 | 允许用途 |
|---|---:|---|
| fit | 16 | G 拟合、fit-only 归一化；固定模型的模拟 Q 候选域 |
| selection | 2 | 完整两协议的有限奖励模型验证，不是最终测试 |
| final_test | 2 | 同 fold 的最终两协议预测和选择评价 |
| Nav external_test | 4 | 外部状态/两药挑战，来源定义和模型资格另核验 |

十个 fold 重复使用同一批 20 条 Viel 观测，不是新增 200 条独立数据；每分支的总 unique observation_id 仍为 24。Nav 10μM 为主解释，11μM 为同四条 Y 的完整敏感性分支，角色完全一致，不增加独立样本数。全部结果此前已公开读取，这是**回顾性计划**，不能称盲测或前瞻性实验。

selection 和 final_test 都是同一论文中的条件均值，共享原图和对照相关，未取得原始生物重复 ID。这里的独立性是拟合、资格选择和最终报告的信息用途隔离，不代表独立研究或新生物重复。不同 fold 的模型分数不能拼接成一次全候选排行；各次评价只比较该 fold 内具有共同模型、范围和终点的完整候选块。

## 3. 真实 G 学习与加载程序

[fit_g](../src/lung/engine/endpoint_protocol_learning.py:97) 通过已核验角色计划取得 fit 标签，而不是接受 caller 任意标签表。一条发表条件均值权重为 1，不给生成细胞复制 Y。输入是审核 P 摘要＋药物指纹、实际剂量、暴露/撤药/恢复协议、来源状态与未知值 mask，终点定义和完整训练 scope 必须相同。

本轮最小 G 是固定 λ=1 的 ridge 基线，不声称已成为最终算法。令 X 为 fit 条件向量，y 为该来源实测 Annexin fraction；只从 fit 计算均值和总体标准差，常数列尺度置为 1。以 Z 表示标准化后的向量，本课题实现的目标为：

$$\min_{b,\beta}\sum_i(y_i-b-z_i^\top\beta)^2+\lambda\|\beta\|_2^2,\qquad\lambda=1.$$

代码采用不惩罚截距的闭式求解，b 为 fit 标签均值，系数使用观测维度的对偶系统；预测出口截到 [0,1]。截断是本课题的比例输出约定，不代表额外概率校准。

Ridge 损失的原文依据是 van Wieringen *Lecture notes on ridge regression*，arXiv:1509.09169v9，Version 0.80，§1.6，正文 p17、Eq1.8；主代理已读该段。[定位原文](https://arxiv.org/pdf/1509.09169v9#page=22)。该笔记标明仍在修订，本轮只用来核对损失与求解形式，不移植统计效果保证。本课题对 Z 的处理、固定 λ、截距和截断由[实际代码](../src/lung/engine/endpoint_protocol_learning.py:59)定义，不归为该笔记的特定实验。

本课题的线性代数检查：对任意非零 u，uᵀ(ZZᵀ＋I)u＝‖Zᵀu‖²＋‖u‖²＞0，因此求解系统可逆。这解释少量观测、多维特征时程序仍可求解；**不证明高维信息已被识别、跨药泛化有效或误差小**。共享控制、条件均值及非独立重复不满足通常的简单独立误差推断前提，本轮不据此构造生物置信区间。

JSON checkpoint 保存具体参数、特征名、比例尺度、scope、真实 fit 观测身份与标签来源摘要、输入/P cache/角色/源码身份。[load_g](../src/lung/engine/endpoint_protocol_learning.py:147) 在不读取标签的路径上检查工件一致性和 exact fit 身份。加载返回新的范围绑定对象；新[rank_checkpoint_candidates](../src/lung/engine/endpoint_protocol_learning.py:480)先核对完整 source/state/scenario/endpoint 及缓存角色身份，才允许排行。

## 4. 奖励资格验证不是自报“已通过”

实际生产程序为 [validate_reward](../src/lung/engine/endpoint_protocol_learning.py:277)：在推理前绑定已存规则文件 → 审核 P cache → 加载固定 G → 对 selection 完整两候选预测 → 冻结预测及模型/规则工件 → 显式读取 selection Y 计算误差/排序 → 写新的预测、评价和 qualification 文件。

规则包含 max_mae_fraction 和是否要求 pairwise_order_match；当前没有真实验证结果，也未替科学研究冻结某个数值阈值。正式执行前需在 protocol 中固定阈值及理由，不能看到 selection 结果后反复放宽。规则文件 hash 只证明内容绑定，不独立证明它在历史上从未根据公开结局调整。

生成 Q 决策样本和 Q 拟合都会再次运行 [_revalidate_reward](../src/lung/engine/endpoint_protocol_learning.py:317)：重算固定 G 的 selection 预测，再从明确的 selection Y 重算指标，与保存工件逐项比较。final/external Y 不用于选择奖励资格。资格仅适用于这一有限 selection 块的模型模拟；两个发表条件不认证广泛生物效力。

模型使用时的 loader 只核验既有工件与 metadata，不进入标签 API。执行记录、文件自洽和生物效果是不同证据；当前输出保持 execution_certified=false、biological_effectiveness_validated=false。

## 5. 私有一次终止 Q 到底是什么

新[generate_terminal_decisions / fit_terminal_q](../src/lung/engine/endpoint_protocol_learning.py:334) 使用固定 P、固定且有相应有限验证工件的 G，以及 fit 角色的合法候选，形成明确 **model_simulated** 单步样本。每个样本 terminated=true，reward 是 G 预测，不是新增实测凋亡；target=reward，没有第二次给药的 bootstrap 或成本惩罚。Q 使用独立参数，G/P 不回传更新，也没有把 G 对象另取名字叫 Q。

动作价值定义的原文依据为 Xiong et al., P-DQN，arXiv:1810.06394v1，§2、Eq2.1–2.2。[定位原文](https://arxiv.org/html/1810.06394v1#S2)。本课题按照一次终止回合定义推导：回报只含这一项终点 reward，因而此处 Q 的目标是其条件期望；不声称实现了论文的混合动作 P-DQN，也不声称已验证多步 Markov 转移。

这是**一次终止候选价值回归/蒸馏对照**，不是已完成的旧 DA-AC 或真实序贯 RL。直接 G 枚举仍是必要对照。若 Q 在全部合法候选上恰好等于 G，两者的最大值选择相同，Q 没有因此创造新的真实药效信息。

本课题另给出有条件的选择误差检查：若同候选域上 sup|Q−G|≤ε，令 aG 和 aQ 分别最大化 G、Q，则 G(aG)−G(aQ)≤2ε，来自分解两项逼近误差及 Q(aG)≤Q(aQ)。这是本课题在显式一致逼近假设下的推导，当前没有测得 ε；它约束对 G 的选择损失，不约束真实 Y，也不证明 Q 优于 G 枚举。正式 RL 方法收益及更多步骤仍需相应数据与实验。

## 6. 独立复审发现并修复的问题

[独立报告](../experiments/20261011-data-alignment-round15/independent_review.md)留下修复前纯 mock 反例和修复后核查：

1. 初版奖励资格信任 caller 误差/排序，甚至能接受负误差及 bool 阈值；现有实际生产和训练重算闭环已替代该入口，类型、有限值和尺度严格检查。
2. 初版模型加载后只保留 endpoint 定义，丢失训练 state/source/scenario；现返回独立范围绑定对象，旧任意评分入口拒绝它，新入口逐候选核对完整范围。
3. Q 曾只核对自身 scope，未核对引用的 G；现重新严格加载被引用 G，并比较同 bindings、scope、feature_names。
4. bool 能冒充 fraction 或终止数字；现预测和决策 row 都检查精确类型，terminated 必须是 True，权重为 1，再逐值核对固定 G 输出。
5. 初版“metadata”计划加载会调用旧 source audit 反序列化全部 Y；现改为标签隔离桥＋ID-only mapping及字节 hash，实际 guard 验证禁止标签反序列化仍可加载。fit/evaluation 明确属于读标签 API，会做原源 Y 审计后只返回对应角色，不把此源审计称为 metadata-only。

没有真实模型运行受到上述初版问题污染；本轮代码在最终统一验证前完成修复。

## 7. 主代理实际验收

统一运行 producer、adapter、supervision、旧桥、learning、evaluation 六组定向检查：**100 passed in 14.30s**。[实际日志](../experiments/20261011-data-alignment-round15/pytest-main.txt)。测试包括纯数组、手写固定参数、mock 调用、真实 metadata guard 和显式标签读取/指标检查，未训练或推理真实模型。

[主代理 verifier](../experiments/20261011-data-alignment-round15/verify_round15.py)重核 v1/v2 角色计划、24 unique Y、各 fold 分组、冻结旧文件、源码绑定与实际 G/Q CLI。G/Q CLI 均为预期 pending；登记 pipeline 缺 response/pool/state/representation，真实 P cache 未生成。实际回执见[validation.json](../experiments/20261011-data-alignment-round15/validation.json)。接口存在不等于整条数值调用已经在真实权重上通过。

本轮两条理论定位还做了[Zotero全库精确核对](../experiments/20261011-data-alignment-round15/theory_source_audit.json)：只读全部60个顶层条目，按规范化标题或arXiv URL未匹配；不据此排除所有别名，也没有写回Zotero。作者教材PDF访问失败后使用上述可读取的原文作为回退，未把不可访问表述为理论不存在。

[只读权重清单](../experiments/20261011-data-alignment-round15/matching_weight_inventory.md)确认所查范围确有旧AE、response及有界真实组件文件和登记，不能说“全机没有权重”。主代理核对最新modular_state及有界run/report：旧完整身份为0560…67b4，有界视图为d6cb…02e4，均不等于当前joint的2909…55d9。83轮、3431次更新是旧AE恢复登记，不是新联合模型的实测进度。本轮没有完整加载大权重或认证其当前内容，所查登记不能直接取得新标准P资格；数据版本/预处理兼容问题与原始数据不足分开处理。

## 8. 当前架构覆盖与剩余条件

| 架构要求 | 已完成 | 尚未完成 |
|---|---|---|
| §2/4.2 完整候选与真实时钟 | 冻结输入、6＋18/持续24区分、同T*=24 | 更广泛协议与状态×时长共同覆盖 |
| §4.3 P/C/实测G分工 | P生产/读取、状态标签保留、G拟合/加载代码及真实条件Y | 匹配权重、真实P、跨来源状态迁移和G留出精度 |
| §4.4/4.5 私有候选价值 | 一次终止Q、固定P/G、有限selection资格和加载/排行代码 | 实际G资格、Q学习及相对枚举收益；旧DA-AC/多步验证 |
| §6 独立测试与整体增量 | observation身份、metadata分组、fit/selection/final用途隔离 | 同一冻结P→G/Q真实整链评价、高维初态功能增量 |

当前是**有真实数据、可运行程序和可证伪评价计划的有条件可行路线**。Viel可检验有限实际暴露协议，Nav保留真实状态/候选挑战，但不同assay/分母不直接混合为统一G，也不将Viel训练头直接用于Nav定量外部评价。共享标准参考P依然是迁移假设；不是每条Y的真实初态或真实药后RNA。

下一步先核对可用checkpoint是否匹配当前真实联合数据身份，在明确训练授权后生成标准P，再执行固定G、selection验证、私有Q及同fold最终评价。若最终效果不支持假设，调整方法或结论；不以增加模拟样本代替真实条件。训练保持停止，未修改架构、旧模型或冻结核心数据，未提交/推送，本轮没有新Cloud/Dot派单。完整目标仍active，尚不能登记“全课题数据可行性已通过”。
