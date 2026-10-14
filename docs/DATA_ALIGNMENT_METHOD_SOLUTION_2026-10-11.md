# 数据对齐的方法依据与实际接入方案

日期：2026-10-11。接续[第十二轮](DATA_ALIGNMENT_ROUND12_2026-10-10.md)，保留[毕业论文研究架构](G:/研究生科研文件/毕业论文/研究架构.md)规定的 A549 状态条件响应、真实凋亡与药物—剂量—实际暴露方案选择。本文为方法与数据审查，不报告模型效果。

## 1. 当前判断

**“没有每条功能终点同时测得的药后RNA”不自动使课题做不成。可实施的研究路线是：合法状态或参考输入产生预测响应描述符，再由完整实验条件对应的真实终点监督功能头。** 这种连接是条件级建模；它没有恢复同一细胞的真实前后配对，也没有先验保证描述符能改善凋亡预测。

CSG2A 是最直接的方法先例。主代理亲读其原文 Fig.1d、§3.2.2–3.3.2，并核查固定源码与 notebook：表达响应模型预训练后，预测表达参与真实 log IC50 预测。该任务不要求每条 IC50 同时测得药后表达，但有细胞系基准表达与药物反应的身份连接。其固定 10 µM/72 h 是特征生成约定，不能冒充本课题的实际处理协议。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11256952/)、[本地逐项审查](../experiments/20261010-data-alignment-method-search/endpoint_transfer_review.md)。

因此应当分别回答：原始身份是否对应、监督单位是否合法、输入能否表示完整协议、独立评价是否可实施、模型是否取得收益。不能用“必须成立”代替验证，也不能用严格超过架构要求的配对门槛把所有路线判死。

## 2. 原始论文实际解决了哪些问题

以下是主代理亲读决定性原文或源码后可采用的依据；四条代理报告另保留更广的审查范围及未核项。这里没有把每篇论文的所有实验或理论保证迁移到本课题。

| 原始方法与定位 | 可用于本课题的依据 | 不提供的证据 |
|---|---|---|
| CSG2A，Bioinformatics 2024，Fig.1d、§3.2.2–3.3.2 | 预测扰动表达可作为真实功能预测的输入；基线表达与终点可按细胞系身份连接 | 不预测 Annexin；统一特征时钟不证明洗脱RNA；不证明本机代码已接通 |
| CellOT，Nature Methods 2023，正文“Predicting perturbation responses…”，Methods“Optimal transport”与“Limitations” | 从真实对照与处理群体学习分布间映射，有明确的最小运输建模假设 | 不能把模型耦合当观察到的同细胞轨迹；原方法不能恢复增殖/死亡造成的群体变化 |
| Tsai & Lin 2020，出版 pp.515–516，§2.1–2.2 | 比例监督可以不拥有每个实例的标签，但输入bag与其自身比例必须对应 | 不能把任意外部比例赋给生成细胞；本文引用的其他理论尚不能当本课题保证 |
| Ilse et al. 2018，PDF p.2 Eq.(1–2)、p.4 Eq.(9)与Interpretability | 集合表达可产生bag表示，attention可作为诊断线索 | 原目标为“至少一实例阳性”的二元标签，非Annexin比例；attention不是单细胞命运真值 |
| Kidger et al. 2020，PDF p.3 §3/Eq.(3) | 时间路径可以驱动隐状态，终端状态再映射成输出 | 表示能力不等于稀疏公开端点能辨识任意药物开关轨迹 |
| Dudík et al. 2011，PDF pp.2–3 §2/2.2–3 | 部分动作日志的DR评价依赖真实采集机制与估计隔离 | 未知实验分配概率不能填均匀值；完整候选块无需凭空制造propensity |
| Elmachtoub & Grigas，arXiv v5作者稿 pp.8–9 Eq.(1–3) | 预测条件期望并根据候选目标选择，可作为单次决策的直接对照 | 不保证单次RL优于枚举；SPO+一致性保证有自身假设，本文未套用 |

原件与定位分别见[响应迁移报告](../experiments/20261010-data-alignment-method-search/endpoint_transfer_review.md)、[弱监督报告](../experiments/20261010-data-alignment-method-search/weak_supervision_review.md)、[暴露协议报告](../experiments/20261010-data-alignment-method-search/exposure_protocol_review.md)、[决策证据报告](../experiments/20261010-data-alignment-method-search/decision_evidence_review.md)。CellOT原始XML在[官方全文副本](../experiments/20261010-data-alignment-method-search/unpaired_response_sources/CellOT2023_PMC10630137.xml)，主代理读取 `Sec3/Sec13/Sec20`，未运行作者实现。

核心原文链接：[CellOT](https://www.nature.com/articles/s41592-023-01969-x)、[LLP](https://proceedings.mlr.press/v129/tsai20a/tsai20a.pdf#page=3)、[MIL](https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf#page=2)、[Neural CDE](https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf#page=3)、[DR](https://icml.cc/2011/papers/554_icmlpaper.pdf#page=2)、[SPO作者稿](https://arxiv.org/pdf/1710.08005v5#page=8)。

源码审查还发现两项影响复现的差别。CSG2A Fig.1d 写 scaled basal 进入下游头；固定 `CSG2A_net.py` 的实际 `forward` 拼接原始 basal，缩放使用共享长度1参数。论文未明确缩放维度，不能说它承诺逐基因参数。示例 notebook 使用行级 `random_split`，不能据此认证论文全部留药评测。TE-CDE 的调用传入跳变网格，但读取链不足以确认真实小时被保留；本课题须自行编码真实时钟。以上属于静态源码事实，尚未运行作者模型。

## 3. 本课题提出的连接：标准P描述符与真实候选分开

以下是本课题设计，不是上述任一论文已经验证的A549方法。

对一个真实条件观测单位 i，分别记录：

- `s_i`：决策前可用的状态类别、培养背景或有来源的参考表达。同期vehicle群体标为reference，不说成同细胞t0。
- `a_i`：药物身份、剂量及完整物理协议，含暴露、撤药/换液、恢复和共同观察终点。
- `o_i`：观测定义，含Annexin类型、分母、门控或图像计数口径。
- `Y_i`：实际来源公布的终点；原始fraction、percent、fold等尺度分别保留。
- `phi_i`：冻结P在声明的标准参考条件下生成的描述符，保留模型/预处理/参考身份。

拟议预测为 `G_o(s_i, chemical_i, dose_i, phi_i, complete_protocol_i) -> Y_i(T*)`。Annexin图像和流式收获群体的定义不同，不能只因数值均落在0–1就无条件混合。可分别设置观测出口或明确来源内评价，具体训练分支尚待接入。

**phi可以是标准连续24 h的响应特征，而实际a是6 h给药＋18 h恢复。** 此时phi的含义是共同参考描述符，不是实际洗脱终点RNA。G必须另读完整协议；如果它去掉协议，两个方案将变成相同输入，无法检验实际暴露方案选择。

当前Vielhauer来源的两臂可共享同药/同剂量的P请求，G/Q保留不同候选ID和数值时钟。Nav-Gal两种群体状态保留类别和密度等背景；标准P请求不是其同孔初态RNA，也不是已证明有状态特异性的P输出。

功能训练的单位是条件/实验观测。生成100或10000个细胞都不会增加真实Y数量。不把一个比例复制成10000个监督标签，也不把10/11 µM两种解释当成两个不同药效实验。生成描述符与Y的对应按真实条件和声明的参考转移连接，不宣称同测RNA配对。

## 4. 这次实际接入的数据

[完整协议输入包](../08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1/manifest.json)由[消费者](../src/lung/data/endpoint_protocol_inputs.py)和[命令入口](../scripts/build_endpoint_protocol_inputs.py)读取：

| 单位 | 数量与含义 |
|---|---|
| 候选输入记录 | 28 = Vielhauer20 + Nav-Gal8 |
| 独立活性观测身份 | 24 = Vielhauer20 + Nav-Gal4；不是24个独立研究 |
| 标准P请求 | 14 = Vielhauer10药物/剂量请求 + Nav-Gal4药物/剂量解释请求 |
| 实际化学输入 | 28×2048位指纹；结构身份与行号从已审核包读回 |
| 直接及递归来源绑定 | 110项 |
| 实际生成P描述符 | **0**；没有占位向量 |

Vielhauer候选为两药×五剂量×两暴露方案，T*=24 h。输入时钟为 `(dose_uM, exposure_duration_h, washout_at_h, drug_free_recovery_h, terminal_time_h)`；持续臂保留无撤药事件，洗脱臂为6/6/18/24。终点倾培养基时的浮游细胞回收未报告，保留unknown；因此评测收获群体比例，不能解释为初始队列累积死亡率。

Nav-Gal有两状态×两药，共4条活性24 h观测。10与11 µM分别成为两个敏感性场景，各映射回同一观测ID；它们不是可选择的两个真实剂量。前药是否撤除未报告，状态诱导与铺板密度共同变化。实际比较范围是所记录群体背景的预测效用，不是隔离密度后的状态因果作用。

候选输入与观测文件分开；模型输入没有Annexin值、像素读图坐标或误差条。source dose对象的数值是合法浓度输入，不属于Y。读取时保留fraction与percent原尺度，未来转为统一单位须同时保留转换记录和观测定义。

## 5. 如何检验原架构的四方面

| 架构职责 | 数据与检验 | 不能提前声称 |
|---|---|---|
| A：状态信息增量 | Tahoe/sci-Plex同参考细胞集合与其均值，公平比较真实药后表达分布；34参考上下文索引已冻结 | 不能由同期参考优势直接推出单细胞未来命运或真实凋亡收益 |
| B：未见条件 | 冻结化学组/条件划分，用真实表达及合格功能留组；条件补全、留药、来源迁移分别命名 | 不把细胞随机划分当新药泛化，不把两个时间点叫任意时间轨迹 |
| C：真实功能一致性 | 相同程序生成的P描述符经相应G，与真实Annexin观测比较，并做去P对照 | C状态代理或IC50结果不能替代Annexin验证 |
| D：联合选择 | 在同一真实可比候选块比较固定方案、直接监督、P→G枚举、P→G→Q；Nav状态块与Viel完整暴露块分别评价 | 分项通过不等于完整链通过；单状态Viel不证明状态×时长交互 |

最终比较必须来自同一冻结程序与同一合法候选域。各来源的检测定义不相同，不把分数拼成一个跨来源排行榜。测试Y不选择参考、评分方向或最好看的方案；当前已查看公开数字，评价登记为回顾性，不追称盲测。

一次选择时Q估计同一终点期望。完整枚举G分数是给定评分器内的选择对照，额外Q价值需落到独立真实Y。部分日志缺采集概率时不用虚构DR；有全部已测候选均值时可直接计算观测集合中的选择差距。

本课题的有限集合代数分析为：对同块发表均值v、冻结评分m和Q所选a，真实选择差距不超过 `2*max|m-v| + (max m - m(a))`。这不是引用论文定理或效果保证。像素提取范围仅作读图敏感性，不能当生物置信区间；总体期望未知时不能把发表均值误差改名总体误差。详见[推导与评价边界](../experiments/20261010-data-alignment-method-search/decision_evidence_review.md)。

## 6. 尚未完成的实际工作

1. 将真实冻结P生成器接入14个标准请求，绑定参考、预处理、化学身份及实际权重。当前没有生成这些响应特征。
2. 实现真正消费完整协议和状态背景的G分支。旧schema2有预测特征监督，但没有因此自动支持6+18洗脱协议。
3. 将完整候选ID、协议向量和共同终点接入实际Q/排序/评测路径，而不是只供数据脚本读回。
4. 冻结适合实际观测数量的训练/选择/评价设计，不机械凑四份非空分区；真实同一Y与共享来源关系保持隔离。
5. 用真实候选块完成整链评价，并报告状态、P描述符和Q的增量；效果可以失败，失败不应改写为正收益。

旧的手工descriptor入口只验证文件格式，不证明数值来自P。独立审查已给出可把Y塞进descriptor的反例；当前实包全部descriptor为空，尚未污染。新增[标签隔离输入桥](../src/lung/engine/endpoint_protocol_features.py)固定已审输入包身份，只读取候选/请求/真实指纹，不反序列化终点文件，并拒绝未经审核的descriptor/cache。[回执](../experiments/20261010-data-alignment-round13/protocol_feature_bridge_receipt.md)明确这只隔离了危险原型入口；生产生成器、固定特征定义及真实cache接入仍未实现。不能把“拒绝假特征”说成“已经生成真实P特征”。

## 7. 文献检索与个人库核对

本轮接续先前SciSpace语义检索、网页检索及原始出版/作者仓库定位，复用下载原件，不将重复入口计作独立证据。四份报告含12篇方法窄审，另亲读CellOT决定性原文；这不是所有13篇均已完成全实验复现或系统综述。

只读Zotero全库顶层条目实际读取60条，穷尽分页；对13个方法按DOI字段或规范化精确标题核对，未取得匹配，论文索引也未命中。此结论仅覆盖这些精确字段，不排除别名条目或其他存储形式。[查询脚本](../experiments/20261010-data-alignment-method-search/crosscheck_library.py)、[查询结果](../experiments/20261010-data-alignment-method-search/library_crosscheck.json)。没有写回Zotero或自动建精读笔记。

本轮未正式训练、未模型推理、未联系作者、未写Memory、未提交或推送。当前数据输入接入有实质进展；**完整数据对齐目标仍保持活动，未验收完成。**
