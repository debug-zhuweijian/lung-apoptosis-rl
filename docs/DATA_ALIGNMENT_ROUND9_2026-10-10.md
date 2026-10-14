# 数据对齐第九轮：新增真实条件接入、Cloud 与 Dot 复核后的状态

日期：2026-10-10。研究目标仍为 A549 起步的可观测状态条件化响应预测，以及有限药物、剂量和真实暴露时长选择。本轮没有启动正式训练。

## 1. 先回答是否解决

**整体尚未解决，但已取得并接入新的真实凋亡数据；目前不能把问题归结成“所有数据都无法对应”，也不能说“只差继续训练”。**

已完成的是条件身份、药物结构、单位、真实时长、观测位置及输入接口的可复查连接。未完成的是：初始表达提供的额外信息能否在真实功能候选块中被独立检验，跨来源预测特征与实测凋亡能否可靠连接，以及完整状态条件化方案选择是否有足够相容结局。

不要求每个条件都有同一细胞给药前后的 RNA。允许群体参考、平行培养和冻结模型产生的特征，但需要保留其身份和迁移假设。不能把一个条件比例复制成很多细胞的未来命运标签。判据沿用[证据要求](DATA_FEASIBILITY_EVIDENCE_REQUIREMENTS_2026-10-10.md)与[最新研究架构](G:/研究生科研文件/毕业论文/研究架构.md)，没有以缩小课题承诺换取“通过”。

## 2. 本轮实际完成的数量

主代理运行[真实数据及消费者验收](../experiments/20261010-data-alignment-round9/verify_round9.py)，exit 0；结果为[validation.json](../experiments/20261010-data-alignment-round9/validation.json)。

| 项目 | 前一有效版本 | 第九轮实际版本 | 应如何理解 |
|---|---:|---:|---|
| 核心实验条件 | 304 | 308 | 包括不同终点、对照和排除项，不全是凋亡训练条件 |
| 源位置／其中有数值 | 842／840 | 846／844 | 新增4个发表均值；没有增加原始生物重复 |
| 可用 Annexin 条件级汇总 | 38 | 42 | 包含对照；不是42种药或42个研究 |
| 单药24h且化学输入可用的 Annexin 条件 | 14 | 17 | 按来源协议区分，不能直接合为统一校准样本 |
| 上述候选的完整化学身份／来源 | 13／3 | 14／4 | 新增一个名义化合物、三个剂量 |
| 核心表实测初始表达条件 | 0 | 0 | 仍使用明确标记的外部参考，不伪造初态测量 |
| 已生成完整表达特征 | 0 | 0 | 当前仍缺与联合数据身份相容的冻结模型 |

当前有效入口为[curated-r5](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/curated-20261010-r5/manifest.json)、[condition-inputs-v6](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/condition-inputs-v6/manifest.json)与[coverage-v5](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v5/coverage.json)。旧 r4/v5、旧化学审查共85个文件前后哈希一致；旧记录和指纹按化合物身份保留。[SJB独立复审](../experiments/20261010-data-alignment-round9/sjb_independent_review.md)及15项定向回归通过。

## 3. SJB2-043：新增的真实24h剂量条件

原始论文为 Wu 等，DOI [10.3390/cimb47030155](https://doi.org/10.3390/cimb47030155)，PMCID PMC11941171。主代理亲读方法、Fig4图注与[原图](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round9/sjb/cimb-47-00155_p08_image2_Im6.jpg)，并独立重算像素读数。[源资料说明](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round9/sjb/README.md)保存完整证据。

Fig4图注直接规定24h。Annexin/PI的四象限印值与右侧发表均值是两类数据，已分开保存。接入核心表的只有以下四个均值估计，每条件一条：

| SJB2-043 | 发表均值读图估计 | 像素读图范围 |
|---|---:|---:|
| 0 µM | 3.6% | 3.3–4.0% |
| 0.625 µM | 8.3% | 8.0–8.8% |
| 2.5 µM | 9.5% | 9.2–9.9% |
| 10 µM | 20.1% | 19.8–20.6% |

图注 n=9，但重复性质未明确，原始重复值和SD没有取得。范围是像素提取不确定性，不能当作生物学置信区间。代表性四象限求和没有重复进入监督；也没有将 n=9 展开为九条假观测。[主代理像素复核](../experiments/20261010-data-alignment-round9/sjb_pixel_readback.json)验证4个均值和16个图印值。

协议保留为作者呈现的 Annexin/PI 分析母门；上游门控、事件计数及浮细胞回收未报告，Annexin 对照的溶剂未明确。不能借 Fig4B 的 DMSO WB 对照补填，也不能声称得到完整原始培养群体的自然凋亡率。该协议与旧来源分别登记。

材料连接是论文明确 MCE 厂家及名称，对应当前官方 HY-15757 名义结构。主代理亲读[官方结构字段](https://www.medchemexpress.com/SJB2-043.html)，材料审查保留原实验货号、lot和纯度未知；不把当前产品记录当历史批次鉴定。[化学审查清单](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round9/sjb/chemical-review-v1/manifest.json)递归绑定旧审查，原来的两个暂缓结构仍暂缓。

三个非零剂量已有真实2048位结构指纹和条件请求。条件输入总包的283条指纹均从登记结构逐位重算；未解决结构仍不可用。[接入验证](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/condition-inputs-v6/round9_validation.json)记录旧表保持和新增范围。

**实际用途：** 可支持这一背景、这一药物的三个24h剂量功能挑战。**尚不支持：** 初始表达增量或多状态选择。论文虽然描述10µM/24h RNA，表达文件和共同样本键未取得，RNA—Annexin实测配对仍为0；不能把方法中的名义重合写成已读矩阵。

## 4. Kotani：已从源表进入独立评价入口

上一轮取得的 Kotani Fig4 数据现在有[构建器](../scripts/build_kotani_challenge.py)、[读取与评价器](../scripts/evaluate_kotani_challenge.py)及[真实运行记录](../experiments/20261010-kotani-input-readiness/README.md)。本轮实际构建并读回，状态为 `input-ready/prediction-missing`。

- 四个活性条件：未作axitinib前处理的ABT-263 24h，以及axitinib前处理后的ABT-263 6/12/24h。
- 两个无ABT参考的时钟未知，单独保留，不用于时间匹配扣底。
- 18个发表均值与24个代表图印值分表。Annexin和PI均以未破碎细胞门ND为分母；ND比例另以全部采集事件为分母。
- 主比较只有两个24h条件。6/12h允许辅助逐点评价，不构成共同终点下的暴露时长选择。
- 当前未提供预测，MAE、预测差异等均为null。两个发表均值之差9.8个百分点只是观测描述，不是模型收益或因果状态效应。

预测入口拒绝错误协议、分母、门控、状态、时长、未知条件和虚构同测RNA。预测文件声明的模型身份仍需外部核实，接口不把声明本身当作真实冻结模型的独立证明。

实现代理和独立审查均完成26项定向测试；主代理再次实际调用读取器，重算结果与保存JSON完全一致。[独立复审](../experiments/20261010-data-alignment-round9/kotani_independent_review.md)未发现阻断问题。这里没有启动模型推理或训练。

## 5. 新17条件已通过真实输入链，但特征仍缺失

[final-v3预检](../experiments/20261010-endpoint-feature-preflight-joint/final-v3/preflight.json)已绑定v6和已完成的联合响应数据。主代理实际初始化只开放T角色的 `EngineData(require_preprocessing=True)`，通过 `mapped_conditions` 得到17个候选输入、14个化学身份；SJB三个剂量与24h时间逐项一致。

参考仍为2,431个fit对照细胞，82条参考来源记录均复核；12条来源/背景/协议映射明确标为未经生物验证的参考假设。17个候选身份均未见于fit，其中8个剂量超出全局fit剂量范围。SJB三个剂量在该全局范围内也不能证明新药预测可靠。

当前预检缺项仍为 response、pool、state 三类相容冻结检查点；因此命令返回非零并保存 `pending`，没有写出占位特征。**缺权重是当前执行断点；初态增量、跨来源功能校准和完整候选覆盖是另外的科学证据缺口。** 不能将两类问题合并为“只要训练完成即可”。

## 6. Cloud、Dot分别做了什么

协作记录见[Cloud/Dot工作说明](CLOUD_DOT_COLLABORATION_2026-10-10.md)。两者均实际参与并已返回，不是仅计划调用。

**Cloud负责新公开来源及定向原文核查。** 首轮给出Kotani、Cutty、Pogacar等线索，主代理随后取得原件和数据。第九轮限定查询保留IJO2019固定顺铂状态挑战，去重Frontiers2023；BRG1定向审查于20:17返回。主代理独立核对11条ENA元数据，其中确有四个A549 RNA run；RNA中的siBRG1与功能中的siHIF1A/PFI3不能互换。有关功能强度、时钟和矩阵未读的边界见[HIF1A/BRG1审查](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round9/hif1a/README.md)。

**Dot负责数据链与论证的独立审查。** 它读取指定本地状态记录、研究架构并核对PerturbCellRL v2，认可分层、可证伪的验证路线；没有认可“完整数据已足够，只差权重”的主张。它特别指出预测价值与选药价值需要分别检验，批次/密度完全混杂不能靠增加一个协变量就消除。Dot未重跑全部源矩阵或原始功能图，主代理保留了这一审查范围。[Dot原文](../experiments/20261010-cloud-dot-coordination/dot_review_extracted.md)。

PerturbCellRL v2已在此前完成原文核查。本轮没有将其未配对表达训练方法解释成凋亡标签或真实初态作用的保证，详见[状态说明第13节](DATA_ALIGNMENT_STATUS_2026-10-10.md#13-perturbcellrl-v2对当前数据问题的帮助和限制)。

## 7. 尚缺的具体连接与下一步

| 层面 | 已有基础 | 仍要解决 |
|---|---|---|
| 条件级响应输入 | 共同基因轴、化学身份、群体对照、实际响应读取及17个外部候选输入 | 需要相容模型才能产生真实预测；预测误差须独立评价 |
| 真实凋亡评价 | 42个核心条件汇总及多个独立挑战 | 不同分母/尺度保持分开，不能直接作为一个统一G训练集 |
| 状态增量 | Sarin、Nav-Gal、Kotani等有限群体状态挑战 | 高维初始表达相对药物、类别、来源及群体均值的增量尚未建立充分验证块 |
| 有限方案选择 | 有局部药物/剂量比较；Hastings有共同72h的有限暴露方案 | 完整预测—选择程序与无状态/简单基线须在相同可比真实结局上接受检验 |

下一步优先补少量能改变上述缺口的源附件或样本条件，而非继续扩大不相容论文的数量。BRG1 Fig6的定量数据、材料与完整时钟、相容初态是一个具体未闭合线索；获取失败记录保留，不声称数据不存在。原始RNA整库下载在这些基本对应尚未核定前没有启动。

已经可读的挑战也可以在未来相容模型冻结后独立评价，允许无增益或负结果；可行性不是预先要求模型获胜。本轮保留 `full_scientific_feasibility_passed=false` 和 `formal_training_ready=false`。没有改课题标题、研究架构或正式训练配置，没有写回Zotero、联系作者、提交或推送。
