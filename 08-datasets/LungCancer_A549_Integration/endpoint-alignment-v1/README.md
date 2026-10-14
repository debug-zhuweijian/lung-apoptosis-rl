# 条件级终点对齐 v1：执行中

日期：2026-10-10。目标保持为“完整实现数据对齐，确保整个课题在数据上的可行性”。当前完成部分数据工程与原文核验，**完整数据可行性尚未通过**。

## 当前有效入口

| 目录 | 实际内容 | 使用状态 |
|---|---|---|
| `curated-20261010-r5/` | 308 条件、846 个源位置、844 实值；全部源位置中16个排除（含2缺失） | 当前条件/观测表；新增4个SJB发表均值，保留r4全部记录，尚非 G 训练 manifest |
| `chemical-links-v2/` | Inde 37 + Macaya 3 个来源药物，40 个名义结构核实；Inde 36个名义完整结构候选 | 厂家证据消歧后的新版本；实验试剂批次/样本配对不自动成立 |
| `condition-inputs-v6/` | 与条件表连接的283个源化合物条目；52个可用结构，59条件有化学输入；含实际2048维指纹 | 预测特征生成请求；实际 RNA 特征尚未生成 |
| `coverage-v5/` | 全308条件复核；42 Annexin汇总中17个单药24h条件有化学输入，14个结构、4个来源 | 按检测定义分别报告；当前四分区计数不是所有研究设计的可行性标准 |
| `source_reviews/state_function_round8/kotani/challenge-v1/` | 4活性条件、2未知时钟参考，18均值与24图印值分表 | 独立读取/评价已运行；ND门内定义，input-ready/prediction-missing，不并入核心G |
| `source_reviews/state_function_round9/` | SJB原件/结构核验及实际接入；HIF1A/BRG1的11条ENA元数据和明确错位记录；其它新来源资格审查 | SJB三个24h剂量进入v6；没有新增实测初始RNA—凋亡配对 |
| `source_reviews/state_function_round10/` | HIF1A原始附件80个比值/8条件；CellCage原文及公开数据入口；BRG1与伴随论文限定审查 | HIF暴露时长/误差口径未知、无siHIF1A高维RNA；CellCage真实CCE连接存在但矩阵/行映射未取得；r5/v6不变 |
| `source_reviews/state_function_round11/` | Vielhauer原图、22个总Annexin均值和2个完整结构；TIDYI三个作者notebook及60行计数/CFSE；克隆/HSP90来源资格 | 有限暴露方案的数据缺口取得实质进展；初始表达增量仍未验证，未混成自然群体或单细胞命运标签 |
| [Round12状态参考索引](../../../experiments/20261010-data-alignment-round12/state_source_manifest.json) | 34真实上下文、3,176个六role处理、2,195个共享候选成员；保留原角色与对照可用数 | 可构造表达分布与同参考均值比较，无需强制每个功能Y同测RNA；不是新划分、训练结果或完整链已接通 |
| `source_reviews/state_function_round12/` | RNF20原矩阵3个A549列与时钟/图文审计；GSE200541元数据；CellCage访问收束 | 来源资格分层；0新增已放行G标签，谱系未完成包不计入验收。最新组合设计见[Round12报告](../../../docs/DATA_ALIGNMENT_ROUND12_2026-10-10.md) |
| [Vielhauer时长挑战](../vielhauer-duration-challenge-v1/manifest.json) / [化学输入附表](../../../experiments/20261010-data-alignment-round11/chemical_condition_inputs.json) | 20 active＋2图对照、10比较；6h给药＋18h恢复与24h持续暴露均T*=24h；20×2048指纹实际可读 | input-ready/prediction-missing；仅1状态，无RNA或生成表达特征，未接入现有G训练。洗脱分母、n歧义等见[来源README](source_reviews/state_function_round11/exposure_duration/README.md) |
| `state-expression-v2/` | 三项外部 bulk 状态表达按既有32,461基因轴重排；新增16个GSE246690 A549样本 | 保留 bulk 原单位及缺失mask；不直接送入单细胞归一化/分类器 |
| `source_reviews/chembl/` | 5 篇原文审查、原始图表及纠错记录 | 数字、检测定义、条件与结构逐项分开核验 |
| `source_reviews/state_function/` | 52个真实功能源值、16个RNA样本/状态及原始资料 | 原始协议存在具体缺口，不能登记精确配对 |
| `source_reviews/state_function_round2/` | Sarin的22个表达样本、10个Annexin fold原值、3个条件连接；含后续线索核验 | 平行24h无药参考，非t0实测；fold不能当fraction |
| `source_reviews/state_function_round3/` | NSH76三状态流式图、GSE246690四状态表达及其它原文核验 | 见新`dose_protocol_readback.md`：NSH76剂量可按全局方法推定，但状态与剂量共同变化 |
| `source_reviews/chembl_round2/` | 新6k、1a、8h等原图和方法 | r4已纳入12个代表图；8h的4个条件保留检测方法hold |
| `source_reviews/chemical_round3/revision-3/` | 1a、8h与6k逐结构核验及原图更正 | 6k已审完12中心，11明确、C4两楔有歧义；不采用已知漏羰基的旧结构。v5输入保留其绑定的revision-2快照 |
| `source_reviews/state_function_round4/nav_gal/challenge-v1/` | 两状态×两活性药及vehicle；6相关曲线的60个发表均值，24h有4个活性处理状态均值 | 有条件的群体挑战；数字化估计、10/11μM冲突、密度/洗脱限制另记，非原始重复或RNA配对 |
| `source_reviews/state_gene_round4/` | Sarin 22个微阵列样本；12,070唯一探针基因形成共同轴矩阵，完整探针层保留 | 原值及mask验证通过；保持微阵列尺度，跨平台模型适配和fold评价尚需验证 |
| `source_reviews/duration_round5/cleaved_caspase_challenge_v1/` | A549 cisplatin 2h去药与持续72h，共同72h的两个相对cleaved caspase-3发表均值 | 有限共同终点时长挑战；不是Annexin比例、原始重复或完整联合选择 |
| `source_reviews/tahoe_state_design_round5/` | 实际v3元数据中154个相同化学身份/剂量/24h的跨板组、325条件 | 候选参考分布比较；表达与生物状态效应未验证 |
| `source_reviews/state_expression_function_round5/` | 2个新重点来源与1个旧来源复核，16个GSM、12个A549 ENA run和18个旧源功能值 | 状态/对象/单位不一致保留，没有新增合格联合配对 |
| `source_reviews/chemical_disambiguation_round2/` | 6项材料/别名问题中5项取得厂家证据 | 已供identity-v3/chemical-links-v2采用；Quinidine混合仍排除 |
| `source_reviews/mphahlele/` | 11个系统命名结构转录、原文及独立复核记录 | 9个转录通过；3e/3k原文分析数据矛盾保留 |

`curated/`、`curated-20261010-r2/`至`r4/`、`chemical-links-v1/`、`condition-inputs-v1/`至`v5/`、`coverage-v1/`至`v4/`及`state-expression-v1/`为前序快照，保留以便追溯；本轮消费者显式使用上表版本。脚本兼容默认值可能仍指旧版，不能省略版本参数后猜测读取了最新版。

Nav-Gal现有[独立读取与评价入口](../../../docs/NAV_GAL_CHALLENGE_EVALUATION_2026-10-10.md)，真实状态为input-ready/prediction-missing；不混入上述核心G训练表。[第五轮覆盖审查](../../../docs/FULL_DATA_FEASIBILITY_COVERAGE_ROUND5_2026-10-10.md)区分局部可检验和完整目标通过。

## 对应单位和表结构

- `conditions.parquet`：一个来源内的完整实验条件，含来源块、初态身份、真实处理时长、背景与协议。不把多个研究的同药同剂量自动合为同一实验。
- `interventions.parquet`：该条件的药物成分、来源化合物ID、剂量和单位。同时组合保留多个成分，不冒充真实序贯。
- `observations.parquet`：真实源单元格或公开图文中的一个数值。保留重复ID、源文件SHA256与定位、检测定义、缺失和排除原因。
- `condition_units.parquet`：条件级汇总。`available_numeric_records`是可用数值条数；`measured_replicates`只计明确取得的独立实验原值。代表图和发表标量不能变成三次原始重复。
- `protocols.json`：Annexin与SYTOX分别定义；保留分母、门控、对照、来源和测量局限。

目前所有初态仍明确标为参考状态，未伪造终点实验自己的初态表达。新状态功能源资料另行保留，待剂量/时长/克隆等连接冲突解决后才能进入对应条件表。

## 复算和读取

在项目根目录运行，重复构建必须选择新输出目录：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/build_endpoint_alignment_v1.py --output '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/<new-version>'
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/extend_endpoint_alignment_round2.py --help
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/build_endpoint_condition_inputs.py --alignment '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/curated-20261010-r4' --chemical-links '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/chemical-links-v2' --additional-chemical-review '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chemical_round3/revision-2' --output '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/<new-input-version>'
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/align_state_function_expression.py --include-round3 --output '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/<new-state-version>'
```

`src/lung/data/endpoint_alignment.py::load_alignment`校验表文件和原始来源hash，并逐行绑定观测与来源登记。它拒绝重复实验伪增、把预测分数当实测、端点与协议不一致，以及没有证据的实测初态/同样本配对声明。

`scripts/align_state_function_expression.py`已实际运行。Jochems2021的20,029基因中18,667个匹配当前轴；GSE133029的19,713符号中15,310个匹配；GSE246690的60,728行中32,432个唯一基因匹配。后者45个身份在去掉数字后缀后出现90行歧义，均排除而不擅自求和；971,648个A549计数与官方工作簿逐项读回一致。未匹配基因以NaN+mask表示，未填为0；原始bulk log2CPM和raw counts分别保留，没有混作同一种测量。

## 已发现的来源问题

1. Klingbeil2016 的部分ChEMBL记录把JQ1 48h写成24h，并误读交换坐标轴后的象限。正式对齐不能直接采用这些二次记录。
2. Mostafa2018的5FU剂量/时长未独立核实；compound19可核对2µM24h，但四象限和不等于100%。后者保留数值并在正式条件汇总中排除，未重新归一化。
3. Mphahlele2023的3e质谱实测/计算值有矛盾，3k质谱分子式漏掉系统名中的四个F。结构转录与试剂身份验证分开。
4. 状态功能资料中存在TRAIL蛋白动作、Caspase取点不明确、RNA撤药协议不同、RNF113A剂量单位/时间/克隆/重复标注冲突。详细原文证据见各来源README；不默认选择最方便的一项解释。

## 与研究架构的对应

已按[研究架构](G:/研究生科研文件/毕业论文/研究架构.md) §2、§4.2、§4.3、§6核对：输入保留状态、药物、剂量、时间、来源及协议；C状态标签和G实验终点不混合；基因轴对应不等于平台迁移有效；没有把bulk表达送入单细胞C后解释成凋亡概率；没有用单药数据补造序贯轨迹。

当前 `formal_training_ready=false`：尚需补齐冻结模型派生表达特征、协议一致的终点划分与独立验证。现有G已经新增schema2“预测特征—真实终点”接口，按条件与物理观测定位而不编造expression_sample_id；schema1保留。接口通过工程测试不改变真实特征仍未生成、独立功能覆盖不足的状态。

外部特征生成器 `scripts/generate_endpoint_features.py` 已实际预检。最新joint入口为`experiments/20261010-endpoint-feature-preflight-joint/final-v3/`，显式读取v6；2,431个fit-only控制细胞、75个CSR、12条参考映射和17个候选输入已核验。当前预检缺匹配的响应/群体/状态权重；跨来源功能校准、初态增量和完整候选选择的数据证据另需验证。它不读取外部Y选择参考，也不把参考群体声明为终点实验的实测初态。旧identity-v3及joint final-v2预检分别保留。研究设计中哪些阻断可以调整、哪些证据不可替代，见[独立设计复审](../../../docs/DATA_FEASIBILITY_DESIGN_REVIEW_2026-10-10.md)。

完整执行、验收状态及接续任务见[执行记录](../../../docs/DATA_ALIGNMENT_EXECUTION_2026-10-10.md)。

面向用户与汇报的解释见[当前状态说明](../../../docs/DATA_ALIGNMENT_STATUS_2026-10-10.md)。

2026-10-10第八轮：[Cloud/Dot联合审查](../../../docs/DATA_ALIGNMENT_REVIEW_CLOUD_DOT_2026-10-10.md)已返回并核验。新增独立来源在`source_reviews/state_function_round8/`，包括Kotani的ND门内Annexin挑战、Cutty的成像命运与原图错误隔离、Pogacar的官方元数据资格。第九轮进一步完成SJB追加、Kotani评价入口和joint final-v3输入验收，见[第九轮报告](../../../docs/DATA_ALIGNMENT_ROUND9_2026-10-10.md)；未恢复训练或伪造初态RNA。

## 第十三轮完整协议输入与方法审查（2026-10-11）

新增独立[完整协议输入包](../endpoint-protocol-inputs-v1/README.md)，28候选/24活性Y/14标准P请求，核心r5/v6表数量不变，不混加训练样本。输入和终点独立保存；真实P、协议G/Q及整链评价仍pending。证据和实际回验见[第十三轮报告](../../../docs/DATA_ALIGNMENT_ROUND13_2026-10-11.md)、[方法方案](../../../docs/DATA_ALIGNMENT_METHOD_SOLUTION_2026-10-11.md)。
