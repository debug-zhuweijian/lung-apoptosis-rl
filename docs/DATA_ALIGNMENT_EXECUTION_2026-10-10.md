# 数据对齐实施与全课题数据可行性验收

日期：2026-10-11，第十五轮更新。用户目标：**完整实现数据对齐，确保整个课题在数据上的可行性**。本任务持续进行；没有把成功条件改为“下载了数据”或“一个子模块测试通过”。

## 1. 当前结论

已从可行性报告推进到实际工程：药物身份、真实条件终点、共同基因轴、联合表达预处理与独立挑战已经整理。[第十四轮](DATA_ALIGNMENT_ROUND14_2026-10-11.md)实现标准P生产/缓存读取、完整协议G/Q路由和真实监督视图；最新[第十五轮](DATA_ALIGNMENT_ROUND15_2026-10-11.md)补G/Q学习及加载、selection奖励验证程序与冻结角色，100项检查通过。实际P描述符0，登记pipeline缺response/pool/state/representation；没有实际G/Q学习、跨来源迁移或整链功能结果，**全课题的数据可行性尚未通过，目标保持active**。此前各节保留原轮次记录，不把历史“未接线”当作最新代码状态。

以下第3—7节记录本目标第一个执行轮：新增并运行了构建器、实际生成了新数据版本和真实元数据索引、完成定向测试与独立审查。后续轮次按节追加，历史状态不作为最新状态使用。

## 2. 不缩小范围的验收要求

| 要求 | 当前权威证据 | 当前判定 |
|---|---|---|
| 同细胞状态标签、基因身份不乱配 | 既有743细胞对应审查、32,461共同基因轴；本轮外部基因按ID实排 | 基础已核验；跨平台有效性另测 |
| 药物身份能进入真实训练读取路径 | identity-v2 + training-identity-v2；prepare→reader→action_features实测 | 新版本接入通过，明确排除待核验材料 |
| 来源、背景、剂量、时间和协议逐条件对应 | curated-20261010-r3 的三表及protocols | 已覆盖一批真实数据；其余候选继续核验 |
| 实测凋亡不混入一般死亡、增殖或预测值 | Annexin/SYTOX分开；状态源的late Annexin/Caspase另存 | 已实现已覆盖部分的明确分型 |
| 实验重复不伪增、不把代表图当原始多重复 | 源位置、重复ID唯一；缺失、排除、代表标量保留 | 已有构建和校验器通过 |
| 新药物结构与真实终点建立可追溯输入 | condition-inputs-v2 的结构、剂量、时间、真实指纹 | 48源化合物/51条件有化学输入；不等于实测RNA配对 |
| 存在可检验初态功能增益的数据 | 新增Sarin同来源两状态×顺铂条件连接；RNA为24h平行对照，终点为fold | 有有限条件级证据；**完整状态增益验证仍未通过** |
| 终点头接受合法预测特征+真实Y且绑定完整特征版本 | endpoints.py新增schema2及world显式协议/生成范围核验 | **工程接口通过；真实特征未生成** |
| 有按药物/来源/状态隔离的独立功能评估 | 候选已知为开发阶段看过的来源，没有重命名为全盲测试 | **待构建足够覆盖的划分与评估** |
| 时间/候选选择与原架构一致 | 显式小时、组合/单药区别、来源协议 | 仅已观测条件可用；不声称任意时长或真实序贯已支持 |

## 3. 药物身份修复已落地

目录：[identity-v2](../08-datasets/LungCancer_A549_Integration/identity-v2/README.md)。原始Tahoe 379个药物标签对应376个唯一非空CID，全部实际从PubChem查回；连同vehicle输出380行目录、1,336行条件映射和380×2048手性指纹。

191个标签补回来源结构遗漏的立体信息。已分开Doxorubicin/Epirubicin与crizotinib/(S)-Crizotinib两组旧碰撞；Paclitaxel恢复11个已定义立体中心。

4个材料描述与CID参考结构不一致：Trametinib solvate、Elimusertib hydrochloride、Carbidopa monohydrate、Quinidine含15% dihydroquinidine。没有猜替换；12个旧处理条件被显式排除。另2个源标签缺CID，原本未进入selected条件。来源结构核验不是试剂实物鉴定。

实际接入文件为data/common.py、prepare.py、splits.py、reader.py，独立配置[configs/data_identity_v2.json](../configs/data_identity_v2.json)。新资产[training-identity-v2/manifest.json](../08-datasets/LungCancer_A549_Integration/training-identity-v2/manifest.json)含1,296个合格处理条件、2,534,156个细胞索引；排除24,114个与待核验材料关联的细胞索引。读取器确实返回新手性指纹，变更身份文件会拒绝复用。

当前状态为`metadata_ready`：仅完成元数据准备，未重建表达预处理。新结构重新计算了分组和划分，旧AE/响应检查点不能自动视为新划分下的合法续训。旧prepared/v1、training-v1与正式modular配置未覆盖。

## 4. 终点和化学输入实际产物

统一入口：[endpoint-alignment-v1](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/README.md)。当前有效条件表为`curated-20261010-r3`：292条件、830源位置、828实值；2缺失、9自发荧光排除、1象限总和/分母未解决的发表点。保留Macaya两个来源块、Mphahlele 11化合物源图、Inde单药24h原表，并新增经主代理亲读原文的发表数值。

本轮5篇ChEMBL来源精查发现：JQ1原文48h被二次库写成24h，部分象限在坐标交换后误标为early；5FU的图中数值虽可读，完整处理条件未独立确认。已建立纠错记录，未将有疑义的候选直接放行。不能把1,364候选记录数量写成可用样本量。

外部药物身份另建`chemical-links-v1`：Inde37+Macaya3源条目，38个名义结构解析；Inde其中34个与Tahoe完整结构相同，Palbociclib HCl仅母体相同，两条歧义保留。Macaya仅Trametinib与合格Tahoe结构连接。所有实验样本paired字段仍为false。

`condition-inputs-v2`已实际把这些来源结构、9个独立核对通过的Mphahlele系统名转录、Puri compound24作者SMILES连接至条件表，生成48个源化合物的有效结构表示，51条件具有化学输入。组合用药、非24h、来源控制迁移及其他未核实结构都有显式原因，未生成占位RNA特征。

## 5. 状态功能缺口取得的新证据与剩余冲突

[源目录](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function/README.md)已经实际取得52个功能源值、4个Jochems A549状态表达及12个GSE133029 RNA样本。

- cFLIP研究有增殖/alisertib诱导衰老×TRAIL的Caspase3/7原始三重复数值。但TRAIL为蛋白动作，取点小时未明确；能对应0.5µM诱导的RNA还有撤药24h差异。
- RNF113A研究有无药表达和顺铂后Annexin源值。主图/方法20h与源表24h冲突，RNA元数据µg/ml与论文µM冲突，RNA未说明是哪条shRNA；原图明确终点为late Annexin+/PI+，不能改为总Annexin。
- Triana研究有bleomycin诱导状态×digoxin的Annexin/Caspase原值，但特定图的给药条件与配套RNA仍未完整核实。
- EphB1图为处理过程时间序列，未证明重新挑战；已撤回其“初态→再次给药结局”候选判断。

主代理已亲读RNF113A原始Methods、Fig2图注及GEO各样本记录，并亲读cFLIP源XLSX；部分补充方法核对由专项代理完成，均保留源定位。

已实际生成`state-expression-v1`：Jochems数据18,667/20,029基因、GSE133029数据15,310/19,713符号连接到既有基因轴；缺失用NaN+mask，保留bulk单位。**基因坐标对齐没有消除上述实验协议冲突。**

## 6. 验证和架构一致性

以下定向测试共48项通过（8.13秒）：

```powershell
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_endpoint_alignment_v1.py tests/test_external_gene_axis.py tests/test_drug_identity_v2.py tests/test_identity_prepare_v2.py tests/test_training_data_v1.py -q
```

另实际执行了源文件重读、源单元格复算、生成结果读回、原始/派生hash检查和真实新元数据构建。独立审查发现并已修复三项漏洞：逐观测来源绑定、实测初态证据要求、独立重复唯一性；后续复测均拒绝相应反例。原落盘数据未出现这些错误。

已对照[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)核查：

| 架构位置 | 对应实现 |
|---|---|
| §2 状态/候选/响应输入输出 | 状态来源、候选条件、实测Y与预测特征分开记录 |
| §4.2 完整药物、剂量、时长、协议 | 身份v2及条件表，不机械生成笛卡尔积 |
| §4.3 来源条件化表示、C与G分工 | bulk单位和mask保留；状态分数不作为实测Annexin |
| §6 独立划分、真实功能证据 | 来源与重复隔离；未将开发已见资料声明为全盲测试 |

本次没有修改已确认研究主线，没有把凋亡目标替换为一般毒性，也没有声称已有真实序贯验证。没有启动正式训练、提交、推送、写回Zotero或长期Memory。

## 7. 下一轮接续点

1. 继续扩大经过原文核实的Annexin来源及化学身份覆盖；优先可取得原始重复、多药/多剂量/明确时间及多个初态的研究，而不以发表标量数量替代独立覆盖。
2. 初态数据进一步定位A549亲本/耐药派生或明确预处理状态与未给药RNA、再挑战Annexin；本轮已发现的具体矛盾保留，不重复当作无缺口来源。不得未经授权联系作者。
3. 对未解决命名/材料问题补查原厂家、CAS、原始库表；名义结构连接、实验配对和平台迁移三个判定保持分开。
4. 实现显式的“预测特征—实测终点”G输入版本，绑定实际使用的表示、P、C、基因轴、预处理、药物身份和检查点；没有权重或特征时保持pending，不造零向量。
5. 在足够真实条件上构建来源/药物/状态隔离的基线与留出评估，再逐项审计本节前面的完整要求。未通过之前不标记目标complete。

## 8. 第二轮实际完成：身份v3、G接口与覆盖复算

本轮有新的可运行代码、真实数据版本与原文证据，尚未遇到无法推进的持续阻塞。面向汇报和教学的说明见[当前状态](DATA_ALIGNMENT_STATUS_2026-10-10.md)。

### 8.1 新身份已实际进入读取器

根据MCE官方产品资料、Selleck原始目录及CAS/结构核对，3个Tahoe材料形式和2个Inde别名获得名义身份确认。主代理已亲读三个MCE datasheet首页与官方SMILES、Selleck目录第72页及两产品页化学信息；名义形式不等于历史试剂lot验证。

- `identity-v3`：376个合格结构参考；原1,336条件全部保留原列。
- `training-identity-v3`：1,305合格处理条件、2,552,588细胞索引，实际prepare→reader→action_features验证通过；比v2恢复9条件、18,432细胞。
- Quinidine混合继续排除3条件、5,682细胞，没有把15%解释成已知摩尔配方。
- `chemical-links-v2`：40个外部名义身份可解析，Inde的完整结构候选由34变36；实验样本paired数量仍为0。
- `condition-inputs-v3`：50个源化合物有效结构、53条件有化学输入；实际279×2048指纹数组和资格mask一致。

实际新训练资产状态仍为metadata_ready；未重建表达预处理，未复用旧响应权重。旧冻结版本全部保留并复核原hash。

### 8.2 G schema2与推断身份

新契约明确`predicted_features_measured_endpoint`，使用条件与真实物理观测定位；T拟合、V选模、审计后C校准，训练不反序列化E标签。完整绑定所有参与的特征阶段、基因轴、预处理、化学输入和来源文件。一个G只接受已核实一致的label_type/denominator/sampling_gate；未生成特征不能导出ready manifest。

独立复审发现并修复两处推断漏洞：新增特征阶段可绕过旧依赖子集检查，逐次n/ODE参数可绕过构造时范围核验。现已严格比较完整阶段集合，并约束正式单条/批量评分。V-only数值诊断另有显式入口，报告正式范围与偏离参数；其缓存与正式评分分开，诊断不改变正式范围。独立复验确认两处均关闭、schema1仍兼容。

### 8.3 真实覆盖仍不通过

`coverage-v2`重读292条件并验证来源与化学输入hash，结果：30个Annexin条件汇总；14个单药24h化学输入候选，13结构，3来源。每种当前记录的观测定义内，来源/实验及药物联系形成1个划分块，无法满足本版四个非空独立分区。此为当前约束下的组合覆盖结论，不是一般统计功效阈值。

当前完整表达特征条件数为0；没有生成正式G训练manifest，没有把SYTOX或fold变成Annexin fraction。新增化学身份没有增加Annexin标签。

### 8.4 Sarin状态功能证据

实际取得29,729探针×22个表达样本和S8表6的10个Annexin原值，连接为2状态、1药物、3条件。主代理亲读两文方法、Figure5、补充源表与GEO样本协议。RNA对照为平行无药24h参考；非t0实测、非同批配对。功能为相对各自未处理对照的fold，不能反推fraction。源表一处均值的0.001印刷/复算差异保留两者。详见`source_reviews/state_function_round2`。

本轮新增化学文献资料另保存在`source_reviews/chembl_round2`；尚未接入curated-r3者不计入上述30/14。

### 8.5 验证与架构对应

- 终点schema2、world、V诊断、核心终点与外部基因轴：**128 passed，53.65秒**。
- identity-v3、v2兼容、旧训练资产读取：**34 passed，9.95秒**。
- 实际condition-inputs-v3、coverage-v2所有登记输入/输出hash及指纹资格一致性通过。
- 已对照研究架构§2、§4.2、§4.3、§6：状态层级和预测身份保留，完整候选可追溯，终点定义与真实监督绑定，独立评价要求未被降低。

这些结果证明工程与已覆盖数据的对应关系，不证明跨平台迁移有效或生物学收益。下一步仍需增加可比真实功能覆盖、完成外部条件的冻结响应特征生成，以及状态增益与候选选择的独立验证；正式长训练仍由用户启动。

## 9. 第三轮实际完成：新终点、参考表达、生成器与设计复审

### 9.1 当前版本和真实数量

`curated-20261010-r4`保留r3全部原记录，新增6k、Li2012的1a及C9MD00494G的8h共12个代表性流式面板。当前304条件、842源位置、840数值；16个源位置排除（含2缺失）。可用Annexin条件汇总由30增至38，新增8个中6处理、2对照。`condition-inputs-v4`实际282源化合物、50个有效结构、53个条件有化学输入。`coverage-v3`当前单药24h候选仍为14条件、13结构，完整模型生成特征为0。

主代理与独立复审均按原图确认：6k为24h；1a为48h；8h是化合物编号，Annexin实际48h而24h为Western。1a的19.4%为早期象限，合计为23.6%；没有把它当总Annexin。8h四行继续保留补充检测方法未读的hold。单面板不登记成原始生物重复。

[独立r4复审](DATA_ALIGNMENT_ROUND3_REVIEW_2026-10-10.md)确认旧表内容、协议、指纹和21个来源hash保持一致。另指出6k的10/20μM未在冻结r4写入显式外推标记；当前化学hold仍有效，未进入14候选。后续获准接入时须按真实fit剂量范围标记，不能静默改写r4。

### 9.2 四状态表达新增与边界

`scripts/align_state_function_expression.py --include-round3`实际生成`state-expression-v2`，保留v1两项来源并新增GSE246690的16个A549样本。新增矩阵有60,728行，32,432个唯一基因对应共同轴。源字符串本来唯一，但去数字后缀后45个ID产生90行冲突，均排除；没有选择一条或擅自求和。缺失为NaN加mask。

主代理重新从原始XLSX读取全部971,648个A549计数，与CSV逐值一致；读取时恢复错误的worksheet dimensions，不修改源文件。GSM与工作簿列一一对应，只取A549，未混入SK-MEL-147。每状态4个样本文件不自动记成4个独立生物重复。具体证据见`state-expression-v2/source_readback.json`与源GEO元数据。

新增表达对应的navitoclax功能主要是活力，未形成可用于Annexin头的联合标签。NSH76来源另保留三状态的原始流式象限。窄复核接收了按全局方法及Fig6推定的NSH76标称剂量：7.38/2.85/2.63μM，各24h；原CSV的“panel未直接印剂量”不改，另有`dose_protocol_readback.json`。IC50来自24h活力测定，不能叫Annexin IC50；状态和剂量同时改变，也不能由三个点分离状态增益。

### 9.3 实际外部特征入口

新`src/lung/engine/endpoint_features.py`与`scripts/generate_endpoint_features.py`复用真实BiologicalWorld的响应推演、pool、历史和特征切片，拒绝48h、组合和未核实结构。G当前输入不含C类别概率；C概率另记diagnostic，不以架构草图猜实际数据流。

候选参考位于`experiments/20261010-endpoint-feature-preflight-v3/reference_inputs-candidate/`：2,431个真实fit-only控制细胞、75个CSR文件内容哈希、11个原协议的显式映射。参考由稳定元数据规则选取，没有读取外部Y；各来源映射均为`cross_source_transfer_unvalidated`。已实际运行interim预检，参考与映射缺项消除，但尚无匹配的response/pool/state权重，未生成占位特征或ready训练表。

已运行新版完整数据准备：`scripts/prepare_training.py --data configs/data_identity_v3.json --device cuda:0`。最终进程exit_code=0，fit表达缓存、GPU读取缓存、归一化、完整基因图及带宽均完成，manifest为complete；该过程不训练模型。已另建最终参考描述重新绑定，旧metadata_ready候选保留作历史记录。最后状态与版本见9.6。

### 9.4 科学判断的修正

[独立设计复审](DATA_FEASIBILITY_DESIGN_REVIEW_2026-10-10.md)指出，四个固定分区以及把同一来源/发表图块整体阻断，是当前工程契约，不是所有研究的普遍必要条件。`source_and_chemical_holdout_blocks=1`不能改写为已证明只有1个有效生物观察。可以针对预定的来源内留药或冻结模型外部挑战重新制定方案；仍须保持真实观测不泄漏、终点定义不混用、泛化范围如实说明。

原主线的关键缺口仍在：固定同一参考状态的终点数据不能证明初态的额外价值；来源与药物无交叉时不能仅靠protocol字符串分离来源偏差；倍数不能在基线未知时反推出绝对比例。数据和代码的改善不能自动通过这些检验。

输出已对照[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)§2、§4.2、§4.3、§6：允许群体状态，保留完整处理协议，区分C与G、实测与预测，主问题仍包含状态增益和候选选择。没有以一个较小的同参考单药任务替代并宣称完成原主问题。

### 9.5 本轮验证

- r4扩展与原终点读取：18项通过；独立复审另对3个新增函数及来源内容复核通过。
- 新外部特征生成路径与相关回归：76项通过；独立复审发现的参考ID/板号及描述对象与文件未强绑定漏洞已修复，最终18项定向测试通过。独立复验确认正确候选仍14动作；错误及缺失ID/板号、对象与JSON不同均拒绝；描述文件生成途中改变不能发布特征。候选文件本身未出现错误映射，冻结文件未改。
- 外部基因轴：4项通过；真实矩阵全量读回与NaN/mask回读通过。

各测试范围有交叠，不机械相加。当前状态仍是`formal_endpoint_training_ready=false`；没有启动正式模型训练、提交、推送、Zotero写回或长期记忆更新。

化学审图另有一项已纠正的误读：第三轮子审查曾误称6k最终图缺C20甲基。主代理亲看原图定位到相应虚楔，子审查重新核验后明确撤回；修订包`chemical_round3/revision-2`保留完整纠错记录，原报告未覆盖。C11羰基的二次数据库错误仍成立。6k剩余手性逐位审查属于待完成工作，不是原数据不存在或原图已证实有缺陷。

### 9.6 本轮最终发布状态

- **完整预处理完成。** `training-identity-v3/manifest.json` SHA256为`95e1c0cbc60bf9410ae699442d8d28f542af5de11f6bb6e68fc3545fba5df18a`；16个登记文件逐个实际哈希通过。32,461维归一化、32,461×33基因图（self+32邻居）及5个核带宽已实际读取验证。
- **新增条件输入v5。** `build_endpoint_condition_inputs.py --additional-chemical-review`递归核对当前与前序审查的来源哈希，再按明确资格读取名义结构。基于已冻结revision-2，将1a纳入结构输入：51个源化合物有结构、56个条件有化学输入；相对v4新增的是3个48h处理条件。`coverage-v4`仍为38个Annexin汇总、14个当前24h候选、13结构、0个完整表达特征。新补充结构读取及coverage回归8项通过。
- **6k没有提前放行。** 后续逐中心反证在C4的两条原图楔键之间发现需要解释的差异，revision-3草稿已撤回“12/12通过”的提前表述；本次v5继续使用冻结revision-2的hold。明确少羰基的旧数据库结构仍拒绝。不能把尚在核验的候选计入已解决数量，也不由单个化合物的待审推出整个课题不可行。
- **最终参考已重绑。** `experiments/20261010-endpoint-feature-preflight-final/`绑定v5输入和complete的v3数据；实际T-only EngineData成功打开并构造14动作。数据身份为`ca580cf39748683d133dd70c53ca0de4c50293d2c53c8aa56f62518a70d203de`。预检仅剩response/pool/state匹配权重缺项；没有生成features/manifest或启动训练。缺权重属于模型准备，独立状态—功能证据缺口属于研究验证，两者分别报告。

最终来源核验、数据准备、对齐代码与报告都维持原[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)的状态条件化主线和完整候选定义。工程里程碑完成不替代主问题的数据可行性验收，目标仍未标记complete。

## 10. 第四轮：实际状态功能挑战与表达坐标补齐

### 10.1 已落地的资料

- Nav-Gal公开发表版Fig4b的60个群体均值已提取为`digitized-v1`，并登记为`challenge-v1`的6条相关时程；24h子集为4个活性药物×状态均值及2个vehicle均值。来源PDF、原图、XML、像素规则与输出哈希绑定，未恢复或伪造逐重复数值。原文10/11μM冲突、4000/6000铺板差异、顺铂洗脱未说明等均保留。没有把连续时程当作共同终点下的撤药时长实验。两个脚本为`extract_nav_gal_published_means.py`和`build_nav_gal_challenge.py`。
- Navitoclax/Nav-Gal的`chemical_review`提供两种独立名义结构和`(2,2048)`手性指纹。主代理亲读合成原文与Fig1b；糖立体重建依据及S1原谱尚未独立核验的范围明确记录。没有改动冻结的challenge-v1、digitized-v1、condition-inputs-v5或既有结构表。
- Sarin/GSE108214的29,729探针×22样本已完成身份映射，654,038原值逐项回读一致；22,901探针对应共同轴16,852基因，其中12,070基因拥有一个已解析探针，形成22×32,461矩阵。重复探针、冲突和缺失保留；原微阵列尺度未改成单细胞计数。代码为`align_sarin_state_expression.py`，结果在`source_reviews/state_gene_round4/`。该RNA仍为跨研究、平行24h参考，不是同样本t0与未来Annexin配对。

### 10.2 验证与科学边界

[独立技术复审](NAV_GAL_CHALLENGE_REVIEW_2026-10-10.md)未发现Nav-Gal挑战集需修订的实质数据错误：60点像素复算、12项哈希及15个文档链接通过；12柱独立目视与全部60点算法复算分开报告。衰老组两药读图区间重叠，不强行确定胜者。化学审查记录5个输入、27个输出哈希和两行实际指纹回读通过。Sarin记录4项定向测试、16输入及9输出哈希、两份NPZ回读通过；不与此前重叠测试机械相加。

[完整证据验收说明](DATA_FEASIBILITY_EVIDENCE_REQUIREMENTS_2026-10-10.md)明确：公开均值、群体类别与带不确定性的输入可以支持相应检验，不强制所有来源都有逐细胞t0或原始FCS；同时保留状态增量、新条件预测、真实功能一致性和完整候选选择的验证要求。有限数据通过来源检查不等于完整主目标通过。

本轮状态：`training-identity-v3`保持complete；核心coverage-v4保持38个Annexin汇总、14个24h化学输入、0个已生成完整表达特征；最终预检仍为pending，仅列response/pool/state匹配权重缺失。工程缺权重与跨来源生物学验证、共同终点时长数据缺口分别登记。目标仍active，未启动正式模型训练，未提交、推送或写回Zotero/长期记忆。

## 11. 第五轮：共同终点资料落地与评价入口实跑

- **Hastings暴露窗已核实并取数。** 从原文处理图登记cisplatin 5 µg/mL、2h去药＋70h恢复与持续72h两方案；`extract_hastings_duration_challenge.py`从原图取出72h相对cleaved caspase-3均值1.2693/2.9116，另存像素包络。`duration_round5/cleaved_caspase_challenge_v1/`的5输入与3输出哈希和数值复算经独立审查通过。这里只是蛋白相对信号，未混入Annexin头或主训练表。
- **Nav-Gal读取/评价入口完成。** `evaluate_nav_gal_challenge.py`实际连接6条件和两种化学结构；真实readiness保存在`experiments/20261010-nav-gal-input-readiness/readiness.json`，状态`input-ready/prediction-missing`。主代理复跑`python -m pytest tests/test_nav_gal_challenge_evaluation.py -q`，19项通过（1.54s）。两个剂量解释分开评价，不扩增Y、猜测RNA或强制衰老组胜者。没有真实模型预测或训练。
- **Tahoe跨板覆盖已实际复算。** 1305合格处理、14板28vehicle，完整化学身份×剂量×24h共1128组，154组跨板重现、含325条件。源条件/控制链接/manifest哈希与两份输出表绑定，独立审查逐组核对通过。只证明候选比较结构，不证明板位是生物状态或参考表达已经改善预测。
- **新来源资格审查已完成。** 2个新重点候选＋1个旧来源复核，保留16个GSM和12个ENA A549 run元数据；旧Nature源表18个Annexin/confluence数值按单元格提取。关键状态/对象/单位缺口保留，无新增合格RNA—功能联合配对。Lee2012主文的A549双阳性流式图与GSE30516的25个非A549 RNA样本分别登记。公开来源的获取失败不是数据不存在的证明。
- **全文和资源匹配。** 搜索记录包含web、SciSpace和Scite失败回退，核心判断回到原文/官方数据；Zotero全库60顶层条目对8个DOI及完整标题均未命中，未写回。PerturbCellRL v2仍按用户顺序留待数据研究之后。

新增来源与原数据版本全部分开；对应[状态说明第10节](DATA_ALIGNMENT_STATUS_2026-10-10.md)和[独立覆盖复审](FULL_DATA_FEASIBILITY_COVERAGE_ROUND5_2026-10-10.md)。本轮有真实新证据、数据与代码进展，完整E09仍未通过，目标保持active。未更改研究架构、启动正式训练或提交推送。

主代理最后复核结果见`experiments/20261010-nav-gal-input-readiness/round5_validation.json`：新时长包5输入/3输出、新来源包4输入/65输出哈希通过；两个均值复算、Nature旧源表18单元格逐值重读、111个本地链接及新增脚本语法检查通过。一次校准对象比较因Python tuple与JSON list的表示差异失败，改用JSON标准化比较后通过，源数据及冻结输出未改动。

接续的具体入口：先依据154组Tahoe跨板实际ID核对参考表达/均值/plate-only的评测条件，再检查既有`sciPlex3/GSM4150378/extracted/v1/`的24/72h表达进入时间模型的真实接口。其README记录244,281个A549细胞已提取，尚未进入当前Tahoe专用v3训练资产；继续时应重读当前manifest、协议与现行消费者，不从旧README直接宣布训练就绪。功能侧继续使用各自真实终点，补齐完整状态—候选—功能验证的共同覆盖。上述工作不启动正式训练，PerturbCellRL v2仍排在数据可行性研究之后。

## 12. 第六轮：进度答复时已完成的核验与接入中的工作

- Tahoe 原参考表达已实际读取：30,065 个 fit vehicle 细胞、28 源样本、14 板，完整 32,461 基因轴。主代理另行从样本累计量逐项重算 14 板细胞加权均值及检出率，完全一致。见 `source_reviews/tahoe_reference_expression_round6/`；数据职责仍为可观测参考，不等于已确认 14 个生物状态。
- sci-Plex 作者保留名单与 A549 hash 筛选逐 ID 一致，24/72 h 分别为 143,015/67,443；20 板均有合法同时间/重复/板的 vehicle。保留 1,920 个计划孔及唯一零合格细胞孔。时间和对照接口见 `source_reviews/sciplex_round6/protocol/`。
- `build_sciplex_structure_sidecar_v2.py` 生成 `chemical_inputs_v2`：188 源目录记录、187 主指纹、S4246 相对立体实体及两个无权重镜像解释。原厂 SDF 的文字语义与原串一致；拒绝用 CAS 查询覆盖为单一对映体。8 项定向测试通过（0.72s），18 个输入/输出哈希及行号复核通过；v1 原包和生成脚本保留。新 manifest SHA256 为 `c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05`。
- [评价契约](SCIPLEX_EVALUATION_CONTRACT_ROUND6_2026-10-10.md)已完成并独立核验：源内条件补全、严格 72 h 留出、允许 72 h 对照的版本、新联合药物留出各自定义。源内时间表达监督不代表共同终点的暴露时长选择已获验证。
- 真实多源路由与新联合划分已接入现有读取/预处理/AE/带宽/模型输入链，`training-joint-sciplex-v1`已生成为`metadata_ready`。实际读取通过Tahoe24h、sci-Plex24h与72h，各核对6条32461基因计数行并执行条件编码前向。sci-Plex纳入209504细胞、1871处理条件和40个控制孔，S4246的8条件954细胞及1个零QC孔的原记录保留。512个保守划分组无跨角色。独立工程复审和新包完整预处理尚待完成，不能把metadata_ready写成训练就绪。旧数据与权重继续保留，未启动正式训练或全量新预处理。

[状态说明第11节](DATA_ALIGNMENT_STATUS_2026-10-10.md)已回答用户最新问题：部分数据对齐有实际证据，完整状态条件化真实凋亡及联合候选选择的数据可行性仍未通过。本轮输出再次核对研究架构§2、§3、§4.2/4.3、§6，没有缩减原主目标后宣布完成。

## 13. 第七轮：新状态矩阵与完成的联合预处理

上一轮分类为实际进展：新联合元数据和真实输入链已完成。本轮先完成独立接入复审，未发现阻断错误；4项反例、15个资产哈希及真实world拒绝条件通过。结论绑定metadata_ready快照，不覆盖随后预处理状态。

- 实际运行 `python -u scripts/prepare_training.py --data configs/data_joint_sciplex_v1.json --device cuda:0`，exec session `28597`，Python PID `101976`，启动UTC `2026-10-10T10:15:53.779109+00:00`。原进程现已正常退出（exit 0），没有重启。扫描outer train共1,705,349行，保留1,453,616个fit细胞、1,566条件；fit/cache、归一化、完整图、带宽均完成。主代理执行`experiments/20261010-joint-sciplex-preprocessing/verify_complete.py`通过：19资产哈希与大小一致，三种来源/时长的T权限实际读取及独立归一化复算一致。新manifest为`513c9650acec6613d61c6f32be26d68ec477d7fb9c3c38f43742829443cbba05`；运行和终验分别见`run.json`、`complete_validation.json`。
- `align_gse279184_expression.py`完成六列100,020值重读、15,697唯一基因映射。新包 `source_reviews/state_function_round7/GSE279184_common_axis_v1/`保留原值、缺失mask和24项哈希；manifest为`ee4af0127093cc6792d7470da6454671fb9797ce693d156aaffe49b9ca59ee52`，5项测试通过。主代理另做全值、映射及基因顺序复核。
- 同来源新增`GSE279184_condition_inputs_v1`：论文两项货号的官方名义结构、两条2048位指纹与六个GSM条件连接完成，manifest为`85bccdd2fe57f90f521b3e8f7ebb0a7adcc1006a2f01f8117041442345feb795`。主代理复验23项输入/输出哈希，重新读取官方结构复算指纹，核对全部诱导/挑战关系；UT时长及挑战指纹仍为null。证据`experiments/20261010-data-alignment-round7/chemical_condition_validation.json`，新增凋亡标签为0。
- `state_function_round7/induced/`新增两篇来源的原图、条件表、GEO表及补充资料。Schepers四条印值保留不完整剂量/时间匹配边界；Raffi的RNA与WB剂量不同，没有伪配。`resistant/`新增两篇全文与官方元数据核验，两项具体错配/终点差异保留。最后引文cFLIP是已审旧来源，不重复计数。
- 最初一条委派检索因平台内容过滤中断，主代理已用公开文献与官方数据字段完成对应复核，没有因此把研究范围缩小成更易通过的任务。所有新增资料保持只读原件、独立版本和真实尺度，没有训练、联系作者、写回Zotero、提交或推送。

本轮进展和完整目标边界见[状态说明第12节](DATA_ALIGNMENT_STATUS_2026-10-10.md)。新原始证据与执行产物属于具体进展；尚未得到新增完整状态—候选—定量凋亡—参考表达块，目标继续active。

补充验收：joint终点特征生成接口已实际完成新reference/context map绑定，当前有效目录为`experiments/20261010-endpoint-feature-preflight-joint/final-v2/`。仅T读取器、2,431个fit参考细胞/75个CSR、82条参考来源记录、14候选/13化学身份及新protocol=1通过。首次配置继承旧protocol容量的问题已在新推理快照内修正；旧配置及失败证据保留。预检仍缺匹配response/pool/state权重，正式特征为0；不混淆科学监督缺口与权重缺项。

按用户顺序，数据轮验收后已亲读Zotero中的PerturbCellRL v2，版本与主文/附录关键内容经定向独立复核。结论和准确页码见[状态说明第13节](DATA_ALIGNMENT_STATUS_2026-10-10.md)：群体表达奖励不要求真实细胞前后一一配对，但仍需要同条件目标RNA；没有补出凋亡终点，也没有证明药物选择策略或单细胞命运。未改研究架构或训练方法。

用户随后授权联合Codex Cloud及Dot推进。新的实际分工、派发文本和状态单列于[Cloud/Dot协作记录](CLOUD_DOT_COLLABORATION_2026-10-10.md)，云端环境准备和任务派发不计为数据可行性通过。

## 14. 第八轮：协作结论归档及实际新来源验收

Cloud两次来源交付和Dot独立审查均已返回，正文归档于`experiments/20261010-cloud-dot-coordination/`。主代理进一步取得官方源文件、实际提取数值并复审关键判断。合并说明：[DATA_ALIGNMENT_REVIEW_CLOUD_DOT_2026-10-10.md](DATA_ALIGNMENT_REVIEW_CLOUD_DOT_2026-10-10.md)。

新增`source_reviews/state_function_round8/`三个独立来源目录：Kotani的6条件18个均值/24个图印值；Cutty的48印值/12分层、2列错误隔离以及8个H1944表达列绑定；Pogacar的34个官方GSM与3个Series资格表。它们分别是ND条件Annexin、成像死亡、状态RNA/克隆实验，未混成统一凋亡标签。

已执行bundled Python的`cutty/build_source_audit.py`与`experiments/20261010-data-alignment-round8/verify_sources.py`，均exit 0。验证57项文件哈希、Kotani18均值像素复算及24印值、Cutty两异常列整列隔离、Pogacar样本背景和技术重复。收据`experiments/20261010-data-alignment-round8/validation.json`明确`full_scientific_feasibility_passed=false`、`formal_training_started=false`。工程和公开证据取得属于实质进展，完整初态—候选—凋亡选择仍待补证，不把目标标记完成。

## 15. 第九轮：真实剂量数据追加与独立挑战执行

实际新增脚本 `extend_endpoint_alignment_round9.py`、`build_kotani_challenge.py`、`evaluate_kotani_challenge.py` 及对应定向测试。SJB的4个发表均值/3个给药条件已进入新r5/v6；当前覆盖实际为308条件、42个Annexin汇总、17个24h输入/14结构/4来源。旧r4/v5和旧审查85文件不变，新增均值未扩成n9原始重复。15项接入/化学回归通过，独立复审又做完整临时构建和正式表核验。

Kotani challenge-v1及真实无预测readiness已运行，4活性条件+2未知时钟参考、18均值+24图印值分开；26项定向测试和独立复审通过。分母限定ND门，未知时钟参考不评分，6/12h不作共同T*选择。

主代理实际运行 `audit_endpoint_coverage.py` 生成coverage-v5、`generate_endpoint_features.py --propose-reference` 生成joint final-v3，并执行 `experiments/20261010-data-alignment-round9/verify_round9.py`（exit0）。T-only EngineData实际读取与17候选输入构造通过；71条条件消费者绑定、82条参考来源记录复核，参考为2431个fit细胞，12条映射保持未经生物验证。预检命令因缺response/pool/state权重按pending保存并返回非零，没有生成占位特征。

Cloud的第九轮限定查询和BRG1定向审查已完成，Dot既有独立审查归档保留；主代理确认4个A549 RNA run存在，但不能换成siHIF1A/PFI3功能状态。HIF1A补充获取超时/404已记录，不推断数据不存在。未训练、推理、联系作者、写回Zotero、提交或推送。

详细科学用途和剩余缺口见[第九轮报告](DATA_ALIGNMENT_ROUND9_2026-10-10.md)，真实验收见[validation.json](../experiments/20261010-data-alignment-round9/validation.json)。目标仍active，完整数据可行性尚未通过。

## 16. 第十轮：新源附件、逐值复核与完整目标独立审查

新文件位于`source_reviews/state_function_round10/`。HIF1A改用官方PMC Cloud Service取得原图、PDF及Data S2；80个归一化表格数值/8条件按坐标提取，药物和时钟冲突保留。CellCage取得原文、补充与官网PDF，确认CCE终末RNA连接但没有下载到矩阵/行映射。BRG1限定文字审查和非A549伴随论文核验分别保存，不以访问失败作不存在判断。

主代理运行`experiments/20261010-data-alignment-round10/verify_round10.py`：复核HIF24个源包文件及6个旧来源引用、80源值/8均值和6均值差、BRG1的10个文件、CellCage的3输入/28审查文件、伴随论文15个文件；r5/v6/coverage-v5内容绑定一致。独立复审另核HIF原件和全目标A/B/C/D缺口，文档保留于同一新实验目录。

科学判定明确承认Nav-Gal已有两状态×两药有限选择块，Hastings已有共同终点时长比较；主要剩余是表达超出类别/均值的功能增量及相容的状态—真实时长—功能连接。Cloud/Dot已有意见被使用，本轮没有新派单。没有训练、模型推理、Zotero写入、记忆写入、提交或推送。详见[第十轮报告](DATA_ALIGNMENT_ROUND10_2026-10-10.md)及[验证结果](../experiments/20261010-data-alignment-round10/validation.json)。

## 17. 第十一轮：真实暴露方案包、化学附表与作者代码复读

新增`source_reviews/state_function_round11/`：Vielhauer的37文件原图/数值包、两药名义身份20文件及2上游引用、TIDYI三个notebook和逐值数表、克隆/HSP90资格审查。新`build_vielhauer_duration_challenge.py`实际构建并读回20活性条件、2参考、10比较；定向18项测试通过。`vielhauer-duration-challenge-v1`冻结保存，不覆盖核心r5/v6。

主代理`experiments/20261010-data-alignment-round11/verify_round11.py`实际通过：全部登记源文件hash、66分量＋22总Annexin几何复算、60CFSE行与12参数原输出核对、PubChem结构一致性和20×2048化学行读取；核心r5/v6/coverage-v5及架构文件绑定未变。真实新RNA配对0、生成表达特征0、没有模型预测。

独立科学审查、数值与loader审查、化学附表审查及TIDYI再审保存于同一实验目录。化学指纹可读不表示G可训练；6h给药＋18h恢复不能省略洗脱协议。TIDYI条件率为每小时模型率、按celltype连接，未补成Annexin标签。Cloud/Dot的5份既有交付hash复核，本轮无新派单。完整目标保留active，详见[第十一轮报告](DATA_ALIGNMENT_ROUND11_2026-10-10.md)。

## 18. 第十二轮：状态表达索引与验收契约校正

独立复审发现旧A最低证据将功能Y设为普遍门槛，比架构更强。已保存旧契约快照，修正现行A-expression/A-function及整体链协议条款，没有改变研究架构或放宽C/D真实功能要求。

实际新增state_build_inputs.py及34上下文、3,176处理、2,195共享候选成员索引。原role/split/精确条件和实际control可读数逐行核验，8输入/6输出hash通过；10条件只有1个同role非空control，未伪补第二孔。fit为1,498处理＋68对照。索引不读取整表达或生成预测。

新来源RNF20矩阵3个A549列56,445值读回，源包59文件核验；CSE/GSE200541取得8元数据但不与Nav-Gal错配；CellCage核对已有10组入口后停止重复请求。谱系代理被服务内容限制中断，其未完成包不计入已验收数据。

主代理[verify_round12.py](../experiments/20261010-data-alignment-round12/verify_round12.py)实际exit0；当前状态为组合设计具备依据、完整协议消费者与整链验收待接入。未训练、模型推理、重新划分、提交/推送或外部消息。详见[第十二轮报告](DATA_ALIGNMENT_ROUND12_2026-10-10.md)。

## 19. 第十三轮：完整协议输入与方法接入（2026-10-11）

完成28候选/24活性终点/14标准P请求/110绑定的实际输入消费，及不反序列化终点的输入桥；56项数据回归和10项桥接回归通过。四条方法审查已保存。旧消费者、冻结包、核心r5/v6和研究架构未覆盖。实际P特征仍为0，完整协议G/Q和整链独立评价尚未接入。详见[第十三轮报告](DATA_ALIGNMENT_ROUND13_2026-10-11.md)、[方法接入方案](DATA_ALIGNMENT_METHOD_SOLUTION_2026-10-11.md)。
