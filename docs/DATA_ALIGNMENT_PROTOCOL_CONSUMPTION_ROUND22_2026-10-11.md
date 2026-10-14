# 第22轮：完整协议 P 消费路径与真实输入准备

日期：2026-10-11。范围包括本项目已登记32个完整请求、Viel20的单细胞代理表达出口，以及供该新参数版本使用的独立 joint AE／完整协议P学习入口。新增实现和真实数据准备已验收；匹配权重、真实模型推理、功能效果及整个课题可行性仍未验收。正式训练停止安排保持。

## 本轮结果

- 新增 [full_protocol_response.py](../src/lung/engine/full_protocol_response.py)，保留旧 native/standard producer 和 full_protocol 模型源码的字节。新增入口组合真实参考表达准备、来源域检查、完整物理协议输入、`E.encode_h → FullProtocolLatentAdapter.sample_latent → member.decode → physical_projection`；真实模型路径未执行。
- 新增 [准备脚本](../scripts/prepare_full_protocol_response.py)。无 `--output` 时只报告静态依赖，有 `--output` 时实际读取并另存输入；两种模式均不加载权重、构造模型或调用生产器。
- [实际数据回执v2](../experiments/20261011-data-alignment-round22/protocol_consumption/actual_data_run_v2.json)：全部2,431个fit参考细胞×32,461基因，75个源CSR分片读取前后分别核对SHA256，3,038,304个选中非零计数。真实输入、20候选数值和物理张量已落盘。运行时将 `torch.load`、`nn.Module.__init__` 和 `qualified_model` 设为拒绝函数，实际PT读取、模型构造、真实forward均0。
- [直接公式读回及v2字节核对](../experiments/20261011-data-alignment-round22/protocol_consumption/direct_oracle_readback_v2.json)：5个输出文件SHA256重核；首参考细胞从原CSR用完整32,461列直接重算 `log1p(10000*counts/common_gene_library)` 及冻结fit均值/scale，与落盘float32逐值完全相同；20候选全部数值字段与10个物理batch字段逐项相同。v2因OrderedDict权重检查修正后producer代码hash改变而另立，5个数据文件与原直接oracle版本逐字节完全相同，旧证据保留。
- producer及学习入口共40项定向测试通过（26项producer＋14项学习），最终浮点摘要及完成恢复增量3 passed、37 deselected。包含真实登记输入/拒绝路径与独立4基因合成模型检查；合成检查不能替代真实预测验收。独立复审指出的调用方自报audited资格、True等于整数1及已早停恢复多更新的问题已修正。当前producer SHA256为 `b8cdefee4a2124031d1a19e4ed2165bdc1b02dcd8b5946962dc058ad1160f2a8`；学习源码SHA256为 `6067f32aac22673a5169f6696fea5204ee516923332eda720869f28d1d17f781`。最终CI产物位于项目内新basetemp，见[完整40项回执v3](../experiments/20261011-data-alignment-round22/protocol_consumption/learning_synthetic_CI_receipt_v3.json)及[最终增量回执](../experiments/20261011-data-alignment-round22/protocol_consumption/final_dtype_and_completed_delta_receipt.json)。

参考float32内容SHA256为 `e916d732ce3562b71f68be965579b642a9680cef0593ca483e5fc04a24aff219`。实际准备manifest见 [real_inputs_v2/manifest.json](../experiments/20261011-data-alignment-round22/protocol_consumption/real_inputs_v2/manifest.json)；该回执绑定当时producer版本。最终资格与消费修复只改producer SHA，所有数据/域/source PINS不变。[readiness v5](../experiments/20261011-data-alignment-round22/protocol_consumption/current_readiness_v5.json)实际重核5个数据文件hash；当前 [v6](../experiments/20261011-data-alignment-round22/protocol_consumption/current_readiness_v6.json)再绑定最终浮点检查源码并复用v5与原真实读取证据，不重复75-shard数据任务。候选状态始终32行，Viel20数值可用、Nav8缺物理输入、Bothe4缺bulk参考/上下文/出口；未把pending候选删掉或改名。

## 逐字段消费与证据边界

| 字段 | 实际来源与准备 | 进入模型的位置 | 已验证与剩余条件 |
|---|---|---|---|
| 参考表达 | 固定Tahoe fit对照2431行；原CSR按 `genes.parquet.tahoe_column` 投影到joint Ensembl轴，使用joint已拟合均值/scale | `FullProtocolResponseAdapter.forward` 把标准化表达传 `SharedAutoencoder.encode_h`，h进入每次latent velocity | 全部真实数据已读；实际E/h和真实forward未运行。同期对照群体不是Viel实测给药前初态 |
| 原来源状态 | `source_state_id=a549_parental_standard_culture` 及source角色/观察ID按冻结请求保留 | 作为来源及结果metadata；不伪装成已训练的状态embedding | 当前未数值编码；代理E来自Tahoe表达。不能据此证明状态匹配或状态增量价值 |
| 药物结构 | 每候选首drug interval的已绑定2048位名义FP；源指纹身份状态保留 | base `ConditionEncoder.drug` 和完整协议每event `FullProtocolConditioner.drug` | 20个真实FP已读；base与首event逐位核对。名义published graph不证明历史药材批次认证 |
| 剂量 | base为源nM/1000后的uM；physical保留nM原值和unit ID | base DoseEncoder及physical values/unit embedding | 真实输入及单位换算核对；两路不一致会拒绝 |
| 暴露时长 | active exposure6或24小时 | base DurationEncoder和完整事件的start/end/exposure fields | 真实读取；base duration必须等于首interval exposure，不能把pulse duration改成观察24h |
| 恢复与洗脱 | pulse药物0–6、wash/media在6、recovery6–24；连续仅drug0–24 | 完整有序event序列经HistoryEncoder/GRU进入条件融合 | 10pulse/10continuous身份保持；合成检查实测两种条件送入velocity的向量不同。transition时钟不是实测洗涤耗时0 |
| 观察终点 | 每候选T24；base elapsed0/horizon24 | base elapsed/horizon与duration/horizon；full endpoint log1p及endpoint_known通道 | 两路时钟核对；生成τ仍 `[0,1]`，不当作治疗小时 |
| 处理历史 | physical event类型、顺序、FP、单位、values和lengths | full history GRU；旧base history缺省零向量，实际物理历史由full分支输入 | 有序字段已准备。full历史没有真实RNA训练资格，结构通过不能证明能外推生物响应 |
| unknown masks | wash_count/retention等源null保持known=false；FP/endpoint各有独立mask | full conditioner在数值占位之外另外输入known/FP_known/endpoint_known；有效前缀由lengths约束 | 未知不变known零。合成检查未知NaN占位与未知零占位同输出，已知零与未知零输出不同 |
| 来源/背景/协议上下文 | named proxy为Tahoe-100M/CVCL_0023/tahoe_continuous_exposure_24h_v1，冻结词表索引0/0/1 | base三个embedding；original Viel来源、状态和检测协议仍单独保存 | 不把Viel字符串强改Tahoe，也不声称存在Viel训练embedding；cross_source_transfer仍unvalidated |
| 来源解码D | member decoder输出joint单细胞标准化域；现有physical_projection恢复共同基因library10000 log1p，再重标准化 | adapter在latent Heun积分完成后只调用一次member.decode | 仅合成结构运行通过。出口不是Viel实测表达、bulk、TPM或源绝对counts；缺匹配D参数不能生产 |

当前未进入网络的培养细节仍包括未报告的stock/material、medium完整配方/体积、未知retention和精确操作耗时。metadata保留缺口；不把本schema能解析的字段表述为全部培养条件已被网络监督。

源码入口见 [producer](../src/lung/engine/full_protocol_response.py)、[独立学习入口](../src/lung/engine/full_protocol_response_learning.py)。原始定义为 [physical模型及mask融合](../src/lung/models/full_protocol.py:131)、[base条件编码](../src/lung/models/encoders.py:175)、[共享E](../src/lung/models/latent_staged.py:41)、[latent采样](../src/lung/models/latent_staged.py:137)、[来源出口的现有单细胞投影](../src/lung/models/objectives.py:170)。这些是本地源码行为，不是论文实测效果。

## 候选合法性与资格拒绝

1. 入口只接受固定版本32个candidate ID，完整请求通过第21轮固定hash及源重构认证；不同drug-dose-duration下的pulse/continuous协议仍保留不同ID。没有调用者自填FP、source字符串或任意候选笛卡尔积。
2. Nav8没有完整physical事件，Bothe4没有该版本的bulk参考/上下文/出口，真实production在读权重前拒绝。Nav的两个浓度解释身份及Bothe四历史保留；参考单细胞域不能冒充bulk域。
3. Viel20可以准备数据及显式代理假设，当前真实production因缺metadata/匹配full参数而拒绝。缺matching RNA是迁移与直接表达验证的科学缺口，不设“每个功能Y必须有独有RNA”的通用执行条件。
4. 权重入口要求representation为latent_ae，response为新 `full_protocol_latent_cfm_v1`、非synthetic、实际epoch/optimizer_step正值且严格int（bool不能冒充1）、data/domain/axis/normalization/context与本bindings相同、response嵌入shared参数与AE完全一致，并以full parameter namespace严格载入。旧latent_cfm不能因更改source字符串获得资格。
5. 未来显式producer按固定参考行一次生成latent noise，再按候选重复相应physical batch分块采样并解码；结果保留原请求、模型bindings、参考行和预测身份。不读取source target RNA或功能Y，不创建新的fit/selection/final角色，也不生成G/Q或排名。

新增权重检查支持实际E的sparse COO graph：对coalesced values检查finite、对indices/values作exact比较。这里只运行合成张量helper，没有加载真实checkpoint。实参检查及static metadata本身不授真实推理或科学资格。

## 新权重的真实训练入口准备

新增 [full_protocol_response_learning.py](../src/lung/engine/full_protocol_response_learning.py) 与 [train_full_protocol_response.py](../scripts/train_full_protocol_response.py)。默认入口为真实data-only预检；`--fit-ae`、`--fit`分别是未来需要额外模型运行授权的显式分支。本轮未执行这些分支。

[真实读取v1](../experiments/20261011-data-alignment-round22/protocol_consumption/learning_preflight_v1/preflight.json)逐项检查1,498个fit与374个selection原生单细胞RNA请求，并实际构造1,872组partial物理张量。读取六个来源／角色／终点组各两参考及两目标计数向量，共24向量、每向量32,461个共同基因，按冻结joint均值/scale转换；保存实际cell ID、nnz和转换后float32内容hash。guard禁止真实PT加载及nn.Module构造，真实模型forward、优化更新均0。不能把24向量当作24生物重复。

| 来源 | fit请求 | selection请求 | 实际监督与mask |
|---|---:|---:|---|
| Tahoe 24h | 706 | 150 | FP、uM剂量、24h观测终点；真实RNA；暴露/洗脱/历史unknown |
| sciPlex 24h | 656 | 168 | FP、uM剂量、24h观测终点；真实RNA；暴露/洗脱/历史unknown |
| sciPlex 72h | 136 | 56 | FP、uM剂量、72h观测终点；另一破坏性采样群体；不是24h后恢复48h |

源字段证据为冻结 [native requests](../experiments/20261011-data-alignment-round17/native-response-units-v1/requests.json) 的 `actual_protocol`、[sciPlex时钟契约](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/time_control_contract.json) 和当前 [joint manifest](../08-datasets/LungCancer_A549_Integration/training-joint-sciplex-v1/manifest.json)。本轮没有重写这些原件。所有请求的exposure_duration_h为空且exposure_duration_unknown=true；withdrawal、prior_drug、washout亦unknown。作者报告的endpoint信息不据此补成已核实连续暴露。

新P的训练physical仅为一个已报告名义drug event：dose known、FP known、endpoint known；start/end/exposure、wash_count、retention、media_change、recovery全部mask=false，事件列表未证明完整。旧base duration通道保留native observation hours，并显式登记其语义。Viel生产请求的base duration为实际active exposure，历史亦包含pulse/wash/recovery；这种数值语义与mask/事件覆盖差异是协议OOD。训练、实际审计与预测输出均登记 `Viel_pulse_washout_recovery_protocol_OOD=true`、协议科学验证false；不会仅凭参数namespace把这些事件称为已监督。

独立B3复用原latent OT抽样、同角色参考well→cell分布、固定E/h缓存、detached目标latent及旧CFM方程；新增完整物理conditioner参与每个velocity条件。optional B4仍为可微latent Heun→单次member decode→旧biased全基因MMD，训练期间不以physical projection替换原B4 loss。selection仍no_grad自由生成后按原投影MMD＋mean MSE筛选；final RNA、功能Y、原状态C和奖励均关闭/未修改。新P明确采用fp32独立numeric版本；原AE按现有precision设置执行，没有改冻结旧B3/B4数值或loss。

14项学习合成测试覆盖B3新参数更新、AE及B3 decoder逐tensor不变、B4 decoder开启、更新中断和B3→B4边界的exact恢复、依赖/角色/协议身份漂移pre-PT拒绝、4基因B1格式导出逐tensor一致、安全serving snapshot、已完成P/AE恢复不再更新及best/last或selected漂移拒绝。最终测试checkpoint位于项目内合成fixture目录，不是真实模型资产。先前34/37项默认TEMP运行的路径偏差见[回执](../experiments/20261011-data-alignment-round22/protocol_consumption/synthetic_test_path_deviation.json)，旧TEMP未删改；不称此前所有临时产物在I。合成counter和完成标志不授实际matching权重或科学资格。

最终 [learning readiness v3](../experiments/20261011-data-alignment-round22/protocol_consumption/learning_current_readiness_v3.json)登记Torch/NumPy/SciPy、完成恢复及最终producer执行依赖，在再次实际准备1,872个mask张量后复用仍有效的v1六组RNA读回。源数据、协议identity、axis/norm和所有数据bindings均相同，仅producer code绑定更新；本次delta没有重复计数正文读取。matching joint AE仍缺失，fit_ready=false。

## matching joint AE及后续运行次序

已保存可审阅的 [独立matching_joint_AE.yaml](../experiments/20261011-data-alignment-round22/protocol_consumption/learning_preflight_v3/matching_joint_AE.yaml)与[实际执行顺序清单](../experiments/20261011-data-alignment-round22/protocol_consumption/future_matching_AE_P_model_plan.json)。这是当前config的副本，data固定 `configs/data_joint_sciplex_v1.json`、response词表source/background/protocol为2/1/2；原config、旧interrupted AE及全局modular latest没有修改。data identity固定为 `29092acd61b8d5d0f5d59642f43bf1df7c55e780e350d3e2f24fa4e55aa355d9`。

`--fit-ae`直接复用现有 `run_latent_ae` 与AEDataSupply：同双端全基因fit MSE、同condition group与参考抽样、同selection、同既定early-stop/epoch，无新增墙钟上限；输出在独立实验内，不进入旧run。该函数的AE shape原n_protocols=1；SharedAutoencoder没有context embedding。未来导出时严格核新AE来源run声明、实际data/shape/epoch/positive int counter及完成状态，再显式将新P的context capacity扩为2/1/2，逐tensor相等并保存原source shape及checkpoint SHA。这不训练或修改任何AE数值参数。合成格式导出验证该机制；真实导出尚未运行。

训练checkpoint保留optimizer/cursor/两类RNG/early-stop/decoder阶段；producer使用另存的tensor＋primitive serving snapshots，避免把训练RNG pickle当推理权重。新AE/fullP文件均绑定最终producer源码及完整bindings；外部独立audit ledger不属于源码、训练bindings或optimizer身份，登记不改变source hash或参数文件，因而没有注册循环。源码或Torch/NumPy/SciPy等执行依赖变化时resume在首次PT前拒绝，须另行核版本而不能偷换旧训练身份。

以下命令是后续授权方案，本轮没有执行：

```powershell
F:/anaconda/envs/study/python.exe scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/learning_preflight_v3/matching_joint_AE.yaml --fit-ae --output experiments/full-protocol-joint-AE-v1
F:/anaconda/envs/study/python.exe scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/learning_preflight_v3/matching_joint_AE.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/full-protocol-native-P-v1
```

接续顺序为：审查并授权新AE→原fit-only归一化cache/AE拟合→新AE完成与精确导出→fit/selection h cache→独立native full-P拟合与selection→下节真实两行执行审计→独立外部登记→Viel20固定参考预测→同候选/信息/目标下的G/Q独立验证。恢复各自新实验需加 `--resume`；不使用旧run ID。optional B4若启用，应另立经确认的schedule/version。

已完成P的resume验证最后快照、early-stop状态及selected serving snapshot的SHA/metadata/实际counter，直接返回既有结果，不再构造P、optimizer或重建未用h cache。新AE wrapper核本namespace完成receipt及best、last两个文件SHA，直接返回；只有last已完成而缺wrapper完成receipt时拒绝并标proof缺失，不再调用旧loop，也不假称complete。中断中的新实验仍按原精确恢复；旧frozen latent_pipeline没有修改。

资源另见主代理 [实际CSR header资源回执](../experiments/20261011-data-alignment-round22/root_model_resource_preflight.json)：原稀疏归一化cache保守上界约36.64GB（3,590源shard headers、1,769,627 fit/selection cache行），当前joint cache尚不存在；这里没有改成229GB dense缓存。另需h/cache索引、模型、恢复checkpoint及系统余量。全h若float32×256约1.81GB，为估算而非实际新AE/P峰值；本轮没有真实训练吞吐/显存实测，须授权后先做具体资源审计再开始大运行。

## 下游输出实际消费

新增producer `read_prediction(path)`只读验证当前内容digest、candidate及完整request、data/axis/norm/model bindings、固定外部registry与audit身份、AE/P文件SHA、固定fit参考行、seed/steps、outlet及全gene mean/variance。它不加载PT，也不拟合G。真实P尚不存在，因此当前可运行结果仍为缺权重/登记拒绝和纯合成容器验证；没有用mock登记为真实模型授资格。

moments要求浮点类型，bool数组不能充当表达均值或方差。此处校验的是完整性及来源绑定；调用方可改写metadata或自算digest，这本身不能证明产生过相应真实执行，也不构成科学效果或无泄漏认证。真实producer/训练运行仍需实际回执、完整来源及独立复审。

该接口供后续预测特征→真实终点G→Q视图读取，保留协议OOD、代理来源迁移未验证、原状态tag未数值编码及科学收益未验证。G/Q必须另核终点定义、角色、共同候选域、同信息基线和独立测试；当前实现不将原始分选状态、Annexin、caspase、生长或死亡互相替换。

与毕业论文 [研究架构](G:/研究生科研文件/毕业论文/研究架构.md) 的§§4.2、4.3、4.5和§6对应：完整候选ID与治疗/观察/生成三时钟分开，E/P/source D和C/G/Q职责分开，fit/selection/final分别登记，多步/真实序贯没有因该入口结构通过获得资格，原目标没有移至展望。

## 实际模型审计与独立登记的顺序

生产入口只读固定路径 `experiments/full-protocol-response-registered-assets-v1/registry.json`。当前该文件不存在，生产资格pending。调用方的 `qualification='safe_tensor_execution_audited'`、`audit_passed=true` 或任意receipt路径不会建立登记。producer及准备脚本均不自动创建ledger，不写资格。

将来取得真实模型运行授权后按以下次序执行：

1. 冻结本producer源码，保存当前完整bindings；以相同data/domain/axis/normalization/context与新full参数身份拟合或登记实际匹配的AE/full-response权重。这里不更改旧checkpoint，旧训练参数不会被自动升级。
2. 将实际模型文件路径、SHA256、阶段、非synthetic、正optimizer_step、response的representation hash及真实training_protocol_coverage写入新的metadata JSON；checkpoint实际metadata必须含同一full_response_bindings和coverage。该metadata此时只是待查声明。当前独立入口会生成这一待查声明，但不生成registered_audit_id。
3. 显式调用 `audit_matching_model(candidate_id, metadata, new_output, seed=..., steps=..., device=...)`。该函数安全载入两个匹配PT，读取完整fit参考，并对前两个固定fit行运行一次E/fullP/D，将 `actual_forward_log1p.npy` 与 `actual_execution_audit.json` 保存到新实验目录。记录这两个行的实际ID；它只授执行证据，不自行登记，不声称完成所有P预测。当前未调用。
4. 独立复审者核对实际checkpoint/输入/输出、真实执行回执和输出内容后，发布固定外部ledger条目。ledger schema为 `full-protocol-matching-model-execution-audit-v1`，`producer_sha256`为冻结源hash，`audits`用audit ID映射到实际receipt的项目内路径及SHA256。这是独立本地资产登记动作，不能把caller自写同名JSON当作已完成独立审查；本轮不创建此文件。
5. production metadata添加登记audit ID，再调用 `produce`。producer查固定ledger、重核receipt/模型/bindings及实际forward NPY的hash、shape、finite、正库质量与观测域，再安全读取匹配PT；生成后重核ledger/receipt/模型身份。实际审计不授模型收益或真实凋亡效果资格。

登记示意（实际路径/hash须来自已运行并独立复核的文件，不能复制占位值登记）：

```json
{"schema":"full-protocol-matching-model-execution-audit-v1","producer_sha256":"<frozen producer SHA256>","audits":{"<audit ID>":{"path":"experiments/<actual audit version>/actual_execution_audit.json","sha256":"<verified audit receipt SHA256>"}}}
```

没有注册循环：ledger及audit SHA不进入producer源码或训练bindings；新增/更新外部ledger不改变producer SHA、checkpoint metadata或参数字节。caller在production metadata添加audit ID也不改checkpoint。若真正修改producer源码或数据/model依赖，旧审计会因bindings不匹配而拒绝，需要另立审计及registry版本，不能把旧执行证据自动移植。

## 与毕业论文架构的对应

按[研究架构§4.2–4.3](G:/研究生科研文件/毕业论文/研究架构.md:77)核对：候选ID代表完整处理记录，Δt/elapsed/T*/τ分开，E—完整协议P—来源D职责显式连接；表达、原始状态C与功能G没有合并。当前出口只明确到单细胞代理观测域。Martin的source E/D及真实protocol/reference使用另一个来源入口；本模块不直接接受log2(TPM+.01)、estimated counts或Bothe bulk counts。整个研究目标和多步/序贯范围保持。

本轮没有更改毕业论文架构、奖励、冻结划分、角色、旧实验和旧producer。真实P与源特异C/G/Q同链、状态增量、未见条件可靠性、功能一致性和联合选择收益仍待模型与独立实验，不能登记整项目完成。

## 接续与可审阅运行顺序

1. 复用本轮已保存的真实输入和读回，不因接口说明重复计算已通过的75-shard读取。只在源hash/版本改变时另立准备版本。
2. 核对 [production_requirements_v6.json](../experiments/20261011-data-alignment-round22/protocol_consumption/production_requirements_v6.json)：目前matching AE/response与audit ID仍缺失；仅存在旧AE/CFM文件不能授fullP资格。独立真实RNA预检、partial覆盖与新AE/P命令已经准备；由已有训练授权边界决定何时进入真实模型分支。
3. 匹配参数取得且真实模型运行被授权后，显式调用 `lung.engine.full_protocol_response.produce(candidate_id, metadata, output, seed=..., steps=..., microbatch=..., device=...)`；`metadata`须是已核对actual bindings的登记JSON。该调用含真实PT载入/forward，当前未授权执行。先按一个Viel24h及一个Viel6+18h候选验证完整reference/context/出口身份、finite和可恢复输出，再扩展全部20；这只验证proxy预测工程。
4. 按既有fold/branch连接真正的G fit/selection/final与独立终点、基线和主问题整体验证。每个fold模型的预测不能拼成一个假称同一冻结模型的20候选全局独立排名。多步滚动和真实序贯分别保留数据、假设与独立验收。

当前可运行的数据预检与合成验证命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' scripts/prepare_full_protocol_response.py
& 'F:/anaconda/envs/study/python.exe' scripts/train_full_protocol_response.py
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_full_protocol_response.py tests/test_full_protocol_response_learning.py --basetemp experiments/20261011-data-alignment-round22/protocol_consumption/synthetic_CI_temp_v5 -q
```

数据准备使用 `--output <新的空目录>`；已有真实输入与预检目录拒绝覆盖。上例CI使用尚不存在的新basetemp，复跑改用另一新目录，合成checkpoint不会落到项目外。当前准备、拒绝路径与合成学习/恢复已实际运行，未来真实AE/P/G/Q训练、trained producer和科学效果未验证。


## 本轮后续独立参考变体的当前版本

状态增量的 observed/zero_h/mean_h 三条独立训练、selection、来源声明、推演及 heldout API 已接通，见 [当前变体报告](DATA_ALIGNMENT_REFERENCE_VARIANTS_ROUND22_2026-10-11.md)。其 phase population 合同分别保留 CFM post-OT有限selected rows、B4/selection原uncoupled有限draw、production/C全部2431fit proxy、native final全允许draw，算子相同但 n/权重/有限均值不同；原OT、RNG、轴、归一化、角色、奖励及 Bothe model PIN未变。最新producer `af7aa029b98ca1e9e0b42d9001d6b49160884db04441727237d7c6eda50f13bf`、learner `103e933fc989e3b379436a5f164fff41705a8a712a64f9a0cdf65fd9434cafed`、CLI `91f53b8a8e90dd293c88d07db96b6451dbdb6b9aff9293b83fe971ba7a0e0bff`、C `6e613e1b50817f602df10b8eff4411c7664d8e72be8050cef3aea2efcbf756d5`。前文旧hash/CI是当时回执，保留，不据此覆盖最新版本。当前78工程项在I新a22t4通过，真实3metadata guard0ML；原真实RNA读回按gene/norm/data/ref身份复用，没有重新打开75shards。C从f39到6e仅AE-only配置与全pop h准备/predict_h精确P复演两个改动，原reader/final/label/gate语义不改。真实模型/状态收益/跨来源功能资格仍pending。
