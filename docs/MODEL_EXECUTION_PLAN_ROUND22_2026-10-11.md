# 第二十二轮：待授权的独立模型运行与正式评测顺序

当前正式训练停止安排继续有效；此文件是可审阅执行方案，不是开训记录，也不自动授予真实参数读取、模型forward、闭式拟合或optimizer更新权限。本轮已执行的数据工程、真实源值审计和合成CI分别见[综合验收](DATA_ALIGNMENT_ROUND22_2026-10-11.md)。原研究架构、冻结划分、奖励与旧中断run保持。

用户本轮再次明确选择“保持模型停止，继续数据与证据工作”，见[模型决定](../experiments/20261011-data-alignment-round22/root_model_execution_decision.json)。随后要求“先做到这里”，并对后台数据作业回复“把这个阶段做完吧”：当前只继续已启动的 Bothe 数据阶段，完成后暂停整个目标的推进，见[接续与停止条件](../experiments/20261011-data-alignment-round22/CONTINUATION_HANDOFF_2026-10-11.md)。下面的模型命令保留为待授权方案；当前决定不批准模型执行或新的训练/评测划分。

## 1. 运行范围与前置资产

工作目录统一为`I:/科研/肺癌细胞凋亡-强化学习`，解释器`F:/anaconda/envs/study/python.exe`。按当前版本执行，输入与源码变化时先刷新身份；已完成参数按其完成证明与SHA复用，中断只恢复同一新namespace，缺证明保留失败，不覆盖或重跑为“完成”。

- Bothe21:41已为12/12 raw官方整对闭合、5/12完整FASTQ、4/12 count；四闭合leaf真实source→joint整数/mask读回通过，完整12生产仍pending。下载以闭合回执正常结束，串行计数、全量source→joint及bulk观察包继续，模型许可不是其前提。不能把该快照当以后仍在运行的证明。
- 当前匹配的joint AE、fullP、原状态C和新G/Q真实参数均缺失；旧AE/CFM不升级成新来源/完整协议资格。真实模型与三角色模型评价为0。
- Martin双域source E/D的数据/工程已准备，独立final-only路径使用原cycle角色；它不建立共享latent、bulk P或Annexin迁移。
- 全P在native上只获得dose/观察endpoint监督，exposure/washout/recovery/维护历史unknown；Viel6+18与24h完整协议为明确OOD。未来真实运行只能检验其限定代理预测与终点关联，不能将“能执行”记为已学会撤药。
- C固定原三分选类和原weighted CE，三类概率进入G只作辅助预测特征。G目标仍为实验定义的Annexin fraction；Q仍采用同一G终止反馈，当前新terminal ridge为诊断模型，旧Actor/RL环境不因此取得资格。

## 2. 单机资源与调度

RTX3080 10GiB、约32GiB RAM。只读实际CSR索引给出稀疏归一化缓存数组上界36,640,016,915字节，另计索引、h、源CSR及参数；没有使用229GB完整稠密缓存。Bothe原中间产物容量接近初始I盘余量，关闭的新SAM已安排经字节身份核验的限定NTFS压缩，实际空间与压缩率随数据作业记录。

匹配AE/P及完整proxy推演安排为GPU串行；不在Bothe高内存阶段据旧空闲快照同时开多个模型。源码参数规模支持分阶段试验，但整链峰值、吞吐与多轮成本尚未实测，资源可行性保持有条件。首次实际运行记录RSS/显存/空间/吞吐，并保持原更新级可恢复断点；不额外设置墙钟暂停、实验时间预算或新的质量阈值。配置的最大epoch、早停和stage schedule保持。

当前源码与配置的[缓冲读回](../experiments/20261011-data-alignment-round22/root_resource_schedule_readback.json)确认AE两个group的float32输出约1.06GB；36.64GB缓存上界位于磁盘mmap，并非一次全部heap常驻。另有原CSR/临时量/权重/optimizer/GPU激活，以上均不是实测峰值。状态输入消融共享一个冻结AE，各自独立拟合三个P；不能用一个observed训练模型改推理输入替代三次训练对照。

## 3. Martin来源E/D：结构、拟合、冻结final

先在原fit RNA执行两个观测域的结构检查。随机未训练forward只核接线，不具有方法效果资格：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/verify_martin_source_model.py --mode real-untrained-forward --system cycle --fold-id cycle_final:HCC2935 --request-id observedRNA:H1975:long_washout:cycle_final:HCC2935 --domain log2TPM --width 32 --seed 23 --output experiments/20261011-data-alignment-round22/martin_source/real-untrained-log2TPM-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/verify_martin_source_model.py --mode real-untrained-forward --system cycle --fold-id cycle_final:HCC2935 --request-id observedRNA:H1975:long_washout:cycle_final:HCC2935 --domain estimated_counts --width 32 --seed 23 --output experiments/20261011-data-alignment-round22/martin_source/real-untrained-estimated-counts-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_martin_source.py --fit --config configs/modular_research.yaml --fold-id cycle_final:HCC2935 --width 32 --output experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1
```

source数据为30唯一fit GSM、15唯一selection GSM，各GSM双域等权后按GSM等权。原AdamW lr1e-4/weight_decay1e-4/clip1、batch128、max1000epoch、patience20/min_delta1e-4/min_updates1000。此fold每epoch一个update，因此min_updates1000具有实际约束，未擅自减轮。

完成本namespace并冻结best/last/current manifest/receipt后，final-only入口评估15唯一final GSM的同域重建误差：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/evaluate_martin_source_final.py evaluate-final --fold-id cycle_final:HCC2935 --trained-run experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1 --device cuda:0 --output experiments/20261011-data-alignment-round22/martin_source/source-final-HCC2935-v1
```

按域报告model-coordinate MSE、raw-source MSE/MAE与单位。final不拟合统计、优化参数或选模型；这种source自重建误差不能充当条件响应P、功能G或干预收益。来源均值/适用线性重建等基线和后续跨来源表示比较另按相同角色开展，未运行的比较不声称已通过。

## 4. 匹配joint AE与native fullP

新B1 AE复用原目标、抽样、fit-only预处理与selection，独立namespace，不恢复旧中断训练：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml --fit-ae --output experiments/full-protocol-joint-AE-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/full-protocol-native-P-v2-observed
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_zero_h.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/full-protocol-native-P-v2-zero_h
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_full_protocol_response.py --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_mean_h.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/full-protocol-native-P-v2-mean_h
```

新P只使用native fit1498/selection374请求；参数导出显式扩展P context容量，AE每个tensor相等且原shape/hash保留。B3复用原CFM耦合/目标；optional B4当前false，若需要另定独立schedule/version。各自中断加`--resume`，完成时只返回已核selected参数而不追加训练。

三版本分别为observed逐参考h、zero_h无RNA状态输入、mean_h当前允许参考样本h均值；P-only配置不会改变共同AE或C身份。[真实三模板预检](../experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/variant_plan_and_guard_v2.json)核相同数据/轴/normalization、旧目标/早停/OT/角色与missing资产，无真实参数或RNA重读。zero_h保持同监督RNA与原OT先验，只去掉P预测输入中的RNA状态，不宣称训练完全不使用RNA。source_state_id没有数值编码；来源/背景/协议等W保持，不把zero_h说成已另行训练的state-ID基线。

mean_h的算子都是`mean(frozen E h)`，不是`E(mean expression)`。CFM使用原OT抽出的h[ii]有限集合；B4/selection为独立control抽样；生产/C为全2431代理reference；native final为本条件允许reference draw。其有限样本、n和权重不同，不能宣称是同一个固定数值总体均值。保留原耦合/RNG，分别保存阶段合同及抽样ID；有限抽样和评测population变化是应报告的边界。

native final入口的实际纯metadata读回为367药后条件/52split components、原200,682 E细胞ID索引。87Tahoe24h、216sciPlex24h、64sciPlex72h；所有chemical均未见fit/selection；相同来源dose/time边际已见，因此本划分不能另称未见剂量/时间外推，完整exposure/history仍未识别。以下三次冻结表达评价在同source、角色、参考/target ID抽样、seed、ODE设置和fit-only坐标开展，不选模型、不重拟合统计：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/evaluate_full_protocol_native.py --evaluate --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml --response-run experiments/full-protocol-native-P-v2-observed --output experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-observed-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/evaluate_full_protocol_native.py --evaluate --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_zero_h.yaml --response-run experiments/full-protocol-native-P-v2-zero_h --output experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-zero_h-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/evaluate_full_protocol_native.py --evaluate --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_mean_h.yaml --response-run experiments/full-protocol-native-P-v2-mean_h --output experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-mean_h-v1
```

指标为共同基因standardized域的mean MSE和MMD_v，另保存共同library10000-log1p gene摘要；不是原count单位误差。对照有未给药reference和fit-only同source/context等权条件mean。三个独立P还须在同E结果做绑定比较，不能用observed模型更换推理输入的敏感性替代训练对照。留剂量/时间/完整协议的数据与实验缺口另记，不从本chemical留出获得资格。当前真实参数、消融及final表现全未运行。

三份评价都闭合后，当前已实现的比较入口核同一AE、仅reference variant不同的P配置、同条件/细胞抽样ID/seed/noise、同fit-only baseline及完整数值工件，输出逐条件和原连接组件等权的配对差。组件不是生物重复；该比较不训练、不选参，也不根据final重选主模型。

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/evaluate_full_protocol_native.py --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml --compare experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-observed-v1 experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-zero_h-v1 experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/final-mean_h-v1 --output experiments/20261011-data-alignment-round22/full_protocol_native_evaluation/paired-three-variants-v1
```

## 5. 原状态C、独立分类评价与fullP三类sidecar

在同新AE原始best.pt上训练，保持TIDYI `1_to_8`的248fit/64selection/63calibration、原三类与weighted CE/温度校准：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --fit --ae-run experiments/full-protocol-joint-AE-v1 --output experiments/raw-sorting-C-joint-AE-v1
```

独立final分类入口在原保留角色上只加载冻结C及同AE；保存三分类NLL/Brier/F1/混淆与逐类recall，不重新训练或校准：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --final-preflight --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --device cuda:0 --output experiments/20261011-data-alignment-round22/raw_state_scoring/final-plan-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/run_raw_state_scoring.py --config configs/modular_research.yaml --evaluate-final --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --final-plan experiments/20261011-data-alignment-round22/raw_state_scoring/final-plan-v1/final_plan.json --device cuda:0 --output experiments/20261011-data-alignment-round22/raw_state_scoring/final-evaluation-v1
```

主分析368原标签不改，原58个gate疑点为预定义310子集敏感性；两分母不直接比较accuracy作为改善。作者原H5另找回plate1四gate字段及102负类导出疑点，未改fit/selection/calibration。原标签通道推导仍未核实，孔位/QC及预先确定的敏感性另报告，不利用final结果调参。

原状态head对新来源表达的迁移仍待检验。`StateClassifier.population_proxy`的旧诊断保留；新sidecar不调用合类proxy。未来每个真实P用`--score --prediction ... --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --output ... --microbatch ... --device ...`计算三类mean/variance。设备与分块读取原fullP batch plan；noise/seed/steps/参考/AE一致，复演表达的float64 gene mean/variance精确相同才发布sidecar。若不一致保留失败，不新增经验阈值或将C(expression_mean)替代逐细胞评分。

## 6. 实际执行审计、登记、20P生产

P真实拟合后先用`full_protocol_response.audit_matching_model`对两个固定fit参考行执行E/P/D，保存实际NPY、输入ID、模型SHA及回执。独立复审核真实内容后发布固定本地registry，再给production metadata添加对应audit ID。该登记只授工程执行身份，不授protocol OOD、跨来源或终点收益资格。

完整20候选由新串行CLI生产，matching metadata取真实完成run的declaration并加入已独立核验的audit ID：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/produce_full_protocol_predictions.py --execute --config experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml --metadata experiments/full-protocol-native-P-v2-observed/registered_model_declaration.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_predictions_v1
```

同20新joint view、同AE/P、参考、seed、steps、microbatch与设备；独占写入、partial保留、完成预测经source/字节核验复用。只有显式`--execute --resume`恢复该同namespace，绝不从数据准备自动推演。由相同batch plan为20候选逐个生成C sidecar，当前这些真实工件均不存在。

后续获授权且真实batch闭合后，以下循环从已核plan读取候选与设备/分块，调用原C逐细胞评分。运行前核全部新输出目录不存在；已经完成的sidecar先经原reader核验，不能无条件覆盖。遇到不匹配或partial，保留失败并诊断，不继续编造该候选特征。

```powershell
$taskBatchPath = 'experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_predictions_v1'
$taskBatchPlan = Get-Content -LiteralPath "$taskBatchPath/batch_plan.json" -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($taskCandidateId in $taskBatchPlan.candidate_ids) {
    & 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/run_raw_state_scoring.py --score --prediction "$taskBatchPath/$taskCandidateId/prediction.json" --state-declaration experiments/raw-sorting-C-joint-AE-v1/state_declaration.json --microbatch $taskBatchPlan.microbatch --device $taskBatchPlan.device --output "experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecars_v1/$taskCandidateId"
    if ($LASTEXITCODE -ne 0) { throw 'C sidecar failed; preserve artifacts before further execution' }
}
```

上面是observed主链的具体命名。zero_h、mean_h功能消融分别采用自身已审计的P声明、独立production目录和对应C sidecar；相同view20、参考2431、seed/steps/分块/设备和原G超参数保持。共同C只训练一次。三次产生的同一个公开Y不变成三份独立实验；该比较也不能补出Viel未测的初态RNA。

## 7. G与有限联合Q的受控拟合、冻结及终点评价

独立新view20条件：dose极端8fit、中位4selection、另8final。旧十fold、split/reward不动。用新CLI build-index生成20实际P索引；有C sidecar时另用build-state-index，将索引通过`--state-sidecar-index`传入fit和freeze。P-only与P+C在同一原目标及角色下各自拟合，明确为表达/原类特征消融，使用同一available-information权限工件。

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py build-index --prediction-directory experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_predictions_v1 --output experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py fit --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-only-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-only-v1 --role selection --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-only-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-only-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-only-metrics-v1.json
```

G与condition-only基线只用fit8真实均值，每Y一次、原lambda=1和fit-only统计。私有terminal Q使用同G的fit候选模拟终止反馈，不用真实Y直接拟合Q、不bootstrap、不加成本。selection只评价，当前固定lambda无选参。锁定方法与资产后才freeze final8并evaluate；全8域报告直接G、简单排序、私有Q、固定drug/dose/exposure限制策略与uniform诊断的MSE/MAE/排序/regret及读图区间敏感性。

P-only独立final的确切命令如下，先冻结全部候选分数和资产，再读取该角色Y，不因final结果更新模型或选择输入变体：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-only-v1 --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --role final_test --output experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-only-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-only-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-only-metrics-v1.json
```

P+C只增加原三类mean特征；相同角色、信息许可与目标保持。先闭合20个真正对应本P的sidecar，再建独立索引：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py build-state-index --state-directory experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecars_v1 --output experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecar_index_v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py fit --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --state-sidecar-index experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecar_index_v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-C-v1
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-C-v1 --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --state-sidecar-index experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecar_index_v1.json --role selection --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-C-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-C-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/selection-P-C-metrics-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py freeze --models experiments/20261011-data-alignment-round22/joint_selection_learning/models-P-C-v1 --prediction-index experiments/20261011-data-alignment-round22/joint_selection_learning/full_P_index.json --state-sidecar-index experiments/20261011-data-alignment-round22/joint_selection_learning/C_sidecar_index_v1.json --role final_test --output experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-C-v1.json
F:/anaconda/envs/study/python.exe -X utf8 scripts/train_joint_selection_models.py evaluate --receipt experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-C-v1.json --output experiments/20261011-data-alignment-round22/joint_selection_learning/final-P-C-metrics-v1.json
```

对zero_h、mean_h重复上述索引、G/Q拟合和两角色冻结命令时，所有预测、sidecar、model和receipt路径使用各自新namespace；绝不将observed的P/C索引传入其它变体。统一评价定义和可比候选域先于最终结局固定；如需要变更模型或角色，另立明确实验版本，保留本计划。

Viel仅一原状态背景/20发表均值、共享控制、读图界而非原始独立重复；源Y此前已公开审阅，因此是回顾性评价。P/C内部改善不是Annexin实测改善，终端Q不是DA-AC/RL收益，也不替代完整状态增量或未见药物评价。

## 8. 完整目标继续保留的关键证据

所有上述运行之后仍需核查：共享latent/bulk P、新来源协议监督与迁移、A/B真实独立消融、C跨来源与功能一致性、高维A549状态—相容候选—Annexin主问题验证、多步滚动与真实序贯功能证据。Bothe4历史×3RNA/T32不提供每段转移或Annexin。没有对应数据时保留完整目标与未完成状态，不把核心缺口移入展望后仍声称已兑现。

新增[状态context诊断](DATA_ALIGNMENT_STATE_CONTEXT_IDENTIFIABILITY_ROUND22_2026-10-11.md)及独立复审确认：原34上下文可检验记录条件下参考表达效用，但跨上下文伴随板/重复改变；固定W和板后多context组为0。R/G/C也没有补出纯剂量或时间留出。原有模型计划仅覆盖其明确的chemical holdout；真正独立的axis实验需要另立配套训练资产与评价版本，不能把原全T归一化或AE复用为未见该axis的证明。

新增[GSE306207来源](DATA_ALIGNMENT_SEQUENCE_SOURCE_ROUND22_2026-10-11.md)已实际读取A375的4历史×2生物重复TPM及完整名义药物图，全部32461共同基因逐值一致，原样本role仍unassigned。它补齐有限历史表达的数据入口，尚无来源模型、完整洗脱字段、反向/单Oxa RNA或Annexin，不在上面A549模型命令中自动消费。CellCage仍缺实际gene×CCE矩阵及行级对应表，不能由论文或metadata创建模型输入。

该方案允许实验得出无增量、条件外推失败或G/Q不优于简单基线的结果；这些是可证伪研究结果。源码/接口和合成测试通过不能记作真实验收。真实模型操作只有获得该阶段额外授权后才执行；已授权的数据作业继续独立推进。
