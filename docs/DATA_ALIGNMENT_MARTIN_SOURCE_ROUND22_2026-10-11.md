# Martin 来源 E/D：实现与数据准备验收（第二十二轮）

日期：2026-10-11。对应毕业论文[研究架构§4.3](G:/研究生科研文件/毕业论文/研究架构.md#43-表示响应与评价层)中的表达编码 E、来源解码 D 与来源观测监督。**来源适配候选、独立fit/selection runner与真实输入准备已实现；43项合成结构/优化/恢复检查通过。真实 RNA 的模型 forward、匹配参数、训练、共享潜空间迁移及科学效果均未验收。**正式训练停止安排继续有效。

## 1. 证据与边界

沿用[第二十一轮](DATA_ALIGNMENT_ROUND21_2026-10-11.md)已经关闭的 `MartinNativeRNA`、`MartinObservedProtocol` 和 `MartinObservationCoordinates`，不修改原 RNA、joint 投影、LOBO/cycle 角色、8fold 坐标或完整协议 schema。

本轮亲自读取原始[PMC10035388 XML](../experiments/20261011-data-alignment-round16/state_function_new_sources/PMC10035388.xml) Methods `sec2-9`：PC9、NCI-H1975、HCC827、HCC2935，500 nM osimertinib，21天 DTP 暴露，随后两次 PBS 洗涤与24h或背景特定恢复；平行 parental 培养21天后 DMSO/药物24h，不是同细胞给药前 t0。Methods `sec2-4` 的每周两次维护属于 Incucyte 生长实验，不填入 RNA 的未知维护日程。

亲自读取[GEO原始样本 metadata](../08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/rna/geo_samples_source.json) 的 `!Sample_data_processing`：Salmon、tximport `countsFromAbundance="lengthScaledTPM"` 和 `log2(TPM+0.01)`。作者处理域分别为连续估计 count 与含合法负值的 log2TPM；不转换为整数 UMI，也不直接套入旧单细胞 library10000/log1p 路径。

Martin 是非 A549 的 RNA/状态与辅助 caspase 来源。本轮不读取功能 Y，不制造三 RNA 重复与功能重复的配对，不将其替代 A549 Annexin 监督。

## 2. 新入口与职责

| 入口 | 实际职责 | 验收范围 |
|---|---|---|
| [MartinSourceInputs](../src/lung/data/martin_source_inputs.py) | 已认证 RNA→protocol/clock→来源观测张量、单位、坐标和掩码 | 真实数据打包运行通过 |
| `MartinSourceBatch.validate` | 两域、float64、exact gene 顺序、sample IDs、NaN/mask、正 scale、角色、背景与身份 | 合成反例及真实打包通过 |
| [MartinSourceModel.encode](../src/lung/models/martin_source.py) | 表达值与独立 mask、GSE193258 来源、四背景、观测域、参考/target 时钟 | 已实现，纯合成 forward 通过 |
| `protocol_context` | full physical history 的 FP、事件、单位、剂量、暴露、洗涤、恢复、终点及 known mask；额外协议 unknown mask | 已实现，纯合成敏感性检查通过 |
| `decode` | 分域观测出口；log2TPM 反 affine、count estimate 反 log1p；缺失维度回为 NaN | 已实现，纯合成 forward 通过 |
| `observation_loss` | 按样本独立 mask 的均方误差；未测维不进入损失 | 已实现，纯合成检查通过 |
| `validate_checkpoint_metadata` | 新参数 namespace、来源、域、背景/协议词表、坐标、fold、轴、源码及 fit/final 身份 | 静态合成 metadata 反例通过，无 PT 加载 |
| [verify_martin_source_model.py](../scripts/verify_martin_source_model.py) | 默认只进行真实数据 readback；独立显式模式才可对真实输入运行未训练模型 | readback 已运行；真实模型分支本轮未执行 |
| [martin_source_learning.py](../src/lung/engine/martin_source_learning.py)、[train_martin_source.py](../scripts/train_martin_source.py) | 独立来源E/D fit/selection、唯一GSM去重、双域目标、AdamW与既有Stopper、更新级断点恢复 | 真实数据preflight通过；合成优化/恢复通过；真实fit未运行 |

新来源词表明确为 `GSE193258` 与 `H1975/HCC827/HCC2935/PC9`，新 namespace 的 source index 0 只表示 GSE193258，不沿用 Tahoe/A549 index。协议 ID 为四种真实 RNA 状态协议，名称和角色由原 reader 认证。没有给未知 source/background/protocol 自动分配旧模型 ID。

新参数身份为 `martin_source_bulk_E_D_v1`。输入契约绑定原生 RNA/观测坐标/协议 manifest、fold、fit GSM、基因轴、单位与新消费器 SHA；模型身份另绑定模型、`full_protocol.py`、`encoders.py` SHA 和 width。旧 AE/CFM 参数不能仅因 shape 相近或改字符串加载为新来源模型。

## 3. 观测域与本课题损失设计

以下是本课题新提出的工程/建模候选，没有归为 Martin 论文算法，也没有声称理论优越性或迁移保证。

令 (m_{ig}\in\{0,1\}) 为实际观测 mask。

* log2TPM：作者原值 (x_{ig}=\log_2(TPM_{ig}+0.01))。采用已冻结、只由对应 fold 唯一 fit GSM 建立的 (mu_g,\sigma_g)，输入 (z_{ig}=(x_{ig}-\mu_g)/\sigma_g)。解码头输出无非负约束的 (hat z)，恢复 (hat x=\hat z\sigma+\mu)，保留合法负值。
* estimated_counts：保留作者非负、可非整数的原值 (a_{ig})。单独输入 (z_{ig}=\log(1+a_{ig}))，解码头用 softplus 输出非负 (hat z)，恢复 (hat a=\exp(\hat z)-1)。不四舍五入、不拟合新增归一化统计、不进行 library normalization。

每域的候选重建损失为：

\[
L_D=\frac1B\sum_i\frac{\sum_gm_{ig}(\hat z_{ig}-z_{ig})^2}{\sum_gm_{ig}}.
\]

该定义只比较来源实际观测坐标，未使用 Poisson/NB 计数似然、逐细胞标签、功能 Y 或前后配对。bulk 每行是来源 RNA 生物重复，不是单细胞。两域来自同批 RNA 的作者处理产品，不能当成两套独立实验。两域损失尺度和权重的科学选择、相对于简单基线的收益仍须 fit/selection 实验判断。

输入原值及 prepared model values 均保持 float64/NaN+mask。仅神经网络内部将缺失维设有限计算占位，mask 作为独立输入且决定监督范围。默认神经网络运算为 FP32，source 反变换输出的 float64 不恢复网络计算所丢失的精度，后续需检查精度敏感性。

## 4. 时钟、协议与共享表示

参考 E/D 重建使用自身的24h DMSO 时钟，严格核对metadata中的已知DMSO小时，保留21天预培养字段与独立 clock-origin token。它不会把 target 的504h暴露或恢复记录当作参考表达的测量时间。target测量clock严格等于已知physical endpoint。target D 消费有顺序的完整**已报告名义阶段**；`decode(..., as_target=True)` 即使输入reference batch，也使用target endpoint/clock-origin作为解码条件，但输出sample IDs仍标记输入样本，不能与真实target重复伪配对。该出口不构成已实现的 P。

未知洗涤操作耗时、体积、细胞内残药、培养基容积、DMSO 浓度继续为 NaN/False。`event_list_exhaustive=False` 是已有来源边界标记；没有新增21天维护事件。已登记的 transition start=end 记号仍不解释为实测操作耗时零。

此 E/D 是独立来源适配候选。与 Tahoe/TIDYI 共享潜空间的联合训练、域迁移、对应响应 P、bulk 与单细胞表示的可比性尚未建立；不能因为 latent 维数相同就送入旧 C 或宣称共享表示已对齐。旧 `native_response_producer` 没有被改为接受 Martin，因此原 pending status 继续有效。

## 5. 实际运行结果

[最终 v3 真实读回](../experiments/20261011-data-alignment-round22/martin_source/data-readback-v3/real_input_readback.json)对应本轮修后代码 hash；[input contracts](../experiments/20261011-data-alignment-round22/martin_source/data-readback-v3/input_contracts.json)保存实际使用的三个 cycle fold 身份。只选择 fit 请求，原8fold不改；本次读回不等于全部8fold的模型评价。

| 检查 | 实际结果 |
|---|---|
| 真实组、事件、样本 | 16唯一RNA组、40已报告名义事件、60唯一GSM |
| 准备视图 | 64个reference/target/domain张量视图，不是64独立实验 |
| 作者原值 | 双域每个视图与冻结 joint NPY 按原 column 精确一致 |
| 轴与 mask | 每样本32,461维；16,167观测、16,294缺失NaN/False |
| 角色 | 每次读取只开放fit；全部样本属于该请求fold的冻结fit GSM，final入口拒绝 |
| affine 回转 | log2TPM 最大绝对误差1.7763568394002505e-15 |
| log1p 回转 | estimated_counts 最大绝对误差4.656612873077393e-10、以max(1,abs(raw))归一的最大相对误差9.849378744601888e-16 |
| 合成模型与runner验证 | 43 tests passed：模型30、runner13；两域/mask/history、严格clock/GSM/请求身份、更新级中断的model/optimizer/RNG/早停精确恢复、执行依赖漂移前置拒绝 |
| 当前真实 ML | 模型构造0、forward0、PT0、optimizer0；P/G/Q输出0，功能Y读取0 |

初次脚本运行因只加入 `ROOT/src` 导致导入已有 `src.lung_data` 失败；修为与项目测试一致的 ROOT 和 ROOT/src 路径。本地失败记录保留，原数据及来源未动。v1为增加源码/严格资格绑定前的版本，v2为独立复审修正前，v3为当前版本，回执不互相覆盖。

独立复审指出原batch只检查clock为正、没有强制fit GSM成员、loss输出缺请求/样本身份，以及reference-as_target仍沿用参考clock。已分别补reference/target clock对应、冻结fit/selection GSM清单、输出及loss逐项request/sample/source/background/protocol/system/fold/role核对、目标D clock切换。30项模型tests包含这些反例，真实v3读回未构造模型。

验证命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_martin_source_model.py tests/test_martin_source_learning.py -q
& 'F:/anaconda/envs/study/python.exe' scripts/verify_martin_source_model.py --mode readback --output experiments/20261011-data-alignment-round22/martin_source/data-readback-v3
& 'F:/anaconda/envs/study/python.exe' scripts/train_martin_source.py --fold-id cycle_final:HCC2935 --output experiments/20261011-data-alignment-round22/martin_source/learning-preflight-v2
```

三条均已实际执行；脚本拒绝覆盖已有 output，复验需新目录。环境为已安装 PyTorch 2.13.0+cu130；本轮只用 CPU 做数据与合成验证。

## 6. 待授权的真实模型步骤与资源

具体候选第一步是**真实 fit RNA 的未训练模型结构检查**，没有任何方法收益资格。本轮未执行。可审阅命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' scripts/verify_martin_source_model.py --mode real-untrained-forward --system cycle --fold-id cycle_final:HCC2935 --request-id observedRNA:H1975:long_washout:cycle_final:HCC2935 --domain log2TPM --width 32 --seed 23 --output experiments/20261011-data-alignment-round22/martin_source/real-untrained-log2TPM-v1
```

第二域需显式改为 `--domain estimated_counts` 和独立 output。该模式只构造随机权重、no_grad、fit RNA 输入，不加载 PT，不更新参数；输出明确 `untrained_random_parameters=true`。结构检查通过仍只证明接线和有限计算。

独立fit/selection runner已经实现，真实fit未启动。数据只由原reader按既有cycle角色准备，不为LOBO虚构selection。真实[learning-preflight-v2](../experiments/20261011-data-alignment-round22/martin_source/learning-preflight-v2/preflight.json)取得30唯一fit GSM（6参考/24target）和15唯一selection GSM（3参考/12target），final角色未打开。fit参考在双域请求中曾出现48行，去重后仅6实际参考GSM；每个GSM是两个观测域视图，不变成两倍生物重复。

新重建目标为每个唯一fit GSM内两域loss等权平均，再按GSM平均；不是论文算法，不预设优于单域。相同reference在多候选请求下仍是同一观测；目标RNA若数值、协议或观测域冲突则拒绝去重。selection单独no_grad计算，只用于记录/早停，不进入梯度、坐标或optimizer；final关闭。

来源width32/FP32为新候选的明确参数。其余设置直接读取当前项目配置：AdamW、lr0.0001、weight_decay0.0001、grad_clip1、AE batch128、1000 epoch、原`_EarlyStop`（patience20/min_delta0.0001/min_updates1000）。没有墙钟限制。该fold只有30唯一fit GSM，所以batch128使每epoch一个optimizer update，继承min_updates1000意味着早停仅在相应更新数后判断；这是现行配置的实际含义，未自行更改。

`last.pt/best.pt`、manifest、epoch内cursor/固定fit permutation、模型/optimizer/RNG/早停与selection历史均有独立来源/fold/样本hash/目标/参数身份。resume在任何PT读取及模型构造之前比较model/full_protocol/encoders/checkpoint源码、`_EarlyStop`类源码和Torch版本的执行依赖身份。合成中断恢复与不中断运行逐Tensor一致，依赖漂移反例前置拒绝。训练只在显式`--fit`时发生；未来可实际调用的命令（本轮未执行）：

```powershell
& 'F:/anaconda/envs/study/python.exe' scripts/train_martin_source.py --fit --config configs/modular_research.yaml --fold-id cycle_final:HCC2935 --width 32 --output experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1
& 'F:/anaconda/envs/study/python.exe' scripts/train_martin_source.py --fit --resume --config configs/modular_research.yaml --fold-id cycle_final:HCC2935 --width 32 --output experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1
```

真实PT加载只发生在上述明确resume；本轮加载的是测试tmp_path自造的合成checkpoint，未加载任何真实研究参数。后续具体顺序：

1. 独立复审数据/参数身份与观测目标；取得真实模型运行授权后，运行上述两个分域结构检查。
2. 取得真实训练授权后运行已实现的独立E/D runner：cycle 两背景fit、一selection、final保持封闭；每域监督与同一样本去重如上。LOBO是另一独立协议，不能混用其三个fit背景建立cycle坐标。
3. 与来源均值、低维线性重建及现有适用来源模型比较，使用selection判断域尺度、模型宽度、float32敏感性与mask处理；所有选项只由fit/selection确定。
4. 再评估共享E或跨来源表示方案、匹配的bulk P与target D；用同一原RNA时钟的独立表达评价，不把四药72h caspase挑战当RNA target。
5. 取得匹配的新来源/新fullhistory参数及冻结方案后，才规划独立final表达评估与辅助功能迁移；不把辅助非A549 caspase结果转称A549 Annexin。

当前数据只有平行reference/target RNA群体，不能识别逐细胞未来命运。真实多步滚动/序贯及功能终点仍需各自匹配数据与独立评价，不从来源重建测试推导通过。

width=32、32,461基因的新两域E/D按源码层尺寸估算4,376,522个参数，FP32参数约16.70 MiB。仅参数、梯度和两份FP32优化器状态四组数组约66.78 MiB，未含激活、缓存、整链P/G/Q或优化器其他开销。此为公式估算，未运行真实模型或测峰值RAM/显存；不能替代整项目单机可行性结论。

## 7. 对齐与科学验收表

| 问题 | 所需及实际数据 | 实现/验证 | 结论与剩余缺口 |
|---|---|---|---|
| bulk来源观测域能否合法进入E/D | 作者双域RNA、单位、joint轴、mask、来源背景 | 数据准备实际通过；E/D合成通过 | 有工程入口，真实模型未运行 |
| 完整处理历史是否实际进入source出口 | Osi500nM、24/504h、wash/media/recovery与终点、unknown mask | fullphysical字段与顺序合成检查通过 | 仅已报告名义阶段；无完整维护日程或真实模型消费证据 |
| 参数能否与数据身份匹配 | 原manifest、fold/fit GSM、轴/域/词表、新模型/source hash | 新identity/strict loader已实现、合成metadata反例通过 | 无匹配训练参数/真实PT加载 |
| E/D训练与selection能否真实执行 | 同源fit/selection各唯一GSM与冻结角色 | 真实preflight30/15通过；独立runner合成更新/精确恢复通过 | 真实fit未运行、目标科学收益待比较；原fit/selection接口仍关闭final，新增独立final仅元数据预检 |
| 共享状态表示是否可迁移 | 共享/跨域目标、独立bulk表达、独立来源/背景评估 | 当前只提供source adapter | 未验证，共享latent接线与表示学习仍缺 |
| 表达能否连接独立凋亡终点 | 对应RNA协议与真正匹配终点 | 本轮功能Y0 | 不授Annexin或caspase连接资格 |
| 对主问题A/B/C/D与候选Q有何贡献 | 状态/未见条件/功能/联合选择相应独立标签 | 本轮只补来源E/D准备 | 整体科学可行性及效果维持未完成 |

本轮只新增来源数据/模型/独立runner/验证/文档与专属实验回执；未改毕业论文架构、用户已有代码、共享协议、旧producer、冻结划分、奖励定义或旧正式训练入口。未提交、未推送。

## 8. 本轮后继 final 入口与版本接续

[独立final评价入口](DATA_ALIGNMENT_MARTIN_FINAL_ROUND22_2026-10-11.md)已经实现，默认只校final角色／来源／固定fit统计的元数据，不重新读取v3 RNA，不打开原fit/selection接口的final权限。learning在完成result尾段新增完成状态与run_manifest／best／last字节凭据，因此当前learning SHA变为`4d630139f4b7ca1d4658a77258b71fbb50a6591b6a1f0daac20405b9f7386cbc`；原数据consumer／model SHA不变。上文43项属于其原版本，后继13 learning＋22 final的35项I合成执行与零模型guard另存新目录，详见[current readiness](../experiments/20261011-data-alignment-round22/martin_source_final/current_readiness_v2.json)。真实source训练、真实final RNA数值／PT／forward／科学效果仍未执行。
