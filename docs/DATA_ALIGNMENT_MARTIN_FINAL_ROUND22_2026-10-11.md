# 第二十二轮 Martin 独立 final 表达评价入口

结果：已实现独立的、只能评价完成后冻结 source E/D 参数的 final 入口。实际只完成冻结元数据预检；**final RNA 数值读取、真实 PT、真实模型 forward、真实误差及来源科学收益均为 0／未运行**。原 fit/selection 入口继续拒绝 final，不把 final 角色重标为 fit。

## 数据与角色身份

[独立 final 工厂](../src/lung/engine/martin_source_final.py)读取原 `native-rna-inputs-v1`、`observed-protocol-inputs-v1`、`observation-coordinates-v1` 的冻结 manifest、样本／角色／组／轴 JSON。构造时不调用旧 NativeRNA、ObservationCoordinates、SourceInputs：旧构造器会重建 RNA 或 fit 统计，新 default metadata preflight 不执行这些操作。

当前 `cycle_final:HCC2935` 原角色为 fit30／selection15／final15 唯一 GSM；final4 个请求包括 DTP、acute、short washout、long washout，3 个 DMSO 参考与12 个 target。参考在4 请求、2域中出现24行，评价去重为3 个原始参考 GSM；target 双域24行对应12 GSM。log2TPM 与 estimated counts 是同批 RNA 的两种观测域视图，不记成30个独立 final 生物重复。4 个背景的原 cycle fold 元数据与 fixed fit coordinates 保持不变。

模型 gene axis 精确固定为32,461 common genes；原作者 log2(TPM+0.01) 负值、连续 estimated counts、缺失 NaN／mask、单位、来源背景、真实 role/fold/request/sample identity 保留。final metadata scope 增加明确 final GSM／请求清单及 fixed coordinate 文件 SHA，但训练模型的 input contract 与已经读回的 `learning-preflight-v2` **逐字段完全一致**；未改变模型 fit/selection contract。

## 完成训练与冻结评价约束

[学习入口](../src/lung/engine/martin_source_learning.py)唯一新增行为是完成 result 的凭据尾段：completion reason、实际 run_manifest SHA、best／last checkpoint SHA。训练目标、fit／selection reader、唯一 GSM 权重、optimizer、采样、划分和早停接口未变。旧结果如果缺这份明确完成 proof，final 入口拒绝，不补造旧训练完成记录。

`completed_run` 在任何 PT 读取前验证：本 namespace、完整 input contract、当前 source learning SHA 与执行依赖、fold／角色样本计数、manifest identity、真实而非 synthetic 标记、完成状态与轮次、best／last 文件字节。记录中的 P/G/Q 输出为0，final 未用于原训练。当前没有满足这些条件的真实 source 模型资产，因此状态为 pending。

显式 `evaluate-final` 后才加载本项目生成的 best／last PT：last 的实际停止状态、epoch、更新计数、selection history、cursor/order 与完成 result 必须相符；best 必须属于同一参数 namespace／域／轴／背景／fold、fit training role、有效选择轮次和有限参数。模型 `eval`、全部参数 `requires_grad=false`，没有 optimizer、fit/selection 准备、选模或坐标重估。

之后只反序列化请求角色对应的 final NPY 行；整个文件的 byte hash 用于 source integrity，不等于把其他角色数值送入模型。固定 fit affine mean/scale、mask/constant-mask 的字节、dtype与形状必须匹配登记文件；estimated counts 使用与原 source reader 相同的 torch log1p，不四舍五入／做UMI似然。真实观察单位分别以其所属域报告 raw MSE／MAE 和 model-coordinate MSE，不把两个原单位误差直接相加为跨域科学指标。

参考 source E/D 使用自身24h DMSO measurement clock；target 的观察 clock 必须等于已知完整名义协议 endpoint。有序事件、nominal dose、暴露、wash/media/recovery、原单位及 unknown mask 进入 target decoder；来源未报告的维护历史、操作时长、体积或细胞内残药不会变成实测零。source identity 仍是 GSE193258 非A549，不授 Tahoe A549、Annexin、共享潜空间、响应 P 或正式奖励资格。

final 每个 GSM 在每个 domain 只计一次。相同参考在多候选请求中的重复视图不重复加权；如同一 GSM 的域、原值、坐标、时钟或 target protocol 出现矛盾则拒绝。实际输出保存各唯一观测的背景／协议／角色／请求、单位、观测基因数及三种误差；before/after 检查冻结训练资产与 source 数据字节，防止运行中漂移。

## 当前实际证据与版本

[真实 metadata guard v3](../experiments/20261011-data-alignment-round22/martin_source_final/real_metadata_guard_v3.json)明确拒绝 `np.load`、PT、nn module init、fit 和旧 reader 构造器；实际调用次数均为0。预检核验 final15 GSM／4请求与合同逐字段一致性，**没有重读旧 v3 RNA 或 final 数值**。

[current readiness v2](../experiments/20261011-data-alignment-round22/martin_source_final/current_readiness_v2.json)绑定复用的旧 `learning-preflight-v2` 字节、fit／selection observation digest和当前执行依赖。data input／model源码仍分别为 `74e1357c...`／`9f12316d...`；learning 新SHA为 `4d630139f4b7ca1d4658a77258b71fbb50a6591b6a1f0daac20405b9f7386cbc`。旧43项 CI只说明旧输入／模型／runner版本，不能覆盖新增 final。

当前新增版本实际命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 -m pytest tests/test_martin_source_learning.py tests/test_martin_source_final.py --basetemp=experiments/m22final3 -q
& 'F:/anaconda/envs/study/python.exe' -X utf8 experiments/20261011-data-alignment-round22/martin_source_final/guard_metadata_preflight.py
```

结果为 **35 passed in5.15s**（原learning13＋新增final22），全部模型数据为3基因合成 fixture；测试证明旧 final 接口封闭、完成 proof／源依赖／best 字节／计数漂移前置拒绝、实际 last counter 核对、真 final role/clock/mask、冻结参数无 optimizer 的双域去重评价。测试的非synthetic metadata正向路径由 mock source factory 在新I临时目录人为构造，仅用于验证生产边界代码，不能登记为真实匹配模型。首次32通过／1失败为测试错误消息匹配的差异，记录保留；修正后33通过。进一步自审补齐best保存轮次的实际selection history／cursor类型／counter／early-stop更新数，再运行当前35项；旧33结果不覆盖这一增量。

独立源码最终SHA：final `8f6848cd34b255b674e134e3806543eeb48ccb9ca516b4d1d8f6fc2f15685e1f`；CLI `dca79c4858866e39d0ea887edc2c1be0c3cd3ee181adddcadc660bc806cb4b92`；tests `0eedb57c753bd6bbd5dd9cb138636fcac4f60f0e5841e535974e8c093c0003db`。未来独立复审／根综合 CI 如改源，需按新 SHA 更新本页或后继回执，不能只按mtime认定同版本。

[独立final复审](../experiments/20261011-data-alignment-round22/independent_martin_final_review/results.json)在相同8f6848cd版本关闭工程范围、无新增源码finding：22个owner final项亲跑通过，另在I新m22r3独立4反例4passed／3.99s。首轮两项review fixture错误与第二轮导入路径错误由reviewer在自身脚本修复并保留日志，不合并成一次无失败26项执行。复审没有重新读取真实RNA、PT或运行真实模型，科学资格仍false。

## 可实际调用的后续顺序

本轮已执行的默认预检只做元数据；路径不可覆盖，复验应另取新输出：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/evaluate_martin_source_final.py --fold-id cycle_final:HCC2935 --output experiments/20261011-data-alignment-round22/martin_source_final/metadata_preflight_next.json
```

取得额外真实模型运行授权后，先按[来源 E/D 方案](DATA_ALIGNMENT_MARTIN_SOURCE_ROUND22_2026-10-11.md)运行必要的分域结构检查与 `train_martin_source.py --fit`／明确 `--resume`。沿现有配置、selection及早停完成 source 模型，审阅并固定比较／模型选择；不得根据 final 改参数或重新挑选模型。source训练完成后可先做不加载 PT 的完成资产预检，再显式开启 final 模型评价：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/evaluate_martin_source_final.py --fold-id cycle_final:HCC2935 --trained-run experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1 --output experiments/20261011-data-alignment-round22/martin_source_final/completed_model_preflight_v1.json
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/evaluate_martin_source_final.py evaluate-final --fold-id cycle_final:HCC2935 --trained-run experiments/20261011-data-alignment-round22/martin_source/source-E-D-cycle-HCC2935-v1 --device cpu --output experiments/20261011-data-alignment-round22/martin_source_final/actual_final_v1
```

第二条目前**未执行且仍需额外授权**。仅完成 source 重建评估仍不能授共享 latent、初始状态增量、bulk P、功能 G、候选选择收益、真实序贯或主问题整体资格。此入口没有增加科学通过阈值，也没有将 final 效果预先保证为优于来源均值／线性等基线。

| 验收层 | 当前结果 | 剩余证据 |
|---|---|---|
| source final角色、域、轴与来源 | 元数据及字节实际预检，独立final消费代码已实现 | final NPY数值角色读回在显式评价时发生，本轮未开启 |
| 冻结训练完成与参数身份 | source epilogue／loader合成反例通过 | 真实完成模型、匹配PT、selection冻结记录 |
| source E/D独立表达误差 | 合成执行通过；真实未运行 | 同固定final source GSM分域误差及适用基线 |
| 对状态／未见条件／功能／决策的科研贡献 | 仅提供来源表示研究的执行入口 | 共享表示与迁移、真实P/G/Q、完整终点与序贯独立验证 |

未改原 `MartinSourceInputs` 的 fit/selection API、旧正式训练、冻结reward／split、他人源码及毕业论文研究架构；未提交、推送、删除或恢复真实训练。
