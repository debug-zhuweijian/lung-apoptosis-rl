# Martin observed RNA protocol inputs v1

2026-10-11。来源监督输入，schema=`martin-observed-RNA-fullphysical-inputs-v1`，沿用已有`full_physical_protocol_v1`。本包不含RNA生成结果、latent h、核心功能Y、模型权重或模型/功能资格。

`groups.json`保留16唯一观察RNA条件与40个已报告名义事件；`role_links.json`独立保存64 LOBO和64 cycle职责请求到group的映射。真实仍48处理RNA样本和12parental reference，各3个biological bulk repeats；不增加128个独立实验。

药物是RNA来源的osi500nM，名义parent FP row0（PubChem CID71496458，stock/salt未认证）。acute暴露24h；DTP及两类恢复前处理504h；short恢复24h，long按H1975/HCC827/PC9/HCC2935恢复72/96/168/240h。endpoint采用名义phase clock24/504/528/576/600/672/744，不把washout72误作连续72h给药。所有function_72h_four_drug_target=false。

恢复协议为drug interval→PBS wash×2→换无药media→recovery。wash/media start=end504沿用既有transition_clock记号，不是实测耗时0。精确操作开始/结束wallclock、操作耗时、液量、medium配方/血清/添加物、DMSO浓度、21d维持换液日程与细胞内药物残留均保留unknown；RNA Methods sec2-9不能借用sec2-4 Incucyte每周两次维护记录。prior_retention不设生物学0。

parallel parental reference自身为无药21d＋DMSO24h，不是同细胞t0。其source/GSM/列号/轴/mask/domain/roles均委托已独立复审关闭的MartinNativeRNA固定来源reader，不重复创建可由caller重hash的RNA根。condition/tensor API只接登记request ID、system、fold、role；不能提交自造events/vector/kwargs获得来源资格。新包的paper/GSM/FP/schema/RNA loader及manifest均固定绑定，并从这些原件重建记录核对。

`base_design`是未验证的设计契约：duration24/504、horizon名义endpoint、elapsed0、dose_uM0.5；论文没有规定这组网络kwargs。source/bg/protocol indices为None，encoder/decoder/权重pending，没有完整numeric kwargs可调用模型。

```python
from lung.data.martin_observed_protocol import MartinObservedProtocol

with MartinObservedProtocol(ROOT, PACKAGE, allowed_roles=('fit',)) as data:
    rows = data.requests(system='cycle', fold_id='cycle_final:HCC2935', role='fit')
    rid = rows[0]['request_id']
    metadata = data.condition(rid, system='cycle', fold_id='cycle_final:HCC2935', role='fit')
    physical = data.physical_batch([rid], system='cycle', fold_id='cycle_final:HCC2935', role='fit')
    ref = data.reference(rid, system='cycle', fold_id='cycle_final:HCC2935', role='fit')
    supervised = data.supervision(rid, system='cycle', fold_id='cycle_final:HCC2935', role='fit',
                                  domain='estimated_counts', coordinate='joint')
```

physical是CPU数据tensor，validate不调用模型。reference不选择target，supervision才选target；角色系统/fold/role不得串用，final必须显式allow_final=True。source/joint以及log2TPM/estimated_counts由closed RNA接口原样提供；negative log2TPM、noninteger estimated counts及joint NaN/mask保持，不重新归一化、不补0。

准备：`F:/anaconda/envs/study/python.exe scripts/prepare_martin_observed_protocol.py`，已有目录拒绝覆盖。24边界测试通过；实际128 computational tensor rows/320重复计算events按角色CPU校验通过，256双domain监督读取及reference-only与登记RNA精确一致；独立生物计数仍16组/40唯一events/48targets/12refs。旧源/角色/coreY字节及共享代码未改，无ML/PT/训练。见Round21 martin_observed_protocol_dependency/actual_readback.json与实施回执。

独立复审后的严格JSON修复：manifest/groups/role_links按canonical JSON标量类型和list次序重建比较，False不能被int0冒替，NaN/Inf拒绝。修复后32项测试通过；真实16/40、128CPU行和256双domain读取再次核对通过，最新证据为`actual_readback_strict_json.json`。原groups/role_links/RNA不变，manifest仅更新loader源码绑定；初版字节快照保留于`pre-review-bool/`。
