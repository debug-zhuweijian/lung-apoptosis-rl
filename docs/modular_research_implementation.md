# 模块化预训练实施回执与 PyCharm 启动说明

日期：2026-10-07（Asia/Shanghai）。执行主目录：`I:/科研/肺癌细胞凋亡-强化学习/`。

W1–W5 的实现、W6 的测试与启动文档已交付。合成数据完整链和有界真实全基因组件已运行；**完整真实数值审计、受审计 World 的真实小批 Actor/Critic 更新尚未运行，计划六项工程开训标准未全部通过**。本回执不将组件单步更新解释为收敛、药效或训练提速结论。

依据：[已确认实施计划](G:/研究生科研文件/paperstudy/02-notes/Modular_Pretraining_RL_Implementation_Plan_2026-10-07.md)。旧一次性训练路线已被淘汰；默认图不重跑旧链，不包含优越性或提速比例准入条件。正式长时间训练由用户在 PyCharm 启动。

## 实际实现

| 工作包 | 改动与接口 |
|---|---|
| W1 | `engine/representation.py` 独立加载已更新 AE；pool/state 优先读取 `pipeline.representation`，消费 h，依赖及缓存绑定 AE 文件哈希。纯速度网络变化不使这些模块失效。 |
| W2 | `engine/response_loader.py` 统一传统与潜空间响应加载；adapter 补齐 `encode_cells` 和 `condition_encoder`；World 核对所有潜空间成员的 AE 身份，再共享同一冻结表征。 |
| W3 | `rl/actor_schedule.py` 与 `rl/daac.py` 接入同 Actor、同 Adam 的 drug/dose/joint 参数选择；双 Critic 持续学习，冻结头保留输入梯度，没有合法选择的头跳过，配额归 joint。 |
| W4 | `engine/decision.py` 区分 Actor-only `init_from` 和严格 `resume`；保存完整阶段、计数、优化器、replay、随机流、累计预算和计时。 |
| W5 | 根 `train_modular_research.py`、`engine/modular_config.py`、`engine/modular_research.py`、独立 YAML；每模块独立 run ID、预算、缓存与检查点，完成组件校验后复用。 |
| W6 | 新增模块化整链、身份拒绝、梯度、阶段恢复、辅助准备、缓存、数值审计恢复及真实视图测试；新增真实有界验证脚本和本回执。 |

缓存分块持久化，`manifest.json` 只在全部完成后发布。pool 特征按条件保存抽样 RNG；state 的角色缓存和温度优化器可以恢复；potential 的参考目标按 prefix 保存。数值审计保留原协议，ODE 样本和人口重复完成后保存，可在预算边界暂停。

未完成 `last.pt` 不能作为完成组件登记。`modular_completion.py` 将最终训练完成证明绑定到选中的权重，生成独立不可变产物；即使选中的 best 早于最终 epoch，后续也不需要依赖可变的 `last.pt`。state 额外核对校准身份、有限正温度及模型 buffer 一致性。

## 默认入口与阶段

PyCharm 设置：

1. 解释器：`F:/anaconda/envs/study/python.exe`。
2. Script path：`I:/科研/肺癌细胞凋亡-强化学习/train_modular_research.py`。
3. Parameters 留空；默认读取 `configs/modular_research.yaml`。
4. Working directory 可设为项目根；入口本身按文件位置定位配置。

默认阶段为：

```text
ae → cache → response → pool → state → ode_audit
   → assemble（人口数值审计、所需校准及冻结证书）
   → potential → daac（drug 20 → dose 20 → joint 160）
```

默认只训练一个响应成员和一个策略 seed。全基因轴保留 32,461；h/z 初值均为 256，测试另外覆盖异维 h=8/z=4。CFM 使用 128 样本、16 条件一次更新；初始关闭 B4。单次 24 h 条件选择、完整合法候选、现有状态代理奖励及固定生物环境保持原协议。

pool/state 可在 AE 完成后单独启动，不必等待 CFM。`backward/correction/uncertainty` 保留独立阶段能力及依赖检查，不进入首轮最低必要图。五成员接口做了微型兼容验证，没有启动五成员正式训练。默认不运行消融、DLPA 对照或旧套件。

仅读依赖与预算预览：

```powershell
F:/anaconda/envs/study/python.exe train_modular_research.py --dry-run
```

单独执行阶段（前置组件必须存在于本次 pipeline 或明确提供的兼容产物中）：

```powershell
F:/anaconda/envs/study/python.exe train_modular_research.py --stage ae
F:/anaconda/envs/study/python.exe train_modular_research.py --stage pool
```

`modular.artifacts` 可显式指定已有组件，或用顶层 `pipeline` 指向已有 `pipeline.json`。这些路径都会核验数据、表征、候选、依赖、训练完成及校准状态；路径存在不代表可以复用。真实有界验证产物绑定不同视图身份，不能作为正式全量训练产物导入。

## 保存、预算与恢复

根实验保存 `modular_state.json`、`pipeline.json`、`modular_report.json` 和 `response_budget.json`。每个阶段另有 `<root-run-id>-<stage>` 实验目录、权重目录及 `module_budget.json`。暂停时记录当前模块和恢复位置；已完成上游产物继续保留。

默认自动恢复科学配置和源码身份相同的最近模块化实验。也可显式指定根 run ID：

```powershell
F:/anaconda/envs/study/python.exe train_modular_research.py --resume <root-run-id>
```

默认响应链累计上限 10,800 秒，AE 子上限 2,700 秒；DA-AC 初始累计上限 3,600 秒。预算在安全边界检查，当前不可分割更新、评分或保存可能跨过上限；实际耗时完整记录，完成产物不会因最后保存跨限而被丢弃。IDE 强制终止最多丢失尚未持久化的当前原子单元；正常中断会结算本次计时。

预算耗尽后不会自动加倍。修改 YAML 的相应 `modular.stages.<stage>.budget_seconds` **累计上限**后重新启动；响应链总预算耗尽时还要显式调整 `modular.response_budget_seconds`。临时只恢复某阶段也可：

```powershell
F:/anaconda/envs/study/python.exe train_modular_research.py --resume <root-run-id> --stage ae --budget-seconds 5400
```

CLI 上限只适用于该次调用；后续无参数运行仍读取 YAML。增加预算保留累计已花时间。数据、模型坐标、候选、奖励或已登记日程变化必须新建实验；新实验可明确复用兼容上游产物。

`daac.init_from` 只在新策略实验导入同一冻结环境下的兼容 Actor，重建 Critic/targets、优化器、replay、随机流和计数。严格恢复继续旧实验全部状态。两者互斥；阶段切换不会调用 `init_from`。增加已完成 Actor 日程的更新配额属于新日程版本，不静默改变旧实验身份。

## 实际验证

联合 CPU 回归（执行时关闭 CUDA）：16 个相关测试文件 **222 passed，49.56 秒**，覆盖新增模块化路径及传统 World、RL、decision、state 和 response 接口。完成证明专项 `tests/test_modular_completion.py` 另外 **15 passed，5.55 秒**；最终预算标记刷新后，五项整链测试复跑 **5 passed，12.64 秒**。CPU 共覆盖 237 项不同测试，复跑不重复计数。

```text
tests/test_modular_research_v1.py tests/test_modular_world_v1.py
tests/test_modular_actor_v1.py tests/test_modular_aux_budget.py
tests/test_modular_aux_preparation.py tests/test_modular_cache_budget.py
tests/test_modular_potential_budget.py tests/test_modular_real_view.py
tests/test_modular_audit_resume.py tests/test_world_execution_v1.py
tests/test_rl_v1.py tests/test_decision_engine_v1.py
tests/test_decision_optimization_v4.py tests/test_auxiliary_v1.py
tests/test_state_pilot_v1.py tests/test_response_research_v1.py
```

运行方式：`python -m pytest <上述文件> -q --tb=short`。CUDA BF16 / 多成员微型回归 `python -m pytest tests/test_response_research_runtime.py -q --tb=short`：**1 passed，15.88 秒**。这些合成测试不提供生物学效果证据。

合成整链实际完成九阶段、drug/dose/joint 各一次 Actor 更新及三次 Critic 更新；第二次全部复用，检查点 mtime 不变。阶段边界和预算暂停恢复与连续执行逐张量一致；初始化不继承旧 replay/计数；错误候选、表征、角色和未完成产物被拒绝。

真实有界 GPU 命令：

```powershell
F:/anaconda/envs/study/python.exe -X utf8 scripts/verify_modular_real.py --run --max-seconds 600 --ode-samples 2 --no-progress
```

运行 ID：`modular-real-verification-20261007-091627-65eb89`。实际总耗时 **48.11 秒**，保留 32,461 基因；固定原划分中的 16 个 T 条件、1 个 V 条件，每条件及同角色对照最多 128 个细胞。读取 2,561 个不同 Tahoe 视图细胞；TIDYI 的既有 fit/selection/calibration 集合保留。源身份与视图身份分别保存，不重新随机划分。

| 组件 | 更新数 / 状态 | 实际秒数 | PyTorch 分配显存峰值 GiB |
|---|---:|---:|---:|
| AE | 1 次更新 | 8.20 | 0.60 |
| h 缓存 | fit 2,177、selection 384 细胞 | 3.57 | 0.20 |
| 潜空间 CFM | 1 次更新 | 4.51 | 0.27 |
| pool | 1 次更新 | 10.02 | 0.20 |
| state | 4 次更新、2,000 步温度校准 | 17.48 | 0.20 |
| ODE 审计部分进度 | 2 个样本，未判通过 | 2.75 | 0.18 |

所有已运行组件权重/缓存有限、没有 OOM。显存数据为 PyTorch 分配峰值，不是整卡总占用；该计时包含相应小规模阶段开销，不能外推全量训练时长或声称相对旧路线的加速倍数。

本地原始证据：`experiments/modular-real-verification-20261007-091627-65eb89/verification_report.json`、`bounded_real_view.json`、`bounded_pipeline.json`，以及对应 checkpoints/logs。报告明确 `synthetic=false`、`is_bounded_real_view=true`、`formal_training=false`、`formal_dataset_reusable=false`。

## 六项工程开训标准的当前结论

| 标准 | 结论与证据边界 |
|---|---|
| 1 完整路径 | **部分通过**：合成数据完成九阶段；真实数据完成 AE/缓存/响应/pool/state，尚无受完整审计保护的真实 World + Actor/Critic 更新。 |
| 2 维度与角色 | **工程检查通过**：异维 h/z、角色隔离、全基因真实视图与不兼容身份拒绝。 |
| 3 梯度归属 | **工程检查通过**：各阶段参数差分、输入梯度、双 Critic 和冻结环境覆盖；不是训练效果证据。 |
| 4 动作与回报 | **工程检查通过**：合法联合候选归一、单步终止不自举、无时间选择时跳过、训练回报与代理评价区分。 |
| 5 保存与恢复 | **工程检查通过**：两个 Actor 阶段边界、缓存/辅助/校准/势函数、预算续训及 Actor-only 初始化测试。 |
| 6 资源与计时 | **部分通过**：真实 AE/CFM 的 128 样本×16 条件更新无 OOM；未完成正式 World 评分、batch=256 策略更新及完整真实数值审计。 |

现有 ODE 审计每成员用 256 个固定 V 样本比较 128/256/512/1024 步，阈值 `1e-3`。人口审计最多 8 个 V roots × 1,024 重复 × `[512,1024,2048]` 人口数，均值邻接差阈值 `.01`，共约 2,936 万个生成细胞评分。它超出本轮真实小批验证范围；没有减少样本、关闭真实门禁或伪造证书。PyCharm 正式入口会执行、计时并保存这些审计，审计未通过则不会进入 RL。

## 复审、Git 与未运行范围

独立代码审查曾发现显式复用的完成状态漏洞，已修复并以生产 `fit_epochs` 的暂停产物重现验证；正常完成的封装产物也已验证可脱离原 `last.pt` 独立复用。辅助准备预算缺口已补齐。审查记录在本地 `.omo/evidence/modular-pretraining-code-review.md`。

GitHub 私有仓库：[debug-zhuweijian/lung-apoptosis-rl](https://github.com/debug-zhuweijian/lung-apoptosis-rl)。修改前代码已单独提交为 `ef05915`；对应可恢复归档为 `.omo/backups/modular-20261007-164004/source-before.tgz`，SHA256 `87f0a6fca2140c53d2cf7713000fbb69c9ff1dad04843b84590ea8efa1c4e3c6`。源码、配置、测试、说明及数据治理 Markdown 可进入仓库；数据本体、权重、日志、缓存、实验产物、IDE 状态和凭证文件被忽略。

未启动正式长时间训练、完整五成员训练、DLPA/消融、新数据下载或旧路线对照；未修改冻结原始数据、旧权重、历史实验、Memory、PROJECT_MEMORY 或 Zotero。
