# 模块化预训练实施回执

当前启动方式见 [训练运行说明](训练运行说明.md)。2026-10-15 按用户要求删除所有训练断点恢复、暂停预算、自动复用已训练阶段和 TensorBoard；只保留从头训练、完整数据缓存复用和模型权重导出。同一次新运行的上游模型仍供后续阶段加载。默认阶段已精简为 AE/cache/P/pool/C/DA-AC，Phi 与完整数值审计不再强制前置；U 五折误差估计旁支与多层权重副本已删除。

下文是 **2026-10-07 的历史验证记录**，保留当时的命令、数字和限制；涉及已删除接口的命令不再适用，不是当前执行要求。

## 当时的验证

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
