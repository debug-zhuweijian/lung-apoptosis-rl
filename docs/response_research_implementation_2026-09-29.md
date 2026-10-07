# 响应研究入口实施说明与回执（2026-09-29）

对应任务书：[Training_Pipeline_Optimization_Implementation_ClaudeCode_2026-09-29](G:/研究生科研文件/paperstudy/02-notes/Training_Pipeline_Optimization_Implementation_ClaudeCode_2026-09-29.md)。
本文交付**静态实现**：所有代码、配置与测试已写入真实调用路径；按用户此前要求，本轮**未运行**任何测试、训练、基准、AE 预训练或缓存生成。文中一切速度、精度、收益表述均为「待运行确认」。

## 1. PyCharm 入口与配置

| 动作 | 操作 |
|---|---|
| 快速路线比较（默认 BF16） | 运行根 `train_response_research.py`（参数留空；等价 `--config configs/response_research.yaml`） |
| 严格 FP32 配对对照 | `--config configs/response_research_fp32.yaml` |
| 限定臂/预算 | 编辑 `research.arms`（可选 full/cfm_only/cfm_then_full/one_step_then_full/condition_only/population_mean）或 `--budget-seconds 3600` |
| 潜空间阶段 | `--mode latent_ae` → `--mode latent_cache` → `--mode latent_cfm`（依赖顺序，见 §2） |
| 五成员批量（潜空间） | `--mode latent_members`（要求 `member_workers: 1`，单 GPU 进程） |
| 冻结权重 V 比较 | `--mode v_compare`＋`research.v_compare.weights: [{name, path}]` |
| 恢复 | `--resume <run_id>`；不允许改 mode/epochs/device |
| 工程冒烟 | `--smoke`（合成数据；不是研究证据） |

入口只开 T/fit 与 V/selection（`open_data(config, "response")`），不会启动注册套件、E/final_test 或下游 RL；V 比较报告 `automatic_promotion: false`。

## 2. 阶段依赖与输出

```
route_compare:  arms 顺序执行（每臂单成员单进程）→ v_comparison/<arm>/ + research-comparison.json
latent_ae:      T 拟合 SharedAutoencoder（双端全基因 MSE，每 epoch 一次优化步）
latent_cache:   冻结编码器 → <run>/latent_cache/{分片 npz, manifest.json（最后原子发布=唯一完成标记）}
latent_cfm:     读缓存 h → 潜空间 CFM（coupling_space: latent|gene 进身份）→ 可选 B4 精修
latent_members: 共享冻结编码器/缓存 + 五成员 vmap 批量（rng_policy: latent_member_batch_v1）
```

- 每个研究 run 目录：`experiments/<run_id>/`（research_report.json、research_state.json、active_config.yaml）、`checkpoints/<run_id>/`（branches|latent_ae|latent_cache|latent_cfm|latent_members）、`logs/<run_id>/`。
- `budget_seconds`：入口级墙钟预算；耗尽时当前臂强制保存并在检查点 `extra.budget_exhausted=true`，**不冒充收敛**（`checkpoint_completion` 仍判 TRAINING_NOT_COMPLETE），`--resume` 续跑；已耗预算累计在 research_state.json。

## 3. 逐项实施回执

新文件（均为真实调用路径，无仅存于文档的开关）：

| 文件 | 内容 | 静态审查要点 |
|---|---|---|
| `src/lung/engine/numerics.py` | PrecisionPolicy(fp32/bf16/tf32)、identity()、apply()、policy_autocast | 三模式互斥；legacy 路径 policy 未设置时 autocast 为 no-op |
| `src/lung/engine/timing.py` | PhaseTimer（host+CUDA event，flush 边界读）、H2DAccounting、ThroughputCounter | 不在算子级强制同步；等待与传输分离 |
| `src/lung/engine/response_research.py` | 独立入口编排：路线臂（复用 suite_jobs 语义与 optimization.candidate_overrides）、ResearchBudget、BranchMonitor、冻结权重 V 比较 | 臂恢复=成员 last.pt 存在即 resume，身份校验在 response 侧；比较前后哈希锁定权重 |
| `train_response_research.py` | 根入口 | 与 train.py 同构 |
| `configs/response_research.yaml`、`..._fp32.yaml` | BF16 默认＋严格 FP32 对照 | 两文件仅 precision.mode 不同，成对比较 |
| `src/lung/models/latent_staged.py` | SharedAutoencoder / LatentMember（B1–B4 模块拆分） | 解码器每成员独立、CFM 期冻结；population_mean 显式不在此模块提供 |
| `src/lung/engine/representation_cache.py` | h 缓存构建/加载（含对照条件细胞） | manifest 最后原子发布；data/encoder 身份不符即拒绝；分片哈希验证 |
| `src/lung/engine/latent_pipeline.py` | B1–B4 编排；coupling_space 进身份；V 自由生成评价 | CFM 主阶段无 ODE 反传；B4 用可微 Heun+成员解码器；缓存后路径不重复全基因编码 |
| `src/lung/engine/async_checkpoint.py` | BackgroundSaver（computed/snapshot/durable 三步计数） | 有界队列背压、失败升级、close 排空 |
| `src/lung/engine/member_batch.py` | 五成员单进程 vmap 批量 | 权重五套独立；梯度按成员切片回写；要求 member_workers=1 |
| `tests/test_response_research_v1.py` | §7.2 第 1/2/6/7 项相关契约（8 个测试） | 已编写未运行 |

修改文件（改动前备份于 `.omo/backups/research-impl-20260929/`）：

| 文件 | 改动 |
|---|---|
| `src/lung/engine/response_training.py` | condition_batch_mode：cfm 阶段与 condition_only 可合批（新 reason 串）；阶段感知 weight；预取缺 MMD 时按需补抽；分组 CFM 前向 autocast+.float() |
| `src/lung/engine/response.py` | train_response/_train_response_members 增 stop_check（预算耗尽→强制保存+break，不标收敛）；stage 感知预取 cfg；precision 进身份与 metadata；PhaseTimer 包裹 train_block/step/checkpoint/validation；CFM 前向（训练+验证）autocast |
| `src/lung/data/reader.py` | 新增公共 `role_cells()`（缓存遍历入口；复用既有私有缓存与角色检查） |
| `tests/test_response_optimization_v4.py` | 串行回退参数表 cfm→one_step（原行：`('compact_full_gene','observed','cfm','auto','serial_training_stage'),`）；新增 cfm/condition_only 分组正向断言 |

未改：`response_prefetch.py`（按 epoch 重建＋调用方 stage cfg 已满足语义）、`models/latent.py`（旧候选保留）、`checkpoint.py`（seed_all 不动，策略经 apply() 分离）、`suite_*`（旧套件零改动）。

## 4. 科学语义变更清单

1. **训练日程**：入口臂 full/cfm_only/cfm_then_full/one_step_then_full；cfm 阶段现在走合批路径（reason 串进 metadata），one_step 仍串行。
2. **表示空间**：潜空间分阶段（B1–B4）为**新函数族与训练目标**，不是 latent_flow 联合训练的等价实现；z 由冻结投影从缓存 h 现算。
3. **耦合空间**：`coupling_space: latent|gene` 显式配置并进入身份；gene 路径按抽取到的 cell_id 查 h，保持与全基因 OT 相同的配对语义。
4. **共享编码器**：五成员共享冻结表示，成员各自拥有速度网络/条件编码器/解码器/优化器/RNG；共享是否引入共同偏差由后续对照检验。
5. **精度**：BF16 只覆盖密集 CFM 前向（训练与 V 的 CFM 诊断）；OT 保持 FP64 Sinkhorn、时间特征 FP64、ODE/MMD/归约保持已登记精度。**未覆盖**：MMD 精修 ODE 的 BF16（B4 与全基因 MMD 路径仍 FP32）。FP32 对照配置成对提供。
6. **随机协议**：全基因臂沿用 `cfm_shared_mmd_separate_v1`；潜空间单成员 `latent_cfm_numpy_torch_separate_v1`；五成员 `latent_member_batch_v1`（per-member 显式 Generator；**不声明复现旧轨迹**）。
7. **population_mean 消融**：保持逐条件串行（不合批），避免跨条件均值混合；潜空间路径按 E(mean(x)) 原定义未实现，显式缺省。

## 5. 新旧身份与恢复

- 源码指纹变化（implementation_identity 含全部 src/lung/*.py）→ 注册套件按现行规则自动新建；旧套件/旧检查点保留为旧实现记录。
- response 检查点身份新增 `precision` 字段（legacy 配置为 None）——旧 last.pt 恢复会被身份校验拒绝，按项目规则开新 run。
- 潜空间各阶段身份绑定 data_identity＋缓存 manifest＋coupling_space＋成员序号；缓存身份不符不静默复用。跨身份复用 AE/成员权重作为初始化须另建实验并登记来源。

## 6. 已知限制与缺口（如实登记）

1. **未运行**：全部测试、训练、缓存生成、基准；「已编写」≠「已通过」。首跑预期需要修的典型点：vmap/torch.func 与本机版本兼容、Windows 下 BackgroundSaver 文件占用重试（atomic_save 已带）、合成数据与真实数据的形状差异。
2. AE 每 epoch 仅一步优化（梯度累积跨整个 epoch）；按条件分组步进留待需要时接入。
3. 五成员 v1 无每成员独立早停与精修解码（成员随预算/epoch 同停）；参数重堆叠每条件一次，成本未测。
4. BF16 未覆盖 MMD/ODE 路径（见 §4.5）；torch.compile 未接入（W4 编译边界仅在设计，未落码——如实标注）。
5. W5 数据供给多生产者未实现（沿用单线程预取；H2DAccounting 已备未接线）；旧 response 循环保存仍为同步（后台保存仅潜空间路径接入）。
6. latent OT 的耦合先验性质（未配对数据）继续按登记口径记录，不称真实配对。

## 7. 用户启动顺序（建议）

1. `python -m pytest tests/test_response_research_v1.py tests/test_response_optimization_v4.py -x`（先补运行证据；失败按报错回修）。
2. `--smoke --budget-seconds 120` 跑通 route_compare 全链（合成数据）。
3. 正式：`configs/response_research.yaml`（BF16）与 fp32 配对各跑 route_compare，或先只跑三臂 A1。
4. 潜空间线：latent_ae → latent_cache → latent_cfm（先单成员），V 指标对比 A1 赢家后再 latent_members。
5. E/final_test 与任何主模型晋级，留给路线定型后的单独决定。

## 8. 返修记录（对应 Codex 复核清单 R1–R7/C1–C3，2026-09-29 第二轮）

本节状态与 §3/§6 冲突时以本节为准。全部为「已修改并静态复核；测试已编写、**未运行**」。

| 项 | 状态 | 修改位置 | 闭环依据 | 测试代码 | 剩余缺口 |
|---|---|---|---|---|---|
| R1 | 已修改 | `models/latent_staged.py`（新增 `forward`，`object.__setattr__` 存共享引用）；`engine/member_batch.py` 重写 | 条件编码进入 functional 图（forward 内调用 condition_encoder）；堆叠仅含成员自身模块；梯度只回写 `velocity_model./condition_encoder.` 白名单，冻结 shared/decoder 零接触；组末各成员独立裁剪＋各一次 optimizer.step；损失无成员数除法；每组堆叠一次 | `test_member_batched_forward_matches_solo_member`（未运行） | 每成员独立早停与精修复仍缺（登记） |
| R2 | 已修改 | `engine/representation_cache.py` | `_tensor_digest` layout 感知（稀疏 COO 对 coalesce 后 indices/values 哈希，图不稠密化）；身份在编码循环前计算；`ensure_representation_cache` 身份匹配即复用、否则建 `<dir>-<hash>` 版本化目录，旧 manifest 不可能当新证据 | `test_encoder_identity_supports_sparse_graph_and_versioned_cache`（未运行） | — |
| R3 | 已修改 | `engine/latent_pipeline.py` | `draw_control_h` 精确镜像 reader 两级有放回（池抽→池内抽、保批内序）；`_couple` 的 latent 分支对双方冻结投影 z 求解 entropic OT 并按耦合索引配对 h/z；独立随机配对不提供任何别名 | `test_control_draw_mirrors_reader_two_level_sampling`、`test_latent_coupling_solves_ot_and_pairs_consistently`（未运行） | — |
| R4.1 | 已修改 | `engine/response.py` | budget 退出在 finally 后直接 break 本轮，含 `update_resume` 的中途快照不再被整轮快照覆盖，也不会误标 max_epochs | `test_budget_exhaustion_is_resumable_not_converged`（新增 `update_resume` 断言） | — |
| R4.2 | 已修改 | `latent_pipeline.py`、`member_batch.py` | 检查点 extra.resume 携带 epoch/组游标/keys/各 NumPy 与 Torch 流状态；恢复重建 keys 校验游标边界；五成员锁步协议（epoch/cursor/step 分歧即 `MEMBER_SNAPSHOT_DIVERGED`）；预算判定先于快照提交，flag 一次写对 | 同上预算测试＋（跨组恢复专测未写，登记） | 跨两组＋恢复的 C2/R4 专项测试待补 |
| R4.3 | 已修改 | `latent_pipeline.py` | 恢复按阶段（epoch0+1>switch）重建含 decoder 的优化器参数组后加载状态；`_EarlyStop` 带 state_dict/恢复/steps 计数，`min_updates` 以优化步数门槛生效；B4 切换显式 `stop.reset()` | 早停恢复专测未写，登记 | 同左 |
| R4.4 | 已修改 | `engine/response_research.py` | `--budget-seconds` 定义为**新的总额上限**：显式传入即覆盖已存 total、保留 consumed 并立即持久化；未传入回落已存值 | `test_budget_cli_total_overrides_saved_state`（未运行） | — |
| R5 | 已修改 | `response_research.py`、`latent_pipeline.py` | 每臂/每权重评价子目录 `mkdir`；`_load_for_comparison`→`load_latent_adapter`（校验 stage/data 身份，加载 shared+member 状态，暴露 evaluator 契约：condition_reference/encode_condition/sample/sample_noise）；`latent_stage_state.json` 持久化阶段产物，`_resolve_ae_checkpoint/_resolve_cache_dir` 按 配置→本 run 状态→默认路径 解析并报出全部尝试 | 适配器专测未写，登记 | `FrozenExpressionDiagnostics` 对适配器的字段需求未静态核验（expression_diagnostics 未读），列待运行确认 |
| R6 | 已修改 | `numerics.py`、`latent_pipeline.py`、`member_batch.py`、`response.py` | policy 挂到 shared/member；AE/CFM/成员密集前向 autocast＋`.float()`；`apply_seed` 在每个种子/恢复/验证边界重应用策略（含 response 成员 seed 与 evaluate_response 固定 seed） | 复用 precision identity 测试 | MMD/ODE 仍 FP32（原披露范围不变）；torch.compile 仍未接入 |
| R7 | 已修改 | `async_checkpoint.py` | `_to_cpu` 递归构造真正驻 CPU 的独立快照（含 optimizer 张量状态）；`close()` 先 None+join 写线程再传播原错误；durable_step 同组同号语义已写入 saver 与 member_batch 文档 | `test_background_saver_tracks_and_escalates`（断言位置已修） | 快照内存预算计量（构造中/排队）未接线，登记 |
| C1 | 已修改 | `response_research.py` | 总额按臂均分：每臂独立 `ResearchBudget`（`budget_<arm>.json`），臂耗尽只标记该臂、其余臂继续；比较报告 `budget_protocol: equal_share_per_arm_wall_clock` | — | 等墙钟≠等 FLOPs 表述沿用 §7.1 |
| C2 | 已修改 | `response_prefetch.py`、`response.py` | 新增 `discard()`（join 丢弃、不提交流状态）；更新组末检测阶段变化→discard＋按新阶段重设 prefetch.cfg，消除 MMD 流回滚 | 惰性补抽测试仍在；跨两组＋阶段切换＋恢复专测未写，登记 | 同左 |
| C3 | 已修改 | `tests/test_response_research_v1.py` | saver 断言包住 flush/close；新增成员堆叠等价、稀疏身份/版本化缓存、控制抽样镜像、潜空间 OT 真调用、CLI 预算扩容共 5 项真实调用链测试；全部未运行 | 本列即测试清单 | 运行证据待用户授权 |

§4 已披露缺口的更新：AE 已改为**每更新组一次优化步**（与响应训练同口径）；W6 每成员早停/精修、W5 多生产者供给、W4 torch.compile 仍为缺口；W4 GPU 区间已由 PhaseTimer 记录（cuda_seconds 进 summary），但响应日志目前只回写 host 项——GPU 区间持久化仍登记未完成。

## 9. 第二轮返修记录（S01–S11，2026-09-29 第三轮交付）

对应 Codex 二次复核清单；全部「已修改并静态核对；测试已编写、**未运行**」。§5 已确认项未回退。

| 项 | 状态 | 修改位置（函数） | 调用链依据 | 对应测试（仅编写） | 剩余缺口 |
|---|---|---|---|---|---|
| S01 | 已修改 | `member_batch._stacked_inputs`（解包序 `(h_reference, z_target, features)`） | `_couple` 返回契约→`_stacked_inputs`→`forward(z, h, …)` 参数序一致；参考=对照细胞 h、目标=处理细胞 z | `test_stacked_inputs_keep_reference_and_target_content`（同维内容断言＋d_model≠latent_dim 形状用例） | — |
| S02 | 已修改 | `latent_pipeline.run_latent_cfm`、`member_batch.train_latent_members`（`device=device, ae_path=…`） | 两入口显式关键字绑定；shared 由 `_load_shared` 内 `.to(device)` 迁移，不依赖成员 `.to()` | `test_entry_load_shared_binding`（两个真实入口捕获 kwargs） | — |
| S03 | 已修改 | `encoders.py` 四个 `forward_unchecked`、`latent_staged.forward(validate=)`、`member_batch._validate_stacked_inputs`＋批量调用 `validate=False` | vmap 域内无值驱动 Python 分支；全部校验（fingerprint/dose/duration/elapsed/horizon/context 范围）外置；单成员 `validate=True` 语义不变 | `test_unchecked_condition_encoder_matches_checked`、`test_validate_stacked_inputs_rejects_illegal`；合法 vmap 前向由成员堆叠测试覆盖 | nn.Embedding 查找在 vmap 下的兼容性待运行确认 |
| S04 | 已修改 | `latent_pipeline.run_latent_cache`（`cache.manifest`） | 加载器返回实例；指标、阶段状态、返回值三出口均读 manifest | `test_cache_manifest_is_the_only_completion_marker`＋R2 版本化缓存测试 | — |
| S05 | 已修改 | `latent_pipeline.run_latent_cfm`（恢复优化器=基础组＋`add_param_group(decoder)`；裁剪集合按 `requires_grad` 现算） | 恢复与运行时切换产生相同组结构；B4 裁剪含 decoder | 结构性专测未写（登记） | 跨 CFM→B4 恢复的运行级检查待补 |
| S06 | 已修改 | `latent_pipeline.run_latent_ae`（中途快照：epoch 不+1、组游标、keys、NumPy 流状态；恢复含游标） | 预算中途耗尽不再把部分 epoch 标为完成；恢复从游标后继续 | `test_ae_mid_epoch_resume_consumes_from_cursor`（假时钟驱动） | — |
| S07 | 已修改 | `response_research.run_route_compare`（臂训练 `try/finally` 双预算结算）、`main`（dispatch `finally`＋措辞如实） | 中断/异常路径均结算臂与父预算；强杀进程显式声明为进程外窗口 | `test_budget_settles_on_interrupt`（可控时钟） | — |
| S08 | 已修改 | `run_route_compare`（`run_status` 汇总各臂真实状态）、`main.monitor.close` 用之 | 各臂 budget_exhausted 不再被父预算误标 complete | 入口状态矩阵专测未写（登记） | 全部完成/部分/全部耗尽/扩容恢复的入口级检查待补 |
| S09 | 已修改 | `configs/response_research.yaml` 跨 run 示例注释（ae_checkpoint/cache_dir 逐步用法） | 每阶段独立 run_id＋显式路径传递；同 run 内仍自动解析 latent_stage_state | —（文档示例即交付物） | — |
| S10 | 已修改 | `async_checkpoint.submit(group=N)`＋完成计数、`member_batch` 传 `members_count`、status 暴露 `pending_group_jobs` | durable_step 仅在整组文件全部替换后推进；失败组永不标记 | `test_saver_group_durable_semantics`（部分/全部/失败三段） | — |
| S11 | 已修改 | `tests/test_response_research_v1.py`（模块级 `scipy.sparse` 导入；假缓存补 `.pool()`） | 替身接口与真实调用一致；本轮新增 7 项真实调用链/恢复检查 | 本列即测试清单 | 全部未运行 |

未回退确认：清单 §5 所列 R1–R7/C1–C3 已确认修复项未被本轮编辑触及相应语义（S 项修复均为增量或独立函数）。§6 性能与功能缺口清单继续有效，另补：S05/S08 的运行级专测、vmap-embedding 兼容性实测。
