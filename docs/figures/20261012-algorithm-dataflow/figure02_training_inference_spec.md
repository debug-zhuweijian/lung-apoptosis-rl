# 图2：算法计算与分阶段训练

日期：2026-10-12。接续图1，重点为当前单细胞主路径的计算、分阶段训练、参数冻结与独立评价。用户“继续”授权生成后续图稿，不恢复科研模型或数据作业。

## 图稿依据与状态

- `G:/研究生科研文件/毕业论文/研究架构.md` §4.3–4.5、§6：模块职责、冻结环境、候选价值和序贯边界。
- `src/lung/models/latent_staged.py:30`：E、线性投影 J、D 与双端全基因重建；`:117` 潜空间 CFM；`:136` 潜空间 Heun 采样；`:149` 解码。
- `src/lung/engine/latent_pipeline.py:141`：真实读取器路径的 OT 统计耦合及参考 h、target z；不代表同细胞前后配对。
- `src/lung/engine/full_protocol_response_learning.py:413`：P/条件编码/完整历史编码的更新参数；`:432` 可选 B4；`:453` 状态变体实际应用。
- `experiments/20261011-data-alignment-round22/protocol_consumption/reference_variants/templates/fullP_observed.yaml:61`：refine.enabled=false；三个变体按各自契约独立训练。
- `src/lung/engine/full_protocol_response.py:223`、`:245`、`:288`：reference 变体、先整群 h 再切块、主推理出口。
- `src/lung/models/full_protocol.py:129`：物理协议与历史编码；实际时间属于条件，生成 τ 为另一变量。
- `src/lung/models/flow.py:11`：Heun 从生成时间 0 到 1 积分，不读取真实目标表达。
- `src/lung/models/objectives.py:170`：固定观测坐标下的表达出口还原。
- `src/lung/models/heads.py:32`、`src/lung/engine/auxiliary.py:458`、`src/lung/engine/raw_state_scoring.py:199`：原始 C 的类别、加权交叉熵、selection 与独立温度校准。
- `src/lung/engine/raw_state_scoring.py:320`：先还原共同表达坐标，再同一 E、逐细胞 C 后汇总。
- `src/lung/engine/joint_selection_learning.py:260`、`:362`：真实 P 特征、可选 C、fit 终点 G、冻结 G 的预测反馈与单次 ridge Q。
- `docs/DATA_ALIGNMENT_REFERENCE_VARIANTS_ROUND22_2026-10-11.md`：三独立 P 实验和 mean(h) 的阶段允许群体含义。
- `docs/DATA_ALIGNMENT_JOINT_SELECTION_LEARNING_ROUND22_2026-10-11.md`：Viel 使用 Tahoe proxy、表达到实测终点迁移、固定角色、Q 诊断边界。

相对路径以 I:/科研/肺癌细胞凋亡-强化学习 为根。图中序号表示依赖层级；C 与 P 在同一冻结 AE 后具有独立训练入口，不声称当前单机同时执行。B4 为当前关闭的可选细化，正式 RL、多步滚动及真实序贯为保留的待接通/待验证研究目标。所有图稿来自当前源码的静态核查，不新增模型运行、指标或效果结论。

## 会话内 image_gen 生成提示词

Generate ONE Chinese scientific algorithm architecture diagram, figure 2, continuing a clean navy/teal/blue/amber research-figure visual identity. It must be a legible landscape illustration with white background, crisp vector-like borders, consistent modern Chinese sans-serif, short labels, aligned nodes, orthogonal arrows, and high resolution. Do not use photographs, 3D, gradients, decorative molecules, purple or performance plots. This is a conceptual architecture, not a numerical chart. Aim for a readable 16:10 landscape page with three horizontal bands. Use enough font size to read at normal viewing scale.

TITLE: “图2｜算法计算与分阶段训练”
SUBTITLE: “先学习表示与响应，再校准功能终点并评价候选”
Status badge: “本轮新协议链：实现已准备，真实模型效果待验证”

The diagram explains the CURRENT CODE and retains the broader research objective. It does not imply any new training was executed. Data-source detail is in figure 1, so avoid repeating large dataset cards.

BAND A (upper ~43%): “① 分阶段训练：明确谁更新、谁冻结”
Organize a dependency graph left-to-right with five positions; P and C are PARALLEL branches from the same frozen E, not a mandatory serial chain.

A1 big node at upper left: “① 表达表示预训练”
A miniature computation path inside:
“真实表达 x → E → h → 投影 J → z → D → 重建表达”
below: “对照与处理两端；全基因重建 MSE”
badge orange: “更新 E、J、D”
A1 solid arrow to A2:
“② 冻结表示并缓存”
“冻结 E 与 J；保存逐细胞 h”
“处理后表达得到监督目标 z”
badge blue: “共享同一完成版本”

From A2, branch to A3P (upper middle):
“③ P：完整协议响应训练”
“参考 h ＋ 条件 c → 潜空间速度网络”
“处理后 z 监督；潜空间 CFM”
badge orange: “更新 P 与条件／历史编码”
badge blue: “E、J、主阶段 D 冻结”
Within or immediately below A3P include three small parallel ablation chips, with exact labels:
“observed：逐细胞 h”
“zero_h：h = 0”
“mean_h：允许群体的 mean(h)”
a readable line: “三套 P 分别训练与选模”
a tiny but readable line: “mean(h) 先于切块计算；不是 E(mean(x))”
Show a small data-preparation annotation near P training:
“OT 统计耦合；无真实逐细胞前后配对”
An adjacent GRAY small optional box:
“可选 B4：表达 MMD 细化”
“当前配置关闭；启用时另核版本”
The optional box should not appear as an active completed training step.

From A2, a parallel branch to A3C (lower middle of top band):
“③ C：原始状态训练”
“冻结 E(TIDYI 表达) ＋ 原始标签”
“加权交叉熵；独立温度校准”
badge orange: “仅更新 C／校准温度”
badge blue: “E 冻结”
a small line: “保留三个原始分选类别”

Next position A4:
“④ 冻结预测并拟合 G”
“同一 P 的表达统计”
“＋ 完整协议 ＋ 可选 C 概率”
“匹配的 fit Annexin → G”
badge orange: “更新 G（岭回归）”
badge blue: “E／P／D／C 冻结”
Inputs to A4: A3P and optional A3C with clear dependency arrows. The arrows mean prediction features, not G gradients updating P/C. A small caption “资格核验后产生真实预测工件”.
A4 can be somewhat wide; do not overpack tiny labels.

Final top position A5:
“⑤ 候选价值诊断 Q”
“同一特征 → Q”
“冻结 G 的预测反馈作监督”
badge orange: “仅拟合 Q（终点 ridge）”
a prominent amber line: “当前诊断入口；正式 RL 待接通”
Use ORANGE DASHED arrow A4→A5 labelled “预测反馈”.
Do not draw measured Annexin directly training Q, and do not label this ridge Q as an Actor-Critic or RL implementation.

One thin strip along the bottom of this training band:
“fit 用于拟合；selection 用于选模；C 另用 calibration 温度校准；final 仅在冻结后评价”

BAND B (middle ~37%): “② 固定模型后的计算：对每个合法候选独立推演”
Show the inference architecture as a strong clear left-to-right flow with two conditioning inputs above the P box and a C branch below the decoded expression.

Input S1: “参考表达群体”
S1 → box “冻结 E” → “参考状态 h”
Input S2: “完整候选”
small “药物结构／剂量／暴露／恢复／历史”
S2 → box “条件与历史编码” → “条件 c”

Both h and c feed a larger central P box:
“P｜条件潜空间生成”
mini path inside “噪声 ε → ODE / Heun → 预测潜变量”
small formula-style label “速度网络 v(z, τ, h, c)”
No need for a dense CFM equation.
P → node D:
“D ＋ 表达出口还原”
small “预测表达群体 ŷ”
small “共同 log1p 表达域”
Near P put a readable amber clarification:
“τ 是生成内部时间”
“药物暴露时长属于条件 c”
Noise is latent-space noise; do not depict adding noise to the original genes.

D → box F:
“条件级特征”
“表达均值／方差 ＋ 协议”
“可选：原始状态概率均值”
D has a secondary downward branch:
“坐标还原 → 同一 E → C”
small “先逐细胞评分，再汇总”
This branch → F with label “可选 C”.
Do NOT classify the mean expression; do NOT let C consume P latent directly.
Near C: “C 为分选状态评分，非 Annexin 实测值”.

F forks with TWO independent solid arrows:
F → “G：预测实验定义的终点”
F → “Q：候选价值”
Q → final node “全部合法候选排序”
small “输出药物、剂量与暴露／恢复方案”
G may connect to a small output “功能预测报告”, but do not draw G→Q as a runtime input. G supervised Q during the training band only.
No gradient/update arrows in the inference band. Add a short badge “此阶段参数全部固定”.
Near F/G include a small amber note “Viel 采用 Tahoe 代理参考；迁移假设仍待验证”.

BAND C (bottom ~13%): “③ 验证与仍需接通的研究目标”
Two spacious boxes side by side:

Left blue outlined box:
“独立验证”
“表达：状态增量与未见条件”
“原始 C：分类与校准”
“功能与选择：留出实测终点”
small “同候选域、同信息、同目标；与简单基线比较”
The held-out labels are only evaluation targets and never flow back into training.

Right gray dashed-outline box:
“保留的完整研究目标”
“正式 RL 价值／策略 → 多步滚动 → 真实序贯”
“冻结环境；策略／价值单独更新”
small “接入、数据覆盖与独立效果逐层验收”
small “单次终点诊断不代表正式 RL 或序贯通过”
Do not render this future box as already connected with a solid feedback loop.

LEGEND at bottom:
orange fill “本阶段更新”
blue fill “本阶段冻结”
gray dashed outline “可选或待接通”
Orange dashed arrow “监督／训练反馈”
Also a concise note: “结构与工程验收不等于科学效果验证”

Semantic constraints:
- J is the existing linear latent projection, not a newly invented module.
- Main P training freezes E,J and the initialized decoder; optional B4 is OFF in current configuration.
- OT is statistical population coupling, not measured cell trajectories.
- The three P ablations are independently fitted, not three settings applied to one trained checkpoint.
- zero_h removes reference state only from P input; do not say all RNA is removed from the learning process.
- mean_h is mean of h for the permitted reference set, not E of averaged expression.
- C fits original sorting labels and has separate calibration; it is not an Annexin classifier by clinical definition.
- G is supervised only by compatible fit measured endpoints; final labels are not training feedback.
- Current Q is terminal ridge diagnostic based on frozen G simulated feedback; formal RL remains an explicit retained target.
- This figure concerns the single-cell main route; do not silently splice Martin/bulk source models into it.
- No performance numbers, invented biological results, arbitrary numeric architectures, or clinical outcome claims.
Keep the figure polished and readable; omit decorative elements rather than reducing text below legibility.

