# 图1：数据如何协同进入算法

日期：2026-10-12。用途：当前算法数据协同结构图。

使用 engineering-figure-codex 的 concept 工作流及会话内 image_gen 生成；仅生成图稿，不启动任何科研模型或数据作业。

## 图稿状态口径

- 实线表示已实现的数据或预测入口，并不表示真实模型效果已验证。
- 橙色虚线表示训练或校准监督。
- 灰色点线表示待接通或待验证连接。
- Q 主图标记当前终点 ridge 诊断；正式 RL、多步滚动、真实序贯目标保留，并单独登记待验证。
- Martin 来源模型单列；Bothe、GSE306207 不绘为已经进入单细胞 P 或 C 的数据。
- Viel 功能路线标明 Tahoe 代理参考，保留表达到真实终点的迁移假设。

## 精确图像生成提示词

Create ONE polished, scientifically accurate Chinese research architecture/data-flow diagram. This is a conceptual figure, not a plot or a quantitative table. Use case: infographic-diagram. Asset type: thesis and group-meeting figure. The image should be a landscape canvas, preferably 3840 x 2400, with very legible Chinese typography, vector-like crisp lines, a white background, ample spacing between nodes, and a restrained navy / teal / blue / amber palette. No purple, gradients, photography, 3D, decorative molecules or stock icons. Put all Chinese labels in clear modern sans-serif. This is figure 1, emphasizing WHAT EACH DATA SOURCE SUPPLIES and HOW it is consumed by modules, rather than neural-network layer detail. Do not add numerical performance results.

TITLE (large): “图1｜数据如何协同进入算法”
SUBTITLE: “A549 状态条件化响应预测与干预筛选”
A small but readable subtitle badge: “当前数据职责与实现路径 · 新协议链真实模型效果待验证”

LAYOUT: Use a clear top-to-bottom structure in THREE horizontal bands:
Top band approx 24% height: “① 数据来源与监督职责” with four spacious cards, numbered A–D, left to right.
Middle band approx 48%: “② 主算法：状态 → 响应 → 功能评价 → 候选选择”. This is the dominant left-to-right flow, with one secondary branch beneath the expression prediction. Align arrows orthogonally and never pass through text.
Bottom band approx 20%: “③ 辅助来源与待验证连接”, compact horizontal source cards plus a separate pending validation box.
Leave remaining space for title, a short governance strip and a clear legend. Keep text concise and avoid oversized margins. You may adjust node placement to make the exact connections below readable.

TOP BAND cards, exact text:
A teal card:
“响应表达｜Tahoe / sci-Plex”
“A549 对照群体 ＋ 处理后群体”
“对照提供参考；处理后表达监督 P”
B teal card:
“状态标签｜TIDYI”
“同细胞表达 ＋ 原始分选标签”
“用于训练与评价 C”
C blue card:
“药物与实验协议”
“结构、剂量、暴露、恢复、观察终点”
“细胞背景 ＋ 有序处理历史 ＋ 缺失信息”
D amber card:
“实测功能终点”
“Viel：A549 Annexin”
“Macaya 等：按条件与检测定义核验”
“fit 监督 G；final 独立评价”

Under these cards put a thin shared governance strip, exact text:
“统一基因身份，保留表达域、单位与 mask；来源、重复和数据角色可追溯；预处理仅在 fit 拟合”
This strip is a rule, not a claim that every source can enter the same model.

MIDDLE BAND:
Create these nodes in a visually dominant horizontal flow, with sufficient width for readable labels:
R “参考表达群体”
small “同板对照／登记参考”
E “E｜表达编码”
small “状态表示 h”
P “P｜条件响应预测”
small “h ＋ 完整协议 ＋ 噪声”
D “D｜表达解码”
small “药后预测表达群体”
F “条件级特征汇总”
small “表达均值、方差 ＋ 协议”
small “可选：C 概率汇总”
G “G｜实测终点模型”
small “匹配定义下的 Annexin 预测”
Q “Q｜候选价值”
small “当前：终点 ridge 诊断”
O “候选排序与方案输出”
small “药物 ＋ 剂量 ＋ 暴露／恢复”
You may vertically stack F,G,Q,O in the right third to avoid excessive horizontal crowding. Preserve semantic ordering and arrows.

Exact solid blue data/inference arrows:
R → E → P → D → F
F → G
F → Q (this is essential: Q consumes the SAME feature information, not G as an input feature)
Q → O
Top card C → a compact box “完整协议编码／合法候选” → P
The same protocol/candidate box → F.
Top card A supplies R (solid teal arrow labelled “参考表达”).
Top card A has a separate orange dashed supervision arrow to P labelled “处理后表达分布”.
Near E and D, a compact orange dashed training note: “E/D 由真实表达重建监督”.
There is NO true same-cell before/after pairing. Put a readable small annotation beneath R/E/P: “对照与处理为未配对群体”.

C branch beneath D:
D → box Cnode “同一 E → C｜原始状态评分”
small “逐细胞评分后汇总”
Cnode → F labelled “可选特征”.
Top card B orange dashed arrow → Cnode labelled “表达＋原始标签监督”.
Put an amber micro-note beneath Cnode: “C 分数 ≠ 实测 Annexin 比例”.
This C path means predicted expression is restored to the same AE coordinate, re-encoded with the same E, then classified per cell. Do NOT draw direct classification of a bulk mean and do NOT draw C as Annexin ground truth.

G learning path:
Top card D → compact gate box “条件、来源与检测定义核验”.
From this gate use ORANGE DASHED arrow → G labelled “仅 fit 终点监督”.
From G use ORANGE DASHED arrow → Q labelled “冻结 G 的预测反馈”.
G feedback is model-predicted, not direct measured Q training labels.
Place a clearly readable amber note alongside G:
“Viel 使用 Tahoe 代理参考”
“表达→终点迁移仍待验证”
This avoids implying Viel has matched initial RNA.

Independent evaluation box near O, visually separated:
“独立评价”
“留出表达 → P”
“留出原始标签 → C”
“留出实测终点 → G／候选排序”
O solid arrow → this box labelled “冻结结果”.
Top card D or its gate has a separate solid dark-teal arrow to evaluation labelled “仅 final 评价”.
DO NOT draw any final-label arrow back to E,P,C,G,Q or training.
At the bottom of this evaluation box, “同候选域、同信息、同目标比较”.

BOTTOM BAND:
Three neutral / teal source cards, connected only to their stated source preparation outputs:
“Martin”
“真实 bulk 双表达域 ＋ 协议”
“来源专用 E_M / D_M”
small “重建入口已实现；共享潜空间待验证”
“Bothe”
“A549 撤药／再处理 RNA”
“计数 → 共同基因轴 → 来源观测”
small “完整数据包尚未闭合”
“GSE306207”
“A375 有限序贯 RNA”
“TPM → 共同基因轴”
small “向 A549 迁移待验证”
Group these three with a GRAY DOTTED connector to a gray dashed-outline node:
“共享表示／bulk P／历史迁移”
“尚未完整接通与验证”
Then an OPTIONAL gray dotted connector toward the main E/P area, labelled “待验证连接”, clearly not an already-working solid arrow.
Include a small separate note in this bottom band:
“caspase、生长与死亡指标分别建模；保留原实验定义”
And a separate gray dashed-outline target box, not an appendix:
“后续分层验收”
“正式 RL 接入 → 多步滚动 → 真实序贯”
“每层单独验证；前层通过不自动转移”
This is a retained research objective, not a claimed completed implementation. Do not draw an active solid feedback loop from O back to P.

LEGEND, readable:
solid blue line “数据／预测路径（已实现入口）”
orange dashed line “训练或校准监督”
gray dotted line “待接通或待验证”
Final short note: “工程接通不等于科学效果通过”

ACCURACY CONSTRAINTS:
- Keep E,D,P,C,G,Q as separate modules with the specified roles.
- Do not claim all sources are pooled into one expression matrix or all are A549.
- Do not turn TIDYI sorted class proportions into natural apoptosis rates.
- Do not treat raw bulk counts, TPM and single-cell log1p as interchangeable.
- Macaya is an endpoint evidence source; do not imply its bulk RNA is A549 or already paired.
- Do not imply source E/D reconstruction establishes cross-domain latent alignment.
- Do not label Q ridge diagnostic as completed RL.
- No clinical claims, no performance numbers, no invented trajectories, no biological sample count table.
- This is ONE coherent figure, not a prose report. Use concise labels, strong hierarchy, well-separated arrow routing, and enough font size to read at normal viewing scale.

## 本轮亲自读取的依据

| 图中关系 | 文件与位置 |
|---|---|
| 研究主问题、E/D/P/C/G、Q 与单次/序贯边界 | `G:/研究生科研文件/毕业论文/研究架构.md` §4.1–4.5、§5–6；SHA256 `34e08ddb0f48e035fc6b3ba4f6db4f9a172bb5e159a30bc9ca8c8ffff5fb188e` |
| E/D 真实表达重建 | `src/lung/models/latent_staged.py:30`，`reconstruction_loss` |
| 参考 h、完整协议 P、D 与物理表达出口 | `src/lung/engine/full_protocol_response.py:234`、`:261`、`:288` |
| 完整物理事件与时钟进入 P | `src/lung/models/full_protocol.py:129`、`:163` |
| 预测表达回到同一 E，再逐细胞 C | `src/lung/engine/raw_state_scoring.py:320` |
| P 均值/方差、条件和可选 C 特征 | `src/lung/engine/joint_selection_learning.py:260` |
| fit Y 拟合 G、冻结 G 反馈拟合终点诊断 Q | `src/lung/engine/joint_selection_learning.py:362` |
| Martin 观测域、mask 与专用 E/D | `src/lung/models/martin_source.py:83`、`:105`、`:132`；`docs/DATA_ALIGNMENT_MARTIN_SOURCE_ROUND22_2026-10-11.md` §2–4 |
| 完整协议与真实模型准备边界 | `docs/DATA_ALIGNMENT_PROTOCOL_CONSUMPTION_ROUND22_2026-10-11.md` |
| Viel 代理参考、实测终点角色及 Q 的诊断性质 | `docs/DATA_ALIGNMENT_JOINT_SELECTION_LEARNING_ROUND22_2026-10-11.md` |
| Bothe、GSE306207、功能与辅助来源职责 | `08-datasets/index.md` 本轮入口；`docs/DATA_ALIGNMENT_ROUND22_2026-10-11.md` |

上述 I 盘相对路径以 `I:/科研/肺癌细胞凋亡-强化学习` 为根。当前 HEAD 为 `7fcbd1f25679ce245a47da37c929d7d3d0c3e7f9`，大量当前实现为已有未提交改动；图稿以亲读的工作树文件为准，不声称这些内容都属于该提交。保留用户已有改动、架构、划分、奖励和停止安排。

## 成图与校对

最终文件为 `figure01_dataflow.png`；复制完整性、尺寸和当前源码绑定见 `figure01_receipt.json`。主代理查看首稿及修订成图，修正了首稿中辅助来源待接通箭头误指向 Q 的问题：删除该可选向上连接，在来源适配框中明确“待接通主模型 E/P，迁移待验证”。主链、C 逐细胞评分、G 的 fit 监督、G 对诊断 Q 的预测反馈、final 独立评价均保留。

本图是用于讲解当前数据分工的结构图，数据规模和模型性能未绘为结果。上方 A/B/C/D 为数据组索引；中间 E/P/D/C/G/Q 为模块符号。图中的 F 仅表示特征汇总，R 和 O 分别表示参考输入及输出，并未新增算法模块或改动源码。三种 P 状态输入消融、神经网络层结构和训练细节留待后续专图按需展开。

### 定向修订提示词

Edit this existing Chinese scientific data-flow diagram with ONE tightly scoped semantic correction. Preserve the entire image layout, all modules, all Chinese text, colors, typography, and every other arrow. In the bottom auxiliary-source band, there is a GRAY DOTTED connector rising from the box labelled 共享表示／bulk P／历史迁移 toward the Q｜候选价值 box above, with a nearby label 待验证连接. Remove ONLY that upward gray dotted connector and its nearby 待验证连接 label. Do not connect auxiliary sources directly to Q. Keep the gray dashed-outline 共享表示／bulk P／历史迁移 box and its incoming dotted source connections intact. Change ONLY the second line inside that box from 尚未完整接通与验证 to 待接通主模型 E/P，迁移待验证. Nothing else should change. Preserve all other text exactly, especially the main solid arrows F→G, F→Q, Q→O; G dashed feedback to Q; C state scoring; independent final evaluation; and all existing source roles. Render a clean high-resolution result.

