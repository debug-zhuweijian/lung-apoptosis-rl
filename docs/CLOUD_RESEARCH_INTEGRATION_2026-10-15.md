# 云端研究成果本地整合与修复（2026-10-15）

状态：已在独立工作区整合已审GitHub分支及Pro最终离线补丁，修复三类审查问题，完成定向回归和真实数据复算。完整科研可行性仍未验收；未修改模型、奖励、冻结划分或原始数据，未启动正式训练。

## 版本与入口

本地分支：`codex/cloud-research-integration-20261015`。工作区：`I:/科研/肺癌细胞凋亡-强化学习/experiments/20261015-cloud-integration`。

| 来源 | 精确版本或交付身份 |
|---|---|
| 已合并的Dot #2–#4 | main `ec5be211874c6ccc64ce81b6238a689fc56fad12` |
| Dot #5 / #6，集成工作区起点包含两者 | #5 `1f92a8458d605ca9a68663e0fb0ef2cf6adac942`；#6 `682355f07516767b3ccdd393ca71681c7f593ce0` |
| 数据闭合 #1 | `0929b8e891a4648f5b1c5898222cb448c6de304e` |
| source-bridge-validation | `68596ecd89b58dc93ec4d9fea38ed33ef8e68d2e` |
| source-contracts | `34efa5cb5504db5c1e6e3ff2bd9f122d071d3486` |
| Pro最终8文件补丁 | 基于ec5be21；补丁SHA256 `b4f9d28643398a8ae3e2c03994870b6046536ff3d9669d392f682f3658a80149` |

Pro代码包来自用户提供的下载目录，46个登记文件哈希核验通过；对应数据包47个登记文件哈希核验通过。源码及数据原交付保留在执行项目的 `experiments/20261015-cloud-review/pro-final/`，不随源码提交。

为消除同名脚本冲突，同时保留双方行为：

- `scripts/recover_public_expression.py` 保留Dot的来源子命令、共用函数和五种来源。
- PR #1的源锁批量恢复入口改名为 `scripts/recover_public_expression_bundle.py`，相应调用说明和测试一起更新。
- `--reference-gene-axis` 核对实际参考文件的有序基因身份；`--gene-axis-manifest` 核对来源回执绑定的文件字节和数量。两者可同时使用，来源回执不自动证明旧训练资产或模型兼容。
- 共享映射保留所有原测量尺度；合并样本重复拒绝、空白/重复轴身份校验、PAR_Y版本后缀处理。单细胞计数、bulk计数、小数估计计数、TPM、FPKM及微阵列分别保存，不暗中换算。
- Pro的 `recover_history_gene_bridge.py` 和基因坐标映射模块已加入；`lung.data` 改为按需导入，现有TrainingData/prepare_training公共接口保留。

## 已修复的三类问题

1. 四个新导出器的矩阵按common_gene_index排序、ID却按CSV物理行序保存。现在统一导出顺序，新增8个乱序/已排序回归，联合核对ID、值、mask、样本及观测零。
2. 数据恢复依赖缺失。新增 [requirements-data-recovery.txt](../requirements-data-recovery.txt) 和 [附加环境说明](DATA_RECOVERY_ENVIRONMENT.md)，在独立目录验证13个附加包和10条生效依赖；原训练锁和study已安装环境不变。
3. NCI测试将缺失值硬编码为None，不能适配锁定pandas3.0.3。改为检查缺失位置，保留唯一有效映射值的精确断言；生产映射不因此放宽。

## 最终测试与复核

在README规定的 `F:/anaconda/envs/study/python.exe` 上启用隔离恢复依赖，运行最终56个相关测试文件：**439 passed in 21.23s**。涵盖各分支新增/修改测试、Pro新增测试和既有训练数据相邻消费者，不是全仓库/GPU/正式模型效果验收。

首次合入Pro后的同集合运行是438通过、1失败，失败为未修改的AE缓存目录rename触发Windows WinError5。相关9项单独重跑通过；随后使用新的短临时目录完整复跑439通过。原失败日志保留，未禁用断言、未修改训练缓存代码，也未证明该系统权限异常的根因。

证据位于执行项目的 `experiments/20261015-cloud-review/`：

- `integration-final-test-files.txt`：最终测试文件清单。
- `integration-final-tests.log`：首次运行及Windows权限错误。
- `integration-final-recheck.log/xml`：最终439项通过。
- `pr1_reviewer/recovery_environment_validation.json`、`recovery_cli_import_validation.json`：依赖、四格式读写及四CLI从仓库外执行--help结果。

独立复审检查了共享映射、51个已有CLI导入符号、跨入口轴回执、9份工作流YAML及20处脚本/测试路径，未发现需返修的集成错误；Pro坐标映射另经独立审查与28项专项检查。附加工作流保留原分支触发条件，本轮未触发远端运行。

## 真实数据复算

组合版本使用已有、已校验的真实原件运行，输出使用新目录，没有覆盖冻结输入：

| 来源 | 本次核验 |
|---|---|
| 公共Tahoe/TIDYI轴 | 368个明确A549细胞×32,461基因，与旧CI全数组相同；实际冻结轴ID及顺序核对通过；仍是单板原导出 |
| HM97662 | 18个源样本，包含三个背景，其中A549仅6个；原轴和共同轴数组与旧CI相同 |
| GSE161800 | 两张源计数表合并与公共轴投影，数组与旧CI相同 |
| Sarin | 22个RNA样本，12,070映射基因；源表达、投影及原功能值按既有合同恢复，数组与旧CI相同 |
| GSE96779原尺度 | 12个FPKM样本，保留四类历史；原尺度数组与旧CI相同 |
| GSE96779 Pro投影 | 从原始FPKM、SOFT、GTF及HGNC重算，17,978映射基因、215,736观测值、31,339观测零；两个NPZ共10个数组、映射台账和条件连接与Pro交付完全一致；导出基因ID及顺序与本机冻结轴相同 |

前五个恢复出口共核16个真实源文件、30个输出文件、41个NPZ数组。使用 `--reference-gene-axis` 的前四包明确记录实际参考身份通过；不将旧CI来源manifest混用为另一种字节回执。

记录：`integration-real-recovery/source-bridge-20261015T080956Z/verification.json`、同目录 `SUMMARY.md`、`integration-pro-history/manifest.json`、`integration-pro-history-comparison.json`。所有比较均包括样本/基因顺序和缺失语义，数值相同不代表不同测量技术可直接混训。

本机冻结轴SHA256：`791ea5bb47b7f09401e46c9dd2d1ab866c1c25c583f07ed3639e6c0165e1af5e`，运行前后不变。Pro原交付中的“冻结轴尚未核对”保留为历史记录；本轮新增核对结果由上述独立回执证明，没有改写原交付回执。

## 与研究架构对应及边界

依据毕业论文 `研究架构.md` 第3节与4.1–4.4节：此次推进的是数据身份、测量域、来源条件、合法投影和可复算入口。实际参考轴相同只证明坐标身份，不证明跨来源共享latent、预处理/角色兼容、旧检查点可用或预测准确。

状态增量、P独立响应效果、预测表达到实测G的效度、Q相对直接排序的收益、多步滚动及真实序贯效果仍需对应实验；`whole_project_qualified` 保持false。未用来源数量、测试数量或局部数组一致性替代整体科研验收。

当前history桥的实际输入是GRCh37.75与本次HGNC源文件，均有哈希。其报告中的固定注释标签不应推广到任意替代注释；今后更换注释需另行核对版本与映射，不沿用本次结果。

## 本地交付状态

源码在独立集成工作树，原主工作树 `I:/科研/肺癌细胞凋亡-强化学习` 的main和已有数据保持不变。本轮没有commit、push或远端PR合并，没有运行正式训练。新恢复进程按附加环境说明启用依赖即可；未改应用全局设置。

原始审查副本、失败日志和云端原交付均保留。提交/推送应以此完整组合树复核，不能只提交四个局部修复而遗漏被整合的新入口和测试。
