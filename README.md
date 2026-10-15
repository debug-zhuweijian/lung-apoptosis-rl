# 肺癌细胞凋亡—强化学习

2026-10-15 本地整合验收：已整合Dot #5/#6、数据闭合#1、两条来源验证分支及Pro最终离线补丁，修复基因ID错列、恢复依赖和pandas缺失值测试问题。最终56份定向测试439项通过；公共轴/HM/pulse/Sarin及GSE96779真实恢复复算通过，有序基因身份与本机冻结轴一致。模型兼容性、独立功能与序贯收益仍未通过。见[整合与验证记录](docs/CLOUD_RESEARCH_INTEGRATION_2026-10-15.md)；下方云端记录保留原时点结论。

2026-10-15 当前 Chat 云端数据接续：GSE96779 的 12 个真实 FPKM 样本经 hg19 符号/坐标证据投影，17,978 基因、215,736 原值逐值核验；另外三来源投影与 TIDYI 原计数重新读取。修复源内读取被训练依赖提前阻断的问题，85 项便携专项测试通过；扩展 Bothe 测试仍有未上传元数据导致的 15 项错误，不声称全测通过。公共轴未认证为旧冻结轴，完整功能与序贯链仍未闭合。见[本轮主要报告与复算入口](docs/REAL_HISTORY_ALIGNMENT_2026-10-15.md)。

2026-10-14 第二阶段：PR #2 已按用户追加授权合并；进一步恢复 DAMTC 的真实表达及条件功能参考、实读 WAVE 原文/补充/源数据，并完成 sci-Plex2 全部24,262细胞的源值核对和独立剂量CPU诊断。59项针对性检查通过；最高剂量低细胞数、来源溶剂/单位及功能监督断点仍明确保留，详见同一[主报告§11](docs/DOT_RESEARCH_DATA_FEASIBILITY_2026-10-14.md#11-持续研究补充功能证据与新近方法2026-10-14-第二阶段)。这不是整体科研验收。

2026-10-14 Dot 云端独立研究：已从公开原件恢复 Sarin、HM97662、GSE96779、GSE161800 四组源尺度表达与样本/协议对应，显式处理估计计数、共处理历史及 GEO/论文剂量单位冲突。49 项针对性检查通过；未启动正式训练，冻结共同轴、状态增量、功能迁移及真实序贯的完整验收仍未闭合。方法、数据与论文论证架构比较见[本轮主研究报告](docs/DOT_RESEARCH_DATA_FEASIBILITY_2026-10-14.md)，实际执行与来源哈希见[证据清单](docs/DOT_RESEARCH_EVIDENCE_2026-10-14.json)。以下为原时点记录，不代表本轮重新执行。

2026-10-11 第二十二轮：21:41实读Bothe **12/12双端官方完整性闭合、5/12完整FASTQ配对检查、4/12真实计数**；四个闭合样本共同基因轴投影已逐值读回。22:19核对count/finalizer/storage/observation四个后台实例的实际PID、创建时刻、运行路径、命令和批准源码，下载支路已完成退出。全量计数、发布及bulk观察包尚待完成。用户最新要求“先做到这里”，随后明确“把这个阶段做完吧”：当前仅完成已启动Bothe数据阶段，其余新科研工作暂停，模型保持停止，阶段完成后收尾。见[接续与停止条件](experiments/20261011-data-alignment-round22/CONTINUATION_HANDOFF_2026-10-11.md)。

Martin双域真实表达与来源E/D/final、三独立P输入合同、原367条件native final/三结果比较、原C及联合G/Q已完成相应数据预检、工程验证和专项复审，真实模型效果未测。新GSE306207已读取8个A375序贯RNA样本并精确投影32,461基因，两药名义结构已核对；不能据此授予A549或凋亡功能资格。原划分只有化学实体留出；状态增量仍含板/重复混杂，独立剂量/时间方案停在未批准草稿。CellCage实际配对矩阵仍未取得。共享latent/bulk P、表达—实测功能连接、正式RL同链及序贯功能仍是完整目标缺口。最新依据见[第二十二轮综合验收](docs/DATA_ALIGNMENT_ROUND22_2026-10-11.md)、[待授权模型运行方案](docs/MODEL_EXECUTION_PLAN_ROUND22_2026-10-11.md)和[本轮实际证据](experiments/20261011-data-alignment-round22/)；下方保留原时点记录。

2026-10-11 第二十一轮：Martin60个真实RNA双域2365440原值、joint mask与LOBO/cycle读取完成；16组/40名义RNA协议事件已与RNA/角色/FP/unknown联动。完整协议32身份保留Viel20/Nav8/Bothe4，Viel20实参可构造，24不同观测ID不代表24统计独立实验。88输入tests及四项独立审查通过；8fold bulk观测坐标已实际准备、root读回及独立关闭。pilot R1整端官方MD5/gzip已通过，23:49:05UTC root61714续传R2、73351恢复0296、5491remaining与10921countwaiter非终止；1277/92729/63888已终止，旧快照保留。真实counts仍0、训练停止；新FOXN3/GATOR1查证未补齐A549 RNA—Annexin关键矩阵。bulk出口/fullproducer/参数与A/B/C/D同链未验收。见[第二十一轮](docs/DATA_ALIGNMENT_ROUND21_2026-10-11.md)与[接续](experiments/20261011-data-alignment-round21/CONTINUATION_HANDOFF_2026-10-11.md)，下方旧轮次为当时快照。

第二十轮最新恢复：先导28793达到声明长度但MD5及gzip CRC失败，已终止并保留错误副本；原登记路径从零重试54677已启动，5491剩余获取与10921计数等待继续，21:55:50 UTC root工具核三者非终止。真实counts/生产manifest/投影仍未完成，模型训练保持停止。[故障与恢复记录](experiments/20261011-data-alignment-round20/pilot_md5_diagnosis/README.md)保留官方复核、失败完整性和保留操作证据；不改校验值或将坏副本交给计数。

第二十轮工程与受限数据补充：HM两条近似Annexin组均值已按来源与两个RNA group连接，未扩为六Y，角色未分配；Bothe官方ID桥17,107可测唯一映射及受控count→joint整数/mask消费接口已完成。131相关tests、源/metadata、文档链接与git diff-check通过。完整数据可行性与模型效果未验收。[完整第二十轮报告](docs/DATA_ALIGNMENT_ROUND20_2026-10-11.md)保留各阶段边界，下方101及十九轮进程描述是较早子阶段快照。

2026-10-11 第二十轮：新增 HM97662 同来源 A549／1 μM／6天 RNA—Annexin 条件桥，真实六列表达已按 Ensembl 精确对齐 30,268／32,461 基因，缺失2,193保留NaN/mask；不补同细胞、逐重复或完整flow协议。功能metadata v2实际核28候选／24Y身份／14标准P请求／20完整物理候选及480角色记录，仍无P/G/Q产物。真实FASTQ与受控RNA→counts入口独立复审关闭，101项联测通过；root已启动12run顺序计数任务，等待各run正式下载回执和双完整gzip后才进入实际验证/比对/count。真实Bothe counts仍0，模型训练继续停止，完整A/B/C/D与同链可行性未验收。详见[第二十轮报告](docs/DATA_ALIGNMENT_ROUND20_2026-10-11.md)和[三个数据句柄接续](experiments/20261011-data-alignment-round20/CONTINUATION_HANDOFF_2026-10-11.md)。

2026-10-11 数据对齐最新进展：新完整协议v3保持24候选/70事件，四Dex事件补齐名义结构，known FP20→24，旧v2仍可读。严格双端过滤与count矩阵接口独立复审关闭异常输入、文件发布及Windows注释编码问题；完整human hg19 index上10对参考DNA人工read实际跑通比对/filter/207,555-feature计数，预定坐标与count oracle一致。主代理138检查通过（127联测+11追加完成验收）；新增只读完成验收逐端重核文件/回执，旧下载协调器complete摘要不授独立完成资格。真实RNA先导与其余11run的两worker下载正在推进，真实counts尚未恢复；source出口/base kwargs/fullproducer、匹配权重、RNA—Annexin连接和整链科学评价仍pending，未恢复模型训练/真实推理。详见[第十九轮报告](docs/DATA_ALIGNMENT_ROUND19_2026-10-11.md)、[真实下载接续](experiments/20261011-data-alignment-round19/CONTINUATION_HANDOFF_2026-10-11.md)及[当前状态](docs/DATA_ALIGNMENT_STATUS_2026-10-10.md)。

2026-10-07 模块化入口：PyCharm 使用 `F:/anaconda/envs/study/python.exe`，运行根目录 [train_modular_research.py](train_modular_research.py)，参数留空，读取 [configs/modular_research.yaml](configs/modular_research.yaml)。按独立 AE、h 缓存、潜空间响应、pool/state、必要审计、固定环境、势函数和同一个 Actor 的 drug→dose→joint 日程运行；默认单成员、单 seed、`run_ablations: false`。完整说明、断点续训及实际验收范围见[模块化实施回执](docs/modular_research_implementation.md)。正式训练不设墙钟时间上限。旧的一次性训练路线已停止作为当前主线；下文旧入口说明保留作历史记录。

2026-10-08 AE 数据供给已接入逐细胞固定表达缓存、整组读取、只读映射复用和后台预取。118 项相关回归通过；真实完整16条件更新由13.42秒降至3.75秒，损失、参数、优化器和随机状态精确一致。已为原3431次更新、第84轮448/644进度登记保留断点的优化续训；缓存预建入口和验证范围见 [AE数据供给优化与续训](docs/AE数据供给优化与续训_2026-10-08.md)。

2026-10-08 后续执行：CFM 续训指标恢复已修正，数值审计接入固定布局的批量推演。生产 AE 两完整轮包含验证和保存，实测约 4.57 分钟/轮；真实数据驱动的隔离 World/Actor/双 Critic 梯度先导已运行，正式审计资格仍未取得。测试模块整理为 66 个，当前续训登记为 `20261008-training-followup-continuation`。完整结果、留出基线比较及边界见 [训练验证与问题修复](docs/训练验证与问题修复_2026-10-08.md)。

当前已通过合成全链和有界真实全基因组件验证；**完整真实数值审计与受审计 World 中的 Actor/Critic 更新尚未完成，六项工程开训标准未全部通过**。新入口保留审计门禁，正式长时间训练由用户启动。代码仓库为 [GitHub 公开仓库](https://github.com/debug-zhuweijian/lung-apoptosis-rl)，数据、权重、缓存和实验运行产物留在本地。

2026-09-28 本轮并行调整：响应阶段默认使用两个独立进程同时训练成员，每成员四条件合批、每十六条件更新一次；同组表达请求合并读取。`member_workers` 可设为 5，计算合批数量也已取消固定八条件上限。按用户要求，本轮没有运行测试、性能基准或正式训练，速度和峰值内存尚未验证。源码与配置变化后，原入口会按版本规则新建套件，旧检查点保留。配置、日志和启动说明见 [响应成员并行训练](docs/response_parallel_2026-09-28.md)。

2026-09-28 数据读取修复：合并同一次读取中的重复路径解析和源检查，保留完整性校验。真实四条件重复读取由 13.23 s 降至 9.53 s，属于数据准备的小样本计时，尚非整轮训练提速结论；配置保持 16 条件更新、8 条件合批和预取开启。验证与检查点续训说明见 [GPU 等待与数据读取优化](docs/GPU等待与数据读取优化_2026-09-28.md)。

2026-09-28 八条件更新：默认 `condition_microbatch: 8`，启用单组CPU预取与有界元数据缓存。后续已取消预取对更新组最多8条件的固定限制；当前用户设置 `conditions_per_update: 16 / condition_microbatch: 8 / prefetch_group: true`。此前8条件更新配置的进度仍保留，但配置或源码变化后不自动作为同一次训练继续。最初八条件修改按用户要求未运行训练、性能基准或显存压力检查。参数含义、YAML修改方法及恢复边界见 [YAML调参说明与八条件合批](docs/YAML调参说明与八条件合批_2026-09-28.md)。

2026-09-28 最新训练优化：直接运行根 `train.py` 仍执行 19 项流程及独立 DA-AC/DLPA。数据使用无损 CSR 持久缓存，紧凑模型默认完整图反传；`1000 epoch` 作为上限，有合法验证集的阶段可按收敛早停，并记录实际完成量。24–48h 不是硬配额。另有分阶段训练、潜空间模型及合法共享24h预训练的独立候选，未自动替换主方法。配置、恢复、验证与候选启用方式见 [训练优化实现与使用](docs/训练优化实现与使用_2026-09-28.md)。此前耗时记录属于相应旧版本，不是新版整套训练 ETA。

主工作目录：`I:\科研\肺癌细胞凋亡-强化学习`。此目录从 2026-09-18 起用于本课题的数据准备、代码、训练、实验和测试。

毕业论文主线、研究架构与正式正文统一进入 [毕业论文项目](G:/研究生科研文件/毕业论文/README.md)，先读其[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)。论文阅读、历史方法讨论和组会资料继续放在 [paperstudy](G:/研究生科研文件/paperstudy/README.md)。执行时以本目录的代码、数据索引和实验记录为准；G 盘旧数据与脚本保留为历史副本。

2026-09-27 一键入口：PyCharm 选择 `F:/anaconda/envs/study/python.exe`，直接运行根 [train.py](train.py)，参数留空。默认 `research_all / PROXY-A549-V1 / both / 1000 epoch / cuda:0`，依次执行响应模型及状态输入对照、独立 E 评测、状态损失对照、辅助链、独立 DA-AC/DLPA 比较，以及 sci-Plex 时间分支，共 19 个任务。当前接口与验收见[架构评测修复与 GPU 验证](docs/架构评测修复与GPU执行验证_2026-09-27.md)；此前[一键训练运行说明](docs/一键训练运行说明_2026-09-26.md)和[实施契约](docs/一键研究流水线_实施契约_2026-09-26.md)保留为 15 节点版本记录。匹配配置下自动恢复未完成套件，源码变更后建立新套件，只有全部任务成功才标记完成。正式真实训练由用户启动，工程微型测试不作为方法效果证据。

2026-09-28 响应训练更新：默认启用 `compact_full_gene` 紧凑全基因骨干，保留全基因 CFM＋MMD、完整 ODE 和独立 DA-AC/DLPA 路线。真实 T 数据的一个 8 条件参数更新实测约 49 秒；Tahoe 和时间响应训练均在每次参数更新后保存。网络定义、运行配置、恢复方式、精度待验证边界和耗时证据见[紧凑骨干与更新级恢复](docs/紧凑全基因响应骨干与更新级恢复_2026-09-28.md)。这是新的模型候选，旧 `gene_token` 骨干保留作独立对照；此前[响应计算与进度优化](docs/响应模型计算与进度优化_2026-09-27.md)记录的是旧骨干的执行优化。

单独先导仍可使用对应 `--config` 参数，见[先导实验运行协议](docs/先导实验运行协议_2026-09-26.md)及此前[先导执行记录](G:/研究生科研文件/毕业论文/先导实现与诊断复跑进展_2026-09-26.md)。[test.py](test.py)提供冻结模型的独立评估。历史单阶段操作见[训练运行说明](docs/训练运行说明.md)，旧代码与规格对应见[架构实现与验收](docs/架构实现与验收.md)，环境见[configs/runtime.json](configs/runtime.json)。一键套件的执行行为以上述新说明为准。

GPU预处理继续使用CUDA双精度统计、分块基因图及距离计算，世界模型推演保留GPU张量。缓存可接续，准备阶段显示独立进度；[GPU优化记录](docs/GPU执行优化_2026-09-24.md)中的4096查询分块和两细胞微批属于旧骨干配置，当前响应默认配置以上述2026-09-28说明为准。

2026-09-26 补充：Ridge 的 `T/V full unique cells` 已改为 GPU 稀疏归一化与汇总，配合单线程有界预取，省去 CPU 稠密中间表达。8,192 行真实训练子集上，512 批量的汇总中位耗时由 4.77 s 降至 0.68 s；范围、精度与重启说明见 [Ridge GPU 优化记录](docs/Ridge数据汇总GPU优化_2026-09-26.md)。

## 目录

```text
肺癌细胞凋亡-强化学习/
├── AGENTS.md       项目规则与研究边界
├── 08-datasets/    官方原始数据、版本、处理子集与数据说明
├── scripts/        下载、预处理、验证及实验入口
├── src/            模型和可复用研究代码
├── configs/        实验配置
├── experiments/    新实验协议、划分、指标和结果
├── tests/          数据与实现测试
├── checkpoints/    模型权重
├── logs/           下载和执行日志
└── docs/           数据准备、复制核验及复现说明
```

保留 `08-datasets` 名称，使复制的现有脚本无需改动相对路径；已冻结的脚本和历史实验协议哈希保持原值。新实验使用新版本，不能覆盖旧结果后继续称为同一次实验。

## 数据入口与当前边界

首先阅读 [数据索引](08-datasets/index.md)。现有来源包括 Tahoe-100M、TIDYI、GSE161218／Macaya 的表达与凋亡结局、Hu 2025 真实序贯候选及 A549 共同基因接口。

初始资料曾复制核验：209 个文件、5,474,180,897 字节，见[历史复制报告](docs/DATA_COPY_VERIFICATION_2026-09-18.md)。Tahoe 已于 2026-09-19 完成全部 3,388 个源分片、337.64 GB 的下载及严格校验，取得 2,567,838 个已发布 A549 表达；范围和审计判据见[下载说明](docs/TAHOE_A549_DOWNLOAD.md)。

当前数据分为两个版本入口：[数据筛选 v1](08-datasets/LungCancer_A549_Integration/selection-v1/README.md)记录保留、暂缓、排除和去重结果；[统一数据包说明](docs/DATA_PREPARED_V1.md)说明整理后的表达索引、药物条件、原始状态标签、终点、共同基因坐标及读取方法。正式产物位于 `08-datasets/LungCancer_A549_Integration/prepared/v1/`，以其 `manifest.json` 的 complete 状态及 `data_checks.json` 为整理完成依据。

当前 A549 可用视图含 2,558,270 个 Tahoe 细胞及 743 个 TIDYI 细胞。其他肺癌背景的扩展来源继续保留。此前筛选仅清理一个可从保留归档恢复的 TIDYI 解压副本，细节和哈希见筛选版清理报告。数据整理不表示完整模型已经实现或训练。

本项目服务于已确认的细胞状态条件化凋亡响应预测与干预筛选主线，DA-AC 与 DLPA 保留为待比较的方法路线。已有 TIDYI 同板／跨板状态分类诊断，尚未完成完整响应模型或两种决策算法的训练。状态分数、实验定义的凋亡终点与真实序贯效果分别验证；数据下载完成不能替代方法有效性验证。

## 运行约定

在本目录运行。当前统一读取入口是 `src/lung_data.py` 的 `PreparedLungData`，使用示例见[数据说明](docs/DATA_PREPARED_V1.md)。构建器 `scripts/prepare_lung_dataset_v1.py` 拒绝覆盖已有版本；早期 `build_lung_a549_panel.py` 属于历史接口生成流程。准备脚本根据自身路径定位数据；不要从 G 盘历史脚本继续产生本课题的新结果。

冻结的 TIDYI 基线重跑前会核对协议与脚本哈希。来源文件及历史结果中的原始路径、下载时间和执行时间保留为溯源信息，不表示新项目仍写入旧目录。

数据本体、下载缓存、模型权重和日志不进入 Git/Git LFS。本目录已纳入公开代码仓库；当前工作树的源码、配置与文档改动按实际范围审查，提交和推送分别按用户指示执行。

## 环境重建与定向验证

当前本机解释器为 `F:/anaconda/envs/study/python.exe`。直接依赖见 `requirements-runtime.txt`，Windows x64／Python 3.13.14 的 44 包依赖版本闭包见 [requirements-lock.txt](requirements-lock.txt)，包含 filelock 和性能测量使用的 psutil。安装使用官方 CUDA 13.0 wheel 索引；`configs/environment-study.lock.json` 保留为原环境快照。移机时在 `configs/runtime.json` 指向新解释器；版本锁尚未做干净环境安装和 wheel 哈希验证。

现有环境不需要重装。新环境可用 `python -m pip install -r requirements-lock.txt`；固定版本安装的说明见 [pip 官方文档](https://pip.pypa.io/en/stable/topics/repeatable-installs/)。

公共来源的数据恢复另需 [requirements-data-recovery.txt](requirements-data-recovery.txt)；按[数据恢复附加环境](docs/DATA_RECOVERY_ENVIRONMENT.md)安装到独立目录并仅为恢复进程启用，保留现有训练环境与版本锁。

本轮测试整合后，AE 与表达缓存场景统一位于 `tests/test_ae_data_supply.py`；辅助预算和势函数预算共享局部启用的 CPU fixture。日常修改只运行相应子集，历史套件入口回归用于修改旧入口时；`test.py` 仍是冻结模型的独立评估入口。当前修复与真实运行记录见 [训练验证与问题修复](docs/训练验证与问题修复_2026-10-08.md)。
