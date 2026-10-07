# 本项目数据索引

主目录：`I:\科研\肺癌细胞凋亡-强化学习\08-datasets`。与 paperstudy 的历史副本区分；后续新增下载、处理和验证只在此项目产生。

| 数据 | 说明入口 | 本次准备范围 |
|---|---|---|
| Tahoe-100M | [Tahoe-100M/README.md](Tahoe-100M/README.md) | 全部 3,388 源分片及 2,567,838 个 A549 表达已通过最终严格审计；筛选 v1 保留 2,558,270 个可用 A549 细胞，缺结构条件暂缓，其他肺癌源数据保留 |
| TIDYI A549 | [TIDYI/E-MTAB-16354/README.md](TIDYI/E-MTAB-16354/README.md) | 官方 CSV、完整作者归档、两板 743 个唯一 A549 子集和历史诊断结果；可恢复的 human_ss3.h5ad 解压副本已清理，归档成员仍保留 |
| GSE161218／Macaya | [GSE161218/README.md](GSE161218/README.md) | 9 个 bulk count 文件、官方补充材料、两块 Annexin 原始数值及匹配表 |
| Hu 2025 序贯候选 | [Hu2025_Sequential/README.md](Hu2025_Sequential/README.md) | 已公开原文与补充附件；逐重复凋亡原始数据和配套表达仍未取得，不能标记完整训练集 |
| sci-Plex3 A549 剂量—时间候选 | [sciPlex3/GSM4150378/README.md](sciPlex3/GSM4150378/README.md) | 完整官方表达已下载并校验；244,281个A549表达按202个分片保存，人基因/32,461共同基因双视图回读通过；新来源训练划分、预处理及模型接入尚未完成 |
| A549 统一数据包 | [整理与读取说明](../docs/DATA_PREPARED_V1.md) | `prepared/v1` 统一索引 Tahoe 表达、TIDYI 原始计数与状态、药物条件、独立终点及 32,461 个共同基因；来源文件继续保留 |
| A549 训练资产 V1 | [先导运行协议](../docs/先导实验运行协议_2026-09-26.md) | 2026-09-26：`training-v1/manifest.json` 已为 complete；原划分保持不变，T 内归一化、完整 32,461 基因图及核带宽已完成；仅表示现有响应训练资产就绪，不表示 sci-Plex 时间适配、真实终点或正式模型效果已完成 |

下载状态必须区分：复制已核实旧文件、首次下载新文件、完整表达提取、数据质控、训练就绪。未完成的完整下载保持为进行中，不以样本读取或观测元数据数量替代。

Tahoe 的完整 A549 下载已完成，`full_expression/download_status.json` 为 `complete`，最终全分片哈希与条码覆盖审计通过，详见[下载与恢复说明](../docs/TAHOE_A549_DOWNLOAD.md)。

当前筛选入口为[数据筛选 v1](LungCancer_A549_Integration/selection-v1/README.md)，记录实际选中、暂缓、排除和去重结果，以及唯一物理清理项。旧元数据快照中的下载状态不代表当前完成状态；没有删除其他肺癌扩展来源。

当前整理入口为 `LungCancer_A549_Integration/prepared/v1/manifest.json`，完成状态和检查记录分别见 manifest 与 `data_checks.json`。[字段说明](../docs/DATA_FIELDS_V1.md)规定行列坐标、标签、单位及配对边界。该包没有归一化、训练划分或模型评测；历史验证材料仍保留为历史记录，不作为本轮数据整理的新增步骤。

各源文件保留官方命名；版本、URL、大小和校验值记录在相应说明与清单。LINCS 等 paperstudy 的其他历史数据本次不复制，仍可在原项目索引查阅。
