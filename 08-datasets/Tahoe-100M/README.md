# Tahoe-100M：A549 样本与对照核验

> 主目录切换补充：本文件下文保留最初的元数据与小样本核验记录。完整 A549 表达下载已在 `I:\科研\肺癌细胞凋亡-强化学习` 启动；新增进度与最终完整性以[下载说明](../../docs/TAHOE_A549_DOWNLOAD.md)及版本目录 `full_expression/download_status.json` 为准，不能把下文的 1,813 个读取样本当作新任务的最终数量。

核验日期：2026-09-18。官方仓库：[`tahoebio/Tahoe-100M`](https://huggingface.co/datasets/tahoebio/Tahoe-100M)。本目录锁定版本 **`2dc57900b7981cfcf5e211527169a0b006546a95`**，避免把论文版本、当前 HF 数据及部分预览的数量混在一起。

## 当前完成范围

- 四个小型官方元数据表已完整下载、实际读取，并与官方 LFS SHA256 核对一致。
- 对 `obs_metadata.parquet` 的全部 **96 个 row group、100,648,790 条源记录**读取必要列，筛选真实出现的 A549；未将 sample 表与细胞系表做笛卡尔积。
- 已保存 A549 全部观测条码及对应 sample、plate、`pass_filter`，并建立真实样本—剂量—同板对照清单。
- 已读取官方表达数据前两个完整 shard，筛得 **1,813 个 A549 细胞**，覆盖 plate4 的 **96 个样本**，其中 **26 个 DMSO_TF 对照细胞**。这是读取验证子集，不是随机、完整或可直接代表全部 A549 的训练集。
- **尚未提取全部 A549 表达矩阵，尚未完成项目质控、训练划分或模型训练。**官方 `full` 标签是官方过滤结果，不等于我们已经完成跨来源任务的质控。

## 实际观测覆盖

| 项目 | 本次完整 obs 扫描结果 |
|---|---:|
| A549 细胞 | 2,664,482 |
| 官方 `full` | 2,567,838 |
| 官方 `minimal` | 96,644 |
| A549 实际出现的 sample | 1,343 / 全库 1,344 |
| plate | 14 |
| `drug` 名称（含 vehicle；不等于去重化学实体数） | 380 |
| `drugname_drugconc` 标签（含 vehicle） | 1,138 |
| A549 的 DMSO_TF sample | 28 |
| 找不到同板 A549 DMSO_TF 对照的已观测 sample | 0 |

必须保留的覆盖例外：

1. `smp_1576`／plate1／Methylprednisolone succinate 0.05 μM：全量 obs 中没有 A549 行，不能伪造为已测 A549 条件。
2. `smp_1509`／c-Kit-IN-1 0.05 μM：只有 1 个 `minimal` A549 细胞，没有 `full` 细胞。因此真正有 `full` A549 的 sample 是 **1,342**。
3. 官方源 obs 同时包含 `full` 和 `minimal`；官方表达仓库卡片登记 95,624,334 条表达记录。不能默认每个 `minimal` 条码都能在该表达发布中取得表达。完整 obs 覆盖是观测元数据覆盖，不是全部表达文件已经落地。

## 字段、时间、对照与重复

- A549：`Cell_ID_Cellosaur=CVCL_0023`，`Cell_ID_DepMap=ACH-000681`。细胞系元数据是一行一个 driver 注释，不能把多条 driver 行当成多个细胞系。
- `sample` 是混合细胞实验样本标识；`plate` 是实验板。只有对应的 obs 行能证明某个 sample 中观测到 A549。
- 条件表中的 `mean_gene_count`、`mean_tscp_count`、`mean_mread_count`、`mean_pcnt_mito` 原样来自官方 `sample_metadata`，本次没有按 A549 子集重算这些均值；后续 A549 专属质控不能将其误当成该细胞系的实测均值。
- 剂量从官方 `drugname_drugconc` 用 `ast.literal_eval` 解析，保留原始字符串、数值与单位。这里没有凭药物名称补剂量。
- sample 表中 Erdafitinib 和 Selinexor 的药物名带尾随空格；保留原始 `drug`，另建仅去首尾空格的 `drug_key` 与药物元数据连接。经此规范化，379 个处理名称均可连接，只有 vehicle 不属于药物注释表。派生条件表保留官方 `canonical_smiles`、`pubchem_cid` 和连接成功标记；未合并盐型或其他化学形式。
- 连接成功不等于结构齐全：**Sacubitril/Valsartan、Verteporfin** 的官方 SMILES／PubChem CID 均缺失，涉及 6 个 A549 sample，表中 `structure_status=missing_in_official_metadata`。没有补造结构；含多成分产品的名称也不能仅凭单个处理标签当作单一化学实体。
- 对照严格使用 **`drug == "DMSO_TF"` 且同 plate、同 A549 背景**。不能用包含 DMSO 字符串来识别对照，否则会把 Trametinib (DMSO_TF solvate) 误判为 vehicle。
- 官方 README 的 Dataset Features 明确 DMSO_TF 与 plate 配合获得同板对照；sample 区分重复处理。[锁定版字段说明](https://huggingface.co/datasets/tahoebio/Tahoe-100M/blob/2dc57900b7981cfcf5e211527169a0b006546a95/README.md#dataset-features)
- **24 小时**来自 Tahoe 预印本 v1 的实验协议，不是 sample／obs 中逐细胞实测时间字段。论文 Results 第一段说明 24 小时处理后解离、固定并测序；Methods 实验扰动段说明 37°C、5% CO2 下暴露 24 小时。派生表以 `time_source` 显式记录来源。
- 该实验是多细胞系混合的 **3D 悬浮 spheroid**；它不是 TIDYI 或 Macaya 的培养体系等价物。
- 论文 Results 的重复性分析明确 **plate14 是 plate6 的 biological replicate**。其他同名药物和剂量出现多次，不自动标为生物学重复；派生表使用 `not_specified_as_biological_replicate`。
- 上述时间及重复证据已读 Zotero 英文 MinerU 全文：论文 key `5EIID7CD`，PDF `97PJJ8YH`，缓存 `FNCD2ZYA`，`full.md` 行 41、82、200、236、258。原文：[Tahoe-100M 预印本 v1](https://www.biorxiv.org/content/10.1101/2025.02.20.639398v1)。本次没有中英文版本混用。

## 与凋亡结局连接候选的精确条件检查

完整 A549 obs 扫描已确认未带 solvate 后缀的 Trametinib 0.5 μM：

| sample | plate | full A549 | minimal A549 | full H1792 | minimal H1792 | 协议时间 |
|---|---|---:|---:|---:|---:|---|
| smp_1625 | plate2 | 2,053 | 144 | 750 | 53 | 24 h |
| smp_2523 | plate11 | 1,997 | 40 | 1,059 | 23 | 24 h |

H1792（`CVCL_1495`）为额外定向读取对应源 obs 行组取得，未冒充 H1792 全量条件清单。结果和同板对照计数保存在 `derived/trametinib_bridge_observed_counts.csv`。这两个条件可以进入与 Macaya 2023 的同背景、同药物、同标称剂量、同处理时长对照审查。**这不代表同一样本、相同培养体系或已得到可直接合并训练的配对监督。**Trametinib (DMSO_TF solvate) 单列，不自动与 Trametinib 合并。

官方 sample/drug 元数据检索 `Lestaurtinib`、`CEP-701`／`CEP701` 未找到对应条目；不能把其他激酶抑制剂补成该药，也不能用 0.5 μM 代替 0.625 μM。此处报告指定名称及别名的查询结果，不代表已经完成所有化学结构标准化。

## 本地文件

所有以下路径均位于版本目录 `2dc57900b7981cfcf5e211527169a0b006546a95/`。

| 路径 | 内容／状态 |
|---|---|
| `README.md` | 原封保留的官方数据卡 |
| `metadata/gene_metadata.parquet` | 62,710 × 3；gene_symbol、ensembl_id、token_id |
| `metadata/cell_line_metadata.parquet` | 1,000 × 10；含多行 driver 注释 |
| `metadata/sample_metadata.parquet` | 1,344 × 8 |
| `metadata/drug_metadata.parquet` | 379 × 9；vehicle 不在药物注释表中 |
| `metadata_source_manifest.json` | 官方 metadata 目录条目、大小与 LFS 哈希 |
| `obs_remote_schema.json` | 原始 obs 行数、schema、范围读取的列及来源 |
| `derived/a549_observed_cells.parquet` | 2,664,482 个 A549 观测条码及 sample／plate／官方过滤标签 |
| `derived/a549_observed_sample_conditions.csv` | 1,343 个真实观测 sample，含剂量、协议时间、过滤级细胞数、重复标注及同板对照 |
| `derived/samples_without_observed_a549.csv` | 无 A549 观测的 1 个 sample |
| `derived/a549_plate_control_links.csv` | 明确列出的 sample—同板 A549 vehicle 关系；vehicle 也保留同板参照关系 |
| `derived/a549_coverage_summary.json` | 全量 obs 覆盖核验汇总 |
| `derived/a549_obs_row_groups/` | 96 份已过滤的行组缓存，可恢复重跑 |
| `data/train-00000-of-03388.parquet`、`data/train-00001-of-03388.parquet` | 仅前两个官方完整 shard，保留原名 |
| `derived/a549_expression_smoke.parquet` | 前两个 shard 中的 A549 读取验证子集，保留 source_shard |
| `derived/expression_smoke_summary.json` | 表达 token、长度、样本、对照与范围验证 |
| `derived/trametinib_bridge_observed_counts.csv` | A549／H1792 在 Trametinib 0.5 μM 与同板对照条件下的真实 obs 计数 |
| `local_file_manifest.json` | 本地文件大小、SHA256、源／派生标识和版本 |

四个源表校验：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| gene_metadata.parquet | 1,326,799 | `d6104d1ca570d94832be0d27ffe8b7e4e54cfd366c9080239219f9c406eb751c` |
| cell_line_metadata.parquet | 19,040 | `67641f5bdd3fb077978ff1fec4d0d617490674bcd42850515ae3997194c29d4d` |
| sample_metadata.parquet | 65,636 | `33167f0ce28cff8357c503cda67d1c7fea200bff6918ee1de3c4f3549175b9d3` |
| drug_metadata.parquet | 40,475 | `7a04c7a6611e74c92253163b994ff0a47191928ee7511f1c2ca508cee0e79f28` |

完整 obs 文件源大小 2,293,981,573 字节，官方 LFS SHA256 为 `e51b24f3666f5f433c59ff01dd6fa33934627091455e89d929f2c1ddc87401fe`。本次通过 HTTP range 实际读取所需列，**没有下载或本地重算整个 obs 原文件的哈希**；源 digest 与本地派生文件 digest 在清单中区分。

## 读取与重跑

依赖：Python、requests、pandas、pyarrow、fsspec。已在当前环境实际运行：

```powershell
python scripts/prepare_tahoe_a549.py --workers 6
python scripts/prepare_tahoe_a549.py --workers 6 --expression-smoke
python scripts/prepare_tahoe_a549.py --workers 6 --bridge-observations
```

这些命令应依次执行，避免同时重写派生表。脚本复用已完成行组，源小表和表达 shard 必须通过大小、SHA256 校验。obs 筛选检查 Cellosaurus ID、条码唯一性、sample 外键和 sample／plate 对应；分组显式 `observed=True`，避免 pandas categorical 隐式创建并不存在的条件组合。

表达读取必须删除 `genes` 和 `expressions` 各自第一项。实际读取子集中，首 token 为 `1`，首表达标记为 `-2.0`，并非负生物表达；去除后 token 均在 62,710 基因词表中，1,694,380 个存储的非零基因计数全部为非负整数。1,813 个表达条码全部与 obs 一一匹配，sample、plate 一致且官方过滤标签均为 full。官方 README 同样明确 CLS 首项应忽略。

## 尚待完成及结论边界

1. 全部 A549 的真实表达抽取、每条件有效细胞数与表达覆盖复核；当前条件表的源位置只指向仓库 shard 通配模式，尚无完整逐条码 shard 索引。
2. 根据目标选择并审查 `full`／`minimal` 过滤，记录可能减少死亡相关细胞捕获的选择偏差；不能把 filtered 表中的凋亡分数解释为原培养群体的凋亡比例。
3. 32,461 共同基因编号是词表交集；完整表达检出率和训练面板由共享分析另行核定。
4. 分割应考虑同药物、剂量、板与已知重复关系。24 小时单终点不提供任意 Δt 的真实动态监督，更不能直接证明多步序贯有效。
5. 数据与派生非 Markdown 文件按 `08-datasets/.gitignore` 保持本地，不提交 Git 或 Git LFS。

## 获取路径核查记录

- HF `datasets-server/filter` 的 obs 查询返回 500「dataset index is loading」，expression 查询 60 秒超时；因此未把服务返回当成完整过滤结果。本次成功路径是锁定版本的 Parquet HTTP range 读取。
- 官方 HF 表达发布由 3,388 个混合细胞 shard 组成；前两个 shard 已本地验读，完整 A549 提取仍需专门执行。
- 也已读取官方数据卡链接的 Arc Institute README 和 Python 教程。教程展示按 plate 的 h5ad 与元数据过滤，未展示直接下载完整 A549 的分区。当前 README 记录 GCS 已迁移，不能继续把旧 bucket 当成已验证可用路径；本次未接入其 Marketplace 授权项目。来源：[Arc README，commit 4f7bb170](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/4f7bb170c61d2fafce70f67db6e52c7d9af1be4b/tahoe-100M/README.md)、[Python 教程](https://github.com/ArcInstitute/arc-virtual-cell-atlas/blob/main/tahoe-100M/tutorial-py.ipynb)。此项是已检查路径的范围，不宣称全网不存在其他子集。
