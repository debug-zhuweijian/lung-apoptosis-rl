# 肺癌课题数据筛选 v1

日期：2026-09-19。范围：按用户确认，保留其他肺癌背景的扩展可能，先排除当前样本集中的明确无关、重复和缺少必要信息的记录。这里是实际生成的筛选结果，不是模型训练或算法效果验证。

## 本轮结果

| 数据 | 本轮保留 | 未进入当前可用视图 |
|---|---|---|
| Tahoe A549 | 1,308 个结构与剂量信息可用的处理样本＋28 个对照；2,558,270 个细胞 | 6 个缺结构条件及其 9,568 个有表达细胞暂缓；96,644 个未发布表达的观测记录不作表达样本；官方样本表有 2 个样本无可用 A549 表达 |
| TIDYI A549 | 743 个唯一细胞，原始 32,858 基因；三类标签分别 304、229、210 | 3,066 条已知非肺癌来源记录不纳入当前样本集；368 条旧 CSV 重复记录不重复计数；361 条来源记录的细胞身份未确认，单列保留 |
| Macaya 肺癌终点 | 两个数据块共 142 个实测数值，保留 A549 和其余五种肺癌背景 | 2 个缺失位置仍保留在完整筛选台账，不填零，不纳入有标签视图 |
| 多肺癌扩展 | Tahoe 原始混合分片完整保留；元数据中的 27 个肺部来源背景另列清单 | 27 是细胞系元数据数量，不声称已将全部背景整理成可训练数据 |

TIDYI 的 361 是两个来源中的记录数，不是声明有 361 个独立的未确认细胞。重复的 368 条记录已按细胞 ID、基因 ID、全部计数和原始状态标签核对一致。

弱响应、无效药物反应、低 RNA 或低细胞数本身不作为本轮删除依据。当前药物名称未被宣称为已完成化学实体去重或已证实有效的肺癌抗癌药目录。

## 输出文件

| 文件 | 作用 |
|---|---|
| `tahoe_condition_selection.csv` | 1,344 个官方样本的 A549 可用性与筛选原因，保留原始处理信息 |
| `selected_tahoe_conditions.csv` | 1,336 个选中处理／对照样本 |
| `selected_tahoe_control_links.csv` | 2,616 条同板对照候选链接；每个处理连到同板两个 DMSO 样本，不是逐细胞或逐生物重复配对 |
| `selected_tahoe_cells.parquet` | 选中细胞及现有 CSR 分片路径、`matrix_row`，不重复复制表达矩阵 |
| `held_or_excluded_tahoe_cells.parquet` | 未取得表达或条件缺结构的记录及原因 |
| `tidyi_record_selection.csv` | 作者数据与旧 CSV 的完整来源记录筛选台账 |
| `selected_tidyi_cells.csv` | 743 个选中细胞，指向已有 A549 NPZ 的行号 |
| `endpoint_selection.csv` | 两个独立源数据块的全部 144 个位置，含缺失 |
| `selected_endpoints.csv` | 142 条实测读数，保留源数据块、单位、原单元格及非配对状态 |
| `lung_backgrounds_to_preserve.csv` | Tahoe 细胞系元数据中的肺部来源背景；原始分片保留 |
| `source_file_policy.csv` | 原始文件、派生文件和资料的保留政策；物理清理实际结果另见下项 |
| `file_cleanup_report.json` | 唯一物理删除项、哈希和离线恢复来源 |
| `selection_summary.json` | 筛选规则和实际统计 |
| `output_manifest.json` | 本轮输出文件大小和 SHA256 |

`csr_shard`、`matrix_path` 均相对项目根目录 `I:\科研\肺癌细胞凋亡-强化学习`。矩阵行号从 0 开始。Tahoe CSR 保留原 62,710 基因顺序，TIDYI NPZ 保留原 32,858 基因顺序；本轮没有预先裁剪基因。

已有共同基因映射仍为 [32,461 个共同 Ensembl ID](../2026-09-18/shared_gene_vocabulary.csv)。基因交集不代表细胞配对、平台相同或补齐了凋亡标签。

## 字段含义

- `selection_status` 区分 keep、hold 与 exclude。hold 表示当前视图暂缓，不表示原始生物数据错误。
- Tahoe 的 `mean_gene_count` 等平均 QC 字段来自混合细胞系样本；`source_sample_qc_scope` 明确它们不是 A549 重算值。
- Tahoe 24 h 来自论文处理协议，通过 `time_source` 记录，未补造逐细胞实测时间。
- 对照链接标为 `same_plate_control_candidate`、`individual_cell_pairing=not_available`。不能把两个链接当两份独立处理实验。
- Macaya 保留 `percent` 原单位。缺失不填零，同时处理不改写为序贯处理。
- TIDYI 保留原始 PI/APC 标签；本轮不合并类别、不将分类标签直接改成药物凋亡比例。

## 文件清理与来源保留

本轮只清理 TIDYI 解压副本 `audit_sources/human_ss3.h5ad`，共 2,201,171,391 字节。删除前重新计算该文件及完整 `classifier_data.tar.gz` 的 SHA256，均与原清单一致。完整归档及已提取的 A549 计数、标签、基因注释全部保留，可从归档同名成员离线恢复该副本。

历史 `human_ss3_metadata_audit.json` 记录的是当时解压文件的审计，不篡改其旧路径。当前物理文件保留状态以本轮清理报告为准。

Tahoe 原始 `data/` 不是本轮删除对象：它保留其他肺癌背景，且现下载器 checkpoint 及严格审计依赖 source、raw_a549、csr、obs 四类文件。原 A549 raw、CSR、obs、参考条码、基因表、校验清单一并保留。

Macaya 的 H1792 RNA 和其他肺癌终点、Hu 的序贯原文与补充材料仍与课题相关，保留为各自用途。TIDYI 未整理的 human 10x、参数表及其他归档成员保留在原压缩包中；小鼠成员不纳入本轮人类肺癌输入。

旧的 Tahoe 条件表是下载前快照，其中 `expression_full_downloaded=false` 不作为本版筛选依据；本版根据最终表达覆盖重新生成可用性。旧快照未覆盖。

## 可复现与阶段范围

筛选脚本：`scripts/select_lung_dataset_v1.py`。它拒绝覆盖已存在的本版本目录；未来改变规则应建立新版本。脚本只生成数据视图，物理文件清理由独立报告登记，不在重跑脚本时再次删除。

已经执行的检查是细胞、矩阵行、标签和来源的一致性检查。没有运行模型训练、算法性能评估、训练／测试切分、归一化、高变基因筛选，也没有清理 paperstudy 的历史副本。
