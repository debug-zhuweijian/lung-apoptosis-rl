# 统一数据包 v1 字段约定

本文件说明新增字段；原始来源字段继续保留，不将原始标签重定义为新的实验事实。

## 公共约定

- `cell_id`：来源前缀与原条码构成的 ID；`source_cell_id`：原条码。
- `condition_id`：来源内处理样本 ID，Tahoe 使用 `tahoe:smp_...`；药物、剂量或时间相同不表示是同一个来源样本。
- `matrix_row`、`gene_index`、`common_gene_index`、`tahoe_column`、`tidyi_column`：均从 0 开始。
- 表与小矩阵位置相对数据包根；来源资产位置相对项目根。缺失值保留空值，不解释为 0。
- `is_sequential=False`：该记录并非已记录的顺序给药实验；不意味着药物没有时间效应。

## 基因与表达

| 表／字段 | 含义 |
|---|---|
| `gene_axes/tahoe.parquet: gene_index` | Tahoe CSR 原列号，共 62,710 列 |
| `gene_axes/tidyi.parquet: gene_index` | TIDYI 新 CSR 原列号，与作者完整基因轴一致，共 32,858 列 |
| `ensembl_id` | 本次精确交集的依据；没有按基因符号模糊合并 |
| `gene_symbol` | 源展示注释；共同表沿用 Tahoe 符号，不作为匹配主键 |
| `token_id / tahoe_token_id` | Tahoe 源词表编号，不等于 CSR 列号 |
| `gene_map: common_gene_index` | 共同读取视图的列顺序，0–32,460 |
| `gene_map: tahoe_column / tidyi_column` | 共同基因在各完整矩阵中的实际列号 |
| `tahoe/cells: matrix_path / matrix_row` | 原 CSR 分片及其内行号；prepared 表按这两个字段排序 |
| `tidyi/cells: matrix_row` | 对应 `tidyi/counts.npz` 的实际行号 |
| `tidyi/cells: source_matrix_path / source_matrix_row` | 作者 A549 子集中的原始来源位置 |
| `tidyi/cells: state_label_original` | PI-APC-、PI-APC+、PI+APC+ 原始类别；不转换成药物后自然凋亡率 |
| `tidyi/cells: source_sample_id` | 原 `sample` 字段，保留可能的源拼写差异 |

读取器输出的基因表按实际返回列排列。共同视图保留 `common_gene_index`，其 `gene_index` 表示所读取数据集中的原列号。Tahoe CSR 已在此前提取时去除 CLS，占位符不能再当成基因重复删除。

## 药物和条件

| 字段 | 含义 |
|---|---|
| `drug_id` | Tahoe 源药物标签去首尾空白后生成的稳定 ID；不宣称是统一化学实体 ID |
| `drug_name_original / drug_name` | 原始标签／去首尾空白标签 |
| `canonical_smiles / pubchem_cid` | 源药物元数据；不自行新增结构或编号 |
| `targets / moa-* / human-approved / clinical-trials / gpt-notes-approval` | 源注释，含作者辅助整理内容，不作实测响应或凋亡监督 |
| `used_in_selected_conditions` | 是否出现在本次已选条件中；False 的药物资料保留用于追溯与扩展 |
| `component_role` | `drug` 或 `vehicle_control`；endpoint 成分表另有 `untreated_control` |
| `dose_value / dose_unit` | 处理药物的标称浓度；当前单位 uM（μmol/L） |
| `source_dose_value / source_dose_unit` | 原始剂量字段；DMSO 源占位 0 uM 留在此处，标准剂量列留空，不伪称测得溶剂浓度 |
| `treatment_hours / time_source` | Tahoe 为 24 h，来自研究级协议，不是逐细胞测得的连续轨迹 |
| `condition_type` | `treated` 或 `control` |
| `n_cells` | 本次条件索引中实际表达细胞数 |
| `source_sample_qc_scope` | 原 `mean_*` QC 是混合细胞系样本的均值，未为 A549 重算 |
| `relationship_type` | `same_plate_control_candidate`，同板对照候选关系 |
| `individual_cell_pairing` | `not_available`，不存在逐细胞前后配对 |

## 终点、辅助来源和候选关系

| 字段 | 含义 |
|---|---|
| `endpoint_condition_id` | Macaya 来源块＋背景＋处理条件，不含重复号 |
| `endpoint_id` | 上述条件加源实验重复号，唯一定位 144 个原始位置 |
| `replicate_id` | 对应来源实验的编号，不跨来源强行对齐 |
| `source_file / source_sheet / source_cell` | 原附件、工作表及单元格，可追溯原始读数 |
| `annexin_v_positive_percent` | 原百分比，保持原数值 |
| `annexin_fraction` | 原百分比 / 100，范围 0–1；缺失仍缺失 |
| `has_measured_endpoint` | 是否实际取得数值；共 142 True、2 False |
| `endpoints/components: component_index` | 同时组合的成分编号，仅用于枚举，不是给药顺序 |
| `bulk_sample_id / bulk_condition_id` | GSE161218 RNA 样本／条件 ID；9 样本、3 条件 |
| `expression_path` | 原 gzip 计数文件位置；没有将 bulk RNA 拼成单细胞矩阵 |
| `pending_data.status` | 待核对、未取得或不在当前人肺癌输入范围；不表示源文件已经删除 |
| `lung_backgrounds_to_preserve` | 27 个源肺癌背景元数据条目；表达准备目前仍是 A549 |
| `cross_source_candidates.link_level` | 仅为条件候选，不是配对样本 |
| `time_match` | 标称处理时间是否相同；False 不被改写为相同终点 |
| `sample_pairing` | `not_established` |
| `chemical_form_equivalence` | 仅源名称匹配，未独立核实化合物形式等价性 |
| `culture_and_sampling_equivalence / control_equivalence` | 尚未建立等价关系 |
| `use_as_paired_training_target` | 固定 False，避免把候选关系直接当监督标签 |

`manifest.inputs` 是本轮使用的输入和程序指纹；`manifest.outputs` 是新产物指纹，不含 manifest 自身。`tahoe/shards` 的 SHA256 明确来自已有 checkpoint，本轮核对其大小、修改时间及全部选中细胞的原 obs 身份，没有声称本轮重新哈希扫描了全部大矩阵。
