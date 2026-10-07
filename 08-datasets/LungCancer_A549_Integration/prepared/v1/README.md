# 肺癌课题：统一数据包 v1

整理日期：2026-09-19。用途：为后续药物扰动预测、状态识别及决策模型提供统一的数据读取入口。当前只整理数据，不进行模型训练、性能评测或训练集划分。

主目录：`I:\科研\肺癌细胞凋亡-强化学习`。

数据包：`08-datasets/LungCancer_A549_Integration/prepared/v1/`。`manifest.json` 是文件入口，`summary.json` 是数量清单，`data_checks.json` 是本次整理的完整性检查记录。

## 整理后的内容

| 数据层 | 内容与数量 | 保存方式 |
|---|---|---|
| 药物扰动表达 | Tahoe：2,558,270 个 A549 细胞，1,308 个药物处理条件、28 个对照条件；其中对照 60,135 个细胞 | `tahoe/cells.parquet` 索引原有 3,388 个 CSR 稀疏分片，不再复制全部表达 |
| 药物资料 | 379 个源药物标签与 1 个 DMSO 对照条目；本次选中条件涉及 377 个药物标签及 DMSO | `drug_catalog.parquet` 保留原名、结构及源注释；缺结构药物未进入当前条件表 |
| 状态识别表达 | TIDYI：743 个明确 A549 细胞，原始 PI-APC- 304、PI-APC+ 229、PI+APC+ 210 | `tidyi/counts.npz` 保存原始整数计数；`tidyi/cells.parquet` 保存原标签及板、孔信息 |
| 基因坐标 | Tahoe 全量 62,710；TIDYI 全量 32,858；共同 Ensembl ID 32,461 | 保留两个完整基因轴；共同基因表显式记录各自在原矩阵中的列号 |
| 实测终点 | Macaya：6 个肺癌背景，48 个来源内条件、144 个重复位置，其中 142 个数值、2 个缺失 | `endpoints/measurements.parquet` 保留原百分比，另加 0–1 比例；`components.parquet` 记录单药、同时组合及对照 |
| 其他表达 | GSE161218：9 个 H1792 bulk RNA 样本，3 个条件，16 h | `auxiliary/macaya_rna_samples.parquet` 引用原始计数文件 |
| 扩展来源 | Tahoe 27 个肺癌背景的元数据条目；TIDYI 保留归档；Hu 2025 真实序贯文献与补充材料 | 保存来源索引和待整理／缺失项，不误标为已制备表达或完整序贯训练数据 |

这里的“条件”是某块实验板上的处理样本，不是独立药物数。药物 ID 按源标签生成，未做盐形式归并或化学实体去重；`377` 不能直接表述为 377 个已核验独立化学实体。Hu 材料中的序贯设计不等于取得了配套逐样本表达与实测终点。

## 文件布局

```text
prepared/v1/
├── manifest.json / summary.json / data_checks.json
├── README.md / FIELD_DICTIONARY.md
├── gene_map.parquet
├── gene_axes/                 两个完整原始基因轴
├── drug_catalog.parquet
├── tahoe/
│   ├── cells.parquet          细胞 → 条件 → 原 CSR 行号
│   ├── conditions.parquet     药物、剂量、时长、板与细胞数
│   ├── control_links.parquet  同板 DMSO 候选关系
│   └── shards.parquet         源文件位置与既有严格审计指纹
├── tidyi/                     counts.npz + cells.parquet
├── endpoints/                 measurements.parquet + components.parquet
├── auxiliary/                 bulk RNA、肺癌背景、文献和缺失项
└── cross_source_candidates.parquet
```

这个包依赖项目内原始数据，不能只复制 `prepared/v1` 后就删除原 CSR 分片。表名在 manifest 中登记；表和小矩阵路径相对数据包，表内 `matrix_path`、`expression_path`、`source_file` 等来源路径相对项目根目录。

## 怎样读取

在项目根目录运行 Python，使用已提供的统一读取器：

```python
from src.lung_data import PreparedLungData

data = PreparedLungData()

# 每批输出：稀疏原始计数、同行序细胞表、同列序基因表。
for counts, cells, genes in data.iter_tahoe(batch_size=1024):
    # 此处接后续自行确定的数据处理与模型代码。
    break

counts, cells, genes = data.load_tidyi()
conditions = data.table("tahoe_conditions")
endpoints = data.table("endpoints")
controls = data.control_candidates("tahoe:smp_1625")
```

默认 `gene_space="common"`，按共同基因编号顺序返回列；用 `gene_space="full"` 读取对应数据集的完整基因轴。Tahoe 按细胞索引分批读取，不拼成完整稠密矩阵。TIDYI 采用 SciPy CSR NPZ 与注释表，无需额外安装 AnnData。所有表达均为原始计数；本轮没有归一化、对数变换、高变基因选择或进一步筛掉弱响应细胞。

## 连接规则

1. Tahoe：`cell_id → condition_id → drug_id`。同板对照表提供两个 DMSO 候选条件，不生成逐细胞前后配对，也不把一条处理条件复制成两个独立重复。
2. TIDYI：每一行计数对应一条原始分选状态标签。保留 `source_sample_id` 的原始拼写；例如原字段中的 `A594` 不覆盖明确的 `Cell_line=A549` 注释。标签含义与命名保留来源，不在本轮改成统一“凋亡真值”。
3. Macaya：保留每个来源块、条件、重复与原始表格单元格。相同药物处理在不同来源块中不自行去重。两个缺失终点保留为空，`has_measured_endpoint=False`；百分比除以 100 仅是单位换算。
4. 跨来源候选：只登记细胞背景、源药名及标称剂量匹配的条件，再显式记录时间是否一致、样本是否配对等字段。16 h RNA 与 24 h 终点标为时间不一致；培养、采样、对照及化合物形式的等价性不假定成立。候选表不构成可直接监督的表达—终点配对集。

## 本次检查与后续边界

整理检查包括：冻结筛选产物哈希、全部选中 Tahoe 细胞与原 obs 的条码／样本／板／背景／行号对齐、CSR 和 obs 大小及修改时间与既有严格校验清单一致、TIDYI 原始计数和标签对应、共同基因列坐标、条件和对照关系、两个真实 Tahoe 条件的完整批读取、小矩阵读取器单元测试。

Tahoe 原始 337.64 GB 数据不重复全量哈希扫描；本轮复用此前已完成的严格审计，并核对使用中的 CSR／obs 文件属性。新的整理产物各自生成 SHA256。上述检查确认数据整理和读取正确，不是模型精度、迁移效度、凋亡预测或序贯方案有效性的验证。

本轮沿用 `selection-v1`，没有新增删除。Tahoe 其他肺癌的混合原始分片、缺结构条件、TIDYI 原始归档及历史记录继续保留。后续算法确定后再设计按条件／药物／板／来源的划分、训练内预处理及整体模型验证；此包没有替研究任务预先固定这些选择。

构建入口为 `scripts/prepare_lung_dataset_v1.py`，读取器检查为 `python -m unittest discover -s tests -p test_lung_data.py -v`。构建器先生成临时目录，完整性检查通过后发布 `v1`；已有版本或临时目录时拒绝覆盖，不自动删除数据。具体运行环境和脚本指纹记录于 manifest。
