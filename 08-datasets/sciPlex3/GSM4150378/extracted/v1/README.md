# sci-Plex A549 完整表达提取 v1

日期：2026-09-24。状态：**完整提取与独立回读通过，尚未接入训练。** 未归一化、未拟合模型、未创建train/val/test划分。

## 文件与覆盖

| 内容 | 数量/说明 |
|---|---|
| 官方源文件 | 3,096,224,606字节；保留上级sources的原名gzip |
| 完整源稀疏记录 | 1,007,419,688条；源坐标范围、有序性、重复与正整数计数检查通过 |
| 全源细胞 | 799,317；每个细胞总UMI与官方pData一致，0不符 |
| A549筛选前细胞 | 244,281，保留全部原始记录 |
| 标记为通过指定hash规则 | 210,458；24 h 143,015，72 h 67,443 |
| A549矩阵分片 | 202个行分片；每片各有human与common两种基因视图 |
| human视图 | 58,347条原始人基因注释，含PAR_Y；236,619,812非零值 |
| common视图 | 32,461基因，与旧主模型轴一致；234,907,013非零值 |
| 稀疏矩阵数据类型 | CSR、int32正整数计数；404个NPZ文件共3,774,678,292字节 |

`a549_cells.parquet`保存细胞身份、药物、剂量、时间、重复、板、hash标记、来源列号、`shard_id`、`row_in_shard`和三种轴的UMI总量。`human_genes.parquet`和`common_genes.parquet`给出对应列顺序。`extraction_status.json`列出每个分片的行数、非零数、字节数与SHA256。

独立[readback_audit.json](readback_audit.json)已重新读取所有分片，逐行核查human/common计数和与元数据相同，按基因映射投影human矩阵后与common矩阵逐元素一致。全源计数证据保存于`all_source_cell_umi_audit.npz`。该审核没有运行效果评价。

## 基因与保留规则

human视图保留全部人基因来源行及PAR_Y后缀；common视图只映射无歧义的原始Ensembl基础ID，不把PAR_Y强行加到其对应行。全A549的PAR_Y来源行共2个UMI，均在human视图保留。

common视图保留24 h人基因UMI的99.547%、72 h的99.426%；这是**当前A549计数比例**，不是基因检出率、模型准确率或普适覆盖保证。两时间都没有共同轴全零的细胞。[表达统计](expression_summary.json)

低hash质量记录完整保留，由`author_hash_qc`选择视图；没有把筛选细胞构成比当作凋亡率。72 h的67,443与论文64,440差异继续保留为未解决项。

## 读取示例

以下只读取原始计数，不形成训练或终点预测：

```python
import json
from pathlib import Path
import pandas as pd
from scipy import sparse

root = Path(r"I:\科研\肺癌细胞凋亡-强化学习\08-datasets\sciPlex3\GSM4150378\extracted\v1")
cells = pd.read_parquet(root / "a549_cells.parquet")
manifest = json.loads((root / "extraction_status.json").read_text(encoding="utf-8"))
shards = {s["shard_id"]: s for s in manifest["shards"]}
cell = cells.loc[cells.author_hash_qc & cells.time_point.eq(72)].iloc[0]
matrix = sparse.load_npz(root / shards[int(cell.shard_id)]["common"]["file"])
counts = matrix[int(cell.row_in_shard)]  # (1, 32461)，按common_genes.parquet列顺序
```

需要训练时，先按药物/条件/孔及重复注册划分，再只在训练部分拟合预处理。不得直接把本文件路径替换到旧Tahoe训练配置中，也不得把24/72 h独立群体配成真实细胞轨迹。

## 可复现检查

[提取程序](../../../../../scripts/extract_sciplex_a549.py)流式读取完整gzip，每块边界保留完整细胞；重启重扫来源并核验已记录分片，不覆盖不匹配的产物。[独立回读程序](../../../../../scripts/verify_sciplex_extraction.py)重新加载所有分片及基因映射。7项小型合成检查覆盖分片边界、基因顺序、重复/非法坐标、gzip CRC损坏和独立回读。

来源、处理协议、化学身份和当前研究边界见[上级数据说明](../../README.md)。
