# Tahoe fit 对照参考表达的实际读取审计

日期：2026-10-10。**28 个源 vehicle 样本、14 块板的 fit/T 对照表达已实际读取并形成保持原计数尺度的参考输入。** 这补上上一轮只核查元数据、尚未读取参考表达的缺口；没有训练或执行预测，没有将板解释为已确认生物状态。

## 实际完成量

| 项目 | 实际结果 |
|---|---:|
| 访问的角色与观测 | `fit` / `train` / `contemporaneous_control` |
| 实际唯一 A549 vehicle 细胞 | 30,065 |
| 源对照样本 | 28；每个 522–1,859 个 fit 细胞 |
| 没有 fit 细胞的源对照样本 | 0 |
| 板 | 14；每板 2 个源对照样本 |
| 涉及的共享 CSR 分片 | 1,838 |
| 共同基因 | 32,461；保持 ENSG、共同索引及 Tahoe 原列对应 |
| 样本摘要 | `28 × 32,461` raw-count 均值与检出率 |
| 板摘要 | `14 × 32,461`，按真实细胞数加权合并两个源对照 |
| 同板两个对照的描述性比较 | 14 对，原均值向量均不完全相同 |
| 不同板参考的描述性比较 | 91 对，原均值向量均不完全相同 |
| 上轮跨板同处理组 | 154 组、325 条处理条件，650 个同板对照链接均有可读 fit 参考 |

参见 [validation.json](validation.json)、[manifest.json](manifest.json)。325 个处理条件的角色只作为元数据读取：fit 156、selection 47、residual_fit 32、aux_selection 28、calibration 26、final_test 36；这不表示读取了这些条件的药后表达。参考输入全部来自单独筛出的 fit 控制细胞。

## 读取方式和资源

先亲读 `SparseRows` / `CSRCache` 的实现，再从 `training-identity-v3/train.parquet` 按 `role=fit` 与 28 个 control condition ID 过滤。未创建 `TrainingData`，因此没有其自动加载其他资产或归一化的行为。没有打开 `val.parquet`、`test.parquet`、TIDYI 分类标签或大体积密集表达缓存。

按 CSR 分片顺序处理，每批最多 256 个已选对照细胞。利用现存的只读 CSR 内存映射，最多保留一个源映射；只抽取合法行和共同基因列后进行稀疏累计，不密集化全部细胞。读取器中的 cache builder 被本审计替换为直接拒绝写入；本次没有创建或修改共享缓存。

原始 CSR 约 3.74 GB，其对应稀疏缓存合计约 14.61 GB，但不是同时载入内存。最大单分片缓存约 12.86 MB；过滤后必要细胞身份表约 5.6 MB，28 份全基因计数/检出累计约 14.5 MB。执行前预计进程内存低于 1 GB，实际采样监测的最大 RSS 为 **666.96 MiB**，逐分片阶段通常约 355–366 MiB。没有 GPU 训练或推理。

为了核对完整性，会流式 hash 共享源文件和缓存的完整字节，并读取 CSR 形状/索引元信息。这些文件还物理包含其他条件的编码值；**分析层面仅抽取上述 30,065 个 fit 对照行**，没有对非选中药后表达作汇总、拟合、预测或评测。`opaque_shared_source_and_cache_bytes_hashed=true` 明确记录了这一边界，不把整文件 hash 虚称为完全未触及共享文件。

## 身份、尺度与缺失

每一个被选细胞都完成以下核验：

1. train 元数据的角色、来源、A549 背景和 `contemporaneous_control` 身份合法。
2. `condition_id → source_sample_id / plate` 与冻结 condition 表一致，源条件确为 vehicle。
3. `(matrix_path, matrix_row)` 指向原始 obs 表的同一行；原始条码加 `tahoe:` 与 `cell_id` 完全一致，sample、plate、`DMSO_TF`、`CVCL_0023` 同时匹配。
4. cell ID 和物理 CSR 行在本次选集内都没有重复。
5. 共同基因轴在训练资产、prepared gene map 与 Tahoe 原列中的 ENSG 顺序一致。
6. 被抽取的计数有限、非负且为整数；原始 CSR 与 obs 均按已登记 hash 核验。

每个基因输出：

- `raw_count_sum`：所选 fit 细胞原始计数之和，int64。
- `raw_count_mean`：上述计数和除以该源样本实际 fit 细胞数。
- `detected_cell_count`：原始计数大于零的细胞数；存储零和重复稀疏坐标不重复计作检出。
- `detection_rate`：检出细胞数除以该源样本实际 fit 细胞数。

没有拟合或应用文库归一化、log 变换、全数据标准化或基因筛选。CSR 隐式零是该已测细胞在该基因的零计数，不能和“某个样本没有 fit 细胞”混同。若后续版本的某个参考没有细胞，代码保留 `n=0` / `reference_available=false`，其均值与检出率写为 NaN；本次没有这种缺失。

板摘要直接累计两个源样本的 count_sum、detected_count 和 n，再计算均值与检出率；没有简单平均两份样本均值。真实摘要中的 14 个板都已从样本累计量独立重算，逐值一致。

## 可读取的产物

| 文件 | 内容 |
|---|---|
| [sample_references.npz](sample_references.npz) | 28 个源样本的计数和、检出数、n、均值、检出率、可用标记及完整基因轴身份 |
| [plate_references.npz](plate_references.npz) | 14 个细胞加权板参考，字段与样本摘要相同 |
| [reference_samples.csv](reference_samples.csv) | 原 condition 元信息、实际 n 与参考可用性；原混合细胞系 QC 字段没有冒称本次 A549 重算 |
| [fit_control_cell_identity.parquet](fit_control_cell_identity.parquet) | 本次实际读取的 30,065 个 cell ID、物理 CSR 行及角色/样本/板对应 |
| [gene_axis.parquet](gene_axis.parquet) | 本次完整 32,461 共同基因轴 |
| [source_reads.csv](source_reads.csv) | 1,838 个源 CSR、obs、缓存 manifest/数组的 hash 和实际所选行数 |
| [reference_differences.csv](reference_differences.csv) | 14 个同板与 91 个跨板描述性差异；均值差单位为每细胞每基因原始计数，检出率差无量纲 |
| [repeated_treatment_reference_links.csv](repeated_treatment_reference_links.csv) | 154 组 / 325 个处理条件的 650 个参考链接，含 fit 参考 n 与未读取处理表达标记 |
| [reader_source_versions.json](reader_source_versions.json) | 本次导入读取器及相关模块的精确源码版本 |

NPZ 使用 `allow_pickle=False` 读取。`ids` 是样本 ID 或板 ID，数组第一维按 `ids` 顺序，第二维按 `ensembl_ids` 顺序；不能忽略这两个轴改按文件排序或近似名称拼接。

## 源码版本与验证

并行任务可能随后增加其他来源读取支持。因此启动前已保存 `reader.py`、`common.py`、`sparse_cache.py`、`preprocess.py` 和 `lung_data.py` 的小型源码快照到 `source_snapshots/`，导入后立即核对期间内容未变化。本次来源绑定使用这些快照，不依赖以后可变的工作树源码来解释当时行为。

当前审计入口：[audit_tahoe_reference_expression.py](../../../../../scripts/audit_tahoe_reference_expression.py)。本次执行副本：[audit_tahoe_reference_expression_snapshot.py](audit_tahoe_reference_expression_snapshot.py)。输出目录存在 manifest 时拒绝覆盖；重跑需要指定新目录。

实际执行先运行合成边界测试，然后运行真实读取：

```powershell
& 'F:/anaconda/envs/study/python.exe' -m pytest tests/test_tahoe_reference_expression.py -q
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/audit_tahoe_reference_expression.py
```

**11 项合成测试通过（0.54 s）**，覆盖非法角色、错误样本/真实条码对应、重复 cell/CSR 行、原计数与检出率、细胞加权、缺失参考 NaN、稀疏存储零和非计数拒绝。随后真实进程正常结束；28 份样本与 14 份板摘要均完整回读，所有源细胞身份检查通过。工程测试和文件回读不是生物学显著性或模型性能证据。

## 与整个课题的关系

本结果证明实际存在不同的、来源可追溯的对照参考表达输入，能支持后续预先设计的匹配参考、统一参考、群体均值、plate-only/QC 等对照研究。**尚未运行这些预测比较，不能说状态输入已经带来收益。** 原计数均值差可能受文库深度、批次和质量影响；本次每个源样本平均共同轴文库计数约 1193–2593，说明尺度与技术背景必须在后续方法设计中处理。

这些是相应采样时点的平行对照群体，不是同一处理细胞的实测给药前 t=0。14 块板不等于 14 种已定义生物状态，154 个跨板组不等于 154 个独立生物重复。本轮没有读取任何凋亡功能终点，没有补造同细胞轨迹，也没有证明表达差异能转化为促凋亡筛选收益。输出与研究架构允许的群体参考输入、统一基因轴和独立功能验证边界一致。
