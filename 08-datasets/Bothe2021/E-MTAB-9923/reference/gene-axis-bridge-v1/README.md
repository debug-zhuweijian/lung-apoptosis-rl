# Bothe 官方 GeneID 到 joint32461 的 ID 桥

2026-10-11 实际构建并回读完成，状态为 **Ready ID bridge only**。23,125 个注释可测基因中有 17,107 个取得严格唯一、当前官方交叉引用且在 joint32461 内的身份。真实 RNA counts 恢复量仍为 0；本桥不授予 raw、counts、source decoder、归一化等价或模型资格。

只使用 annotation gene_table 的 `ncbi_gene_id`，精确匹配官方 `GeneID`，读取 `dbXrefs` 中 `Ensembl:` 数据库条目，再精确匹配冻结 `genes.parquet` 的 `ensembl_id`。不用 symbol，不改 ID 版本，不按染色体/assembly 坐标推测，不合并一对多或多对一。many-to-one 检查覆盖全部当前 tax9606 官方记录，而不只检查所选注释。joint 顺序保持其 `common_gene_index` 0–32460；未映射身份和索引为 NA，mask false，无表达或零值填补。

## 官方来源与定位

- [NCBI 官方 human gene_info](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz)，下载前 HEAD 确认 5,194,256 bytes，Last-Modified Fri, 09 Oct 2026 08:44:13 GMT；实际原件在 `official/`。
- [NCBI 官方说明](https://ftp.ncbi.nlm.nih.gov/gene/README)，本地 `official/README.txt` §gene_info 第695–738行定义 tax_id、GeneID、以竖线分隔的 database:value dbXrefs；第184–188行明确 Ensembl gene identifiers 可在 gene_info dbXrefs 获取，且与 gene2ensembl 的结果可有差别。文件每日重算，当前映射不证明作者2021年所用注释年代相同。
- 实际原件包含 193,925 条 tax9606 和 73 条其他 taxonomy 记录；后者全部排除，保留 `official_tax_excluded.tsv`。所有输入/output SHA256 和 bytes 记录于 `manifest.json`。
- 冻结 annotation manifest `c50c69db266c8de1b5ce91f8f0aa364cf2cffbb5a7e185625314037f6200e9c2`；joint genes `4d38a758179d9123fff1ffda1f1eae85fe5f57faa31a3a6df01fcdefbb5314fc`；joint manifest `513c9650acec6613d61c6f32be26d68ec477d7fb9c3c38f43742829443cbba05`，构建时实际重核。未读取或重复hash大型genome index。

## 产物和状态

`joint_axis.tsv` 保存完整32,461行、mask和身份；`source_status_ledger.tsv` 保存全部29,634个选择基因及候选官方ID；`scope_status_ledger.tsv` 保留全部99,517个源transcript的原始reason、NA/mask。scope记录不是独立gene输入，不因同一GeneID存在另一个合格primary transcript而将非primary记录升级。

| source status | 数量 |
|---|---:|
| mapped | 17,107 |
| not_in_joint_axis | 7,039 |
| no_ensembl_dbxref | 2,705 |
| annotation_unmeasurable（有唯一joint对应但注释无feature） | 2,337 |
| ambiguous_many_to_one | 437 |
| absent_current_ncbi_deprecation_unconfirmed | 8 |
| ambiguous_one_to_many | 1 |

状态采用代码规定的顺序：不存在/无xref/一对多/多对一/不在joint/不可测/映射；完整原始annotation_status仍保留，故不可测基因并非只出现在最后一种状态。8个当前缺席的GeneID没有通过gene_history确认退休或替换，保持 unresolved，不能称为已确认deprecated，不追随替换ID。

## 固定 API 与复核

`lung.data.bothe_gene_axis_bridge.load_bridge(directory)` 返回 `(source_ledger, joint_axis)`，实际从已登记官方 ID 原件重新构建，重核全部输入/output hash并与保存表逐字段比较。它只返回元数据，不接收count或模型。

独立复审后当前manifest登记 `validation_revision: 2`，原manifest字节保存为 `manifest.pre-review-v1.json`。Loader将所有8项输入绑定到本轮实际snapshot的固定SHA256与登记resolved路径，并将gene_table/scope身份对照冻结annotation manifest的output_files；不是信任caller自行声明的hash。它要求完整4项ledger并全部从源重建核对，summary重新计算，status只能为ID桥ready、counts只能为整数0，raw/count/source_decoder/normalization_equivalence/model/function资格必须全部严格boolean false。13项小测试包括伪造source及自报hash、缺ledger、伪造model/status/counts与缺官方身份字段；实际源和四ledger未改。

构建：`F:/anaconda/envs/study/python.exe scripts/build_bothe_gene_axis_bridge.py`。已发布manifest存在则拒绝覆盖。独立回读：`F:/anaconda/envs/study/python.exe experiments/20261011-data-alignment-round20/bothe_gene_axis_bridge/independent_readback.py`；其直接读取官方源并重新构造所有xref edges，不调用builder。报告位于同实验目录的 `validation.json` 和 `independent_validation.json`。

本产物符合研究架构§4.3的统一基因身份要求，保留 A549、完整 A/B/C/D 和 E/D/P/C/G/Q 职责。Bothe bulk unique intronic RNA 与joint单细胞来源观测域不同；尚需真实counts/QC、来源出口和域适配，不能据身份交集认定状态、凋亡或序贯监督已成立。原unique-intron划分与全部冻结资产未修改，不是作者exact复现。

## 生产 counts 的最小消费接口

`project_production_counts(version)` 只有一个版本参数，内部调用既有source-bound `bothe_count_production.load_production(version)` 与固定登记桥loader，不接收caller array、CountReadback、路径、hash或runner。它核对完整29,634 GeneID/status和12个sample原顺序，再将17,107个唯一对应投影到32,461 Ensembl轴。返回冻结 `JointBulkCountProjection`：`values` 和 `observed_mask` 均为 gene-major 32,461×12 的嵌套tuple，合法实测值保留Python int（含0及大整数），无对应/不可测/缺sample为None/false。不会转换float或保存object NPY/pickle。

返回域明确为 `bulk_total_RNA_unique_intron_fragment_counts_on_joint32461`，不是单细胞表达或凋亡概率；model/function资格恒false。生产manifest缺失或异常时原样抛错，不返回占位值。metadata桥manifest的counts0快照保持原样；投影对应独立生产version，partial/full状态从source loader取得并再次核对。

CLI：`python scripts/build_bothe_gene_axis_bridge.py --mode readback` 只核对元数据；`--mode project --version <registered-version>` 执行生产source-bound验收后返回摘要，不发布/覆盖数据。该生产loader会按原契约验证真实source链，待真实生产完成后再使用；本轮没有调用它访问raw或大型index。单测对内部loader做隔离monkeypatch仅验证join与整数/NA行为，永远不登记actual counts证据。
