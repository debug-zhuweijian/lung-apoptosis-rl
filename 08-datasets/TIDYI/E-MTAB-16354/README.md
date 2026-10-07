# TIDYI A549：E-MTAB-16354 数据审计与来源内初步基线

> 2026-09-19 筛选与文件保留状态：当前选中 743 个唯一 A549，旧 CSV 的 368 个重复细胞不重复计数。已核对完整作者归档及成员 SHA256 后，清理 `audit_sources/human_ss3.h5ad` 的可恢复解压副本；`classifier_data.tar.gz`、A549 NPZ、标签与基因注释仍保留。下文 H5AD 读取与审计是历史执行记录，当前物理状态及筛选结果见[数据筛选 v1](../../LungCancer_A549_Integration/selection-v1/README.md)。

核验日期：2026-09-18。状态：**官方 A549 CSV 完整下载、实际读取、质控审计、固定协议来源内分类基线已运行；进一步取得作者训练对象中的第二块 A549 板，并完成固定参数跨板诊断**。这不是 TIDYI 论文复现，也不是 Tahoe 跨来源验证或真实药物凋亡率预测。下文原 368 细胞结果保持不变，第二板结果独立记录。

## 来源与原始文件

- 官方研究：[E-MTAB-16354](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-16354)，标题 *In vitro growth signature identification from SmartSeq3 sequencing of 5 human cancer cell lines and 1 mouse cancer cell line*，公开日期 2025-12-05。
- 当前只下载指定 A549 导出，不下载整个研究或原始测序读段。
- [SS3_Human_A549_counts.csv](https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_A549_counts.csv)：49,296,053 字节；SHA256 `b33aa1b03d71972f8e8c33864be0f0e2747ffa8b56289d00f27cd8c276cc3360`。
- [SS3_Human_A549_metadata.csv](https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_A549_metadata.csv)：250,782 字节；SHA256 `6ee300a8ba36642950c94d3be7455beb51255e42d8f311273d00aa9e0868a38b`。
- [官方 study JSON](https://www.ebi.ac.uk/biostudies/api/v1/studies/E-MTAB-16354) 已保存为 `E-MTAB-16354.study.json`，16,912 字节；SHA256 `eeeb15af1e95877bff4af43f6b30f2162ffc456883bef97e453393329e192687`。
- 下载 UTC 时间、URL、字节数、SHA256 见本目录 `download_manifest.json`。原始 CSV 未修改，派生表使用独立文件名。

## 官方实验说明与标签解释边界

官方 study JSON 的 Study/Description 说明：培养细胞经过分选，将 BOBO3−/AnnexinV−、BOBO3−/AnnexinV+、BOBO3+/AnnexinV+ 分别描述为 live、dying/early apoptotic、dead/necrosis or late apoptosis，再进行 Smart-seq3xpress。该实验保留培养上清以保留死亡细胞。

**当前 CSV 的原始标签是 `PI-APC-`、`PI-APC+`、`PI+APC+`，其 PI 命名与官方 BOBO3 说明不同。** 2026-09-18 追加核查已确认作者代码确实使用这些字符串，并将前者映射为 Alive、后两者合并为 Dying；但尚未找到从原始 FACS 通道/阈值生成 PI 字符串的处理证据。本次三分类基线仍直接预测三个原始字符串，不重标标签、不将 `PI+APC+` 统一命名为纯凋亡。详细原始出处见下方追加核验。

官方 Paclitaxel 1 μM 条件明确作用于 MDA-MB-231 clone L，不能移植成 A549 的处理条件。本 A549 表没有可用于监督不同药物/剂量/处理时长—真实凋亡比例的逐条件标签。分选后各组细胞数量也不是未选择总体的凋亡率。

官方 Protocol `P-MTAB-161163` 说明原研究分析有 10,000 reads/cell 再 log1p 的归一化步骤；**本次下载的 counts CSV 全部为非负整数，不能当成已经归一化的矩阵**。本次基线从该计数矩阵重新归一化。

## 完整矩阵审计

| 项目 | 实际结果 |
|---|---|
| 表达矩阵 | 369 个细胞 × 32,858 个唯一 Ensembl 基因编号 |
| 元数据 | 369 行 × 89 字段，另有 cell ID 索引 |
| 样本对齐 | 两表 ID 集合完全一致、唯一，分析按元数据 ID 重新索引 |
| 非有限/负值/非整数 | 均 0 |
| 最大单个计数 | 3,222 |
| 全零细胞/重复表达行 | 均 0 |
| 全零基因 | 8,417 个；不在全部数据上预先做用于模型的方差筛选 |
| 表达矩阵非零比例 | 18.9330% |
| 细胞总计数范围/中位数 | 1,095–133,069 / 40,650 |
| 检出基因范围/中位数 | 209–10,639 / 7,046 |
| 板号 | 全部 `Plate=8` |
| 来源 predicted_doublets | 全部 False；该字段不是本次重新检测的结果 |
| 原元数据 QC 与导出计数差异 | 269 个细胞至少一项不同，原 `total_counts`、`n_genes_by_counts` 均较本矩阵重算值最多大 9；原因未在所读官方说明中确认 |

小差异保留于 `A549_source_qc_discrepancies.csv`，不能据此声称两者统计完全一致。本次表达处理使用实际导出矩阵重算值。未以常见健康细胞的线粒体比例或低 RNA 阈值批量排除分选死亡相关群体，以免直接删除研究目标；其低计数/高线粒体比例局限单独报告。

| 原始状态标签 | 全部记录 | 严格 A549 且表达非空 |
|---|---:|---:|
| PI-APC- | 168 | 167 |
| PI-APC+ | 93 | 93 |
| PI+APC+ | 108 | 108 |
| 合计 | 369 | 368 |

缺失 `Cell_line` 的一个记录是 `ATAGGAGGCGCACAGATGGA`，Well=E20，标签 PI-APC-。原始矩阵继续保留，本次严格 A549 基线排除此记录，不按文件名推断并填补细胞系。

## 新发现：状态与孔板位置有关联

**本次已读 CSV 导出的全部样本**来自同一块板，孔列 1–8 全部是 PI-APC-；9–16 主要是 PI-APC+；17–24 主要是 PI+APC+。每列原始分布见 `A549_well_column_by_state.csv`。这不代表原研究 A549 只有一块板；追加查阅原始 Fig. S2c 时看到 A549 含两种板颜色，具体见下文。

因此，按单细胞随机划分可能给出过于乐观的来源内结果。本次改为**按孔列分组**的外层交叉验证，训练与测试不共享同一列；但仍会在同一块板、同一状态位置区带间插值。**这种划分不能排除状态与板位/技术过程的混杂，也不能证明表达预测全由技术伪影造成。** 需要独立板、独立实验或外部标注来区分。

## 已执行的基线协议

协议在第一次评估前保存为 `A549_baseline_protocol.json`，并记录脚本与原始计数 SHA256。没有依据外层测试表现调整超参数，也没有从多次随机划分中择优报告。

1. 固定 Tahoe 快照 `2dc57900b7981cfcf5e211527169a0b006546a95` 的基因词表，与 TIDYI 按 Ensembl ID 求交，得到 32,461 个共同 ID。只使用词表 ID，不用 Tahoe 响应值或标签筛选特征。
2. 使用严格 A549 的 368 个细胞。固定 seed=20260918，外层 4 折 `StratifiedGroupKFold`，分组为孔列；每折训练/测试均有三类且孔列不重叠。测试折大小为 92、92、91、93。
3. 各细胞在共同基因面板内按总量归一到 10,000，再 log1p。这一步只依赖该细胞自身，不拟合全样本参数。
4. 每个训练折内筛选至少 5 个训练细胞检出的基因，取训练方差最高 500 个，训练折内标准化；固定 L2 多类逻辑回归 C=0.1，max_iter=3000。
5. 校准版在每个外层训练集内进一步做 3 折按孔列分组的 sigmoid 校准，`CalibratedClassifierCV` 包住整个表达预处理＋分类 Pipeline。基因筛选与标准化均在各内层拟合子集中重新学习。校准版集成内层模型，未将全外层数据预拟合的变换泄漏给内层校准集。
6. 同时运行固定的训练类别先验、未校准表达模型、孔列位置单变量树（深度 3 的混杂诊断）、QC-only 逻辑回归。QC-only 只用本矩阵 log 总量/log 检出基因数及元数据线粒体比例，表达模型不使用这些元数据字段。流式 gate、孔号、细胞状态不作为表达预测器输入。

这只是本项目自建的初步诊断基线，不是已优化的最终评价模型。没有导出可部署终点奖励模型。

## 实际来源内结果

全部为 368 个细胞的外层 out-of-fold 预测，三类顺序固定 `PI+APC+`, `PI-APC+`, `PI-APC-`。

| 固定方法 | Accuracy | Balanced accuracy | Macro F1 | Macro OVR AUROC | 多类 Brier↓ | Log loss↓ |
|---|---:|---:|---:|---:|---:|---:|
| 训练类别先验 | 0.4538 | 0.3333 | 0.2081 | 0.4892 | 0.6443 | 1.0663 |
| 表达逻辑回归，未校准 | 0.7500 | 0.7437 | 0.7415 | 0.8735 | 0.3653 | 0.6222 |
| 表达逻辑回归，内层 sigmoid 校准与集成 | 0.7527 | 0.7098 | 0.7018 | 0.8795 | 0.3493 | 0.5625 |
| **孔列位置单变量诊断** | **0.8560** | **0.8789** | **0.8566** | **0.9099** | **0.2388** | 1.9199 |
| QC-only 诊断 | 0.6304 | 0.5443 | 0.4967 | 0.8073 | 0.4748 | 0.7965 |

Brier 定义为每细胞三个类别平方误差之和再取均值，范围 [0,2]，非除以类别数版本。AUROC 是 one-vs-rest 三类宏平均。校准前后模型训练集组合不同（校准版使用内层集成），因此两者差异不能完全归因于 sigmoid 映射。10 个等宽置信度箱的 top-label ECE，未校准为 0.1048，校准版为 0.0839；小样本 ECE 仅作辅助诊断。

**未校准和校准版的 PI-APC+ 召回率分别为 50/93=0.5376 与 26/93=0.2796。** 校准版概率分数改善不等于三类识别全面改善，更不等于凋亡识别已经可靠。

孔列本身即可取得很高分类成绩，结合状态×列分布，提示必须优先解决实验设计混杂和外部验证；不能把本表中的 AUROC/accuracy 作为可泛化到 Tahoe 的凋亡评价精度，也不能称其为真实凋亡率校准。未计算把 368 个细胞当作 368 个独立实验重复的显著性结论。

## 文件、读取与复现

主要派生证据：

- `A549_audit.json`：机器可读审计结果。
- `A549_cell_qc.csv`、`A549_gene_qc.csv`、`A549_exclusions.csv`：逐细胞/基因审计与严格入组排除。
- `A549_gene_ids.csv`：32,858 个基因 ID，供跨数据词表审计。
- `A549_expression_audited.npz`：全部 369 行 float32 计数、cell_ids、gene_ids、eligible 严格入组掩码；不包含已拟合的降维或基因选择。
- `A549_source_qc_discrepancies.csv`、`A549_state_expression_summary.csv`、`A549_well_column_by_state.csv`：来源 QC 差异、分组描述、位置关联。
- `A549_baseline_protocol.json`、`A549_baseline_results.json`：固定协议、环境版本、分折孔列与完整指标/混淆矩阵。
- `A549_baseline_out_of_fold_predictions.csv`：每个样本外层折号、原始标签及各固定方法预测概率。
- `A549_baseline_calibration_bins.csv`：校准箱计数、平均置信度、实际准确率。

在 paperstudy 根目录已实际运行：

```powershell
python scripts/prepare_tidyi_a549.py
python scripts/tidyi_a549_baseline.py
```

准备脚本重复执行时复用原始文件并校验 SHA256，不覆盖已存在来源文件。基线脚本若发现协议脚本 SHA 改变会停止，防止在同一实验名下悄然变更已评估协议；新模型应另存版本并明确重新验证。

本次环境为 Python 3.13.12、NumPy 2.3.1、pandas 2.3.3、scikit-learn 1.8.0（以机器可读协议实际记录为准）。依赖 requests、threadpoolctl、parquet 读取后端；后者用于 Tahoe 基因字典。

```python
from pathlib import Path
import numpy as np

p = Path('08-datasets/TIDYI/E-MTAB-16354')
a = np.load(p / 'A549_expression_audited.npz', allow_pickle=False)
x = a['expression'][a['eligible']]
ids = a['cell_ids'][a['eligible']]
genes = a['gene_ids']
```

数据 CSV/NPZ/JSON 均由数据目录现有规则忽略，不进入 Git/Git LFS；本 README 与项目 scripts 可版本化。本轮未提交 Git。

## 待补条件与明确不能声称的结果

- [ ] 原论文补充实验/分选代码核对 PI/APC 与 BOBO3/AnnexinV 的实际对应，核查异常分选标签和缺注释细胞；不自行补标签。
- [ ] 查清原元数据 QC 与公开计数导出的小差异原因；本次分析已使用实际矩阵重算值。
- [x] 在作者公开训练对象中找到 A549 第二板并核对计数、ID、标签，完成固定参数跨板诊断；结果见后文。
- [ ] 补充独立实验/平台验证，或能打破共同孔列位置—标签关联的数据；同研究两板仍不等价于充分外部验证。
- [ ] 验证表达模型跨平台转移到 Tahoe，尤其死亡相关低 RNA 状态的筛选偏差、归一化、特征可测性、类别比例差异。共同基因只是接口条件。
- [ ] 对对应药物、剂量、时长和细胞背景的实测凋亡比例做单独结局校准；不得将本次 sigmoid 分类校准当成这一任务已经完成。
- [ ] 真实序贯用药的独立验证仍需另行匹配数据。

当前可支持“公开 A549 三种原始分选标签与表达的来源内学习有可计算起点，同时观察到较强板位关联”；不能支持药效因果、真实凋亡比例、跨药物/跨细胞系/跨平台泛化，也不能直接用于宣称 DA-AC 或 DLPA 优于其他方案。

## 追加原文与作者代码核验（2026-09-18）

本轮 Zotero Local API 对用户库顶层条目按 `TIDYI`、DOI 前缀 `10.21203/rs.3.rs-8252082`、v1/v2 标题检索，均有效返回空列表，因此改查作者公开原文与代码。该结论只针对这些精确查询，不声称已逐条检查全部 Zotero 条目。

1. **原始补充图已读。** Research Square v1 补充材料第 3 页 Fig. S2c 明确标注 BOBO3/AnnexinV 四象限：Q1 live、Q2 early apoptotic、Q3 late apoptotic、Q4 necrotic；并说明主动富集 dying/dead 群体，故其高比例不等于培养物总体低存活率。A549 面板可见板 1 与板 8 两色。原图及 PDF 保存于 `audit_sources/Rosenbauer2026_TIDYI_v1_FigS2.png` 和 `audit_sources/Rosenbauer2026_TIDYI_v1_Supplement.pdf`。[官方补充材料 v1，Fig. S2](https://assets-eu.researchsquare.com/files/rs-8252082/v1/8c523a853dbaa2640d07e243.pdf)
2. **作者包标签用法已读，尚未运行作者包。** `jhausserlab/TIDYI` commit `d8643cec80fb73c566df8b7f835870b65e50b21e`，`tidyi/tidyi_main.py` 的 `train_death_classifier`，L97–108 明确使用 `PI-APC- → Alive`、`PI-APC+ → Dying`、`PI+APC+ → Dying`。这解决了“作者是否使用此 PI 字符串”的问题，并明确作者进行二分类；它没有解释该命名的实验来源，也不能直接提供本项目三分类的原始门映射。[固定 commit、L97–108](https://github.com/jhausserlab/TIDYI/blob/d8643cec80fb73c566df8b7f835870b65e50b21e/tidyi/tidyi_main.py#L97-L108)
3. **官方 A549 h5ad 远程元数据核对。** 对 `SS3_Human_A549_adata.h5ad`（200,429,210 字节）使用 HTTP Range 只读取 HDF5 `obs`，仍是 369 条、全部 Plate=8、368 个明确 A549＋1 个缺注释，同 CSV 三类计数；不是补充图另一板的直接来源。本轮没有下载整份该 h5ad。[官方文件](https://www.ebi.ac.uk/biostudies/files/E-MTAB-16354/SS3_Human_A549_adata.h5ad)
4. **分析档案定位。** [Zenodo 17711696](https://zenodo.org/records/17711696) API 可用；公开 `tidyi_notebooks_11_2025.zip`（3,157,740,883 字节）中央目录通过 Range 读取并存为 `audit_sources/Zenodo17711696_zip_inventory.json`，其中有 `development/00_raw_sc_data.ipynb`，但该条目加密。所尝试的官方主文 PDF 读取返回 HTTP 403，因而未成功读取该 notebook，不推测其阈值、重标注或过滤步骤。
5. **另一板相关来源定位。** 作者 `data_utils.py` L39–46 规定公开 classifier_data 中包含 `human_ss3.h5ad`；L11–12 给出公开下载入口。该训练对象是确认 A549 另一板是否可用的直接候选，应核对其样本和原始表关系；与 CSV 来源保持分开，不覆盖已冻结的基线。[作者数据列表](https://github.com/jhausserlab/TIDYI/blob/d8643cec80fb73c566df8b7f835870b65e50b21e/tidyi/data_utils.py#L39-L46)

本轮原文补充材料是 v1；搜索发现 v2 题为 *Absolute proliferation and death rates in tissues revealed by single-cell transcriptomics*，但没有把尚未读到的 v2 方法当成已核实证据。

## 作者训练对象：第二板取得与逐值一致性核验

上述第 5 个入口随后实际下载成功。公开 Google Drive 文件由作者固定 commit 的 `data_utils.py` 指定；保存为 `audit_sources/classifier_data.tar.gz`，只提取了与本任务直接相关的 `human_ss3.h5ad`，没有解压其中无关的其他矩阵。

下载过程先尝试顺序流读取，在至少 1,400,053,760 个压缩字节后持续无进度而停止；随后用 16 MiB HTTP Range、四并发、每段最多三次、420 秒总界限完成。各段核对 HTTP 206、Content-Range、实际字节数，并使用 If-Range 与 Last-Modified 固定来源版本，下载后再次探测版本。

- 来源无 ETag；Last-Modified 为 `Fri, 31 Oct 2025 15:49:42 GMT`，各段一致。
- 完整压缩包：2,935,132,934 字节；SHA256 `b672a598576140058c656e05f6dc84a7a32e191b0148cc8d473f69fad48f1919`。
- `human_ss3.h5ad`：2,201,171,391 字节；SHA256 `8b6a32e7049c1b80d6c332a52de0ec26b44b795e81001dd8e0460853ab318a04`。
- 完整下载证据为 `audit_sources/human_ss3_download_manifest.json`；获取脚本 `audit_sources/fetch_author_archive.py` 保留于本地。以上 SHA 是本次完整下载实算值，不冒充作者发布的校验值。

`human_ss3.h5ad` 实际含 4,169 个细胞 × 32,858 基因；其中明确注释 A549 的 743 个细胞分布如下：

| 板 | PI+APC+ | PI-APC+ | PI-APC- | 总数 |
|---|---:|---:|---:|---:|
| Plate 8 | 108 | 93 | 167 | 368 |
| **新增 Plate 1** | **102** | **136** | **137** | **375** |

严格 Plate 8 的 368 个 A549 ID 与此前 CSV 严格入组集合完全一致，原始标签无冲突；按 ID 和 Ensembl ID 对齐后，`layers/counts` 与此前 CSV **每个计数值完全一致，最大绝对差为 0**。Plate 1 的 375 个 ID 均不与原 CSV 重复，不是原样本换名后的伪独立集。

两板 743 个细胞所读计数均非负、有限、整数；新增板无全零细胞，计数总量范围 1,177–145,416（中位数 18,672），检出基因范围 201–10,490（中位数 4,519）。两板仍有类似的孔列状态区带，不能认为增加第二板已经消除了共同实验设计的混杂。

可复用证据：`audit_sources/human_ss3_metadata_audit.json`、`human_ss3_A549_metadata.csv`、`A549_author_collection_counts.npz`、`A549_author_collection_cell_qc.csv`、`A549_author_collection_plate_column_by_state.csv`、`A549_author_gene_annotations.csv`。NPZ 中 `expression/cell_ids/gene_ids/plates/states` 保存 743 个明确 A549 的计数、原始 ID 和标签；旧 369 行文件及旧基线未覆盖。

## 固定参数跨板诊断：Plate 8 → 新增 Plate 1

第一方向预先指定为 **Plate 8 的 368 个细胞训练 → Plate 1 的 375 个此前未用于训练的细胞测试**。反向 Plate 1 → Plate 8 仅为敏感性分析，不当作第二份独立验证。无超参数搜索，没有依据第二板结果改变基因面板、模型参数或 seed。

表达模型复用已冻结来源内基线的全部预处理和 LR 参数；内层三折校准只在训练板内按孔列划分，每折训练/校准均包含三类，完整 Pipeline 位于内层。QC-only 改为从原始计数与共同面板重算三项特征；`var.mt` 对应的 37 个线粒体基因与 `gene_symbol` 以 `MT-` 开头的规则完全一致，37 个均在共同面板内。

**主方向结果（n=375）：**

| 固定方法 | Accuracy | Macro F1 | Macro OVR AUROC | 多类 Brier↓ | Log loss↓ |
|---|---:|---:|---:|---:|---:|
| 训练类别先验 | 0.3653 | 0.1784 | 0.5000 | 0.6814 | 1.1209 |
| 表达未校准 | 0.5520 | 0.5271 | 0.7415 | 0.7384 | 1.6637 |
| 表达内层校准集成 | 0.5520 | 0.4813 | 0.7736 | 0.5657 | 0.8773 |
| 孔列位置诊断 | 0.8987 | 0.8992 | 0.9150 | 0.2300 | 1.7188 |
| QC-only | 0.5147 | 0.4393 | 0.7591 | 0.5961 | 0.9795 |

| 主方向逐类召回 | PI+APC+ | PI-APC+ | PI-APC- |
|---|---:|---:|---:|
| 表达未校准 | 82/102 = 0.8039 | **16/136 = 0.1176** | 109/137 = 0.7956 |
| 表达内层校准集成 | 86/102 = 0.8431 | **2/136 = 0.0147** | 119/137 = 0.8686 |

校准版 Brier/Log loss 改善，但 PI-APC+ 的 argmax 识别几乎失败。未校准表达模型的 Brier 与 Log loss 甚至比训练先验差，说明不能仅凭 AUROC 宣称概率可靠。孔列诊断高分仍是混杂警示，不能作为有效生物分类器成果。

**反向敏感性（n=368）：** 未校准表达模型 Accuracy=0.5679、Macro F1=0.5833、AUROC=0.7891、Brier=0.6253；校准集成为 0.5679、0.5638、0.7924、0.4975。反向同样未显示稳定的三类识别，不能用反向较好的某项值替换主结果。

实际执行：

```powershell
python scripts/tidyi_a549_cross_plate.py
```

固定协议为 `A549_cross_plate_protocol.json`，含新脚本及复用旧脚本 SHA；完整指标、逐类召回、混淆矩阵见 `A549_cross_plate_results.json`；每个测试细胞每个固定模型的概率见 `A549_cross_plate_out_of_sample_predictions.csv`，两方向标记清楚。校准箱另存 `A549_cross_plate_calibration_bins.csv`。概率有限且每行和为 1，训练测试 ID 无重叠。

这轮把“是否有独立板可检验”推进为**取得第二板并观察到跨板识别薄弱**。它没有证明所有更强表达模型必然失败，也没有验证 Tahoe 跨平台迁移、药物结局或真实序贯效应；现阶段不能把本三分类器直接当成可靠终点奖励模型。
