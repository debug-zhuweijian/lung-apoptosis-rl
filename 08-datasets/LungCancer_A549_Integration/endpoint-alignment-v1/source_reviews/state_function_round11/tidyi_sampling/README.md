# TIDYI A549 分选分母与同来源功能数据再审（Round11）

日期：2026-10-10。范围：TIDYI 原研究的 A549 资料、直接官方数据入口及相关作者 notebook。只读复用既有 743 个细胞对齐和 58 个门归属未确认记录；本轮没有重建标签、训练、模型推理或执行 notebook 单元格。

**结论：在核查范围内仍未取得 A549 未富集群体的 Annexin 分母、原始 FCS 或分选抽样概率，不能把富集后的三类频数恢复为自然凋亡率。新进展是取得并亲读了 TIDYI v2 全文、补充和三个作者 notebook，确认有同培养条件的细胞计数/CFSE 功能资料与 10x RNA 的条件级连接，并从作者保存的输出读到了 A549 时间序列。该连接没有达到 RNA sample↔具体 CFSE well/细胞的一一配对；作者保存的 A549 拟合还带有未报告收敛的限制。**

## 1. 本轮版本和读取范围

| 来源 | 本轮状态 |
|---|---|
| [ResearchSquare v2 官方页](https://www.researchsquare.com/article/rs-8252082/v2) | 实际读取 HTML；标题 *Absolute proliferation and death rates in tissues revealed by single-cell transcriptomics*，发布日期 2026-05-27，页面标为 latest preprint、未同行评审 |
| [v2 主文 PDF](TIDYI_v2_main.pdf) | 15 页；从官方页真实 Download PDF 链接下载并解析 |
| [v2 补充 PDF](TIDYI_v2_supplement.pdf) | 26 页；从官方页真实补充链接下载并解析 |
| E-MTAB-16354 现有 study JSON、A549 metadata、v1 Fig. S2 | 只读复用并核对应字段；没有重跑 743/58 对齐 |
| [新取 SDRF](E-MTAB-16354.sdrf.txt) | 8 行 × 66 列；核查样本/因素/衍生文件关系 |
| [Zenodo 17711696](https://zenodo.org/records/17711696) | 只读 Range 核对 54 项中央目录；与旧库存路径及大小一致，45 个非目录项加密，未发现 `.fcs` 文件名 |
| 三个相关 notebook | 按 v2 主文第 10 页作者公开说明读取，未转写访问字符串、未执行代码；总计 Range 读取 7,365,163 bytes，没有下载 3,157,740,883 bytes 整库 |

旧 README 曾因归档加密无法读取 notebook。本轮新读取的 v2 主文第 10 页给出了作者公开的读取说明，故该历史障碍已解除。原始 PDF 保留不改，派生正文文本中的访问字符串已遮蔽；脚本从原 PDF 读取，不含访问字符串常量。

正文与页码可直接查看 [TIDYI_v2_main_text.md](TIDYI_v2_main_text.md) 和 [TIDYI_v2_supplement_text.md](TIDYI_v2_supplement_text.md)。本报告的页码均为 PDF 文件物理页码，包含封面。

## 2. 分选前 Annexin 分母：已得与未得

| 需要的资料 | 本轮实际证据 |
|---|---|
| 未分选 A549 的四象限比例、原群体总事件数 | 未取得 |
| A549 原始 `.fcs` 或未筛选事件表 | EBI study 登记 20 项文件和 Zenodo 54 项库存均未见；所读 notebook 也没有原 FCS 读取流程 |
| 各 gate 的分选数/总体数、入选概率、采样比例 | 未取得；没有能逆校正富集的参数 |
| 已分选单细胞的 FACS 通道与 gate 字段 | 现有 A549 CSV 确实有，但有效行的 `All Events #Events` 均为 1，其他门 count 为 0/1、percentage 为 0/100 |
| 743 个已分选 A549 状态—RNA 对应 | 本轮复用已有核验；不把它作为自然总体抽样框 |
| 58 个原作者阴性标签的具体来源 | 新读 raw notebook 没有生成标签/门映射步骤，没有可据以修订这些标签的新证据 |

实际读取 `SS3_Human_A549_metadata.csv` 的 369 条原始导出记录，仅核查事件计数字段，未重新判标签：

- `All Events #Events`：368 行为 1，1 行缺失。
- `All Events %Parent`：369 行全部缺失。
- `cells #Events`：360 个 1、8 个 0、1 个缺失。
- `dead/APC-/APC+ #Events`：仅出现 0/1；有值的对应 `%Parent` 仅出现 0/100。

这符合逐个 index-sorted event 的导出粒度，字段名中的“Events/%Parent”不能被当成培养物总体数量/百分比。精确各列频数和源文件 hash 见 [existing_event_fields_readback.json](existing_event_fields_readback.json)。

亲视既有 v1 Supplement PDF 第 3 页 Fig. S2：左侧 debris/singlet 示例 gate 有 72.9、86.8 的数字，但没有标定为 A549 的未分选 Annexin 四象限统计，不能与右侧富集后的散点频数组合恢复自然比例。v2 Supplement 第 12 页 Fig. S2 图注明确高 dying/dead 比例来自主动富集。

若要从富集样本比例复原自然群体比例，需要各状态的入选概率或分选前计数。目前缺少这些量；增加状态分类精度也不会自动补出分母。

## 3. 三个 notebook 具体提供了什么

单元格编号在本报告中使用 **原 notebook 的零基索引**。提取出的 `.py` 仅供静态阅读，未执行。

| 文件及实际大小 | 已亲读的关键行为 |
|---|---|
| [00_raw_sc_data.ipynb](00_raw_sc_data.ipynb)，6,234,796 bytes，8 个 cells | cells 0/1 直接载入作者目录中已预处理、已带 `celltype/cell_state` 的 10x/SS3 H5AD；后续标准化、UMAP 和调用分类程序。没有读取原 FCS、原始 gate 生成、采样比例或总体分母步骤 |
| [FitPIModelToCFSE.ipynb](FitPIModelToCFSE.ipynb)，3,411,650 bytes，8 个 cells | cell 0 读取各条件的 `*-n_cells.csv`、`*-integrated_intensity.csv`、`*-integrated_intensity_whole_image.csv`；cell 2 按名为 `well` 的行索引拟合；cell 4 汇总每条件中位数输出 `rates_FinalExp_week2.csv`；cells 6/7 另存 initial implementation |
| [Fig1_panels.ipynb](Fig1_panels.ipynb)，590,404 bytes，24 个 cells | cells 5、16、20 在 `(celltype,sample)` 内计算 RNA 汇总，但取 rate 时只按 `celltype` 选择一个条件级 `a/b`；没有 CFSE well、时间或细胞 ID join |

可读源码：[raw_sc](00_raw_sc_data_source.py)、[CFSE](FitPIModelToCFSE_source.py)、[Fig1](Fig1_panels_source.py)。原始 notebook、仅文本/HTML输出提取与成员 hash 分别保留。没有执行其中的训练、拟合或推理调用。

原 CFSE 的三个 per-well CSV 在 notebook 中指向作者本地目录，未出现在这份 54 项公开归档库存。原始 well 名没有出现在 A549 打印输出；不能用本轮给出的序号假装生物学 well ID。

## 4. 新取得的 A549 功能序列及质量边界

v2 Supplement 第 2 页给出：24、32、48、56、72 h 获取 brightfield 与 CFSE 影像，每 well 60 张视野；用分割量化细胞计数和 CFSE。第 3–4 页说明从两类测量拟合 proliferation、death、clearance 等参数，并对 clearance 时间施加软先验。第 11 页 Fig. S1 指向这些时间序列与参数。

`FitPIModelToCFSE.ipynb` 原 cell 2、output 4 的 A549 保存输出，实际包含 **12 条序列 × 5 个时间点**：

- 已整理为 [A549_author_saved_cfse_curves.csv](A549_author_saved_cfse_curves.csv)，共 60 行；保留 source cell/output 索引和打印序号。
- source code 将时间减去最早观测时刻，保存输出为 `0,8,24,32,48`。表中的 `time_h_from_paper_schedule` 依照方法加回 24 h；原 per-well CSV 表头并未取得，因此这个时钟还原的证据性质已单列。
- 第一条打印序列的计数为 `132,147,193,236,340`，CFSE 值为 `835423,669793.5,366739.5,260644.5,128331`；它们是作者保存输出值，未经本轮重算，也未确认原 CSV 的全部视野聚合细节。
- 这里的“12”是按源码 well 行循环打印的序列数；没有物理 well ID、实验批次或生物重复表，不能把它自动报告为 12 个独立生物重复。

同时保留 [A549_author_saved_fit_parameters.csv](A549_author_saved_fit_parameters.csv)。**12 次 A549 主拟合的 `best_res.success` 保存值全部为 False**；输出没有给出每次失败原因。代码仍按目标值挑选最佳解并继续汇总。因此这些参数是“作者保存的拟合输出”，没有在本轮被证明收敛或可靠，不能直接登记为经验证的训练标签。False 也不单独证明每个数值必然错误，具体原因需另行复核。

`Fig1_panels.ipynb` cell 4 保存的条件表已提取为 [Fig1_author_saved_condition_rates.csv](Fig1_author_saved_condition_rates.csv)。其中 A549 只有一行，显示 `b=0.060744 /h`、`a=0.042319 /h`、`g=0.1866681 /h`。这些是模型率，量纲为每小时；绝不能将 `a` 填成 4.2319% Annexin 阳性率。另一个 initial implementation 的输出与主拟合不同，本轮没有据效果择优替换。

上述观测能够支持的新增内容是：作者公开保存了 A549 计数/CFSE 时间数据及相应条件级模型率记录。它没有提供各时间点的实测 Annexin 阳性比例或 Annexin 分母。

## 5. E-MTAB-16351 的 RNA 与 CFSE 能对应到哪一层

v2 主文第 3 页、Fig. 1 和 Supplement 第 7 页将 10x scRNAseq 与相同 cell-culture condition 的纵向计数/CFSE 率用于校准。Data availability（主文第 10 页）直接登记 E-MTAB-16351 与 E-MTAB-16354，两者职责不同：

- E-MTAB-16354 的三状态富集 Smart-seq3 用于状态签名。
- E-MTAB-16351 的 10x 样本未按上述三状态进行 FACS 富集，但仍经过捕获、QC 等技术选择，不能自动当成无偏自然 Annexin 分母。

本轮只读复用并实际核对现有 10x 小元数据：

| 文件 | 行数 | sample | batch | celltype |
|---|---:|---|---|---|
| Chromium10x_1001_A549_metadata.csv | 1,208 | 1001 | 0 | A549 |
| Chromium10x_1002_A549_metadata.csv | 1,251 | 1002 | 1 | A549 |

它们各有 24 列（含 cell ID 索引列），其余为 barcode、sample、batch、celltype 和 RNA QC 等元数据，**没有 CFSE well、时间、原培养重复或逐细胞 Annexin 键**。完整列名/hash 见 [A549_cfse_and_10x_linkage_readback.json](A549_cfse_and_10x_linkage_readback.json)。

作者 Fig1 notebook 的实际 join 是：对每个 `(celltype,sample)` 汇总 RNA，再按 `celltype` 从六条件率表选同一行 `a/b`。既有固定作者源码 `tidyi_main.py.source.txt` 的 `train_death_rates`（153–159 行）也按该规则赋值。不同 sample 使用同一个条件率，不能作为两次独立的 CFSE 终点测量。

因此可确认“同作者、同培养条件层面”有连接依据；**尚无 sample 1001/1002→某条 CFSE well 曲线的映射，也无单细胞 RNA→同细胞未来命运配对**。RNA 中的死亡 fraction 在代码内来自状态分类器预测，仍不是实测 Annexin 百分比。

## 6. 药物、时间和直接引用

v2 Supplement 第 5 页和现有 EBI study JSON 一致：Paclitaxel 1 µM 处理对象为 MDA-MB-231 clone L，A549 只是本研究培养条件之一。新读 notebook 列出了 A549、4T1、MDA-A/L/S 及处理条件，未给 A549 新增药物或剂量。

A549 新核实的时间数据是计数/CFSE 功能时间序列，不是相应每个时间点的 RNA–Annexin 测量，也不是多种用药持续时长的随机干预比较。

Supplement 第 26 页列出的相关直接引用中，ref. 1 的题目是 MDA-MB-231 spheroid 的显微表型研究；ref. 5 是 lymphocyte CFSE time-series 方法；refs. 3/4 是 Smart-seq3xpress 方法。A549 培养来源写为实验室赠与，功能实验由本研究完成。在所读 A549 方法、数据可用性和这些直接引用条目中，未定位另一个 A549 未富集 FACS 分母的公开入口。本轮未据引用题名推断被引全文，也未扩大搜索其他不相关研究。

## 7. 明确未取得的字段与当前使用范围

未取得：分选前 A549 各 Annexin/BOBO gate 的全事件数和阳性数；原始 FCS；各状态入选概率/分选比例；RNA sample 与物理 CFSE well/批次/时间的映射；CFSE 原 per-well CSV；A549 多药多剂量实测 Annexin 标签。

本报告的负面结论限定于：v2 正文/补充、现有 EBI study/SDRF 与 A549 小元数据、54 项 Zenodo 库存及三个最直接相关 notebook。未解析整库大型表达对象或所有其他 notebook，不能据此宣称作者从未采集、任何地方都不存在这些字段。

当前可保留三种不同用途：已分选同细胞状态—RNA 用于有偏抽样下的状态识别；10x A549 用于来源内分布/迁移研究；计数/CFSE 与条件率用于有明确模型假设和质量限制的条件级校准研究。自然 Annexin 比例监督仍缺分母，现有 743/58 标签处理和正式训练资产保持原样。

## 8. 关键 hash 与验证

| 文件 | SHA256 |
|---|---|
| TIDYI_v2_main.pdf | `1c037016a661107df429535b0a7dfeae59e1d98f60d2882e8e190d8e55eefe41` |
| TIDYI_v2_supplement.pdf | `c7e02b3529763c74aa35f43d8a281d1920dd26fb8f8411b28cc743e9a1a44638` |
| 00_raw_sc_data.ipynb | `9647697a923a99f4f70fa7dccd4f61f34c68b00955af02f21c7e4bf7bea45c18` |
| FitPIModelToCFSE.ipynb | `f42092a8819ebd24a2a2ccb0b290294a3c8cd732fcbcd38a57ea1b11687eac16` |
| Fig1_panels.ipynb | `19eb5191864f51dfb00e028b30df17eaf1b0fa0f73a59c7376677ef9378de619` |

每个 Range 的范围、长度和 hash 见 `selected_notebook_fetch.json` 与 `figure1_notebook_fetch.json`；ZIP 成员解压通过其 CRC 检查。服务未返回 ETag/Last-Modified，本报告不声称已计算整库 SHA256。Zenodo API 两次超时的事实另记；网页、实际 Range 和解压证据分别报告。

完整审查文件与只读输入绑定见 `manifest.json`，本地校验结果见 `verification.json`。校验只重读、比对已有文件和保存输出，不拟合、不执行 notebook、不运行模型。所有新增文件均限于本目录。
