# Kotani 2024：A549 诱导状态与 ABT-263 后 Annexin 终点核验

核验日期：2026-10-10。论文：*Therapeutic Senolysis of Axitinib-Induced Senescent Human Lung Cancer Cells*，Cancers 16(16):2782，DOI [10.3390/cancers16162782](https://doi.org/10.3390/cancers16162782)，[PMC11352446](https://pmc.ncbi.nlm.nih.gov/articles/PMC11352446/)。本轮只保存原件、协议核验和图中数据；未改核心表、代码、配置，未训练、未联系作者。

**原文确实有“无 axitinib 对照/axitinib 诱导状态 → ABT-263 再挑战”的可读取定量 Annexin 数据。关键是 Fig4B/C，Annexin 分母是未解体细胞（non-destroyed，ND）门内事件。按这个定义，数据可以用于独立的状态类别挑战。它不是原始培养群体的整体凋亡率，也没有形成多小分子候选的 Annexin 排序数据。**

## 1. 实际取得和亲读的材料

| 文件 | 实际内容与定位 |
|---|---|
| `PMC11352446.xml` | Europe PMC 官方全文，Methods §2.1–2.12、Results §3.3–3.6、Fig3–6 图注及 Data Availability 已读 |
| `cancers-16-02782.pdf` | MDPI 官方论文 PDF，17 页；Methods 第 3–5 页，Fig3 第 8 页，Fig4 第 9 页，Fig5 第 10 页，Fig6 第 11 页 |
| `cancers-16-02782.md` | 对上述 PDF 的逐页文字抽取；用于文本行号定位，排版与原图以 PDF 为准 |
| `cancers-16-02782_p09_image1_Im0.png` | 从官方 PDF 第 9 页直接提取的 Fig4 图像，2219×2370，未重画 |
| `cancers-16-02782-g004.jpg` | PMC 官方 CDN 的 Fig4 缩略大图，766×802；仅作额外原图复核，不用于数值提取 |
| `cancers-16-02782-s001.zip` | MDPI 官方补充 ZIP，152,475 字节，实际只有下一份 PDF |
| `cancers-3131450-supplementary.pdf` | 1 页，Supplementary Figure S1 原始 Western blot；没有 Annexin 源表或原始 FCS |
| `cancers-16-02782_p08_image1_Im0.png` 等 | PDF 中的 Fig3、Fig5、Fig6 原始嵌入图片，保留用于状态/终点类别核验 |

Data Availability Statement 说明 raw data 可向作者索取。本轮没有联系作者；公开补充没有提供 Annexin 数值源表。因此取得的是发表的流式图百分比与可数字化的均值，不能称为原始事件或原始重复数据。

## 2. 前置状态、药物与时间的实际连接

依据 Methods §2.1、§2.3、Fig4 图注和 Results §3.4：

- A549 来自 ATCC；培养于 RPMI-1640，10% FBS、20 µg/mL gentamicin，37 °C、5% CO2。
- 两种群体历史：不加 axitinib，或 **axitinib 2.5 µM × 3 天（72 h）**。
- 然后 **harvest，再培养于 ABT-263 1 µM**。原文有这一先后操作，不能说仅为同加药实验。
- 重新培养时是否换为完全不含 axitinib 的培养基、洗涤次数、撤药后间隔、axitinib 残留/继续给药，已读主文没有明确。`harvest` 不自动等于经核实的完整洗脱。
- Fig4 的无 axitinib 群体给出无 ABT 与 ABT **24 h**；axitinib 群体给出无 ABT 与 ABT **6、12、24 h**。因此有两个状态同为 1 µM ABT-263、24 h 的直接条件比较。
- 图中两个无 ABT 条件的横轴是 “−”，没有标共同采样时钟。保留 `challenge_duration_h=null`，不自行视为 24 h vehicle，更不复制成 6/12/24 h 三套对照。
- 无药/无 axitinib 是图示条件标签；Methods 未指定相应溶剂种类和浓度，本轮不补写 DMSO vehicle。

6/12/24 h 是不同的处理后测量时点。它们没有给出“不同暴露时长后洗脱、在共同 T* 观察”的设计，不能据此验证固定终点下的暴露时长选择。

## 3. 分母、门控与重复

Results §3.4 明确，流式收集培养孔中的 **whole cells and debris**。Fig4B 上排标注 All Events，通过 FSC 得到 ND 群体；下排是 **ND cell-gated** 的 Annexin V-FITC/PI 图。

| 读数 | 分母/含义 | 可保留的用途 |
|---|---|---|
| ND cells (%) | 所采集事件中属于 ND 门的百分比；包括 cells/debris 的采集背景 | 记录解体与门保留情况的伴随终点 |
| Annexin V+ (%) | ND 门内 Annexin 阳性事件比例，包含 PI 阴性与阳性象限 | 原文定义下的凋亡相关条件终点 |
| PI+ (%) | ND 门内 PI 阳性事件比例 | 膜完整性/死亡相关伴随终点，不能单独等同凋亡 |
| Fig4B 三个印字象限 | ND 门内 Annexin−/PI+、Annexin+/PI+、Annexin+/PI− | 图示样本的各象限百分比，区别于 Fig4C 均值 |

该分母条件不使比较失去合法性，但限定了预测目标。严重解体的事件不会以完整 ND 细胞进入 Annexin 分母；不能把低 ND 比例当作技术无效，也不能把 ND 内 Annexin 阳性率当作最初接种细胞的总凋亡率。没有原始事件数、培养总细胞数或回收率，不能据这些图恢复自然群体死亡率。

试剂：Annexin V-FITC Apoptosis Detection Kit（BioVision）和 PI；仪器 CytoFLEX。精确事件数、补偿矩阵、各 gate 的数字阈值、完整 singlet/debris gate 顺序未在已读文本中报告，不能补造。

Fig4C 图注写 **mean ± SD of three replicates**。未说明是三次独立实验、三个技术孔，还是其他重复结构，故 `author_reported_n=3`、`replicate_kind=not_specified`。不能把图中六个条件或同条件三个终点当作独立重复。

## 4. 已提取的数据与数值层级

### 4.1 Fig4B：24 个作者印值

`Fig4B_author_printed_values.csv` 保留图中醒目的单小数百分比。四类终点 × 六个条件 = 24 个印值，不是 24 个独立实验，也不是 Fig4C 的均值。

| 前置状态 | ABT 时间 | ND (%) | Annexin−/PI+，ND 内 (%) | Annexin+/PI+，ND 内 (%) | Annexin+/PI−，ND 内 (%) |
|---|---:|---:|---:|---:|---:|
| 无 axitinib | − | 78.1 | 0.4 | 0.7 | 0.5 |
| 无 axitinib | 24 h | 90.9 | 0.5 | 1.7 | 1.3 |
| axitinib 2.5 µM、72 h | − | 77.1 | 1.1 | 2.9 | 0.5 |
| 同上 | 6 h | 20.3 | 2.8 | 1.7 | 3.8 |
| 同上 | 12 h | 16.8 | 5.7 | 2.2 | 3.0 |
| 同上 | 24 h | 19.6 | 16.8 | 9.9 | 4.0 |

源图 Tube3/6/7/10/13/16 标题也记录在表中，但不知道这些 Tube 的实验重复编号，不把它们包装成可配对的原始重复 ID。图面小字还含更多小数；本轮采用醒目印值，不混合不同显示精度。

### 4.2 Fig4C：18 个带读图范围的均值

`Fig4C_digitized_means.csv` 是三类终点 × 六条件的**发表柱均值数字化估计**。主终点 Annexin 的读图结果如下；范围反映像素/坐标读出不确定性，不是 SD、SE 或统计置信区间。

| 前置状态 | ABT 时间 | ND 内 Annexin V+ 均值估计 (%) | 保守读图范围 (%) |
|---|---:|---:|---:|
| 无 axitinib | − | 2.6 | 1.9–3.4 |
| 无 axitinib | 24 h | 2.5 | 1.7–3.2 |
| axitinib 2.5 µM、72 h | − | 3.2 | 2.5–4.4 |
| 同上 | 6 h | 5.2 | 4.5–5.9 |
| 同上 | 12 h | 4.5 | 3.8–5.4 |
| 同上 | 24 h | 12.3 | 11.6–13.0 |

伴随 ND 均值依序约 **86.5、89.7、78.2、17.6、17.2、18.3%**；PI+ 均值依序约 **2.5、1.8、3.9、4.2、8.2、25.3%**。两者也带读图范围，详见 CSV。它们仍各自使用前述分母。

提取方法与坐标完全保留在 `extract_fig4.py`、`Fig4_extraction_audit.json`：源图 2219×2370，按黑色柱顶及线性纵轴标定；没有提取灰色 SD 误差棒。黑色阈值 50/70 的柱顶一致；阈值 100 在少数很低的柱上会包含相接的灰色误差棒帽，已纳入更宽的读图范围。没有从图均值/SD 反造三个重复。

复算命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '<本目录>/extract_fig4.py'
```

## 5. 哪些可以进入独立挑战

1. **状态类别挑战可用**：同一研究的两个已定义前置群体，各接受 ABT-263 1 µM、24 h，比较原文定义的 ND 内 Annexin 终点。可用发表均值及其读取不确定性，必须标注图像数字化、重复类型未知。
2. **处理后时间响应辅助挑战可用**：axitinib 诱导群体在 ABT-263 1 µM 下的 6/12/24 h 观测；这是不同观测时间，不是共同 T* 的时长优化证据。
3. **ND 和 PI 保留为伴随终点**：用于展示分母群体随处理发生的变化，不能覆盖、替换为 Annexin 标签。
4. 两个无 ABT 参考可以展示基线图示，但未经确定其采样时钟，不做严格时间匹配的差中差或复制生成多个 vehicle 行。

单一后续小分子与少量条件不能独立证明跨药排序、复杂模型的总体增益或高维 RNA 状态表征的增量。此来源本身存在合法的有限挑战价值；无需因为没有 RNA 同测就排除它。

## 6. 状态观测与其他图的边界

Fig3（PDF 第 8 页）对 2.5 µM axitinib/lenvatinib 处理 3 天的群体测 p21、IL-8 qRT-PCR，并检查 Ki-67、BrdU；Fig1 还有 β-gal/细胞体积。它们可作平行群体的状态证据，时序上处在三天预处理阶段；不是 Fig4 流式细胞的逐个 RNA—命运配对。qPCR 以 β-actin 归一化、2−ΔΔCt 计算。本轮未获得高维转录组或公开 RNA accession，不用这两个基因冒充完整 RNA 状态表示。

Fig5 只报告 ND 百分比，其中 A549 的 ABT-263 终点为 12 h；不能把 PC9 的 24 h/H2452 的 72 h 混到 A549。Fig6C（ABT263±NAC，6 h）是 FSC/ND；Fig6D 的阻断实验与 Fig6E 的 ABT263/ABT199/另一个 Bcl-xL 抑制剂比较使用 **CCK-8**，不是 Annexin。原文对该 Bcl-xL 抑制剂的拼写在 Methods、Results 和 Fig6E 存在 H1331852/A-1331852/A-1331851 差异，本轮未擅自纠正，也没有把它加入 Annexin 动作表。

动物 Fig7 的 TUNEL 属于体内瘤组织，不与体外 Fig4 的分母或状态样本混并。

## 7. 访问失败、验证与文件清单

初次 PMC 网页返回 browser check；MDPI 网页工具返回 429。转用 Europe PMC 官方 XML 成功。Europe PMC supplementaryFiles 请求读超时后，改用 MDPI 单文件官方地址取得 PDF 与补充 ZIP；未反复请求同一失败点。一次 MDPI 单图 PNG 路径返回 404，随后采用实际下载 PDF 内的原图，不把失败说成图不存在。

`download_manifest.json` 记录下载结果，`pdf_embedded_images.json` 记录 PDF 页码/图片来源，`local_file_inventory.json` 记录本目录文件字节数与 SHA256。提取检查确认 24 个印值、18 个图均值、6 个条件，全部明确来源、分母和时间；两个无 ABT 的时长为空，未生成任何原始重复或原始群体率。
