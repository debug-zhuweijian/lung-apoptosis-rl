# Round 10：HIF1A / A549-PTX / Fig2K 原始附件与条件功能核验

核验日期：2026-10-10。论文：Gronkowska et al., *HIF1A, BRG1, and p300 interaction confers paclitaxel-induced drug resistance by enabling the overexpression of ABCC genes*. 正式版 DOI [10.1016/j.omton.2025.201049](https://doi.org/10.1016/j.omton.2025.201049)，PMC12493206，PMC Cloud version 1，`is_manuscript=false`。

## 本轮结论

**已取得并实际读取作者 Data S2 数表，可恢复 A549-PTX 的 8 个真实条件、80 个已归一化源数值。** 数据是 Caspase-Glo 3/7 信号除以 resazurin 荧光，属于作者定义的凋亡相关功能读数；不是 Annexin 阳性比例，不乘 100，不转换为细胞死亡率。每条件有 10 条数值，但独立生物重复数、技术重复数和跨条件配对均未说明。

这提供了 A549-PTX 中 siCTRL 与 siHIF1A 两种明确遗传处理状态，在无化疗、多柔比星、MK571、二者联合配置下的有限条件功能证据。**目前仍不能形成时间明确的 RNA 状态—候选—凋亡闭环：Fig2K 暴露时长未知；本轮未建立 siHIF1A 高维 RNA 对应。** 该块保留在独立来源审查目录，没有加入 r5/v6、模型输入或训练。

重要限定：原图 Fig2K 下图和源表都写 A549-PTX + **Dox 0.25 µM**；PTX 0.25 µM 是上图 **MDA-MB-231-PTX** 的挑战。正文对应段却使用“before PTX”的叙述。本轮保留这一不一致，按原图和数表登记 A549 的 Dox 条件，不虚构第二种 A549 化疗药物。

## 原件与准确定位

本目录：`I:/科研/肺癌细胞凋亡-强化学习/08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/hif1a/`

| 文件 | 已亲读范围与用途 |
|---|---|
| `PMC12493206.1.json` | 官方版本、DOI、license、原件路径、MD5 登记。正式版、CC BY-NC-ND；下载时 metadata 标注未撤稿，非额外撤稿数据库审查。 |
| `PMC12493206.1.xml` | Fig2 图注、相关 Results、Methods §4.1、4.2、4.3、4.5、4.6、4.13–4.15。 |
| `PMC12493206.1.pdf` | PDF 第 5 页是 Fig2 图，第 6 页图注及 Results，第 14 页材料/RNA方法，第 15 页转染方法，第 17 页 Caspase 与统计方法。 |
| `gr2.jpg` | 官方 Fig2 图，764 × 1042；亲读 J/K 的细胞系、药物、剂量、图例、纵轴。 |
| `Fig2K_PDF5_native_X29.png` | 从主 PDF 第 5 页原生嵌入图 X29 无重采样提取的 A549 K 面板，650 × 339；亲读误差棒。提取绑定见 `native_image_extraction.json`。 |
| `mmc3.xlsx` | 8 个工作表均列举和文本/注释检查；`Cell viability` 全表亲读；A549 K 的数值在 `B57:D101`。 |
| `mmc1.pdf` | 8 页；Figures S1–S7 的所有图注/可提取文字已读；S5–S7 没有补出 Fig2K 暴露时钟或重复分组。 |
| `primary_text_readback.md` | 从新 XML 提取的精确原文：22 行 J 的时间；23 行 K 的剂量；27 行材料；43 行通用转染；57 行 Caspase assay；61 行统计；65 行药名不一致的 Results。 |
| `mmc3_cell_values.md` | 所有工作表非空单元格的坐标和值，方便全文查找；公式需回到 XLSX 的 `data_only=False` 视图。 |
| `mmc1_extracted.md` / `PMC12493206_pdf_extracted.md` | 页码分隔的 PDF 文本提取，仅作定位；图像、公式、特殊符号以原件为准。 |

本轮没有下载 `mmc2.pdf`（WB 图）、`mmc4.xlsx`（Bioinformatics）和 24.5 MB 的 `mmc5.pdf`（文章与补充整包）；本任务数值/时间判断限于上述已读原件。未下载 FASTQ。

## 可直接复算的数据

Data S2 / `Cell viability`：`B57=A549 PTX`；`C58=NT`；`D58=Dox 0.25 μM`。

| 原表组名 | 行区间（C=NT，D=Dox） | NT 均值 | Dox 0.25 µM 均值 |
|---|---:|---:|---:|
| sictrl | 59–68 | 0.0111530 | 0.0635974 |
| sictrl + iABC | 70–79 | 0.0160462 | 0.1495495 |
| siHIF1A | 81–90 | 0.0287069 | 0.1638916 |
| siHIF1A + i ABC | 92–101 | 0.0190902 | 0.1514697 |

表中 iABC 为 MK571，25 µM，依据 Fig2K 图注。NT 指没有 Dox 挑战，**不意味着该组没有 siRNA 或 MK571**；原件没有给出可独立指定的 NT vehicle 成分。

输出：

- `fig2k_a549_normalized_source_values.csv`：80 条作者已归一化数值；每条保留工作表、单元格、剂量、状态和源文件 SHA256。原仪器 luminescence / fluorescence 两项分量未独立给出。
- `fig2k_a549_condition_summary.csv`：8 条条件摘要。均值和样本 SD 是本轮从 10 个源数值重算，未替换成图中读图值。
- `extract_fig2k.py`：固定源 hash，核对所有表头/块标签，逐单元格提取并读回核验。
- `extraction_validation.json`：80 个源坐标唯一、8 个条件唯一、所有 CSV 值按坐标与原表一致；作者 6 个 posthoc mean difference 与本轮分组均值差一致，误差小于 0.00001。

`source_row_ordinal_not_pair_id` 只是块内顺序，不能作为跨条件配对、培养批次、细胞 ID。`computed_sample_sd_ddof1` 只描述这 10 条记录的离散程度。额外的 `computed_sd_over_sqrt_record_count_not_inferential_sem` 是算术诊断量，**不在重复独立性未知时冒充有效 SEM/CI**。Methods 通用文字写 mean ± SD；本轮未把图中误差棒按某个假定重复层级反推，也未从误差棒生成重复。

**图示误差定义尚未闭合。** 独立复审后，本代理又亲读了原 PDF 内嵌 A549 K 面板：末组 siHIF1A + iABC + Dox 的上误差端约 0.187（人工目视量级，非统计区间），而 `D92:D101` 的 mean + sample SD 为 `0.1514697 + 0.1115128897 = 0.2629825897`，明显不能重现原图。mean + SD/√10 为 `0.1867331720`，与图接近，但重复聚合方式未知，**不能据此确定作者误差棒是 SEM**。8 条均值和 80 个源数值仍保留；计算的 SD 与 SD/√10 都不登记成作者图示误差。这个差异不靠更改源值、丢弃极端数值或把 10 条记录假定独立来修正。

## 条件和时间：可确认与不能填补

| 项目 | 已确认内容 | 边界 |
|---|---|---|
| 细胞背景 | A549-PTX，作者的 PTX 耐药派生群体；A549 来源 ATCC。 | 诱导历史由本研究引用 refs 15、48；本轮未重读这两篇，不推定与别的 A549/PTX 同名株相同。 |
| 状态 | siCTRL 与 siHIF1A；作者称 HIF1A proficient/deficient。 | 不是同一细胞前后 RNA 追踪；siRNA 剂量在本次 K 证据中未明确。 |
| 材料 | Doxorubicin hydrochloride，Sigma-Aldrich；MK-571，Sigma-Aldrich M7571；HIF1A Silencer Select siRNA n336611；control sc-37007。 | Dox 历史货号/lot 未印；没有据此自动生成新化学指纹。 |
| 名义剂量 | Dox 0 或 0.25 µM；MK571 0 或 25 µM。 | 两种因素同时为非零时是联合配置，不展开成等价单药训练样本。 |
| 转染通用流程 | §4.5：siRNA-RNAiMAX 6 h 后加培养基，再培养 48 h。 | 此节列 24/96 孔、chamber 和 spheroid；K 为 384 孔；未规定 K 的加药时点或化疗暴露长度。 |
| Fig2J 时间 | 转染后 24 h 加药，再处理 48 h。 | 明确属于 **resazurin viability J**；不移植给 K。 |
| Fig2K 时间 | 转染、inhibitor/药物孵育之后做 assay。 | 化疗暴露、MK571 预处理窗、撤药、换液、采样钟点未明确，输出留空。 |
| K 试剂读取 | §4.14：播种于转染前一天，384 孔白板，2,000 cells/well；Caspase-Glo 或 5 µM resazurin 末端孵育 1 h，37°C。 | 这个 **1 h 是末端检测试剂孵育时间**，不是 Dox 暴露时长。 |
| 分母 | resazurin 荧光，作为活细胞代谢活性/相对活细胞量的代理。 | 不是活细胞直接计数、总细胞母门或 Annexin 流式分母；比值升高可受分子和分母共同影响。 |

Fig2K 同协议条件比较在它定义的 ratio 内有意义，分母不同不使它自动无效。但它不能校准成 Annexin 概率，也不能直接支持已知时长的行动选择。

## RNA 和有限状态观测

Fig2A/B 有 A549-PTX 的 qPCR（ABCC3/5/10 与 HIF1A 等有限基因），图注是转染后 72 h；S5 对应非耐药细胞；S6F/G 比较非耐药/耐药 HIF1A mRNA/蛋白。这些可作为该状态定义的有限表征依据，**不能称为 Fig2K 实测给药前 t0 RNA，也不能伪装成全转录组表示**。

本轮亲读 Round9 已落盘的 `PRJNA1095909_run_metadata.tsv`：A549 相关 RNA runs 为 SRR28539878（A549）、SRR28539877（A549-PTX）、SRR28539876（A549-PTX siCTRL）、SRR28539875（A549-PTX siBRG1）。**siBRG1 不是 siHIF1A**；不能根据同一个项目 accession 把这些测序记录一一配到 K 的 8 条条件。本轮新增高维表达值/合法 siHIF1A RNA 匹配均为 0。该原始元数据保持在 Round9，只读不改。

相邻 qPCR 的额外注意项：`qPCR gene expression ptx res!G57` 是 `AVERAGE(B57:F57)` 公式，不能在将来提取 HIF1A qPCR 时把 G57 当第 6 个独立原始观测。本轮功能提取只用 `Cell viability!C59:D101` 的 80 个非公式数值，不受该相邻公式影响。

## 获取和验证记录

所有原始文件来自 **NIH NLM NCBI PubMed Central Article Datasets** 官方匿名 Cloud Service。此次先读取 `README.txt` 和 PMCID 前缀目录，再从官方 JSON 的精确 URI 转换为文档允许的 HTTPS 获取。没有绕过登录、验证码或权限限制。

- `retrieval_log.md`：旧路径失败、新官方路径成功、软件依赖回退、实际读取范围。
- `cloud_article_listing.xml`：精确 PMCID 下的 16 个对象、大小和 ETag；`IsTruncated=false`。
- `official_asset_downloads.json`：metadata URI、实际 URL、响应、文件大小、SHA256、官方 MD5 比对。
- `manifest.json`：本目录所有固定文件的 SHA256，以及只读旧源引用。

复算命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/hif1a/extract_fig2k.py'
```

实际运行：exit 0，`conditions=8`、`source_numeric_records=80`、`validation=passed`、`paired_rna=0`。没有训练、联系作者、Zotero 写入、Git commit/push；没有修改 Round9、r5/v6、通用 builder、中央文档或其他代理的目录。
