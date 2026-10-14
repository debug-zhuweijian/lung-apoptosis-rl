# 第九轮：SJB2-043 2025 原文与 RNA—Annexin 条件核验

核验日期：2026-10-10。来源：Wu et al., *SJB2-043, a USP1 Inhibitor, Suppresses A549 Cell Proliferation, Migration, and EMT via Modulation of PI3K/AKT/mTOR, MAPK, and Wnt Signaling Pathways*，Current Issues in Molecular Biology 47(3):155，2025-02-27。DOI [10.3390/cimb47030155](https://doi.org/10.3390/cimb47030155)，[PMC11941171](https://pmc.ncbi.nlm.nih.gov/articles/PMC11941171/)。

**已核定真实 Annexin/PI 功能数据为 A549、SJB2-043 0/0.625/2.5/10 µM、24 h。24 h 直接来自 Fig4 图注，没有借用 RNA 时长。该来源目前可以提供单一 A549 背景下的剂量条件功能挑战；还不能形成实际可读取的 RNA—Annexin 配对数据集。**

RNA 方法独立写明对照/10 µM、24 h，因此至少有明确的名义协议重合。但原文未给表达 accession，公开补充没有表达矩阵，Data Availability 是向作者合理请求。本轮没有取得高维表达，不以论文画了火山图就声称 RNA 已可用。

## 1. 原件、图与精确定位

| 本地原件/记录 | 核验定位 |
|---|---|
| `PMC11941171.xml` | Europe PMC 官方全文；Methods §2.1、§2.2、§2.4、§2.8、§2.11；Results §3.2、§3.4；Fig4 图注；Data Availability |
| `cimb-47-00155.pdf` | MDPI 官方 PDF，18 页；RNA 方法第 3 页；Annexin 方法第 4 页；Fig4 第 8 页；Data Availability 第 15 页 |
| `cimb-47-00155.md` | 按 PDF 页码抽取的文字；RNA 从第 139 行，Annexin 方法第 184 行，Fig4 图注第 395 行，Data Availability 第 859 行 |
| `cimb-47-00155_p08_image2_Im6.jpg` | 官方 PDF 第 8 页原始嵌入 Fig4 图像，1610×1083；四象限和均值均从此图读取 |
| `cimb-47-00155_p07_image1_Im4.jpg` | Fig2 的火山图/富集分析原图；不是表达矩阵 |
| `cimb-47-00155-s001.zip` | 官方补充，235,978 字节，实际包含一份 PDF |
| `cimb-3475498-supplementary.pdf` | 1 页，仅 Beas2B 的 CCK-8 补图和方法；没有 Annexin 源值/FCS、RNA 矩阵或测序 accession |

本目录还保留下载 URL、SHA256、PDF 嵌入图片清单、原文关键段落抽取及复算脚本。未更改核心数据表、源码、配置、研究架构或他人文件。

材料补核另见 `material_identity_readback.md`：论文 §2.1 位于 PDF 第 2 页、抽取 Markdown 第 117–118 行，明确 MCE 厂家但没有货号。当前官方名义产品为 HY-15757、CAS 63388-44-3，厂家结构和论文引用已核；三个名义材料—24 h 剂量链接保存在独立 JSON，未修改功能 CSV，历史批次仍未知。

## 2. Annexin 的材料、时间、分母与重复

依据 Methods §2.1–2.2、§2.8、Fig4A 及其图注：

- A549 来自 ATCC，培养于 RPMI-1640、10% FBS、1% penicillin–streptomycin，37 °C、5% CO2。
- SJB2-043 来自 MedChemExpress，已读原文未给商品 catalog number、lot 或 CAS。化学身份接入仍需遵循项目自己的材料核对，不能只凭名字把当前动作库状态标成通过。
- Annexin 四组为 **0、0.625、2.5、10 µM，全部 24 h**。剂量由 §2.8 和 Fig4A 图面直接支持；时长由 Fig4 图注直接支持。
- 收集细胞并以 PBS 重悬，加入 195 µL loading buffer、10 µL PI、5 µL FITC（Beyotime），室温染色 10 min，再流式采集，用 FlowJo 10.8.1 分析。
- 此处染色的 10 min 不是药物处理时长；不得与 24 h 混淆。
- 这是 A549 培养群体的单药剂量实验，没有预先诱导出的两个初态再挑战设计，也没有同细胞前后跟踪。
- 药物处理期间是否换液、补药、是否仅采集贴壁或同时纳入漂浮细胞、脱落细胞回收率，已读方法未明确。不能补造“完整回收所有原始细胞”的条件。

Fig4A 以 Annexin-V 为横轴、PI 为纵轴，四象限 Q1–Q4 给出百分比。四象限合计在显示舍入误差内约为 100%，分母登记为 **作者呈现的 Annexin/PI 分析母门事件**。上游 FSC/SSC、singlet/debris gate、事件数、补偿矩阵未报告，因此不把它直接改写成最初接种群体的自然凋亡率，也不沿用 Kotani 的 ND 门定义。

Fig4A 柱图的标签是 apoptosis rate (%)。正文 §3.4 说明同时观察初期与晚期凋亡群体；本轮分别保存四象限印值，以及作者报告的柱均值。Q2+Q3 可由图的阳性横轴象限相加得到，但这一派生图示值不等于 n=9 的柱均值。

Fig4A 图注明确 **mean ± SD，n=9**，但没有说明 n=9 的技术/生物重复结构。`replicate_kind=not_specified`，不推定为九次独立实验。Fig4B 的 Bax/Bcl-2 WB 则为 2.5 µM、24 h、n=3，不把 WB 的 n=3 移用于 Annexin。

Fig4B 的 WB 对照图面写 DMSO；Fig4A 的流式对照只写 0 µM，不能仅凭邻图给 Annexin 补入具体溶剂浓度。

## 3. 实际提取的功能数值

### 3.1 四象限作者印值：16 条

`Fig4A_author_printed_quadrants.csv` 保存图上原有精度的数字。它们来自各剂量显示的一张流式图，不是恢复的九个原始重复。

| SJB2-043，24 h | Q1：Annexin−/PI+ | Q2：Annexin+/PI+ | Q3：Annexin+/PI− | Q4：Annexin−/PI− | 四象限合计 |
|---|---:|---:|---:|---:|---:|
| 0 µM | 1.20 | 2.52 | 1.12 | 95.1 | 99.94 |
| 0.625 µM | 2.70 | 5.06 | 3.07 | 89.2 | 100.03 |
| 2.5 µM | 1.99 | 6.94 | 3.52 | 87.6 | 100.05 |
| 10 µM | 5.02 | 14.9 | 6.07 | 74.0 | 99.99 |

单位均为分析母门内百分比。`Fig4A_displayed_plot_Annexin_positive_derived.csv` 另外保存由 Q2+Q3 算出的 **3.64、8.13、10.46、20.97%**。清楚标记为本轮对印字的求和，不写成作者源表均值。

### 3.2 柱图均值：4 条数字化估计

`Fig4A_digitized_apoptosis_means.csv` 保存如下值。作者误差棒为 SD；本轮未数字化 SD，更未反推出重复。下列范围只是坐标/像素读出范围。

| SJB2-043，24 h | 作者凋亡率柱均值的估计 (%) | 保守读图范围 (%) |
|---|---:|---:|
| 0 µM | 3.6 | 3.3–4.0 |
| 0.625 µM | 8.3 | 8.0–8.8 |
| 2.5 µM | 9.5 | 9.2–9.9 |
| 10 µM | 20.1 | 19.8–20.6 |

读图范围不是 SD、SE、统计置信区间，也不是生物变异范围。n=9 只保留图注所述，不生成额外行。横轴的四个条件不是四个独立初态。

`extract_fig4.py` 从源图保存的坐标标定纵轴，识别柱顶并对黑色轮廓中心作 2 像素修正；图像阈值 225/235/245 的边界差异≤1像素，柱顶和坐标厚度计入读图范围。原图 hash、校准参数、16个印值和4个均值的检查见 `Fig4_extraction_audit.json`。

复算：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '<本目录>/extract_fig4.py'
```

## 4. RNA 条件：有协议重合，未形成可读取配对

§2.4 明确写出 control cohort 和 **10 µM drug-treated cohort，处理 24 h 后 RNA-seq**。§3.2 也报告 A549 经过 SJB2-043 24 h 后做表达分析。与 Fig4A 的 10 µM、24 h 在同一论文中有名义条件重合。

但这与“已获得同条件表达—功能数据表”不同：

| 项目 | 已核实状态 |
|---|---|
| A549/药物/10 µM/24 h 协议重合 | 是，两个测定有各自原文依据 |
| 完整 RNA 样本数与重复 ID | 未报告明确清单 |
| RNA-seq accession | 主文/补充未列出，本轮定向公开查询未取得对应编号 |
| counts/TPM/FPKM 表或 FASTQ | 未取得，实际读取高维表达样本数为 0 |
| RNA 与流式共同样本/重复键 | 未报告 |
| 初态 RNA | 没有取得；药后 24 h RNA 本身不能当作给药前 t0 |
| 实际可投入的 RNA—Annexin 对齐行 | 当前 0 行；协议候选不等于可读取的配对样本 |

§2.4 前半只写 control/10 µM 两组，后半又说求 control-versus-low-dose 与 control-versus-high-dose 的共有差异基因，但没有给出该 low-dose RNA 的浓度或样本清单。这是原文内部尚未解决的范围描述问题；本轮不自行填入 0.625/2.5 µM，也不擅自认定作者的哪句话是笔误。

TCGA/Xena、HPA、UALCAN 是该论文另行使用的患者/组织关联资料，不能当成 A549 药后 RNA 或其初态参考。火山图及 DEG 数量也不能反推出完整表达矩阵。

## 5. 定向 accession 核查与访问范围

论文 Data Availability：所有研究数据可向通讯作者合理请求。官方补充仅 Beas2B CCK-8；没有高维 RNA 文件或 Annexin 数表。本轮未联系作者。

为避免仅凭没有 accession 就作“无公开数据”的判断，本轮对 NCBI GEO GDS/SRA 作了窄查询，完整响应在 `official_accession_queries.json`、`official_accession_queries_corrected.json` 和 `SRA_SJB2_record_summaries.json`：

- GDS 的原始 `SJB2-043` 查询被 Entrez 改写为仅 `043`，返回 6079 个不合格宽匹配，**全部作废，不计为该药数据**。
- 更正为 GDS `SJB2[All Fields]` 后为 0；GDS 的 PMID 查询也为 0，且被转换为 All Fields，不能视为穷尽 PubMed/GEO 关联。
- SRA `SJB2-043 AND A549` 为 0；再查 SJB2 得到 7 条，逐条读取官方 ESummary，全部为 mangrove sediment、fish 或 sea-turtle 等宏基因组样本名字，非本研究、非 A549，全部排除。

这些定向查询未找到对应公开 RNA；不据此宣称其他命名或未索引位置绝对不存在数据。当前能确定的是本轮实际未取得可读表达。

## 6. 当前可用性判定

**可用：**一个细胞培养背景、一个小分子、三个非零剂量和对照、共同 24 h 的发表 Annexin/PI 条件功能挑战。保留原始方法定义、n=9 类型未知、图均值数字化及读取误差，不将图中样本冒充原始重复。

**尚不可用：**实际可读取的 RNA—Annexin 联合监督表；多个先验初始状态的增量评测；跨候选药物排序；同细胞初始 RNA 到未来凋亡命运配对。本轮发现的名义 10 µM、24 h 交集值得保留，但没有用它替代缺失的高维表达文件。

本轮只做指定论文和其公开数据的定向核验，没有另行广泛检索，没有训练、联系作者、修改其他目录、提交或推送。验证与哈希见 `validation.json`、`local_file_inventory.json`。
