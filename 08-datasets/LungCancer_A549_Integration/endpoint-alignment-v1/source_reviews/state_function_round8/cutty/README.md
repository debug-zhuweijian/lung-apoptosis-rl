# Cutty：A549 起始周期—成像命运，与 H1944 RNA 分开登记

核验日期：2026-10-10。论文 [Pro-survival roles for p21(Cip1/Waf1) in non-small cell lung cancer](https://doi.org/10.1038/s41416-024-02928-9)，2024-12-20 在线发表，British Journal of Cancer 132(5):421–437（2025）；[PMC11876327](https://pmc.ncbi.nlm.nih.gov/articles/PMC11876327/)。

**本来源确有给药前 A549 周期状态与给药后命运的观测；公开 RNA 则来自 NCI-H1944。它支持有限的成像死亡挑战，不能拼成 A549 初始 RNA—凋亡配对。Fig5c 还存在两列作者百分比内部不一致，已整列隔离。**

## 原件和实际读取

- `PMC11876327.xml` 为 Europe PMC 官方全文；`PMC11876327_text.md` 为检索定位副本，原始排版以 PDF 为准。
- `s41416-024-02928-9.pdf` 为出版社 PDF。Fig5c 在 PDF 第12页、印刷页432，图注在下一页。`Fig5_page12_render.png` 是该整页的渲染图，主代理和只读复审代理分别亲看了四张百分比小表。
- MOESM7 DOCX 为补充图表说明，MOESM8 PDF 为补充图；MOESM9/10/11 XLSX 分别为 Table S1/2/3。本轮未下载视频，没有把视频目录当作数值轨迹。
- `GSE266945_samples.soft` 是官方 GEO 的全部八个 GSM 记录。`download_manifest.json` 记录七个下载资产的 URL、大小和 SHA256。

三个 XLSX 已实际逐行读取：

| 表 | Sheet | 数据行 | 内容 |
|---|---|---:|---|
| S1 / MOESM9 | p21High-v-p21Low_goDOWN | 308 | GO 富集，九列含 ID、Description、GeneRatio、p.adjust、geneID 等 |
| S2 / MOESM10 | p21High-v-p21Low_goUP | 5 | GO 富集，同上九列 |
| S3 / MOESM11 | 220118_p21-FACS-subsets-RNAseq_ | 60,394 | gene_id、gene_symbol、p21High/Low 的 Plate1–4 共八表达列 |

S3 共483,152个表达值被实际读取，均为有限非负数。补充 DOCX 明确单位 **TPM**。列名与 GSM 标题逐一核对，绑定见 [GSE266945_sample_bindings.csv](GSE266945_sample_bindings.csv)。不是单细胞计数或 A549 数据。

GSM8257311–14 为 p21High_Plate1–4，GSM8257315–18 为 p21Low_Plate1–4；八样本均为 **untreated NCI-H1944 mRuby-PCNA p21-mVenus**，经 FACS 分为 p21/PCNA 高低群体后提取 RNA。GEO 的 replicate 编号不在本审计中自动升级为独立培养批次。三个补表中没有 A549 轨迹、死亡时间或命运列。

## A549 成像实验的真实连接

主文方法定位：[文本第103行](PMC11876327_text.md:103)、[第107行](PMC11876327_text.md:107)；Fig5c 图注：[第243行](PMC11876327_text.md:243)，结果：[第249行](PMC11876327_text.md:249)。

| 字段 | 核实内容 |
|---|---|
| 材料 | A549-mRuby-PCNA-CDK2-GFP，p21WT 或 p21KO C1 |
| 状态 | 先成像48h，加入药物时按 G1/S/G2 定义起始阶段 |
| 处理 | gemcitabine 5µM 或 cisplatin 16µM |
| 观察 | 加药后72h；每10分钟成像一次，总成像约120h |
| 终末分类 | G1、S、G2 或 D（成像死亡） |
| 细胞纳入 | 被分析细胞应全程存在于影片中，死亡者例外 |
| 重复 | 方法称展示一个实验，代表至少两个生物重复；不能套用 Fig5b 的 n=3 |
| 凋亡特异性 | 此处没有 Annexin/caspase 检测，不能将全部 D 改成凋亡 |
| 洗脱 | 此成像挑战未交代期间洗脱；另一 relapse assay 的三天后洗脱不移植到此表 |

## 48个印值与两列异常

[Fig5c_author_printed_percentages.csv](Fig5c_author_printed_percentages.csv) 保存4个背景—药物面板、3个起始阶段、4种终末命运的48个作者印值。这是12个起始阶段分层，**不是48个独立实验**。[Fig5c_strata_audit.csv](Fig5c_strata_audit.csv) 登记分层资格。

横向表头为 Start phase，纵向为 End fate。其余10列各合计100%，强烈支持按起始阶段分列归一化；这一分母解释标为本课题依据图表结构的推断，尚无原始细胞计数核验。两个原图异常为：

| 面板/起始阶段 | G1、S、G2、D 原值 | 列和 | 处理 |
|---|---|---:|---|
| p21KO + gemcitabine / 初始 S | 0、6.3、21.3、62.5 | 90.1% | 四个值整列隔离，包括死亡项 |
| p21KO + cisplatin / 初始 G1 | 4.3、39.1、39.1、21.7 | 104.2% | 四个值整列隔离，包括死亡项 |

未自动归一化、补足100或由百分比反推人数。不能知道错误落在哪个单元格，因此不能只删非死亡项后仍使用 D。其余10列仅通过数值和检查，保留为描述性成像挑战，未接入正式 Annexin 标签。

作者正文称 G1 初态细胞并未表现出预想的化疗保护；这可用来反驳“G1必然更耐药”的简单命题。但原文措辞不构成统计等效检验，也不能推广为所有高维表达状态无用。主代理没有从代表性实验反造独立重复或显著性。

## 复算与资格

`build_source_audit.py` 在无网络、无训练条件下核对下载哈希、GEO—表达列绑定、三表全部行及图表列和。执行环境为 Codex bundled Python，依赖 openpyxl。记录在 [source_audit.json](source_audit.json)。

此来源新增 A549 RNA—凋亡配对 **0**，核心 Annexin 标签 **0**。可以测试低维起始周期状态与成像命运的有限关系；不能验证高维初态表达增量、自然群体凋亡率或完整药物—剂量—暴露时长选择。源文件和异常均保留，未修改核心表或启动模型训练。
