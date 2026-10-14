# Round12：A549 多状态 RNA 与功能终点的条件级来源核验

核验日期：2026-10-10。职责仅限本目录；没有改动核心 curated r4、condition-inputs v5、训练配置或既有来源，没有训练、模型推理、联系作者或提交。

本包用于增强“表达状态—真实功能”的外部桥接。**它不是研究架构 §3(A) 表达分布预测实验的普遍前置门槛**；§3(A) 应按自身的表达预测任务和现有数据验证状态增量。以下功能配对限制只约束本包可以承担的主张。

## 已完成的结果

| 来源 | 实际取得 | 可以验证 | 尚不能验证 |
|---|---|---|---|
| Zhang2017 RNF20/H2Bub1 | 官方 GEO 18,815 行×15 列 normalized counts；A549 的 3 列和 GSM 元数据；官方 JATS；六张真实图像 | 对照与两种独立 siRNA 的表达差异；平行未暴露状态参考与顺铂凋亡差异的候选条件级连接 | 不是用药起点实测 RNA；RNA siRNA 单独处理与功能等量 pool 不同；精确标签的文字/图像矛盾待协调 |
| Carrà2021 IκBα/NFKBIA | 官方全文、补充 DOCX、Fig2 原图 | shCTR/D2/D7 三背景、顺铂 5/10 µM×48 h Annexin 条件网格确实存在 | RNA 六份文库只在方法中描述；未取得矩阵、accession、逐样本到克隆映射；因此目前不能读回 RNA—功能输入 |
| Ramesh2023 propionate | 官方 JATS 与附件清单；GSE224740 公开记录已读 | 多时间 A549 RNA 和后续顺铂死亡表型有来源 | 该药后图用 Cytotox Green object counts；不能改为 Annexin 比例 |
| Wang2025 FTO，已启动但未完成深入核验 | 官方 JATS、Fig2、两条 BioProject 的实际页面 | A549 shNC/shFTO RNA 与 Annexin-FITC 图存在；项目登记可核验 | 逐样本元数据读取遇阻；未取得表达矩阵与 RNA/功能时钟；不登记为可用配对 |

RNF20 的实际读取结果在 [validation.json](validation.json)。本包 **0 条已放行的核心 G 训练标签、0 条模型预测**。这不是“所有功能来源均不可行”的结论，而是把已取得、可作为参考、尚需假设和尚未取得的环节分开。

## RNF20：取得了什么，怎么连接

原文为 [Loss of H2B monoubiquitination is associated with poor-differentiation and enhanced malignancy of lung adenocarcinoma](https://doi.org/10.1002/ijc.30769)，Int J Cancer 2017，PMC5527746。使用 PMC 官方作者稿/JATS 与原图，未声称另行核验 Wiley 最终排版版。

- [GSE88943_normalized_counts.txt.gz](GSE88943_normalized_counts.txt.gz) 是官方矩阵。GEO 标为作者 edgeR normalized counts，不能因为文件数字都是整数就叫 raw counts。
- [RNF20_RNA_samples.csv](RNF20_RNA_samples.csv) 对应 GSM2355674（A549 control siRNA 2）、GSM2355675（RNF20 siRNA 1）、GSM2355676（RNF20 siRNA 2）。
- [GSE88943_A549_normalized_counts.csv.gz](GSE88943_A549_normalized_counts.csv.gz) 保留原矩阵行号和原列名；18,815 行中有 18,787 个不同 gene symbol，28 个超出首次出现的重名行，**没有自动合并**。
- GSM 的 “all 14 samples” 文本与实际 15 个表达列不一致；A549 实际只有 1 个 control 列，不能补造第二个 control。不同 siRNA 也不能自动当作同一 pool 的两个重复。
- [RNF20_candidate_condition_links.csv](RNF20_candidate_condition_links.csv) 只提出两个条件级候选联系；两个 siRNA 的 RNA 不会复制出两个独立实测 Y。

论文方法 P10：RNA 在转染后 72 h 获取、没有顺铂处理；后续功能实验改用两种 siRNA 的等量 pool。方法 P13：转染后 24 h 开始加顺铂、持续 72 h，终点为转染后 96 h。因而 RNA 是**平行未加药状态参考**，既不是转染后 24 h 的实测起点，也不是顺铂处理后 RNA。若后续使用，必须显式承担“该参考可代表相应干预背景”以及“单独 siRNA 可代表 pool”的迁移假设，不能把这些假设写成已验证的配对事实。

目前有明确数值描述的 A549 Annexin 是顺铂 20 µM、72 h。Western blot 的其他浓度不自动扩展成有 Annexin 均值的剂量网格。Fig5 图注说明 early/late apoptotic cells、实验重复三次，统计方法说明均值±SD；详细 FACS 母门、漂浮细胞回收范围及原始 FCS 未在已读段落中明确。这里没有将其改写为所有起始细胞的死亡概率。

### 数值和图像身份问题

[RNF20_source_text_statistics.csv](RNF20_source_text_statistics.csv) 保留原文 P22 的两个 A549 均值/SD：70.33±10.70% 与 53.97±11.02%。句子顺序与“敲低后下降”的语义存在歧义。本包记录“对照对应较大值”的**解释性分配**，同时保留 literal order，且不将其作为已放行训练标签。

实际下载六个 PubMed DOM 图链接后，发现图文件编号与正文图注错位：

| DOM/文件 | 根据真实内容匹配的正文图注 |
|---|---|
| f1 / f2 | Fig1 / Fig2 |
| f3 | Fig4：迁移、侵袭 |
| f4 | Fig6：组织 IHC/生存 |
| f5 | Fig3：BEAS-2B 分化 |
| **f6** | **Fig5：顺铂、Annexin/PI** |

实际凋亡图为 [nihms874355f6.jpg](nihms874355f6.jpg)。它的图例明确深灰为 control、浅灰为 RNF20，A549 对照柱更高；但柱高目视约 75/57%，不能精确等同于正文 70.33/53.97。该目视近似**没有作为数字提取值输出**。保留正文统计和图像身份问题，待同版本最终图/源表协调后再选择数值来源。真实 URL 和映射分别在 [DOM 链接表](RNF20_pubmed_observed_figure_links.json) 与 [图像身份审计](RNF20_figure_asset_audit.csv)。

## IκBα：实验存在，但 RNA 输入尚未取得

原文为 [IκBα targeting promotes oxidative stress-dependent cell death](https://doi.org/10.1186/s13046-021-01921-x)，Carrà et al. 2021，PMC8050912。

[Fig2 原图](13046_2021_1921_Fig2_HTML.jpg) 的 C 面板明确显示三个 A549 背景（shCTR/shIKBD2/shIKBD7）与 NTC、5 µM、10 µM cisplatin。处理组为 48 h；均值±SEM，至少三次独立实验。[条件清单](IKBa_condition_inventory.csv) 记录 6 个活性条件＋3 个 NTC，**尚未数字化柱值**。该条件规模不是 9 个 RNA 样本或 9 条独立状态。

补充材料的 RNA-Seq lines 段实际写明四份 siNFKBIA A549 和两份 scrambled A549，GRCh38.p5/Gencode v24、STAR、DESeq2；未给出六份样本各自对应 D2/D7 的 key，也未给出本包可下载的矩阵或 accession。正文数据声明仍是“Rna-seq data will be available.”。全文、补充附件和定向 GEO/标题检索均未补到可实际读取的 accession，本轮停止在**尚未取得公开输入**，不把这个状态解释为实验没有做过或数据永远不存在。

Fig2E–H 的表达分析及 SI 没有建立逐孔 RNA 与后续 Annexin 对应。仅凭“同一 A549 敲低背景”可以提出群体级连接，但不能伪造同批/同时 RNA，也不能把 RNA 的四份 knockdown 当作四份已配对功能观测。该来源后续需要实际矩阵、样本-构建体表、RNA 取样/用药时钟与门控说明。

## 筛查与读取边界

[screening_summary.csv](screening_summary.csv) 登记已启动的新来源和去重/排除理由。RNF20、IκBα 是本次保留的主要支线。RIF1/RBIS 已读的全文把 GEO 主要用于患者表达资料，未确认有其 A549 干预的公开高维矩阵；这与“所有 A549 RNA 不存在”不同。FTO 已读取 BioProject，但 SRA/ENA 访问失败不能写成没有公开原始 reads。

旧 Sarin、Nav-Gal、Kotani、HIF1A、Seah、Tan、YangHSP90 分支不重复扩查；Nav-Gal 本轮由主代理负责。本目录没有据此修改研究架构或宣称模型效果。

## 可复算检查

```powershell
& 'F:/anaconda/envs/study/python.exe' '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round12/condition_state_rna/audit_source_package.py'
```

实际通过：矩阵形状与三列身份、数值有限非负、A549 子矩阵导出后逐项等值、保留重名 gene symbol、两个候选联系的错位时钟和材料状态、9 个 IκBα 图条件的记录。检查只是输入审计，没有运行模型或证明桥接假设。冻结文件及 SHA256 见 manifest.json。

