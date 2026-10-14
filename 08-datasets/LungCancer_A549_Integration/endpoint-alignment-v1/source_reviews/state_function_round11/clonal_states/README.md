# Round11：A549 克隆、预分选群体与凋亡功能的定向补证

日期：2026-10-10。范围：A549 药前状态、克隆/姐妹细胞或分选群体，能否连接实际 Annexin/caspase 测量与公开表达；不把存活、克隆形成、代谢活力当凋亡，不要求破坏性 RNA 前后来自同一细胞。

## 结果

本轮取得两项来源的官方原件，保存了 **15 个作者图中印值**：Seah2015 的 9 个 Annexin-FITC 区域注释，以及 Tan2023 的 6 个 early Annexin/DAPI 注释。**它们不是 15 个原始重复，也没有补成新的 A549 高维初态表达—凋亡配对块。** Watanabe2002 的克隆/caspase 研究只取得可读作者全文文字，原 PDF/原图下载受限，保持未完成数值核验。

| 来源 | 已确认的价值 | 未补齐的关键部分 |
|---|---|---|
| Seah et al. 2015；[10.1038/cddis.2015.299](https://doi.org/10.1038/cddis.2015.299) | A549 WT、R1、R2 克隆，cisplatin 24 h 的 caspase/PARP WB；补图有三群体 TRAIL 4 h 的 Annexin-FITC 印值。 | 已读方法/附件中没有取得高维状态 RNA；TRAIL 是蛋白；流式印值是 R5 区域，不能替代总 Annexin 比例。 |
| Watanabe et al. 2002；[10.1152/ajplung.00025.2002](https://doi.org/10.1152/ajplung.00025.2002) | G4S/D3R 克隆的 4 种处理与 caspase3/9；基线抗氧化活性/蛋白表征。 | 非 RNA 表征；原图未取得，未填每个处理的剂量、时长和数值。 |
| Tan et al. 2023；[10.1093/jmcb/mjad039](https://doi.org/10.1093/jmcb/mjad039) | A549 TRA-1-81+/− × GEM 0/1/10 µM，Fig2J early Annexin+/DAPI−，6 个印值。 | RNA 明确来自 SW1990；J 的药物时长与给药前分选/药后 gate 的区分未明确，不能升级为已核定的药前分选 RNA 对应。 |

以上均是各自来源的有限证据；R1/R2、positive/negative 等简称不跨论文拼接。没有改核心数据、旧包、r5/v6 或模型，没有生成表达特征、训练/推理或写 Zotero/Memory。

## 1. Seah2015：克隆身份与真实功能，缺高维 RNA

原文标题：*Activation of surrogate death receptor signaling triggers peroxynitrite-dependent execution of cisplatin-resistant cancer cells*，PMC4632318。

已读并保存：`PMC4632318.1.xml`、官方 Fig1 `cddis2015299f1.jpg`、Supplementary Methods `cddis2015299x1.docx`、Figure Legends `cddis2015299x10.docx`、原 FigS2 `cddis2015299x3.tif`。TIFF 为 4444×5810，显示工具不接受 TIFF，另保存无重采样 PNG `cddis2015299x3_display.png`。官方 metadata 中相应 MD5 均逐件核对一致。

**克隆建立和同研究连接。** 主文 Methods “Cell culture and generation of cisplatin-resistant variants”明确：亲本 A549 持续 5 µM cisplatin 6 个月后，用 cloning cylinders 从耐药异质群体分离 R1/R2；三种 A549 群体培养于 DMEM/F12、10% FBS 等。它们是本论文来源明确的派生群体，不是同一细胞纵向读数。未把别的论文同名 A549.R1/R2 接入。

**小分子挑战。** Fig1E 原图确认 WT/R1/R2 各有 0、25、50 µM cisplatin；图注明确 24 h。Y 是 caspase3/8/9 processing 与 PARP cleavage 的 Western blot 图，GAPDH 对照。没有取得对应标量 densitometry 数表，不从条带生成伪百分比。Fig1A–C 是 MTT，Fig1D 是 PI DNA-content/sub-G1，不改称 Annexin。

**TRAIL Annexin。** Supplementary FigS2B 图注明确 WT/R1/R2，0/50/100 ng/mL TRAIL，4 h；代表 3 个独立实验。Supplementary Methods 指 FITC-Annexin V 染色、采集 10,000 events。原图纵轴是 FS Lin、横轴 Annexin-FITC；R4/R5 上方有印值，R6/R7 没有印值。

| 群体 | 0 ng/mL，R5 % | 50 ng/mL，R5 % | 100 ng/mL，R5 % |
|---|---:|---:|---:|
| WT | 5.47 | 4.60 | 2.32 |
| R1 | 4.05 | 27.20 | 33.26 |
| R2 | 6.82 | 23.41 | 26.23 |

这些是 `Seah2015_A549_S2B_printed_region_values.csv` 中的 9 条来源印值。**R5 不是已证明包含全部 Annexin 阳性事件的分区**，低 FS 区域可能另有事件，母门没有完整说明；也没有 PI/DAPI 辅助区分早晚期。保留 `R5_Annexin_FITC_positive_above_FS_gate` 定义，不命名为 total 或 early Annexin fraction。10,000 acquisition events 不能转成每个印值的独立生物样本数；代表图不是三次重复的均值。TRAIL 是 soluble human recombinant protein BML-SE721，不能用小分子 SMILES 动作替代。

Supplementary FigS2C 是 WT/R1 的 TRAIL 50 ng/mL 时间系列 caspase3/6/8/9。方法先按蛋白量归一化成 RFU/µg protein；该图又相对**各时点**未处理组做 fold。保留原图和方法；本轮没有数字化柱均值，不把 fold 当绝对酶活或 Annexin 比例。

已读主文和上述两个 DOCX 未取得 RNA-seq/microarray 矩阵、表达 accession 或克隆 RNA-ID 对照表。基线 WB、DR4/DR5 表面表达与代谢表征可以说明群体差异，不能冒充高维 RNA。未读取的其他 TIFF 不被宣称“全部没有数据”。

## 2. Watanabe2002：克隆及 caspase 证据，数值与 RNA 未接入

原文标题：*A549 subclones demonstrate heterogeneity in toxicological sensitivity and antioxidant profile*。

读取 [PubMed 12225949](https://pubmed.ncbi.nlm.nih.gov/12225949/) 及作者 Karen Iles 公开的 [ResearchGate 全文](https://www.researchgate.net/publication/11163589_A549_subclones_demonstrate_heterogeneity_in_toxicological_sensitivity_and_antioxidant_profile)。已读 Methods 的 caspase 测量、Results “Isolation of A549 subclones”“Identification of mode of cell death”、Fig3 图注及 Discussion 相关段。不是只读摘要。

该文用 limiting dilution 得到 D3R/G4S，比较 actinomycin D、anisomycin、DMNQ、HNE 的 caspase3/9。Fig3 图注为 2 或 4 h、mean±SD、n=2 的代表实验；测量使用 AFC 底物并按外部标准换算。基线测量是抗氧化酶活性、GSH/Trx 等；已读范围未提供转录组测量或 accession。

出版社全文/PDF 入口失败，作者网页的普通下载也失败。**没有取得 Fig3 像素，不能确认每种药物的印刷剂量、逐条件时长或数值；没有导出标签。** Fig2 的 HNE/DMNQ 活力剂量与时长没有借给 Fig3。该线索目前只支持存在克隆功能差异，不能补 RNA—功能连接。失败范围见 `search_and_access_log.md`。

## 3. Tan2023：A549 的 marker 群体 early Annexin，RNA 对象不相容

原文标题：*A targetable pathway to eliminate TRA-1-60+/TRA-1-81+ chemoresistant cancer cells*，PMC10847630。

已取得：`PMC10847630.xml`（Europe PMC）、`mjad039fig2.jpg`、`mjad039_supplemental_file.pdf`（10 页），及官方 PMC version 1 metadata。原图与补充 PDF 的 MD5 验证通过。`Tan2023_primary_focus.md` 提供原文定位：13 行 flow 方法；17 行分选/viability 描述；21 行 early apoptosis；25 行 SW1990 RNA 对象；29 行 Fig2 图注。

Fig2J 是 **A549**，GEM 0、1、10 **µM**。不要混入 Fig2B viability 的 nM。图内 6 个 early Annexin+/DAPI− 印值已亲视并保存：

| TRA-1-81 标记群体 | Control，% | GEM 1 µM，% | GEM 10 µM，% |
|---|---:|---:|---:|
| negative | 2.1 | 5.6 | 8.0 |
| positive | 1.9 | 1.9 | 3.8 |

输出 `Tan2023_A549_Fig2J_printed_panel_values.csv`。图注指出 graphs 为 mean±SEM，n=3 independent biological replicates；本轮记录的是**流式面板上的印值**，不把它们复制为三个原始重复，也不把印值与右侧柱均值混成两份独立数据。

保留三个具体缺口：

1. **时间**：主文 Flow cytometry 与 Fig2J 图注没有给出药物暴露时长；Supplementary Methods 第 2 页的 5 days 指 cell viability，colony 及 sphere 又有不同日程。因此 J 暴露时间留空，不写 120 h。
2. **状态定义时点**：Fig2B 明确细胞群体已 isolated；Methods 确实有 sorting 方法。但 J 的图注/段落没有独立写明“先分选，再加药做 Annexin”或“加药后按 TRA marker gate 分群”。当前不能把 J 当作已经确认的药前分选状态比较，也不能断言其一定是药后分群。
3. **RNA**：Results 明确 RNA-seq 比较 SW1990 TRA-1-81+/− 和 SW1990/SW1990GR；补充 TableS1 同属该体系。不能将其作为 A549 表达参考。A549 高维 RNA 对应新增 0。

流式分母以作者细胞分析为背景，但 FSC/SSC、单细胞及上游母门未完整交代，输出保留 unknown。记录早期 Annexin 这一真实定义，不把晚期群体补入。

## 4. 检索、排除与停止边界

先读已有 source_reviews；Sarin、NavGal、Kotani、Cutty、Pogacar、HIF、BRG1、CellCage、SCCP、early-cisplatin-persister/NOTCH1 等按 DOI/PMCID 去重。SciSpace 语义查询返回 10 条，原始精简元数据保留 `scispace_discovery.json`；网页入口使用克隆、姐妹细胞、分选、ALDH/CD44 与 Annexin/caspase/RNA 的交叉词，随后只深入以上三项。

有限内容检查的排除来源：

- Deben2021，[10.20517/cdr.2020.91](https://www.oaepublish.com/articles/cdr.2020.91)：A549.R2 有 transcriptome，但本文功能读出为 SRB、IncuCyte Cytotox Green 与存活/死亡；未把这些当 Annexin/caspase。RNA 数据声明仍为 acceptance 后将存库；本轮未核定 accession。
- Novel triple-positive markers，PMC6072294：预分选 EpCAM/CD166/CD44 亚群有 viability/克隆/sphere，初筛未获得对应 Annexin+RNA，未深入。
- A549 side-population expression profiling，PMC2234429：公开描述 SP/non-SP expression，未在本轮初筛建立药后真实凋亡连接，不跨来源拼接别的 SP 研究。
- Haley2014，[10.3389/fonc.2014.00344](https://doi.org/10.3389/fonc.2014.00344)：初筛为 RNA/phosphoproteomics EMT 表征，未取得本任务直接功能连接；不作为完成项。
- Schaff2024 及其他 SciSpace 非 A549/未确认对象结果只作初筛，不据摘要保留为 A549 数据。

本轮停止于三项来源已形成可核验的有限用途和缺口；**不是断言整个领域不存在可用资料**。HSP90/AXL/eIF4E 线索由主代理独占核查，本代理未重复下载或计入本轮成果。

## 文件与验证

`initial_downloads.json` 保留 Python 初始成功/失败；`search_and_access_log.md` 登记后续 PowerShell 与网页回退。`source_manifest.json` 绑定原件、派生文字、显示副本和两个 annotation CSV。

本轮没有新增高维表达矩阵、Annexin 训练表或模型输入。15 个印值仅用于来源审查，两个 endpoint 定义、protocol 和数据层级均分开。
