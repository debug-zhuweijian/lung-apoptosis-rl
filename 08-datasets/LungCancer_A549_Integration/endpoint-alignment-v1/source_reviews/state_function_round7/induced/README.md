# 第七轮：诱导 A549 状态后的凋亡再挑战核验

核验日期：2026-10-10。范围：预处理诱导衰老/持留/耐受的 A549 群体，状态参考表达，以及该群体后续处理的真实凋亡相关终点。只在本目录保存新材料；未改训练数据、冻结表、代码或研究目标。

**本轮增加了可复核的局部证据，没有取得新的“多初始状态 × 多个可比较小分子候选 × 定量凋亡结局 × 状态参考 RNA”完整数据块。** 这不是“此类数据不存在”的结论：本轮精读了两个新候选及其可取补充，又核对了一条原文引文，范围到此收敛。

| 来源 | 实际得到什么 | 当前能支持什么 | 仍不能支持什么 |
|---|---|---|---|
| Schepers 2021，PMC7611779 | SBI0206965 诱导/对照 A549，ABT263 再挑战；Fig4C 作者印出的 4 个 Caspase3/7 百分比 | 有限的类别状态与单次小分子再挑战证据 | Fig4C 的 ABT263 剂量、诱导天数未能从已读主文落实；对照时间配对有边界；没有取得对应公开 RNA；不能据此验证 RNA 输入的增量价值或跨药排序 |
| Casagrande Raffi 2024，PMC11670111 | Pro/Sen A549 × salinomycin 2 µM、48 h 的 cCASP3/cCASP8 Western blot；GSE279184 六份 Sen RNA 的实际计数表 | 明确诱导历史下的 Sen 参考 RNA，以及单药、单剂量、两状态的凋亡蛋白证据 | WB 未提供可复算定量结局表；RNA 没有 Pro；药后 RNA 为 2.5 µM，WB 为 2 µM；多剂量/多候选数值主要是活力，不能转成凋亡率 |
| Wang 2022 cFLIP/DR5，DOI 10.1038/s43018-022-00462-2 | 已在第一轮取得 12 个源重复 | 复用第一轮证据，无新增下载或结论升级 | 主要 Caspase 挑战为 Lz-TRAIL 蛋白；navitoclax 相关读数为 CellTiter-Blue；不是新增完整小分子数据块 |

## 1. Schepers 2021：保留印值，保留协议缺口

论文：*Identification of Autophagy-Related Genes as Targets for Senescence Induction Using a Customizable CRISPR-Based Suicide Switch Screen*，Mol Cancer Res 19(10):1613–1621。DOI [10.1158/1541-7786.MCR-21-0146](https://doi.org/10.1158/1541-7786.MCR-21-0146)。[PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611779/)。本地全文 `PMC7611779.xml`，官方原图 `1613fig4.jpg`。

已读定位：XML `sec2` 全部 Methods，尤其 `sec2-1`、`sec2-5`、`sec2-9`、`sec2-10`；Results；`fig2`、`fig3`、`fig4` 图注；Fig4 原图 A–C。

### 数值与分母

`Schepers2021_Fig4C_printed_percentages.csv` 登记的是图中 A549 第一行四个绿色印值：

| 图中列标题 | 前置状态 | 印值（%） | 读数角色 |
|---|---|---:|---|
| Control | 对照 A549 | 1 | 对照/加药前图；不能未经确认当作 24 h vehicle |
| ABT263 | 对照 A549 | 5 | 图注给出加 ABT263 后 24 h |
| 5 µmol/L SBI | SBI0206965 预处理 A549 | 0 | 诱导状态的对照/加药前图；不能未经确认当作 24 h vehicle |
| SBI + ABT263 | SBI0206965 预处理 A549 | 81 | 图注给出加 ABT263 后 24 h |

这是作者在代表性图像上印出的百分比，**不是本轮按像素估计的柱高，不是恢复出的单细胞计数，不是四组原始重复**。Methods `sec2-5` 说明用 Caspase-3/7 Green reagent（#4440）并基于 biological triplicates 的图像量化；但图没有说明这四个印值各自是否为三次实验的平均值，故 `summary_statistic` 保留未明确，不能把 4 个数扩增成 12 个数。

原图 733 × 967，坐标原点为左上。CSV 的坐标框用于定位印字，不用于重新计算分母；原始细胞总数、阈值明细和每次实验的数值未取得。

### 剂量/时长窄复核

- Fig4C 印出 SBI **5 µM**；Fig4B 印出 SBI **2.5 µM** 和 ABT263 **2 µM**，图注对 B 明确 SBI 预处理 **4 天**。不把 B 的两项参数自动转移到 C。
- Fig4C 图注明确加 ABT263 **24 小时后**成像，这个时长可以直接登记。
- 全部主文 Methods 中，ABT263 仅在 `sec2-10` 采购清单出现；`sec2-5` 只说按图示浓度加药。**没有查到可覆盖 C 的通用 ABT263 剂量规定**。因此 C 的 `challenge_dose_uM` 为空，另记邻图 B 的 2 µM 为尚未接受的候选值。不是仅因 C 未印数值就认定无协议依据，而是在查完通用方法后仍未落实。
- SBI 是否在 ABT263 再挑战期间维持、是否洗脱、洗脱到再挑战的间隔，已读主文未明示。图面 SBI + ABT263 不能独自回答残留及维持处理问题。
- 图注描述的是 before and 24 hours after addition。Control/SBI 两列的共同终点 vehicle 地位没有明确；保留该歧义，不计算严格 24 h 处理效应差分。

这仍是一条有效的类别状态依赖凋亡线索，不能因缺少完整 RNA 而抹去。但它目前只给一个后续药物，不能单独检验多药排序，不能证明表达表征相对状态标签有增量。

### RNA 与附件可用性

`sec2-9` 有 RNA-seq 方法，且说明每个 cell line/condition **1 个样本**。Fig2E 为 shATG9A 感染后第 8 天，Fig3D 为 SBI 对照比较；二者不能混为同一初态。已读主文没有找到公共 accession。

本轮四个官方 GEO GDS ESearch 窄查询均返回 0，原响应保留为 `GEO_search_1.json`–`GEO_search_4.json`。这仅是对应查询未发现记录，并非不存在表达数据的证明。

Europe PMC `supplementaryFiles` ZIP 已实际下载，但其中只有主图 1–4 的 JPG/GIF，清单见 `Schepers2021_supplement_access.json`。出版商页面列出的 Supplementary Figures 1–5 PDF 未成功取得，不能说已核完整补充，也不能以此排除该 PDF 中另有协议细节。

## 2. Casagrande Raffi 2024：新 RNA 可用，功能标签边界明确

论文：*An antibiotic that mediates immune destruction of senescent cancer cells*，PNAS 121(52):e2417724121。DOI [10.1073/pnas.2417724121](https://doi.org/10.1073/pnas.2417724121)。[PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670111/)。本地 `PMC11670111.xml`、`pnas.2417724121.sapp.pdf`（14 页）、`pnas.2417724121fig03.jpg`。

已读定位：XML `s9` Cell Lines、`s14` Drug Treatments、`s19` Western Blot、`s21` Bulk RNA Sequencing，Fig3 及其图注；补充 S3–S5（PDF 第 6–11 页，图/图注各占一页）。

### 状态历史与功能条件

- A549 为 ATCC CCL-185，培养背景见 `s9`。
- 通用诱导方法 `s14`：alisertib **0.5 µM × 7 天**；其他诱导药物也列出，但本次 Fig3B/GSE 核心块用的是 ali 诱导。
- 再处理前明确写有 **PBS 洗涤、胰酶处理、重新铺板**。次数、残留量、重新铺板后到 Fig3B 加药的精确间隔未单独给出。
- Fig3B 的四个 WB lane：Pro−Sal、Pro+Sal、Sen−Sal、Sen+Sal；**salinomycin 2 µM × 48 h**。检测包含 cCASP8、cCASP3，HSP90 是加载对照。见 `Raffi2024_Fig3B_lane_manifest.csv`。
- 通用 Methods 有一周 senolytic 验证中第 1 天开始、每 3 天补药两次并换液的安排；RNA/蛋白组另明确 **48 h**。不能把一周给药刷新日程强加到 Fig3B 的 48 h WB。
- 本轮未取得 Fig3B 的逐重复定量灰度/百分比源表，故 lane manifest 的数值为缺失，不从条带亮度临时生成凋亡率。

该文同时检测 apoptosis、pyroptosis、necroptosis。cCASP3/8 是可保留的凋亡相关蛋白终点；总死亡、IC50 或 CellTiter-Blue 活力不能统称凋亡率。

### RNA：已下载并实际读表

[GSE279184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE279184) 含 6 个 GSM。官方 deposited 文件 `GSE279184_Gene_counts_replicates_UT_and_salinomycin_treated.csv.gz` 已实际读取：**16,670 行、17 列，6 个样本计数列，GeneSymbol 唯一 16,670 个**。6 列均有限、非负、为整数；完整检查见 `GSE279184_validation.json`。

| GSM | 样本状态与处理 | 对应源表列 | 合法角色 |
|---|---|---|---|
| GSM8563730–GSM8563732 | alisertib 诱导 Sen，未加 salinomycin，3 个生物重复 | ut_r1、ut_r2、ut_r3 | untreated_senescent_reference_not_observed_t0 |
| GSM8563733–GSM8563735 | 同类 Sen，salinomycin 2.5 µM × 48 h，3 个生物重复 | tr_r1、tr_r2、tr_r3 | post_salinomycin_expression |

原始 GSM 标题有共同 R1 前缀及最后一个样本的 Saliomycin 拼写，`GSE279184_samples.csv` 原样保留，并依据组别/biological replicate 后缀登记列映射；不是条形码匹配。

边界：

1. **没有 Pro RNA**。不能把其他论文的 Pro 直接装成此实验的同批对照。
2. Untreated Sen 可以作为该诱导群体的参考表达；资料未建立它是每个再挑战样本的实测 t0，也未提供 RNA 与 WB 的共同样本 ID。
3. RNA 再处理剂量 **2.5 µM**，WB 是 **2 µM**。前者为药后表达，不能当初态，也不能称为同条件 RNA—凋亡一一实测对应。
4. 源文件名含 counts；本轮只确认 deposited 数值，不额外声称未经归一化的原始 UMI/library counts。保留原始文件和原列名。
5. 16,670 个 gene symbol 不是项目的完整 32,461 维基因轴。主代理已另派 state_gene 在独立目录审查表达接入；本目录没有改动其处理结果。

### 为什么补图中的多个处理没有补成所需凋亡矩阵

| 图/条件 | 实际终点 | 保留/排除理由 |
|---|---|---|
| Fig3A，Sal 1/2/3 µM，3 天 | 电镜形态 | 有多剂量，但没有对应可复算 Annexin/Caspase 百分比 |
| Fig3F，死亡通路抑制剂组合 | CellTiter-Blue IC50 | 可研究活力/机制阻断，不能转写凋亡率 |
| S3B，Sal 2.5 µM，48 h | WB | 细胞是 SK-Hep1，不是 A549 |
| S3C，A549 SLC25A23 shRNA | WB | 遗传干预，不是新增可比较候选小分子 |
| S4A，Sal × conatumumab | CellTiter-Blue | 多候选/组合的活力数值；conatumumab 是 DR5 抗体，不能冒充小分子 SMILES 动作 |
| S5A/B，多治疗臂 cleaved-Caspase3 IHC | A549 移植瘤组织图像与区域统计 | 体内瘤组织，不是 GSE279184 的体外 A549 群体；不能配成同条件功能标签 |

## 3. 最后单篇引文补漏：Wang 2022 已覆盖

PNAS 引用的 *cFLIP suppression and DR5 activation sensitize senescent cancer cells to senolysis*，DOI [10.1038/s43018-022-00462-2](https://www.nature.com/articles/s43018-022-00462-2)，正是第一轮 `../../state_function/README.md` 第 1 节的来源，并非此前 Nature Aging 的 TGF donor/recipient 研究。

本轮只核对已有证据与身份，不将旧结果重新记为新发现。第一轮已保存官方 Fig3 源表 `43018_2022_462_MOESM7_ESM.xlsx`、作者学位论文全文、GSE102639 和 SENESCopedia 表达。12 个 Caspase3/7 重复对应 Pro/Sen × 对照/Lz-TRAIL 200 ng/ml；Lz-TRAIL 是蛋白。Navitoclax Fig4e 为 CellTiter-Blue IC50，不能填补小分子凋亡终点；此前记录的诱导浓度/撤药协议差异仍在。详见第一轮原文定位和表格，本轮没有覆盖或重新抽取它们。

## 4. 交接与停止位置

本轮可新增到证据登记的是：**4 条作者印出的 Caspase 百分比、4 个明确条件的 WB lane、6 个已实际读取的 RNA 样本及其状态/处理元数据**。三类证据的数值层级不同，应分别保存。4 条百分比没有独立重复信息；4 个 WB lane 不等于 4 个定量凋亡标签；6 个 RNA 样本不等于 6 个配对功能标签。

若要继续使用 Schepers 的数据，需要针对其未取到的补充文件或原作者公共存档补齐 C 图剂量/诱导天数，以及表达 accession；在此之前不自动进入标准药物剂量评测。Raffi 的 Sen 参考 RNA 适合作为来源明确的表达资源，但单独不能解决多状态候选排序的监督需求。

本轮完成预定两个新候选、一次缺口定向补查及一条指定原文引文复核，到此结束。未联系作者、未写 Zotero、未启动训练、未创建新的研究目标；没有把公开数据的空缺表述为数据绝对不存在。

## 文件导航

- `Schepers2021_Fig4C_printed_percentages.csv`：四条 A549 图中印值。
- `Raffi2024_Fig3B_lane_manifest.csv`：四个 WB lane 条件；定量值留空。
- `protocol_evidence.json`：原 XML 相关章节与图注的直接抽取，便于主代理亲读。
- `dose_protocol_readback.md`：Schepers 的主文剂量/时长窄复核。
- `search_and_access_log.md`：互补检索、旧来源去重、失败回退和剩余盲区。
- `GSE279184_samples.csv`、`GSE279184_validation.json`：六个 RNA 样本及源表检查。
- `local_file_inventory.json`：本目录保存文件的字节数与 SHA256，不包含其自身。
