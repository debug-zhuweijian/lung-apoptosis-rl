# ChEMBL A549 Annexin 原文审查：第二轮

日期：2026-10-10。审查者：endpoint_pairing_review。范围是现有 ChEMBL 候选的定向原文核验；未训练、未联系作者。本包独立保存，第一轮 `../chembl/` 与主表 `curated-r3` 均未改写。

本包包含 **6 篇原文审查 JSON、17 个药物—剂量—时间审查行、8 个 ChEMBL 分子 ID、6 张原始流式图**。17 行不是17个已可训练终点：9行可从原图复算完整Annexin双象限比例，2行有完整数值但协议冲突必须hold，3行只有early，3行只有“相对对照增加”的正文百分数且绝对分母解释待定。核心表仍以主代理当前落盘的30个汇总Annexin条件和14个化学输入候选为准；不得提前把本包加入这两个计数。

| 来源与最短原文定位 | 条件和已读原始数值 | 当前可用层级 | 关闭缺口 |
|---|---|---|---|
| [C8MD00620B](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec11)，sec11/sec46/Fig6；化学sec37 | 6k，24h，5/10/20μM；early+late=11.80/29.42/40.38%，对照1.58% | 3个原文数值条件；5μM属于优先剂量时长，但三行化学接点均hold | 原文C39H53FN4O5/HRMS677.4073，ChEMBL却C39H55FN4O4/MW662.891；须重建并独立复核结构，不能直接接入；Annexin重复数未明确 |
| [C9MD00494G](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489057/#fig4)，Fig4A | 8h，48h，0.3/1/3μM；early+late=11.84/17.43/97.64%，对照6.20%；膜染料7-AAD | 3个48h原图数值条件；完整协议尚未读取 | 读取SI生物实验方法、重复/培养背景；结构核对；严格48h，不并入24h |
| [Li2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587690/#F3)，FigF3/secS31 | 1a，48h，5/7.5/10μM；total=9.56/13.00/23.60%，对照6.88% | 3个48h原文条件；完整XML可读 | 结构核对、主代理复核；重复数/事件筛选明细缺失 |
| [D4MD00405A](https://pmc.ncbi.nlm.nih.gov/articles/PMC11376208/#sec16)，sec16/sec5/Fig2C | SLL-1A-16，24h10/30μM；代表图total12.14/18.99%；正文mean±SD12.20±0.13/18.87±0.14 | 2行hold；只能记录数值事实 | 乙醇固定与Annexin合写协议未解决；caption20μM与图/Methods30μM矛盾；不可自动修正纳入 |
| [D4MD00364K](https://pmc.ncbi.nlm.nih.gov/articles/PMC11324042/#sec12)，sec12/sec28/Fig6 | tadalafil210、avanafil340、udenafil256μM，48h；reported22.2/44/55.9% | 3行reported-percent-only，absolute total为空 | 需要明确绝对百分比或相对增加、取得未处理对照数字；高剂量48h扩展；图上cisplatin50μM无数字，未创建终点行 |
| [C8MD00487K](https://pmc.ncbi.nlm.nih.gov/articles/PMC6336084/#sec19)，sec19/Fig4toprow/Table4 | L1，24h1/2/5μg/mL；early12.47/30.64/76.86%，对照early6.09% | 3行early-only，total为空，质量剂量保留 | late象限缺失；原晶体MW459.58与ChEMBL无水盐成分不同，须解决剂量物种/结构，换算值不当作原文μM |

原图都显示横轴Annexin和纵轴膜完整性染料。不同论文的Q编号不同，JSON按真实坐标记录。8h的PI专属字段为空，使用7-AAD通用字段。百分比是被分析的流式群体分母；没有原始FCS/完整门控，不等于初始播种细胞的存活或死亡概率。

## 文件与复算

- `reviewed_conditions.csv`：17行扁平审查表；`total_percent`、`reported_percent_only`、`reported_mean_total_percent` 分开。空值保持空值，不能用0填充。
- `CHEMBL*.json`：沿用第一轮schema v1并增加绝对终点/膜染料/均值/剂量换算/结构待核验字段。所有记录 `use_for_training=false`、`added_to_curated_r3=false`。
- `raw/*_fig*.jpg`：6张下载的原始流式图；`raw_manifest.json`记录原始缓存大小与SHA256。
- `raw/PMC3587690.xml`：已实际读取的完整作者稿XML。其他 `PMC*_read_extracts.json` 是审查者结构化抄录，并非完整原文。
- `raw/primary_text_6k.md`、`primary_text_8h.md`、`primary_text_SLL1A16.md`、`primary_text_PDE5.md`：单独的英文原文逐字段落/图注导出，保留URL、读取日期及DOM定位。浏览器翻译改变了无关段落，故未宣称导出完整英文全文。8h的SI仍无法读取，未虚构Annexin具体步骤。
- `raw/*_activities_targeted.json`：仅3篇指定文献的ChEMBL活动，供ID回接，不作为原文数值核验的替代。目标文献活动总数以各文件page_meta为准。
- `raw/documents_*.json`、`raw/OpenAlex_*.json`：针对既有候选的元数据缓存。UCSD/BindingDB数据记录被OpenAlex列为greenOA时，不当作已取得原始全文。

复算时，9个无协议冲突的完整四象限行验证 `total_percent = early_percent + late_percent`。SLL的代表图与正文均值分开验证；PDE5和L1不得被这条规则补成不存在的total。药物、剂量、时间条件不是独立生物重复；跨来源/跨药物推断仍需预先规定划分。

## 修正与未覆盖边界

8h真正的Annexin assay在ChEMBL已写48h。它进入24h候选池的9条记录实际是G0/G1、S、G2/M细胞周期比例，描述误含Annexin。本轮撤回先前“ChEMBL把该48h凋亡错标24h”的说法，正确处理是排除细胞周期行。

没有取得单个生物重复的原始值、FCS或可用的Annexin source XLSX。本轮没有形成新增Tahoe表达—终点一一配对。所有8个分子的结构与Tahoe身份都仍待主代理核验；RDKit只复算ChEMBL标准化SMILES/InChIKey的内部一致性，不能替代原始化学物种核对。

PDE5文章的sildenafil只有未见细胞毒性的结论，未继续测Annexin；SLL是低至中等阳性而非阴性。本轮因此没有取得新的药物Annexin阴性组，不把“没测”伪造为阴性。Albert/10.1111/bph.12408已读到24hCDK9药物A549凋亡，但关键值是柱图/相对倍数，未新增绝对数值行。β-lapachone/PMC4607627的Annexin图对应6μM20h，不能借24h细胞活性图改写成24h凋亡；未新增本轮目标行。

6k、Li2012提供可复算数值，8h提供48h数值但生物方法SI缺失。PDE5、SLL、L1的待解决问题已落盘为hold，不能因想扩大数量或赶汇报而自动进入正式训练或验证。

6k的原文结构公式/HRMS与ChEMBL结构已发现实质矛盾，三条件同步标为 `hold_primary_vs_chembl_formula_mismatch`。原文sec37的HRMS与sec3氧化步骤已由本审查者亲读，关键原句另存 `raw/primary_text_6k.md`。未完成Scheme原图逐原子和立体核对前，不直接修结构。
