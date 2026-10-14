# Round11：真实暴露时长与共同功能终点的来源核验

审查日期：2026-10-10。主来源：Vielhauer GA, Swink M, Parelkar NK, Lajiness JP, Wolfe AL, Boger D. **Evaluation of a reductively activated duocarmycin prodrug against murine and human solid cancers.** Cancer Biology & Therapy. 2013;14(6):527–536. DOI [10.4161/cbt.24348](https://doi.org/10.4161/cbt.24348)，[PMCID PMC3813569](https://pmc.ncbi.nlm.nih.gov/articles/PMC3813569/)，[PMID 23760495](https://pubmed.ncbi.nlm.nih.gov/23760495/)。

**结论：该来源提供 A549、两种化合物、五种重合浓度、两种真实暴露协议在共同 T*=24 h 的 Annexin/PI 均值图。已取得原图并建立 20 个活性条件＋2 个独立图对照的透明读图表。它可作为独立的有限联合条件挑战候选，不能单独验证多状态表达输入的增益，也不能解释为最初接种全部细胞的累积死亡率。**

本包只保存原始来源核查、图像、读取坐标与发表均值估计；未改核心 curated/condition-inputs、模型或训练配置。无模型推理、训练、原始重复补造、RNA 配对补造或实验结果预测。

## 1. 实际取得的材料及定位

| 文件 | 来源与状态 |
|---|---|
| `Vielhauer2013_PMC_fulltext.md` | 实际普通浏览器打开 PMC 后读取的全文正文；含图注与 Methods。不是 PDF 的 OCR。 |
| `pmc_browser_readback.json` | 浏览器 DOM 的原始文本、图片地址和 PDF 链接，建立下载来源链。 |
| `cbt-14-527-g1.jpg` | PMC 原 Figure 1；D 为化合物结构图。结构与立体的机器转录不属于本次已完成项。 |
| `cbt-14-527-g3.jpg` | PMC 原 Figure 3，775×1725 像素；C 明确标 A549，A/B 分属 PC3-MM2/EMT6。 |
| `cbt-14-527-g5.jpg` | PMC 原 Figure 5，712×1237 像素；C 左为24 h，C右为72 h。 |
| `figure5_zoom_manifest.js` | 官方图像放大器明确列出原分辨率1425×2475及瓦片身份。 |
| `figure5_tiles/`、`figure5_highres_assembly.json` | 普通浏览器放大后实际观察到的9个官方高分辨率瓦片，逐文件URL/hash留存。 |
| `figure5_C_left_highres_context.png` | 无缩放拼接原图坐标[0,1024,768,1792]的矩形；每个像素都有上述原瓦片，不是增强或生成图像。 |
| `pixel_landmarks.json`、`digitization_overlay.png` | 人工审核的柱界/轴标坐标及覆盖线，供独立复核。 |

全文精确定位（上述 Markdown 行号）：

- **89**：Fig3 图注；连续24 h；C=A549；parent population 分母。
- **101**：washout Results。段落把24 h连续实验写为Fig3B，但图注与图中细胞标签明确B=EMT6、C=A549。本包采用Fig3C，不随正文错引混入EMT6。
- **105**：Fig5 图注；6 h后冲洗，drug-free培养至24或72 h剩余时间；C为凋亡/坏死。
- **152**：A549来自ATCC；F-12K培养基＋10% FBS；37°C、5% CO2。培养段中A549-luc2是后续体内实验，不替代体外Fig3/5的细胞身份。
- **164**：Annexin Methods；接种3×10^5细胞于T25，培养24 h后处理；终点收集floating与adherent；Invitrogen PHN1018、LSRII/FACSDiva；mean±SEM和n说明。
- **172**：Washout Methods；6 h时倾去培养基、冲洗、加新鲜培养基；无药培养至总24/72 h；段落另写n=3。

目前未取得 PDF 文件。已观察官方PDF链接，但本地下载403；publisher PDF下载TLS失败、web访问403。PMC Cloud官方prefix查询返回KeyCount=0；旧OA API404、EuropePMC XML500，均保留访问记录。公开页面和原图已实际读取，这些失败不被写作“全文未读”。本文可见两张数表是增殖IC50与动物存活数，未发现Annexin均值数表或原始FCS附件；这一判断仅限本次读取的论文正文/所列链接，不声称所有可能附件均不存在。

## 2. 处理协议与三个时钟

以**开始给药**为t=0；接种后先培养24 h属于前置培养，不能算成药物暴露。

| 原图 | 真实暴露Δt | 洗脱 | 无药恢复 | 功能测量T* | 本包职责 |
|---|---:|---|---:|---:|---|
| Fig3C | 24 h | 未设6 h洗脱 | 0 h | 24 h | continuous组 |
| Fig5C左 | 6 h | t=6 h倾培养基、冲洗、补新鲜培养基 | 18 h | 24 h | pulse组 |
| Fig5C右 | 6 h | 同上 | 66 h | 72 h | 本包保留原图作证，不加入共同24 h数值表 |

18 h与66 h来自原文明示“remainder of the 24- or 72-h timepoint”的算术，而非额外猜测培养时间。不能将Fig5C右误记为72 h持续暴露，也不能将其与Fig3C跨终点做纯时长比较。

## 3. 药物、剂量、状态与样本对应

两图亲视确认的重合浓度均为：**50 pM、500 pM、10 nM、40 nM、1 μM**，统一nM后为0.05、0.5、10、40、1000。化合物分别为BocNHO-CBI-indole2（论文简称BocNHO）与seco-CBI-indole2（论文简称CBI-indole2）；不可把前药与母体合成同一药物ID。

两种给药协议使用相同论文定义的A549培养背景。本文没有本次功能实验的初态RNA、匹配表达参考或逐细胞前后对应；状态只能登记为**source-defined parental A549 standard culture**。不会创建expression_sample_id或声称初态已实测。

10个名义药物—剂量配对各有两种协议均值，形成20个活性条件。每张图有一个vehicle，共2个图对照；不把一个vehicle复制成每药独立样本。体外vehicle具体成分在本次读取的Methods中未明确，不能借体内DMSO填写。

跨图均值可按相同名义条件并列，**没有相同培养瓶/同批原始重复的一一对应证据**。比较表的“comparison”是条件级关联，不是paired biological replicate。

## 4. 凋亡定义与分母限制

- early apoptosis：Annexin V阳性、PI阴性。
- late apoptosis：Annexin V阳性、PI阳性。
- necrosis：Annexin V阴性、PI阳性。
- 分母：原文的parent population；详细父门、事件数、singlet或debris门未列明。
- 总Annexin阳性=early+late。原堆叠柱顶包含黑色necrosis，**不能直接把总柱高当总Annexin**。

本包分母固定为`post_protocol_harvested_parent_population`，采样门固定为`parent_population_gate_not_further_documented`。该定义标记测量的局限，不证明两种处理保留了相同的原始细胞队列。

特别是洗脱t=6 h倾掉培养基时，原文没有说明是否回收其中的浮游细胞。终点收集浮游＋贴壁不解决这个早先可能丢失的亚群。因此本包只能支持**各自给药/收获协议后的父群体比例比较**，不能直接估计最初接种队列的累积死亡概率，也不能将差异归结为随机化控制下的纯暴露时长因果效应。完整候选身份必须包含洗脱和收获协议。

## 5. 统计层级与读图方法

数据是发表的**独立实验组均值**，不是单细胞事件或重造重复。

| 位置 | 原文n/误差说明 | 本包记录 |
|---|---|---|
| Fig3C一般浓度 | mean±SEM，n=3 independent experiments | n=3 |
| Fig3C每药1 μM | 每药去除一个outlier，n=2 | n=2；保留排除说明 |
| Fig5C左非1 μM | washout段mean±SEM、n=3 | n=3 |
| Fig5C左每药1 μM | washout段n=3；一般Annexin段1 μM异常值例外是否也适用未明 | published_n为空，n_status明确2/3适用范围歧义 |

没有根据条形SEM长度推算原始重复。没有将早/晚凋亡SEM相加或按独立假设平方和开根号，因两成分的协方差未公开。图星号是各条件对vehicle的unpaired t-test，不是连续vs洗脱的显著性。

提取使用三条边界：总柱顶、early/late分界、necrosis/early分界；均取线条中心，避开中间的SEM误差帽。Fig3C用0–60未断轴部分（y=1521.5至1230.5）；Fig5C左高分辨率crop用0–40未断轴部分（y=497.5至207.5）。100附近断轴不参与计算。

每个柱界采用±2像素读取范围，每个轴标采用±0.5像素范围，通过单调比例换算保守上下界。**上下界只是几何读图范围，不是SEM、SD、置信区间或模型预测区间。** 低值黑段小于边界精度时下界为0。表中多位小数便于复算，并不表示同等实验精度；汇报宜保留一位小数。

对于总Annexin，early/late共用边界抵消，直接用necrosis/early边界减总柱顶计算；不把两成分读图区间机械相加。这是相同柱段的算术，不是增加独立样本。

## 6. 稳定读取接口

| 文件 | 行数 | 关键列 |
|---|---:|---|
| `bocnho_conditions.csv` | 22 | condition_id、protocol_id、protocol_arm、cell_line、state_id、drug_id/name、dose_label/nM、exposure_duration_h、washout_at_h、drug_free_recovery_h、terminal_time_h、is_vehicle、source_figure、published_n/n_status、denominator/sampling_gate |
| `bocnho_component_means.csv` | 66 | condition_id、component（necrosis/early_apoptosis/late_apoptosis）、estimate_percent、reading_lower/upper_percent、原图和边界坐标 |
| `bocnho_total_annexin.csv` | 22 | condition_id、label_type、percent与fraction均值/读图区间、derivation、空SEM及原因 |
| `bocnho_duration_comparisons.csv` | 10 | drug/dose/T24、continuous/pulse condition_id、非重复配对说明、采样限制 |
| `extraction_audit.json` | 1 | 实际计数、输出SHA256与未生成的RNA/重复/模型预测/SEM数量 |

统一protocol_id为`vielhauer2013_a549_annexin_pi_harvested_parent`。所有行的`terminal_time_h=24`；`exposure_duration_h`必须与`drug_free_recovery_h`同时读取。两个protocol_arm分别为`continuous_24h`和`pulse_6h_washout_recovery18h`。标准化label_type仅描述Annexin类型，不覆盖本包分母/收获局限。

## 7. 数值候选示例与允许的验证

以下为去掉黑色坏死底座后的总Annexin均值读图估计，百分比；完整读图区间见CSV：

| 化合物 | 浓度 | 连续24 h | 6 h＋18 h无药 |
|---|---:|---:|---:|
| BocNHO | 0.05 nM | 5.2 | 2.3 |
| BocNHO | 0.5 nM | 8.5 | 2.6 |
| BocNHO | 10 nM | 31.1 | 2.5 |
| BocNHO | 40 nM | 19.6 | 16.0 |
| BocNHO | 1000 nM | 7.4 | 9.7 |
| CBI-indole2 | 0.05 nM | 4.5 | 2.3 |
| CBI-indole2 | 0.5 nM | 6.2 | 2.3 |
| CBI-indole2 | 10 nM | 30.3 | 2.6 |
| CBI-indole2 | 40 nM | 10.7 | 3.0 |
| CBI-indole2 | 1000 nM | 7.6 | 9.7 |

可以检验有限候选的条件均值预测/排序，保留来源与读图不确定性，并按协议处理采样差异。数据使真实暴露时长比较不再只依赖Hastings单一来源，但**尚未证明本项目模型能读取这些条件、能准确预测或能优于基线**。本包没有给出模型性能。

不能单靠本来源声称：状态RNA增益成立、跨来源初态已校准、同细胞未来命运恢复、患者个体化疗效、任意剂量/时长泛化、真正序贯策略有效。两化合物属于相关的前药/母体，不等于广泛药物化学空间的独立覆盖。

## 8. 复算与冻结核验

在项目根目录使用：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 -B '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/extract_bocnho_published_means.py'
& 'F:/anaconda/envs/study/python.exe' -X utf8 -B '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/verify_audit.py'
```

提取脚本核验源图hash，验证22条件唯一、20活性/2vehicle、66分量/22总量/10比较、6+18=24与24+0=24、early+late算术一致且区间合法。它只重算本目录的发表均值候选，不调用任何模型。

正文和两原图的独立只读复审在主代理的`experiments/20261010-data-alignment-round11/duration_independent_review.md`，已确认名义剂量/时钟/分母边界；数值提取的独立复审状态另由该审查记录说明，本包不代替审查人签署。

## 9. 本轮另一个已核候选及排除理由

Tierno et al., **Microtubule Binding and Disruption and Induction of Premature Senescence by Disorazole C1**, J Pharmacol Exp Ther. 2009;328:715–722，[DOI10.1124/jpet.108.147330](https://doi.org/10.1124/jpet.108.147330)，[PMC2649750](https://pmc.ncbi.nlm.nih.gov/articles/PMC2649750/)。

其Fig5A确有A549：disorazole C1 10 nM、nocodazole1 μM、vehicle，1 h脉冲洗脱与持续暴露的trypan blue viable-cell count比较。Fig5B为另一连续24/48 h实验的TUNEL（doxorubicin50 nM、disorazole2/10 nM），不是同T的pulse-vs-continuous Annexin/caspase。不能借相邻Fig5B把Fig5A活力终点改成凋亡终点。原文已保存于`primary_exa_fulltexts.md`后半；未提取其图点或把它接入Annexin表。

按主代理后续指示，本轮收敛到BocNHO这一会改变缺口判断的来源，不为凑候选数继续扩展检索。Hastings未重复提取，NavGal/Kotani采样曲线未改写成持续暴露时长。

