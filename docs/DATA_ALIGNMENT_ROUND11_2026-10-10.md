# 第十一轮数据对齐审查：真实暴露时长、Annexin 与初始状态的证据边界

后续校正（第十二轮）：本页的数值与来源核验继续有效；将“高维初态表达的真实功能增量”当作整个课题普遍前置的表述过强。架构允许先在表达分布预测上检验状态信息增量，再由真实功能块检验完整筛选链。详见[第十二轮的契约更正与实际索引](DATA_ALIGNMENT_ROUND12_2026-10-10.md)，不把本页历史缺口概括当作最新全项目否决结论。

日期：2026-10-10。研究目标仍为**面向肺腺癌促凋亡治疗的细胞状态条件化响应预测与干预筛选**。本文接续[第十轮](DATA_ALIGNMENT_ROUND10_2026-10-10.md)，依据现行[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)，没有改变课题目标、启动正式训练或运行模型推理。

## 1. 直接结论

**数据问题取得了实质进展，但完整课题的数据可行性尚未全部通过。** 本轮补到了同一 A549 背景、共同 24 h 终点的真实 Annexin 数据：两种药物、五档浓度、两种实际暴露方案，共 20 个活性条件、2 个独立图对照和 10 组条件级时长比较。两种药物的完整名义结构及 2048 位化学指纹已经核验，20 条条件已实际连接并读回。

这使得“完全没有真实凋亡端点可比较实际暴露方案”的判断不再成立。此前 Hastings 来源仅提供共同终点的相对 cleaved-caspase-3；本轮新增的是真实 Annexin/PI 图，早期与晚期凋亡相加，坏死部分明确排除。

**仍缺少的核心证据是：高维初始表达状态，在相容的实验条件下，对真实功能结局带来超出药物、剂量、背景类别或群体均值的预测价值。** 不能把一个背景的 20 个条件、TIDYI 的分选标签，或模型生成的表达向量组合后，直接宣布这一主张已经被数据支持。

| 本轮检查对象 | 已落实 | 尚未落实 |
|---|---|---|
| 药物—剂量—实际暴露方案—共同终点 | 2 药 × 5 剂量 × 2 方案，T*=24 h | 这不是多次序贯用药数据 |
| 真实凋亡读数 | 22 个总 Annexin 发表均值估计，另保留 66 个分量 | 原始重复、合成总量的 SEM、跨方案配对重复未取得 |
| 化学输入 | 2 个完整立体结构、20 个条件绑定、实际读取 20×2048 指纹 | 历史批次成分与纯度未实测 |
| 初始表达—未来功能 | 已有分层外部资料；本轮继续核查 | 新来源无配套 RNA；仍未验证表达增量 |
| TIDYI 的 RNA—功能连接 | 作者代码、60 行 CFSE/计数、12 组保存拟合参数可读取 | 无逐孔连接键，无自然群体 Annexin 分母 |
| 当前模型是否能评价该新包 | 数据读取契约已实现 | 6 h 洗脱＋18 h 恢复不能直接冒充旧单次 24 h 动作；模型适配与预测未执行 |

因此当前交付是可复算的数据与来源契约，以及更清楚的剩余问题。`formal_training_ready=false`，全课题数据可行性仍未标记通过。

## 2. 新来源究竟测了什么

原始研究为 [Vielhauer et al. 2013，Evaluation of a reductively activated duocarmycin prodrug against murine and human solid cancers](https://doi.org/10.4161/cbt.24348)。主代理亲读[原文存档](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/Vielhauer2013_PMC_fulltext.md)的 Fig3/Fig5 说明、Annexin 和 washout 方法，并查看原始图像。证据属于**作者对 A549 培养细胞所做实验的公开图表**；本课题做了读图与计算，没有新增湿实验。

### 2.1 两个方案有相同终点，不是相同暴露时间

| 条件字段 | Fig3C：持续暴露 | Fig5C 左：短暴露后洗脱 |
|---|---:|---:|
| 从首次给药开始的暴露时长 | 24 h | 6 h |
| 洗脱时刻 | 无洗脱 | 6 h |
| 后续无药培养 | 0 h | 18 h |
| 检测终点 T* | 24 h | 24 h |
| 来源图 | [Fig3 原图](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/cbt-14-527-g3.jpg)的 C | [Fig5 原图](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/cbt-14-527-g5.jpg)的 C 左 |

原文 Fig5 说明和 washout 方法写明：6 h 后更换无药培养基，继续到总计 24 h 或 72 h 的终点。**不能写成 6+24=30 h。** Fig5C 右是另一终点，本包没有将其当作持续暴露 72 h 的 Annexin 数据。原文 Results 一处将持续暴露图交叉引用成 Fig3B；实际 A549 面板为 Fig3C，本包按面板标签核对。

两药为原文简称 `BocNHO` 与 `CBI-indole2`，完整名义身份分别是 BocNHO-CBI-indole2 与 seco-CBI-indole2。五档浓度是 50 pM、500 pM、10 nM、40 nM、1 µM。全部 20 个组合均来自原图实际交集，没有补出未测组合。

### 2.2 总 Annexin 怎么得到

Fig3C 和 Fig5C 是三段堆叠柱。包中保留三个分量：Annexin−/PI+ 的坏死、Annexin+/PI− 的早期凋亡和 Annexin+/PI+ 的晚期凋亡。目标值只取后两段之和。

下表是本轮从原图边界复算的**发表群体均值估计**，单位为百分比，显示两位小数。它们不是原始流式事件或逐重复测量；完整四位小数估计和几何读图上下界在[源表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/bocnho_total_annexin.csv)。

| 药物原文简称 | 浓度 | 持续暴露 24 h | 暴露 6 h＋恢复 18 h |
|---|---:|---:|---:|
| BocNHO | 50 pM | 5.15 | 2.34 |
| BocNHO | 500 pM | 8.45 | 2.62 |
| BocNHO | 10 nM | 31.13 | 2.48 |
| BocNHO | 40 nM | 19.59 | 16.00 |
| BocNHO | 1 µM | 7.42 | 9.66 |
| CBI-indole2 | 50 pM | 4.54 | 2.34 |
| CBI-indole2 | 500 pM | 6.19 | 2.34 |
| CBI-indole2 | 10 nM | 30.31 | 2.62 |
| CBI-indole2 | 40 nM | 10.72 | 3.03 |
| CBI-indole2 | 1 µM | 7.63 | 9.66 |

均值不是随浓度单调变化；本轮按原图保留，没有平滑、纠正或据此推断剂量安全性。这里也没有把最高均值宣称为经过验证的最优干预。

读图使用原图线性轴区间。每条分量/总量记录均可追溯到图、柱位置、上下边界和轴标定；±2 像素边界与±0.5 像素轴点形成的是**读图几何范围**。原图的 SEM 没有被数字化，更没有将早、晚凋亡的 SEM 相加来制造总量 SEM。

### 2.3 这个对齐有什么限制

检测时作者收集浮游和贴壁细胞，但 6 h 洗脱步骤记载倒掉培养液，未说明保留当时已漂浮细胞。因此标签分母登记为**各自方案完成后收获的父群体**；不能宣称两组均代表原接种队列的累积凋亡概率，也不能将差异全部归因为暴露时长本身。洗脱/换液属于完整处理方案的一部分，配套 sham-wash 的执行细节仍未核定。

一般 Annexin 方法称 n=3，但每药 1 µM 组排除一个离群值后 n=2；washout 方法又统称 n=3。本包将持续组高剂量记为 n=2，短暴露高剂量保留 n=2/3 适用范围未明的状态，没有自动填平。两张图的同药同剂量只能形成**条件级比较**，不是已证实配对的生物重复。图中显著性符号是各组与 vehicle 的比较，不是持续组与洗脱组之间的 p 值。

这些限制决定了可支持的主张范围，并不抹去作者已发表的有限实验结果。该包可用于后续来源内回顾性预测/选择挑战；原始结果已经参与数据设计，不能再称为完全未见的盲测集。没有同测 RNA，只有一个 A549 背景，不能承担多状态表达增量的主验证。

## 3. 两药的结构输入也完成了实际连接

本轮同时核对 2013 Fig1D、Glossary、原引用 Lajiness2010 的材料身份和官方 PubChem 完整结构。2013 的 CBI-indole2 是明确的 seco 形式；匹配依据是完整分子图、立体和分子式，不是名字相似。

| 原文简称 | 完整名义身份 | 官方记录 | 指纹行 / 置位数 |
|---|---|---|---:|
| CBI-indole2 | 1S-seco-CBI-indole2 | [PubChem CID11016947](https://pubchem.ncbi.nlm.nih.gov/compound/11016947) | 0 / 60 |
| BocNHO | 1S-BocNHO-CBI-indole2 | [PubChem CID24755111](https://pubchem.ncbi.nlm.nih.gov/compound/24755111) | 1 / 74 |

两结构都保留原图的立体中心和共价 Cl，不作母体替换或人为增减盐。原始图与实际编码结构图、官方 JSON、SMILES、MOL、复算脚本和 hash 均保存于[化学身份包](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/duocarmycin_chemical_identity/README.md)。本轮确认的是论文图示的名义结构；历史批号、实际盐/溶剂化组成及光学纯度没有得到实测核定。

主代理从官方结构重新生成 Morgan radius=2、2048 位、启用立体的指纹，逐位回读成功；随后按明确药物键连接条件，**CBI 对应 row0，Boc 对应 row1**，没有用表格顺序猜映射。实际按 20 条条件索引得到 20×2048 数组。记录位于[chemical_condition_inputs.json](../experiments/20261010-data-alignment-round11/chemical_condition_inputs.json)。这份附表没有填入 RNA 或预测表达特征，也没有写入现有 G 训练表。

## 4. TIDYI 再审：公开代码读到了，配对的层级也查清了

本节的 v2 指 **TIDYI** 的 Research Square v2，2026-05-27，rs-8252082；与用户此前指定的 PerturbCellRL v2 是两篇论文。PerturbCellRL v2 已按 Zotero 原件审查，其结论保留在[总状态第13节](DATA_ALIGNMENT_STATUS_2026-10-10.md#13-perturbcellrl-v2对当前数据问题的帮助和限制)。TIDYI 本轮依据[原件和来源包](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/tidyi_sampling/README.md)，未改写旧标签。

### 4.1 取得了哪些此前没有读到的资料

TIDYI v2 主文第 10 页的作者公开读取说明，使分析归档中的三个相关 notebook 可以读取。本轮按范围取得 7,365,163 bytes，没有下载整个约 3.2 GB 归档，也没有执行任何 notebook 单元格。主文 PDF、补充 PDF、三个原 notebook、静态源码和保存输出均已落盘；公开访问字符串没有写入派生文档或脚本常量。

- `00_raw_sc_data.ipynb`：直接加载已有的、带标签的 H5AD。没有补出分选前总体门、抽样概率或标签生成程序。
- `FitPIModelToCFSE.ipynb`：保存 12 条 A549 序列，每条 5 个时间点。已转录成[60 行计数/荧光表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/tidyi_sampling/A549_author_saved_cfse_curves.csv)，与原 `cell2/output4` 逐值一致。
- `Fig1_panels.ipynb`：显示条件率与 RNA 汇总的实际连接方法，以及作者保存的 A549 条件率。

原输出的相对时间为 0、8、24、32、48 h。按补充材料第 2 页的方法，对应日程为 24、32、48、56、72 h；CSV 字段明确注明这是按论文日程重建。原始 per-well CSV 表头及物理 well ID 没有恢复，不能把输出顺序 1–12 改称孔号或独立生物重复编号。

### 4.2 不能将模型死亡率改成 Annexin 比例

原保存输出中，12 组主拟合的 `best_res.success` 都是 `False`，已经与[参数表](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/tidyi_sampling/A549_author_saved_fit_parameters.csv)核对。它只说明该次保存输出中的优化器成功标志；**不能据此判定原观测无效或整篇论文无效**。本轮没有重拟合，也没有将这些参数当作已经验证的训练标签。

源码 cells3/4 实际用 `np.median` 汇总，而非注释所写的 mean。12 组 A549 参数的中位数为 b=0.06074358995/h、a=0.0423190064/h，在六位小数精度下与 Fig1 保存表的 0.060744/h、0.042319/h 相等。缺少中间率文件，且 I0 有微小差异，故尚不能证明两个 notebook 的完整输出生成链完全一致。[静态原源码定位](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/tidyi_sampling/FitPIModelToCFSE_source.py)

这里 a 的量纲是每小时模型率，来自计数/CFSE 及作者的模型假设，**0.042319/h 不能写成 4.2319% Annexin 阳性率**。作者模型中的死亡、清除和实验测量定义也不能被省略。

### 4.3 真实连接键是什么

Fig1 原 cells5、16、20 对 RNA 按 `(celltype, sample)` 汇总，却只按 `celltype` 从条件率表取 a/b。A549 的 10x 元数据 sample1001/1002 分别有 1,208/1,251 行，但两个 sample 引用同一条件率；不能算成两次独立功能终点，更不能逐个细胞复制为独立实测凋亡标签。[实际 join 源码](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/tidyi_sampling/Fig1_panels_source.py)

原分选 CSV 的单事件记录及 0/100 门比例，也不能恢复分选前自然培养总体分母。无新证据改变此前 743 个细胞标签的限定用途；58 个未确认门标签的问题没有在本轮被消除。A549 没有因此新增药物/剂量功能条件；PTX 1 µM 属于另一细胞背景。

**本轮增加了可检查的作者证据，确认了条件级关联；没有恢复逐孔或逐细胞 RNA—未来 Annexin 配对。**

## 5. 其他新候选为什么暂未补上关键缺口

以下是本轮定向来源审查，不是全领域不存在相应数据的证明。读取失败也不等于资料不存在。

| 来源 | 实际取得的证据 | 与本课题关键缺口的距离 |
|---|---|---|
| [Seah2015 / PMC4632318](https://pmc.ncbi.nlm.nih.gov/articles/PMC4632318/) | A549 WT/R1/R2、TRAIL 图中 R5 区印值与顺铂 WB 等，来源包已核验 | R5 门不能直接当总 Annexin；没有配套高维初态 RNA；TRAIL 是蛋白材料 |
| [Watanabe2002](https://doi.org/10.1152/ajplung.00025.2002) | 支线已留下定位和访问记录；主代理尚未完成相关原图的核验 | 本轮没有登记可计算的 RNA—功能配对，不把它列为已通过的关键来源 |
| [Tan2023](https://doi.org/10.1093/jmcb/mjad039) | A549 TRA1-81±、GEM 0/1/10 µM 的图上早期 Annexin 印值 | RNA 是 SW1990；A549 凋亡时钟和状态分类顺序未明确，不能与邻图剂量/5 d 时间混用 |
| [Yang2019 HSP90](https://doi.org/10.1038/s41389-019-0158-7) | 原文、补充、holoclone/paraclone RNA-seq 描述及药物生长图；A549/A549R 凋亡相关 WB | 未取得对应克隆 RNA 数表/accession；%control growth 与 cleaved 蛋白不能冒充 Annexin 比例 |
| [患者代谢研究](https://doi.org/10.1038/s41467-019-11808-3) | 原文方法 pp12–14 核查 | 12 h Annexin 对象是 Patient30 MPE；邻近 A549 代谢实验不能接成 A549 RNA—Annexin |

逐项证据和主代理读取范围分别见[克隆来源包](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/clonal_states/README.md)、[HSP90 来源包](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/hsp90/README.md)及[其他原文定位](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/discovery/other_source_readback.md)。克隆支线由本地来源代理补证，主代理亲读了 Seah 的相关原文段落/图S2和决定 Tan 排除的原文/图；Watanabe 保持未完成主代理原图核验的状态，不用代理总结补成确定的机制或数据结论。

## 6. “数据一一对应”现在解决到哪一层

1. **条件字段到真实测量：本轮新包已解决。** 每个终点都有来源图、明确药物、剂量、实际暴露/恢复时钟、细胞背景和实验方案；同一条件 ID 连接实际测量与名义结构。
2. **基因轴和化学编码：已有工程基础。** 共同基因轴及新版身份包继续有效，本轮两药又完成真实结构输入；这不等于跨平台表达可直接互换。
3. **预测特征到条件级真实终点：接口允许，真实特征仍缺。** 可以研究“合法参考初态＋冻结响应模型生成特征＋真实条件终点”的路线，必须把生成特征标明为预测，验证跨来源迁移与独立终点表现，不能声称获得了实测 RNA 配对。
4. **不同初始表达状态到未来功能：核心证据仍未齐。** 不要求同一个细胞先测 RNA 再活着随访，也不要求每个候选都有完整笛卡尔网格；但需要有足够可比较的独立实验单位，使状态信息的贡献能与药物、剂量、批次、密度和粗类别区分。
5. **单细胞个体未来命运与序贯收益：尚未建立。** 群体比例或群体表达匹配不能自动确定每个细胞的未来命运。需要与预定主张相符的独立数据，不能用模拟轨迹或内部奖励补成实验事实。

当前数据足以继续推进若干有边界的预测与验证环节。要让完整主线通过数据审查，下一步应集中在第4层，而不是继续把更多单背景毒性或 WB 文献加总成样本量。

## 7. 工程产物、验证和架构一致性

| 产物 | 实际状态 |
|---|---|
| [原图、条件、分量、总量、比较及像素依据](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/exposure_duration/README.md) | 37 个文件 hash 一致，66 分量＋22 总量及几何上下界独立复算一致 |
| [构建与读取脚本](../scripts/build_vielhauer_duration_challenge.py) / [定向测试](../tests/test_vielhauer_duration_challenge.py) | 18 项测试通过；真实包构建并 `--verify` 通过 |
| [有限挑战包 manifest](../08-datasets/LungCancer_A549_Integration/vielhauer-duration-challenge-v1/manifest.json) | 20 active、2 references、10 comparisons；`input-ready/prediction-missing` |
| [化学输入附表](../experiments/20261010-data-alignment-round11/chemical_condition_inputs.json) | 20 条映射及实际 20×2048 数组读取通过；RNA/预测特征为空 |
| [整轮复核脚本](../experiments/20261010-data-alignment-round11/verify_round11.py) / [收据](../experiments/20261010-data-alignment-round11/validation.json) | 源包 hash、88 图值、60 CFSE 行、12 组参数、官方化学结构及核心旧资产绑定通过 |
| [架构一致性审查](../experiments/20261010-data-alignment-round11/architecture_alignment_review.md) | 本轮输出符合架构声明范围；不等于完整架构或模型效果已完成 |

独立复审分别保存为[科学协议审查](../experiments/20261010-data-alignment-round11/duration_independent_review.md)、[数值与读取契约审查](../experiments/20261010-data-alignment-round11/duration_numeric_and_contract_review.md)、[化学附表映射审查](../experiments/20261010-data-alignment-round11/chemical_sidecar_review.md)及[TIDYI 独立审查](../experiments/20261010-data-alignment-round11/tidyi_independent_review.md)。数值复审抽查 12 根柱的 36 条边界，并逐值复算全部 88 行；不是声称对全部柱的每个像素都做了盲法复测。高分辨率 Fig5 的 9 块原始瓦片重拼与取值 PNG 逐像素一致。

运行时 loader 每次绑定的是 **8 个直接源输入、builder 和 4 个派生文件**；全部 37 个上游文件审计另由本轮验证与独立复审执行，不混称为运行时递归检查。

核心 `curated-r5 / condition-inputs-v6 / coverage-v5` 内容未变：仍为 42 个 Annexin 条件汇总、17 个单药 24 h 化学输入、14 个结构，核心真实初态表达与生成特征计数均为0。本轮新增独立挑战没有暗中混算进这些数字。既有代码、数据和文档含多项未提交工作，未回退、提交或推送。

复算命令：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 -m pytest tests/test_vielhauer_duration_challenge.py -q
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/build_vielhauer_duration_challenge.py --output 08-datasets/LungCancer_A549_Integration/vielhauer-duration-challenge-v1 --verify
& 'F:/anaconda/envs/study/python.exe' -X utf8 experiments/20261010-data-alignment-round11/verify_round11.py
```

## 8. Cloud、Dot、检索及版本记录

Cloud 的来源补证和 Dot 的独立逻辑审查已在此前轮次真实返回。本轮继续使用其归档意见，并复核 5 份交付的 hash；**本轮没有新发 Cloud/Dot 任务，也没有将本地新增结果说成它们已经审过。** 当前协作状态见[协作记录](CLOUD_DOT_COLLABORATION_2026-10-10.md)。本轮新增数值/源码审查由主代理与本地专业代理完成。

本轮以 SciSpace 完整问题检索和网页定向检索补充线索，再回到作者原文、官方 PMC、Research Square、PubChem 与实际代码。SciSpace 的 10 个元数据结果不等于 10 篇全文审查；关键词和返回记录在[发现记录](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/discovery/initial_searches.json)。原图访问、XML/PDF 失败和替代官方入口分别留在各来源包，不把失败算成阴性文献证据。

只读 Zotero 全库顶层 API 本轮有效返回并穷尽 60 条条目。7 个新核查 DOI 在 DOI 字段无精确命中，TIDYI 标题查询也无命中；`01-papers/index.md` 的相应名称/DOI 检索无匹配。这个记录说明此次元数据查询结果，不排除缺失 DOI 或异名条目；没有自动新增 Zotero 或论文笔记。见[库内核对](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/discovery/library_crosscheck.json)。

## 9. 可以用于汇报的表述

> 本阶段完成了药物身份、剂量单位、暴露与检测时钟、凋亡检测定义及条件连接的分层核验。新增 A549 两种药物、五档剂量和两种真实暴露方案在共同24小时终点的 Annexin 条件表，并建立可复算的结构输入和数据读取接口。现有结果为有限干预筛选提供了真实功能比较依据。高维初始表达对真实凋亡预测的额外价值及其跨来源迁移仍待独立数据验证，尚未宣称完整模型或课题效果成立。

后续重点是继续取得相容的**状态表达—功能实验单位**及其实际矩阵/行映射，或明确已有条件参考能支持的可证伪状态增量实验；同时保持真实暴露协议。缺口是否关闭以实际源数据和连接证据判定。训练时间、网络规模和已立项状态都不能代替这一步的数据证据。
