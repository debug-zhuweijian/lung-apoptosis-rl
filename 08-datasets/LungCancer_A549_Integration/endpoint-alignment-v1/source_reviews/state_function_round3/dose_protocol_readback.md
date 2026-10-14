# NSH76 剂量协议窄复核

日期：2026-10-10。仅复核同一论文、原图与原始补充；不扩搜。本文是对既有“panel 未标剂量”判断的细化，**不覆盖既有 CSV**。

## 判定

**可以把 Fig6/S2c 的 A549、A549-CisR、A549-DoxR 分别记录为 7.38、2.85、2.63 µM 的 `protocol-derived nominal dose`（依据论文协议推定的标称剂量）。** 原因是 Methods 明确规定 NSH76 默认使用 IC50，Fig6i 和同节正文给出各状态 IC50，Annexin 的暴露时长又单独明确为 24 h。不能仅因 S2c 没逐图印浓度，就将全部剂量信息降为完全未知。

这层剂量信息仍不同于“该 panel／每个孔的浓度已直接记录”。新增旁表同时保留 `dose_origin=global_method_plus_Fig6i_context` 与 `panel_directly_labels_dose=false`。旧表中的 `panel_dose_uM` 空值仍如实表达“图内没有直接标值”；读取时应合并这个附加证据层，而不是把空值解释成没有任何剂量依据。

## 原始证据链

原文：[PMC12538875](https://pmc.ncbi.nlm.nih.gov/articles/PMC12538875/)，DOI [10.1186/s12967-025-06588-y](https://doi.org/10.1186/s12967-025-06588-y)；本地原始 XML 为 `PMC12538875.xml`。

| 定位 | 原文已核实内容 | 支持的字段 |
|---|---|---|
| Methods `Sec10` Drug treatments | NSH76 在所有实验中默认使用其 IC50；DMSO 为 vehicle | 未另外注明剂量的 NSH76 功能实验采用 IC50 规则 |
| Fig6i 原图；Results `Sec36` | A549 7.38 µM，A549-CisR 2.85 µM，A549-DoxR 2.63 µM | 本次三状态耐药比较的状态特定标称浓度 |
| Methods `Sec19` Cell viability assay | 先接种培养 24 h，再加药 24 h；使用 Alamar Blue，570 nm 读数。耐药细胞的 NSH76/cisplatin/doxorubicin/DMSO 处理也是 24 h | IC50 来自 **24 h 药物暴露后的代谢活力检测**，不是 48 h 持续用药，也不是 Annexin IC50 |
| Methods `Sec23` Apoptosis assay | A549及其两种耐药株用 NSH76 等处理 24 h，随后 Annexin-V/PI 流式 | 功能端点暴露时间 24 h |
| Fig6k 图注与 S2c 原始 PDF | 三种状态的 NSH76/control 流式实验；S2c 本身不印浓度 | 允许按同节协议推定，不声称 panel 直接记录 |

需要保留一个来源内差异：Results `Sec35` 对一般癌细胞实验的描述给 A549 IC50 **7.52 µM**（指向 Fig5a）；而耐药比较 `Sec36` 与 Fig6i 给 **7.38 µM**。本次是 Fig6/S2c，故选择该比较块的 7.38 µM 作为标称剂量有上下文依据；未获得逐孔用药日志，不声称已排除实际使用 7.52 µM 或另一批次值的可能，也不擅自解释两值差异为笔误。

IC50 的精确拟合函数、完整原始活力读数、每个 Annexin 重复的配药记录未取得。文章的全局“所有实验”应与明确写有剂量梯度的 viability 实验一起阅读；有明确梯度的实验以其局部剂量为准，不能把整篇所有处理强行改写成单一 IC50。

## NSH76 化学身份的原始依据

NSH76 是论文的研究化合物名，**其结构并非未公开**：

- `Sec6` NSH76 synthesis 给出从 4,7-dichloroquinoline 出发的合成叙述，并引用参考文献 27。
- 已下载并亲看 `NSH76_Figure2.png`：Fig2b 明确标 NSH76 的二维化学结构及名称，Fig2g 给出作者所称 normal/nitreone 两种表示。
- 已取得原始 `12967_2025_6588_MOESM2_ESM.docx`（Electronic structure analysis of NSH76）：包含 neutral 与 nitreone 形式的能量及原子坐标；文档分别标 `0 1` 与 `1 1`。这里只忠实记录作者的两种计算形式，不新增对其化学解释或有效性的判断。
- 原文参考文献 27 为 Bhagat et al., *Design, synthesis and biological evaluation of 4-aminoquinoline-guanylthiourea derivatives as antimalarial agents*, DOI 10.1016/j.bioorg.2019.103094。**本次只核验到该引用条目，未读取被引文的原始合成表，因此不将其某个编号化合物直接等同于 NSH76。**

这已足以将本研究内部的 NSH76 暴露识别为同一有公开结构依据的小分子。当前可用内部 ID 为 `NSH76__10.1186_s12967-025-06588-y`。本次没有确认公共 CAS/PubChem ID、盐形式、作者提供的机器可读 SMILES 或跨数据库同义名；没有从结构图临时编造 SMILES。后续接入 drugSMILES 时仍需单独核对结构标准化、质子化／互变异构体选择。这里的“有可追溯化学身份”不等于“现有模型已经拥有这个动作的可信响应数据”。

## 能支撑的有限挑战与不能证明的内容

可以把这组数据表述为：**同研究、同一小分子、24 h 暴露、按每种状态各自的 24 h viability IC50 选择浓度的三状态反应挑战**。状态可用明确定义的群体标签与诱导史，不机械要求同细胞 t0 或完整转录组。

本次可识别性分析：观测的是 `Y(s, d_s, 24 h)`，其中 `d_s` 由该状态的 viability IC50 决定。三状态使用不同浓度，没有共同固定剂量下的完整响应，故状态与剂量同时变化。仅凭这三个点，不能分离状态作用与浓度作用，也不能证明状态条件化模型优于只用剂量的模型。它是一种按预先测得活力敏感性调整剂量的实验规则；测试新状态时，不能假设其未知 IC50 已免费可得。

另外，这里只比较 NSH76 与 vehicle，并没有形成三个状态均具有明确剂量的多种活性药物排序矩阵。cisplatin／doxorubicin 的各 panel 剂量问题没有被本次 NSH76 证据解决。S2c 仍只有各条件一张显示的代表流式图，不能替代完整独立重复。

功能标签继续使用原图象限：总 Annexin 阳性为 Q2+Q3，而非 Q1+Q2+Q3；“按 viability IC50 给药”不意味着 Annexin 阳性恰为 50%。本次接受协议推定剂量，并不撤销端点区分、代表图重复限制或跨药排序缺口。

因此建议接收为**有限、协议有依据的 NSH76 群体状态挑战记录**，暂不作为已经验证状态增益、RNA 表示迁移或跨药排序效果的证据。机器可读附加记录保存在 `dose_protocol_readback.json`；原 CSV 和先前文件保持不变。
