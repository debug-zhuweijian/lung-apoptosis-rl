# GSE161218 表达—凋亡条件匹配核验

> 核验日期：2026-09-18。源数据为公开实测结果；此处做数据审计，没有训练预测模型。

## RNA 样本逐项登记

所有 RNA 样本均为 H1792、16 h。组别由 GSM 的 title/source_name 与正文 Fig. 1F/G 共同确认。

GSM 的 treatment_protocol 对全部九条记录复用同一段两种药物剂量说明，连 CTL 也如此；不能将它解析为九条组合处理。

| GSM | 原始标题 | 条件 | Trametinib μM | Lestaurtinib μM | 组内重复编号 |
|---|---|---|---:|---:|---:|
| GSM4892080 | CTL_1 | DMSO control | 0 | 0 | 1 |
| GSM4892081 | CTL_2 | DMSO control | 0 | 0 | 2 |
| GSM4892082 | CTL_3 | DMSO control | 0 | 0 | 3 |
| GSM4892083 | Lest_1 | Lestaurtinib | 0 | 0.625 | 1 |
| GSM4892084 | Lest_2 | Lestaurtinib | 0 | 0.625 | 2 |
| GSM4892085 | Lest_3 | Lestaurtinib | 0 | 0.625 | 3 |
| GSM4892086 | Tram_1 | Trametinib | 0.5 | 0 | 1 |
| GSM4892087 | Tram_2 | Trametinib | 0.5 | 0 | 2 |
| GSM4892088 | Tram_3 | Trametinib | 0.5 | 0 | 3 |

## Annexin V 阳性百分比原始值

来源：官方 Source Data 工作表 `Fig Suppl 2`；处理均为 24 h。Tram=0.5 μM，Lest=0.625 μM，Combo 为这两剂量的同时双药处理。

补充 PDF 第 5 页图注规定 n=3 independent experiments。每行的三个数按源表 Exp 1、Exp 2、Exp 3 排列；它们不自动对应 RNA 的 _1、_2、_3。

| 细胞系 | 条件 | Exp 1 (%) | Exp 2 (%) | Exp 3 (%) | 源单元格 |
|---|---|---:|---:|---:|---|
| H1792 | Control | 1.16 | 0.96 | 1.05 | C12, C13, C14 |
| H1792 | Trametinib | 9.2 | 8.43 | 9.11 | D12, D13, D14 |
| H1792 | Lestaurtinib | 1.23 | 1.41 | 0.991 | E12, E13, E14 |
| H1792 | Simultaneous combination | 15.9 | 17.2 | 17.6 | F12, F13, F14 |
| H2009 | Control | 1.38 | 1.06 | 2.14 | I12, I13, I14 |
| H2009 | Trametinib | 1.89 | 2.04 | 2.11 | J12, J13, J14 |
| H2009 | Lestaurtinib | 4.03 | 3.89 | 3.35 | K12, K13, K14 |
| H2009 | Simultaneous combination | 6.63 | 7.15 | 6.96 | L12, L13, L14 |
| A549 | Control | 0.479 | 0.658 | 0.558 | O12, O13, O14 |
| A549 | Trametinib | 1.99 | 1.85 | 2.18 | P12, P13, P14 |
| A549 | Lestaurtinib | 3.13 | 2.96 | 3.01 | Q12, Q13, Q14 |
| A549 | Simultaneous combination | 6.23 | 6.69 | 6.54 | R12, R13, R14 |
| H1568 | Control | 4.53 | 3.36 | 3.69 | C19, C20, C21 |
| H1568 | Trametinib | 3.14 | 3.06 | 2.77 | D19, D20, D21 |
| H1568 | Lestaurtinib | 5.92 | 5.86 | 5.31 | E19, E20, E21 |
| H1568 | Simultaneous combination | 6.23 | 6.42 | 6.36 | F19, F20, F21 |
| H1993 | Control | 0.498 | 0.55 | 0.607 | I19, I20, I21 |
| H1993 | Trametinib | 0.53 | 0.597 | 0.524 | J19, J20, J21 |
| H1993 | Lestaurtinib | 0.761 | 0.8 | 0.644 | K19, K20, K21 |
| H1993 | Simultaneous combination | 0.929 | 0.769 | 0.814 | L19, L20, L21 |
| H1437 | Control | 3.18 | 3.51 | 2.64 | O19, O20, O21 |
| H1437 | Trametinib | 1.06 | 1.04 | 1.02 | P19, P20, P21 |
| H1437 | Lestaurtinib | 1.45 | 1.47 | 1.35 | Q19, Q20, Q21 |
| H1437 | Simultaneous combination | 1.84 | 1.33 | 2.09 | R19, R20, R21 |

## 连接层级与允许的用途

| 连接候选 | 同研究 | 同细胞系 | 同药物及剂量 | 同处理时长 | 同一样本/可配对重复 | 判断 |
|---|---|---|---|---|---|---|
| H1792 Tram RNA → Annexin | 是 | 是 | 是，0.5 μM | 否，16→24 h | 未建立 | 可作条件级、跨时间辅助验证；不是同终点配对标签 |
| H1792 Lest RNA → Annexin | 是 | 是 | 是，0.625 μM | 否，16→24 h | 未建立 | 可作条件级、跨时间辅助验证；不是同终点配对标签 |
| H1792 CTL RNA → Annexin Ctrl | 是 | 是 | RNA 图注为 DMSO；Annexin 图注为 untreated，溶剂一致性未明确 | 否，16→24 h | 未建立 | 单独标记对照定义缺口，不默认相同 |
| H1792 Combo RNA → Annexin Combo | 是 | 是 | RNA 中没有组合组 | 不适用 | 不存在该 RNA 组 | 只有同时组合终点，不能构造对应 RNA |
| A549 等其余五细胞系 RNA → Annexin | 是 | RNA 不含这些细胞系 | 未有对应 RNA | 不适用 | 不存在该 RNA 组 | 可保留外部终点；需其他表达来源另行匹配 |

同研究并不建立个体级联合观测。只有 H1792 的三个名义条件在两种实验中都出现，其中两个单药条件的药物和剂量明确一致；各自组内三重复不能交叉扩增为独立配对训练样本。

可支持：限定条件下的外部药效排序或生物学一致性检查，以及在显式建模 16 h 表达→24 h 结局假设下的探索性分析。不能单独支撑：通用高维终点评价器的可靠训练/独立校准、单细胞死亡状态监督、任意剂量外推或真实序贯验证。

本次没有将 cleaved PARP Western blot、72 h viability 或肿瘤体积变化混入 Annexin V 百分比标签。

## 原文定位

- [GEO Series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161218)：Overall design；各 GSM 的 source_name/title/treatment_protocol。
- [论文](https://www.nature.com/articles/s41467-023-41828-z)：Results 中单药 RNAseq 段落；Fig. 1F/G；Methods 的 Apoptosis。
- [补充 PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-41828-z/MediaObjects/41467_2023_41828_MOESM1_ESM.pdf)：PDF 第 5 页，Supplementary Fig. 2C/D 图注。
- [源数据 XLSX](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-41828-z/MediaObjects/41467_2023_41828_MOESM7_ESM.xlsx)：`Fig Suppl 2` C12:F14、I12:L14、O12:R14、C19:F21、I19:L21、O19:R21。
