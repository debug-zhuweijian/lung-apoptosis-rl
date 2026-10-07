# Midostaurin 补充终点核验

> 独立保留 Supplementary Fig. 3D/E 的数据块，不能当作 GSE161218 中新增的 RNA 组。

来源：补充 PDF 第 7 页图注；官方 Source Data 的 `Fig Suppl 3`。所有处理为 24 h，Trametinib=0.5 μM，Midostaurin=0.625 μM。Combo 为同时组合，无给药先后顺序。

图注写 n=3 independent experiments，但源表 A549 的 Control Exp 3（O29）和 Tram Exp 3（P29）为空。因此共 72 个预期单元格、70 个实际数值，不能填零或伪造第三重复。

| 细胞系 | 条件 | Exp 1 (%) | Exp 2 (%) | Exp 3 (%) | 源单元格 |
|---|---|---:|---:|---:|---|
| H1792 | Control | 0.151 | 0.178 | 0.12 | C27, C28, C29 |
| H1792 | Trametinib | 2.01 | 1.77 | 1.64 | D27, D28, D29 |
| H1792 | Midostaurin | 0.172 | 0.15 | 0.599 | E27, E28, E29 |
| H1792 | Simultaneous combination | 5.99 | 5.95 | 6.01 | F27, F28, F29 |
| H2009 | Control | 0.16 | 0.127 | 0.256 | I27, I28, I29 |
| H2009 | Trametinib | 0.619 | 0.946 | 1.03 | J27, J28, J29 |
| H2009 | Midostaurin | 0.172 | 0.25 | 0.187 | K27, K28, K29 |
| H2009 | Simultaneous combination | 6.12 | 6.69 | 4.12 | L27, L28, L29 |
| A549 | Control | 0.271 | 0.368 | 缺失 | O27, O28, O29 |
| A549 | Trametinib | 1.07 | 1.16 | 缺失 | P27, P28, P29 |
| A549 | Midostaurin | 0.391 | 0.946 | 1.74 | Q27, Q28, Q29 |
| A549 | Simultaneous combination | 3.6 | 3.65 | 3.5 | R27, R28, R29 |
| H1568 | Control | 6.01 | 4.35 | 4.76 | C34, C35, C36 |
| H1568 | Trametinib | 5.77 | 5.35 | 5.19 | D34, D35, D36 |
| H1568 | Midostaurin | 6.53 | 5.15 | 5.01 | E34, E35, E36 |
| H1568 | Simultaneous combination | 6.8 | 5.16 | 7.74 | F34, F35, F36 |
| H1993 | Control | 1.22 | 1.43 | 0.875 | I34, I35, I36 |
| H1993 | Trametinib | 2.8 | 3.83 | 4.08 | J34, J35, J36 |
| H1993 | Midostaurin | 1.99 | 1.18 | 0.984 | K34, K35, K36 |
| H1993 | Simultaneous combination | 5.06 | 4.72 | 4.98 | L34, L35, L36 |
| H1437 | Control | 5.88 | 5.71 | 5.61 | O34, O35, O36 |
| H1437 | Trametinib | 6.7 | 4.48 | 4.89 | P34, P35, P36 |
| H1437 | Midostaurin | 3.48 | 2.71 | 3.62 | Q34, Q35, Q36 |
| H1437 | Simultaneous combination | 3.57 | 3.31 | 2.81 | R34, R35, R36 |

## 对照复用检查与解释边界

- 按细胞系和表内 Exp 编号比较两个附图的数据块，有 17 个共同非空对照位置，其中 0 个数值相同；未发现整组直接重复使用的对照数字。
- 数字不同不能证明跨附图实验在统计上完全独立；源表 Exp 1/2/3 不是全局样本编号。因此保留 source_block，不合并成六个独立重复，也不建立跨图个体配对。
- 两块 Trametinib 对照药的数值也并不相同，不能把来源块差异抹掉后宣称测得稳定的统一药效比例。
- GSE161218 无 Midostaurin RNA、无 Tram+Midostaurin RNA，且 RNA 仅在 H1792、16 h。新增的是外部终点参考，非新增配对表达监督。
- Annexin V 阳性百分比是论文在其门控与采样协议下的操作性终点，不自动等于初始培养细胞总体最终死亡比例。

[补充 PDF 第 7 页](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-41828-z/MediaObjects/41467_2023_41828_MOESM1_ESM.pdf)；[源数据 XLSX 的 Fig Suppl 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-023-41828-z/MediaObjects/41467_2023_41828_MOESM7_ESM.xlsx)。
