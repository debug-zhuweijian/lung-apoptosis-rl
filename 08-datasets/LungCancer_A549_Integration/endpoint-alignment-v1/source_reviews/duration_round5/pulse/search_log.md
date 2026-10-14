# 第五轮 pulse/withdrawal 定向检索记录

日期：2026-10-10。范围：A549，不同暴露时长、共同观察终点、真实凋亡相关指标；只 GET 官方/作者机构来源，不联系作者、不训练。

## 查询组

网络检索用于发现线索；方法与结论回到全文、官方图、source data。

1. `A549 cisplatin pulse continuous treatment Annexin washout 24 48 apoptosis Hastings 2020`
2. `A549 "exposure" "washout" "Annexin" hours`
3. `A549 "pulse" "continuous" "caspase" treatment duration`
4. `A549 "washout" "Annexin V" "hours"`
5. `A549 "pulse exposure" apoptosis duration caspase`
6. `A549 "drug removal" "Annexin" "exposure"`
7. `"elife-53367-fig1-data1"`
8. `"Analysis of pulsed cisplatin" "washed"`
9. `"A549" "pulse" "caspase" "wash" duration exposure`
10. `A549 "pulse" "washout" "Annexin"`
11. `A549 "pulse" "continuous" "paclitaxel" apoptosis`
12. `A549 "exposure time" "Annexin" wash`
13. `A549 "drug removal" "caspase" paclitaxel`
14. `"Paclitaxel-induced Apoptosis May Occur without"`
15. `"A549" "sevoflurane" "Annexin" "Konishi"`
16. `A549 "pulse" "apoptosis" "48 hours total"`

互补原始入口：PubMed/Europe PMC fullTextXML、eLife 官方 API/CDN/IIIF、Springer 出版记录、Nagoya 官方机构库作者稿。没有把多个入口中的同一论文计为多份独立证据。

## 收敛结果

| 来源 | 实际审查程度 | 结论 |
|---|---|---|
| Hastings 2020 / 10.7554/eLife.53367 / PMC7282820 | 原文、Fig1supp1/2、source XLSX全部工作表、supp1/2/3、透明报告、API附件目录与同行评审相关段落 | 核心有限通过：A549 cisplatin2h/持续72h，共同72h relative cleaved-Caspase-3均值图。XLSX仅另一pulse面板 |
| Hirai 2020 / 10.1007/s00540-019-02701-w | 作者稿Methods/Results/Fig3与图注、出版商元数据/版权更正 | 补充有限通过：错开给药起点、共同撤药后24h，2/4/8h sevoflurane；Annexin相对荧光，麻醉剂背景 |
| PMC8131322 Early nucleolar responses... | 原文Methods Pulse-chase、Fig2/3 | 排除：放射性RNA标记pulse，不是药物撤药duration挑战 |
| PMID15015572 Paclitaxel-induced apoptosis may occur without a prior G2/M-phase arrest | PubMed摘要 | 排除本轮细胞范围：BCap37/KB，不是A549；未读全文，不作机制结论 |
| PMC6688629 Patterns of early p21 dynamics... | 检索线索，未本轮全文核实 | 主要增殖/衰老线索，未纳入已通过凋亡duration源 |
| PMID19706883 Acute dasatinib exposure... | 检索线索 | CML模型，不作A549证据 |

PMC 网页对 PMC8131322 返回浏览器检查页后改为 Europe PMC 官方 XML 成功，没有把网页失败当成原文不存在。Hastings 原件由 eLife 官方 CDN 成功下载，无需重复 PMC 附件路径。

本轮以两个实际有用来源与一个全文排除完成定向范围，不声称系统性穷尽所有时长论文。后续最有价值的新证据是同一A549状态体系下多个候选暴露窗口的Annexin/Caspase数表及初始分子观测，而不是继续收集没有时间键的活力曲线。
