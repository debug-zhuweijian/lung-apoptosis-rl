# Dynamic caging companion：限定来源核验

结论：**排除为本轮 A549 数据缺口的直接补充来源，保留作方法参考。** Methods 的实验对象为人 Treg、Hs 675.T 人结肠成纤维细胞和鼠 BMDM；全文 41 页未检出 A549、Annexin 或 caspase。已核范围没有 A549 实验或可直接接入的 Annexin Y。

论文：Orcutt-Jahns et al., *Linking live-cell behavior to transcriptional responses across perturbations using dynamic caging*，DOI [10.64898/2026.05.05.723043](https://doi.org/10.64898/2026.05.05.723043)。[官网条目](https://www.cellanome.com/publications/link-live-cell-behavior-to-transcriptional-responses-across-perturbations-using-dynamic-caging)标 May 5；实际作者 PDF 页眉及 bioRxiv 条目为 **posted May 8, 2026**。身份为预印本，不能写成同行评议正式论文。

官网实际链接的 PDF 已下载：`OrcuttJahns2026_DynamicCaging_EN.pdf`，4,786,835 字节，SHA256 `65d9a9dc9e6150bd0cd76f4c366c91c811ceddf439a8d4c09dfce009cdf6d1ab`。同名 `.md` 为逐页原文提取；`primary_page_19.png` 等是该 PDF 的直接渲染。

| 核查点 | 实际证据及边界 | PDF 页／印刷行；本地 MD 行 |
| --- | --- | --- |
| 数据公开 | 只承诺以后公开至如 Zenodo，未给 accession；未取得表达矩阵、影像对象表或映射表 | p19／583–585；713 |
| RNA 时点 | 活细胞观察后裂解捕获；不是同一细胞的连续 RNA 测量 | p20／590–620；726 |
| 对应单位 | counts 按 CCE 汇总；Treg/成纤维细胞分析筛选单细胞 CCE | p21–22／668–695；812、844 |
| 对象与处理 | 人 Treg；人结肠成纤维细胞培养 72 h；鼠 BMDM 的 LPS 100 ng/mL、12 h | p20–21／599–664；736、759、789、802 |
| 功能读出 | 形态、免疫荧光/分泌标记；BMDM viability 为 gRNA×lane 相对细胞丰度，不是 Annexin 比例 | p22–23／680–687、747–757；827、905 |

本轮没有发现可沿已读来源取得的具体公开矩阵 accession；这是限定来源检查结果，不是全网不存在数据的结论。补充入口返回摘要和日期，未暴露可下载附件链接；PDF 已含 Extended Data。作者引用的另一篇技术论文未被替换成本文数据，官网其他 A549 宣传或论文也未混入。

本轮已结束定向检索，没有构建非肺癌条件表、虚构初态 RNA、影像死亡/Annexin 标签或模型结果。`evidence_index.json` 保存完整定位和状态；`manifest.json` 与 `verify_audit.py` 用于复核。
