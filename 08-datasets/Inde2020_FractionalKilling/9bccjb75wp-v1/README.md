# Inde 2020：A549 单药死亡终点源表

日期：2026-10-09。来源：[Mendeley 10.17632/9bccjb75wp.1](https://data.mendeley.com/datasets/9bccjb75wp/1)、[Cell Reports 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7409774/)。

已取得 `261 library + DMSO.xlsx`：1,455,082 字节，SHA256 `2f51796806586e48501753b6aba6d152a42a49e78a0028d3c9b0ed1575a36c71`。保留原文件名；该表为单药臂，不是 pimasertib 联合臂。另保存公开原文 XML 与来源核验清单。

A549^N、261 种化合物、标称5 μM、3次独立重复。源表时间0–144 h，每4 h；论文方法说明120 h。本轮只抽取两者均覆盖的24 h，783值均有效且在[0,1]。

- `single_drug_24h_review.csv`：原sheet、行列位置、数值、重复和作者自发荧光排除标记。
- 原论文排除 Sunitinib Malate、Mitoxantrone HCl、PHA-665752，留下258种；保存原行不覆盖源表。
- `tahoe_name_dose_time_candidates.csv`：37个名称/别名、5 μM、24 h候选接点。仅名称候选，化学身份仍待审，不作为真实跨源配对监督。

终点为 SYTOX lethal fraction，**不是 Annexin 阳性率或纯凋亡率**。所有派生行 `use_for_training=false`；没有接入正式训练。完整时序仍含需独立审查的无效值，24 h的通过不能扩展为所有时点质控通过。

提取脚本：[collect_a549_endpoint_review_sources_20261009.py](../../../scripts/collect_a549_endpoint_review_sources_20261009.py)。数据保持本地；[研究说明](G:/研究生科研文件/paperstudy/02-notes/LungCancer_Data_Gap_Resolution_Study_2026-10-09.md)。
