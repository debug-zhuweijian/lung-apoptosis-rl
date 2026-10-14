# ChEMBL 37：A549 Annexin 候选记录

核查日期：2026-10-09。状态：**公开数据库记录已实际取得；未形成可训练的人工核验终点集**。

来源为 [EBI ChEMBL REST](https://www.ebi.ac.uk/chembl/api/data/assay.json?assay_cell_type=A549&description__icontains=Annexin&limit=1000)，版本 ChEMBL_37，发行日期 2026-05-01。原始分页、查询 URL、获取时间及 SHA256 位于 `audit_2026-10-09/`。

- `assays_all.json`：2,513 个 A549/Annexin 描述 assay。
- `activities_24h_*.json`：描述提及 24 h 的 1,023 个 assay 对应 1,364 条 activity。
- `documents_24h_*.json`：131 个 document 元数据；不表示全文全部已读。
- `activities_24h_candidates.csv`：241 个 molecule 的逐条候选及风险标记。
- `tahoe_structure_candidates.csv`：完整结构匹配 5 种名称、25 条 activity。
- `tahoe_structure_dose_time_candidates.csv`：当前完整结构/自动剂量/字面24h条件规则的交集为空。
- `summary.json`：计数、数据状态、文件哈希。

所有记录 `manual_primary_review=pending`、`use_for_training=false`。early/late/total/viable/necrotic 可能是同次实验的多个读数，不得当成独立条件或重复。自动无 flag 不等于通过人工审查；目前为 554 条、143 分子。

脚本为 [audit_chembl_a549_annexin_20261009.py](../../../scripts/audit_chembl_a549_annexin_20261009.py)。不改正式 prepared/training manifest。数据 JSON/CSV 保持本地，研究解释见 [补救研究报告](G:/研究生科研文件/paperstudy/02-notes/LungCancer_Data_Gap_Resolution_Study_2026-10-09.md)。
