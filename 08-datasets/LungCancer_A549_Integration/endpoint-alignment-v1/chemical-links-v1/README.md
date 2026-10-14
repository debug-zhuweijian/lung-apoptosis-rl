# 外部终点药物的名义化学身份与 Tahoe 分层链接

核验日期：2026-10-10。本目录仅建立药物参考结构的可追溯对应，保留实验来源身份；不把结构对应写成同一实验样本配对，也不把 SYTOX 死亡终点写成凋亡。

## 输入与范围

- Inde2020：读取原 `tahoe_name_dose_time_candidates.csv` 的全部 **37 个源标签**，保留原拼写、括号别名和尾空格。原论文 [Methods: Chemicals and reagents / Chemical library screening](https://pmc.ncbi.nlm.nih.gov/articles/PMC7409774/) 记录筛选库为 Selleck Chemicals L2000，筛选体系是 A549^N，标称 5 μM、三次独立重复。本链接对应预先提取的 24 h 视图。
- Macaya2023：读取[原论文 Methods: Reagents](https://www.nature.com/articles/s41467-023-41828-z)及保留原文 `article_extracted.md` 第 184 行。Trametinib/GSK1120212 来自 Selleckchem；Lestaurtinib/L6307 来自 LC Laboratories；Midostaurin/PKC412 来自 MedChemExpress。本目录按这三个主名获取**名义参考结构**，原文厂家记录不等于精确批次或盐型已核实。
- Tahoe：严格读取已冻结的 [identity-v2](../../identity-v2/README.md)，未修改其任何文件。输入 manifest SHA256 为 `1ee71d908de3529e933565ce445189be232c513531aed65f771ca6b62fd02e67`。

## 实际结果

| 项目 | 数量与含义 |
|---|---|
| 源条目 | 40：Inde 37 + Macaya 3 |
| 名义参考结构经唯一名称及 Synonyms 支持 | 38：Inde 35 + Macaya 3 |
| 保留名称/立体歧义 | 2 个 Inde 标签，见下一节 |
| 同时满足源名义结构已核验、Tahoe 条目合格、完整结构相同的源条目 | Inde **34**；Macaya **1**（Trametinib） |
| 全部分层候选链接行 | 46；一个源条目可能对应多个不同 Tahoe 盐型或立体形式，行数不能当成独立药物数 |
| 同一样本或实验配对已证实 | **0** |

全部 37 个原 Inde 名称候选的结果可分为：34 个名义完整结构相同、1 个仅主片段相同（Palbociclib HCl 对 Tahoe palbociclib）、2 个名称/结构歧义。这里的“完整结构相同”仍是数据库参考结构粒度，不证明源试剂批次、培养背景、剂量基准或实验体系完全相同。

Macaya 三药的名义参考为：

| 源药名 | PubChem CID | 完整 InChIKey | Tahoe 结果 |
|---|---:|---|---|
| Trametinib | 11707110 | `LIRYPHYGHXZJBZ-UHFFFAOYSA-N` | 有合格的完整结构链接；标称 DMSO solvate 那一 Tahoe 标签仍不合格 |
| Lestaurtinib | 126565 | `UIARLYUEJFELEN-LROUJFHJSA-N` | 当前 Tahoe 目录没有结构链接 |
| Midostaurin | 9829523 | `BMGQWWVMWDBQGC-IIFHNQTCSA-N` | 当前 Tahoe 目录没有结构链接 |

## 两个不能自动放行的名称问题

1. **Dexamethasone (DHAP)**：完整标签查询返回 CID 40467075；主名 Dexamethasone 返回 CID 5743。两者完整 InChIKey 分别为 `UREBDLICKHMUKA-UWJHKKLZSA-N` 和 `UREBDLICKHMUKA-CXSFZGCWSA-N`，立体结构不同；DHAP 单独又返回 CID 668。该源条目保存 literal 标签返回的结构供审查，但 `nominal_structure_resolved=false`，不能因此断言实验实际用了某个异构体。
2. **Temsirolimus (CCI-779, NSC 683864)**：完整标签返回 CID 9876533、18293306、129009966；主名 Temsirolimus 返回 CID 6918289，二者集合不相交。补充查询 CCI-779 也指向 6918289，NSC 683864 未返回条目。本版本保留歧义，未选定单一源结构。下一步需要原 Selleck L2000 库的对应化合物清单、CAS/产品编号等与原实验相连的证据；不能只为增加交集而选择 Tahoe 已有的 CID。

`name_resolution_candidates.csv` 保存 literal 和主名候选的结构、InChIKey 与属性响应定位，便于继续审核。此处暴露的是名称到结构的数据库歧义，不是实验中真实异构体差异的证明。

## 文件、字段与连接规则

| 文件 | 内容 |
|---|---|
| `endpoint_compounds.csv` / `.parquet` | 40 个源条目的主表，含 `source_id`、`source_compound_name`、`compound_id`、`canonical_full_smiles`、`identity_id`、`parent_id`、`PubChemCID`、`lookup_evidence`、`name_resolution_status` |
| `tahoe_candidate_links.csv` / `.parquet` | 46 条候选关系，保留原 Tahoe `drug_id`、分层结果和对应条件 ID |
| `name_resolution_candidates.csv` | 每个标签的 literal / primary 查询候选与实际结构，保留未采纳候选 |
| `name_resolution_details.json` | 全部查询路径、明确别名、候选 CID、冲突及未确认别名 |
| `sources.json`、`raw_pubchem/` | 当前构建所使用的 PubChem 原始查询响应、Synonyms、属性表、HTTP 状态及 SHA256 |
| `raw_response_inventory.json` | 当前使用和诊断过程保留的全部原始响应，分别标记 `used_in_final_resolution` |
| `summary.json`、`manifest.json` | 实际计数、原输入哈希、构建脚本哈希及派生产物哈希 |

主表中 `source_id` 是与原终点观测连接的键：`Inde2020:<full original name>` 或 `Macaya2023:<drug>`。例如原 `Pioglitazone ` 标签含尾空格，本版本保留该源身份；查询时仅清理首尾空白。**不能用 `compound_id=pubchem:<CID>` 替代源身份后合并不同实验。**

PubChem 先查询完整源标签；有括号别名时再查询主名，并检查候选交集和 Synonyms。只拆明确别名，保留 HCl 等后缀；不会为了命中 free form 去掉盐型。查询存在多个不一致 CID 时保留歧义。完整结构采用同一已验证 RDKit 解析函数，保留盐、电荷、同位素和已给出的立体信息。

`structure_relation` 取值：

- `same_full_form`：完整立体规范结构一致。
- `same_largest_organic_fragment_only`：仅唯一最大有机片段相同；含盐/水合物差异时不能提升为完整对应。
- `same_connectivity_different_stereochemistry`：非立体连接关系相同，但立体表述不同；可能含未指定信息，不自动认定为某两个已证实的试剂异构体。
- `unresolved`、`different_structures`、`no_tahoe_structure_link`：按当前证据保留未解决或未匹配状态。

`eligible_nominal_full_structure_link=true` 仅在源名称与参考结构通过、Tahoe 条目具备结构特征资格、完整结构相同时成立。**该字段只允许作名义化学身份连接**；`experimental_sample_pair_verified`、`dose_basis_verified`、`use_as_paired_training_target` 仍全部为 false。

Inde 链接的条件候选筛为 Tahoe 标称 5 μM / 24 h，并显式保留 `engineered_to_parental_background_transfer`。Macaya 链接只筛化学结构，返回同药的全部 Tahoe 条件，必须由终点主表再核验背景、剂量与时间；不会把所有剂量都称作终点对应。Inde 的 A549^N 带核 mKate2，Tahoe 的 A549 不是该工程株；来源内及跨来源评估需要分别组织。

## 复现与适用范围

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/link_endpoint_chemical_identities.py
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/link_endpoint_chemical_identities.py --offline
```

每条响应有哈希和可追溯 URL，离线模式拒绝缺失或修改的响应。此构建不修改 frozen identity-v2、旧终点表或训练资产，没有启动训练，也没有认定凋亡数据可行性已经整体通过。
