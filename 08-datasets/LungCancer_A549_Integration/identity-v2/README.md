# Tahoe 药物身份 v2：完整立体结构与条件引用

核验日期：2026-10-10。数据源为锁定 Tahoe 版本 `2dc57900b7981cfcf5e211527169a0b006546a95` 的原始 `pubchem_cid`，通过 [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) 取得完整立体 SMILES、InChI、InChIKey 及组成信息。原始 HTTP JSON、请求 URL、取得时间、大小和 SHA256 均已保存。此处是新数据版本，不覆盖 `prepared/v1`、`training-v1` 或旧实验。

## 实际完成范围

| 项目 | 实际结果 |
|---|---:|
| Tahoe 药物标签 | 379 |
| 连同 vehicle 的目录行 | 380 |
| 非空、唯一原始 CID | 376，全部取得并核对返回 CID |
| 非空 CID 对应药物标签 | 377，全部 SMILES 可解析且 RDKit InChIKey 与官方值一致 |
| 补回立体结构信息的药物标签 | 191 |
| 当前符合新结构特征条件的药物标签 | 373 |
| 材料形式需补核验的标签 | 4，涉及旧 selected 条件 12 条 |
| 原始 CID/SMILES 均缺失的标签 | 2，原 prepared 条件集已不包含它们 |
| 原始条件逐行映射 | 1,336，condition_id、原 drug_id、剂量、时长和条件顺序保留 |
| 有多个不相连片段的参考结构 | 98 |
| PubChem 明确记录存在未定义立体信息的药物标签 | 28 |
| RDKit `FindPotentialStereo` 发现未指定潜在立体元素的标签 | 39；检测定义较宽，与上一行不是同一个计数 |

373 是“原 CID 结构核验成功，且未触发本轮明确材料形式冲突”的计数，不是 373 种经过实验验证的试剂，也不是可直接跨来源配对的 373 种药物。未定义立体信息保留原样，不擅自选择一个异构体。

## 已修正的明确表示错误

1. **Doxorubicin hydrochloride 与 Epirubicin hydrochloride**：源非立体 SMILES 相同，新结构分别来自 CID 443939 与 65348。官方完整 InChIKey 分别为 `MWWSFMDVAYGXBV-RUELKSSGSA-N`、`MWWSFMDVAYGXBV-FGBSZODSSA-N`，新完整身份、主片段身份和含手性 Morgan 指纹都不同。
2. **crizotinib 与 (S)-Crizotinib**：源非立体结构碰撞，新结构分别通过原 CID 11626560 与 56671814 保留各自立体信息，完整身份已分开。
3. **Paclitaxel**：通过原 CID 36314 恢复 11 个已定义立体中心；官方完整 InChIKey 为 `RCINICONZNJXQF-MZXODVADSA-N`。

每条对应记录都含 `source_response_path`、`source_response_sha256` 和 `source_url`，可直接定位保留的官方响应。三项修正不是根据名称猜测的新 CID。

## 仍需明确保留的问题

| 源标签 | 原 CID 参考中的问题 | 当前处理 |
|---|---|---|
| Trametinib (DMSO_TF solvate) | CID 11707110 返回 Trametinib 单组分结构，源参考与无 solvate 标签的 Trametinib 相同 | 保留原材料标签，`source_material_form_review_required`，不生成合格结构特征 |
| Elimusertib hydrochloride | CID 118869362 返回结构没有氯化物片段 | 同上，不能擅自当成完整盐型 |
| Carbidopa (monohydrate) | CID 2563 返回结构没有水分子片段，也未定义立体信息 | 同上，不能擅自补水或选择立体异构体 |
| Quinidine (15% dihydroquinidine) | CID 441074 只描述 Quinidine，不能表达标签所称组成比例 | 同上，混合物比例不是单分子身份 |
| Sacubitril/Valsartan、Verteporfin | 源 CID 与结构缺失 | `source_cid_missing_or_invalid`；不按名称搜索结果静默补入 |

上述 4 个明确形式矛盾已在特征资格中排除，**没有删除原条件或改变旧实验**。其原 CID 结构仍完整保留供审核。全目录还保留原名与 PubChem title，因此可继续核查命名、配方及供应商信息；本轮只检查明确可观察的形式矛盾，不声称已审核所有产品或原实验试剂。

剂量仍使用来源的标称 μM；是否以完整盐、溶剂化物或活性母体的物质的量计算，统一登记为 `nominal_source_uM_salt_or_active_moiety_basis_unverified`。不能仅凭相同标称数字转换为经核验的跨来源剂量等价。

## 产物与接口

| 文件 | 含义 |
|---|---|
| `drug_identity.parquet` / `.csv` | 一行一个原 `drug_id`，含源结构、新参考结构、CID、完整 `identity_id`、`parent_id`、证据字段、资格与指纹行号 |
| `condition_identity.parquet` / `.csv` | 按原 `drug_id` 严格 many-to-one 连接全部原条件；保留 `source_condition_id` 与 `identity_version` |
| `morgan2048_chiral.npy` | `380 × 2048` uint8；按 `fingerprint_row` 取行；373 个合格药物生成含手性 Morgan 指纹，其他行全零且必须结合资格字段使用 |
| `identity_review_required.csv` | 材料形式矛盾、缺结构、多片段或未指定潜在立体信息的显式复核视图 |
| `raw_pubchem/properties_*.json` | 8 个原始属性响应，各有 `.provenance.json` |
| `sources.json` | 请求 CID 列表、URL、取得时间、原始响应大小及哈希 |
| `summary.json` | 完整计数、原结构碰撞、新身份重复、21 组主片段重复及边界 |
| `manifest.json` | 版本、原输入哈希、构建代码哈希和各派生产物哈希 |

`identity_id` 是 RDKit 完整立体规范 SMILES 的 SHA256 标识，保留盐、同位素、电荷及已指定立体信息；没有实施互变异构归一化、去盐或中和。`parent_id` 只表示**按重原子数选择的唯一最大含碳片段**，保留片段立体信息和电荷；它不是 PubChem parent 关系，也不是完整材料或药效等价结论。存在同大小不同有机片段时不指定 parent。`source_material_id` 继续使用原 `drug_id`，所以即使两个参考结构相同，源实验标签也不会丢失。

下游可接受的非 vehicle 条目要求同时满足：

```python
identity_status == "original_cid_structure_verified"
eligible_for_structural_features == True
```

Vehicle 使用 `vehicle_control`，不能因为零向量而被误当作未知药物。复用主片段或全结构相同的结果，只能建立化学身份候选；跨数据实验配对还要核验背景、培养体系、剂量基准、暴露时间、结局定义及重复。`experimental_material_verified` 与 `cross_source_pair_verified` 当前全部为 false。

## 构建、验证与架构对应

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/build_drug_identity_v2.py
& 'F:/anaconda/envs/study/python.exe' -X utf8 scripts/build_drug_identity_v2.py --offline
& 'F:/anaconda/envs/study/python.exe' -X utf8 -m pytest tests/test_drug_identity_v2.py -q
```

离线构建校验保存的请求 CID 列表和响应哈希，不重新访问 PubChem。构建前后核对三个原输入文件哈希；已完成重建及回归检查。实现入口为 `scripts/build_drug_identity_v2.py`，可复用函数在 `src/lung/data/drug_identity.py`，无依赖新增。

本产物对应[研究架构第 4.2 节的“候选 ID、药物身份与结构”](G:/研究生科研文件/毕业论文/研究架构.md)及第 4.3 节的条件编码输入；只修正药物表示和条件引用，不新增实测表达或凋亡标签，也不证明功能终点或初始状态增益。当前模块默认使用哪个身份版本，以正式配置及训练资产清单为准；此处生成产物本身不等于旧训练已切换。
