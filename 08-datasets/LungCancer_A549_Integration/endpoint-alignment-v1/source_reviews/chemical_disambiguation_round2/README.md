# 第二轮化学身份消歧：6 个指定问题的厂家证据

日期：2026-10-10。范围只覆盖上一轮留下的 6 个问题；没有重跑已通过条目、修改源码或覆盖冻结的 identity-v2 / chemical-links-v1。

**结果：5 个问题已有足够证据提出新的名义身份映射；Quinidine 混合物继续保留一个具体缺口。**“名义身份”指原研究标签所指产品的化学参考定义，不代表原实验试剂批次已实测鉴定，也不代表跨来源样本、剂量或生物背景已配对。

## 1. 可提出的新映射

| 原来源标签 | 新建议 CID | 核心证据 | 建议 |
|---|---:|---|---|
| Tahoe：Trametinib (DMSO_TF solvate) | **50992434** | MCE HY-10999A；CAS **1187431-43-1**；官方 SMILES 明确包含 DMSO；由厂家 SMILES 算得的完整 InChIKey 与 CID 一致 | 使用完整 DMSO 溶剂化物参考，保留原材料标签；不再复用 free Trametinib 的完整身份 |
| Tahoe：Elimusertib hydrochloride | **129893299** | MCE HY-101566A；官方分子式 **C20H22ClN7O**；厂家 SMILES 包含 HCl 且立体信息与 CID 完全一致；厂家该页未给 CAS，因此没有填 free-base CAS | 使用明确 HCl 形式 |
| Tahoe：Carbidopa (monohydrate) | **38101** | MCE HY-B0311A；CAS **38821-49-7**；厂家给出 **(S)** 形式与一个水分子，完整 InChIKey 与 CID 一致 | 使用 (S)-Carbidopa monohydrate，替代未指定立体、无水的旧参考 |
| Inde：Dexamethasone (DHAP) | **5743** | Selleck 原始厂家目录 PDF 第 **70、72 页**把完整标签明确登记为 **S1322**；当前 S1322 产品页给 CAS **50-02-2**，该 CAS 唯一返回 CID 5743 | 按厂家产品/CAS 消除 DHAP 短别名误导，不使用仅按复合字符串返回的 CID 40467075 |
| Inde：Temsirolimus (CCI-779, NSC 683864) | **6918289** | Selleck **S1044** 官方产品页同时给出 **CCI-779、NSC 683864、CAS 162635-04-3**；CAS 唯一返回 CID 6918289 | 按厂家同一产品的明确别名与 CAS 统一名义参考 |

上述结论分别由原厂家资料独立支持，并非因为这些 CID 恰好能够与 Tahoe 对齐而选择它们。具体结构、`identity_id`、`parent_id`、源键及复核字段见 [proposed_identity_corrections.csv](proposed_identity_corrections.csv)；另有同内容 Parquet 供新版本构建使用。

三个 MCE 产品的完整 InChIKey：

- Trametinib DMSO solvate：`OQUFJVRYDFIQBW-UHFFFAOYSA-N`。
- Elimusertib HCl：`KWQNBYGUBHMRPY-BTQNPOSSSA-N`。
- Carbidopa monohydrate：`QTAOMKOIBXZKND-PPHPATTJSA-N`。

Selleck 当前页面的 SMILES 没有指定立体信息；本轮只把它用于连接图核验，**没有从非立体 SMILES 猜出立体配置**。完整立体参考来自厂家 CAS 唯一对应的 PubChem 记录。两条完整 InChIKey 分别为 `UREBDLICKHMUKA-CXSFZGCWSA-N`、`CBPNZQVSJQDFBE-FUXHJELOSA-N`。

## 2. 为什么原厂家证据可以用于本次消歧

- Tahoe 原论文英文 MinerU 全文 `full.md` 第 198–200 行、Methods 的 Compound sourcing and treatment 明确药物来自 **MedChemExpress**；源标签也明确写出 solvate、hydrochloride、monohydrate。因此本轮定向核对该厂家的同名形式，而非任意找一个其他厂家的同主名产品。原研究未在现有元数据中给每个试剂的订单号/批次；MCE SKU 的对应属于标签与供应商层的名义对应。
- Inde 原论文的 Chemicals and reagents、Chemical library screening 及 Key resources table 明确 261 药筛库来自 **Selleck L2000**。本轮用该厂家自身的产品记录解释完整源标签。没有把当前任何 L2000 库的全量清单当成原实验清单；L2000 还可能表示定制库，原实验批次清单仍未取得。
- Dexamethasone 的关键增量证据是确实读到了厂家目录中 **S1322 与 Dexamethasone (DHAP)** 的对应。仅有当前 Dexamethasone 主名网页本身不足以解释旧标签；本轮补上了该连接。便于检查的 [目录第 72 页单页 PDF](selleck_catalog_dexamethasone_page72.pdf)由原始 74 页 PDF 的第 72 页无改写抽出；完整原件也已保存。
- Temsirolimus 的两个括号别名在 S1044 厂家页面明确并列，故不把 PubChem 对整个复合字符串返回的三个 CID 直接视为三个同等实验候选。

Trametinib 原标签里的 `_TF` 原样保留；本轮不推断该研究内部后缀的命名原因。建议结构依据是同一原厂家明确的 DMSO solvate 产品与原标签所述溶剂化物形式。

## 3. Quinidine 的剩余缺口已经缩小到什么

MCE **HY-B1751** 官方产品和已下载 datasheet 的名称确实为 **Quinidine (15% dihydroquinidine)**。因此不是我们误把纯 Quinidine 读成混合物。

但官方 datasheet 及 SDS 第 2 页的组成部分仍只列 Quinidine 的分子式 **C20H24N2O2**、分子量 **324.42** 和 CAS **56-54-2**，没有提供两组分的定量构成、15% 的质量/摩尔/色谱计量基准，也没有原实验批次 COA。厂家显示的单分子 SMILES 不能代表整个混合物。

可以登记两个**名义组分参考**：Quinidine CID 441074、Dihydroquinidine CID 6432105；后者已获取结构和完整 InChIKey。但不能据此写成已证实的 85:15 摩尔混合物，也不能继续只用纯 Quinidine 的单结构指纹。`quinidine_components_review.json` 把名称中的 15% 保留为原始文字，摩尔分数、组成基准和完整混合物 SMILES 均为空。

建议：该条仍保持单结构模型不可用；后续若得到原批次组成文件，建立显式组分/配方特征的新版本。没有把该缺口扩写成“所有 Quinidine 资料不可用”。

## 4. 原始来源、失败回退与可追溯文件

| 证据 | 实际读取及本地状态 |
|---|---|
| [MCE Trametinib DMSO solvate 产品页](https://www.medchemexpress.com/us/en/Trametinib-DMSO-solvate.html)及官方 datasheet | 网页化学字段通过 web 工具亲读；[本地 PDF](raw_sources/mce_trametinib_dmso_datasheet.pdf)已下载、读取，含 SKU/CAS/分子式/分子量 |
| [MCE Elimusertib HCl 产品页](https://www.medchemexpress.com/gb/en/BAY-1895344_hydrochloride.html)及官方 datasheet | 同上；[本地 PDF](raw_sources/mce_elimusertib_hcl_datasheet.pdf)明确 HCl 形式，未给 CAS |
| [MCE Carbidopa monohydrate 产品页](https://www.medchemexpress.com/carbidopa-monohydrate.html)及官方 datasheet | 同上；[本地 PDF](raw_sources/mce_carbidopa_monohydrate_datasheet.pdf)明确 (S) 名义形式与 monohydrate CAS |
| [MCE Quinidine 产品页](https://www.medchemexpress.com/Quinidine.html) | [datasheet](raw_sources/mce_quinidine_mixture_datasheet.pdf)和 [SDS](raw_sources/mce_quinidine_mixture_sds.pdf)均已下载；组合名称确定，定量组成仍不充分 |
| [Selleck Dexamethasone S1322 产品页](https://www.selleckchem.com/products/dexamethasone-glucocorticoid-receptor-agonist.html) | 完整 HTML 已下载，读取化学信息表的 CAS；[高分辨率原始目录](raw_sources/selleck_inhibitor_catalog_high_resolution.pdf)45,057,694 字节，第 70、72 页有完整旧名称与 SKU |
| [Selleck Temsirolimus S1044 产品页](https://www.selleckchem.com/products/Temsirolimus.html) | 完整 HTML 已下载，读取别名与 CAS；与 PubChem CAS 检索一致 |
| PubChem | CAS/名称/完整 InChIKey 检索、属性和 Synonyms 的原始 JSON 全部保留于 `raw_pubchem/` |

失败没有被当作成功原文：MCE 四个 HTML 的直接 requests 下载返回 **403**，这些本地 HTML 仅是失败响应；厂家事实来自可读的官方 web 页面与成功下载的官方 PDF。Selleck 两个 SDF 下载也返回 **403**，未用它们建立结构结论。低分辨率 Selleck 目录没有检到 S1322，随后下载高分辨率原件并实际定位成功；两者分别保留，不混用页码。

主要机器可读产物：

- `proposed_identity_corrections.csv` / `.parquet`：6 行建议，5 行名义身份可消歧，1 行混合物待定。
- `manufacturer_product_fields.json`：厂家结构字段、HTML 提取字段和读取方式。
- `pubchem_disambiguation_lookups.json`、`proposed_reference_properties.json`：官方查询与具体候选属性。
- `study_source_evidence.json`：原研究指定供应商的定位和来源哈希。
- `sources.json`、`manifest.json`：所有下载的来源/状态/哈希、派生产物和冻结输入哈希。

这些是供新版本采用的审查建议，`applied_to_frozen_versions=false`。两条 Inde 建议与现有 Tahoe 正确参考的完整结构相同；采用后原 37 个候选可从 34 个提升为 **36 个名义完整结构候选**，Palbociclib HCl 仍只能作母体层对应。三个 Tahoe 建议可解除相应 9 个已选条件的材料形式歧义；Quinidine 的 3 个条件继续保留。上述数字是新版本采用建议后的预期，不是冻结版本已经改变。

本轮没有补造凋亡结局、相同细胞前后测量或跨来源实验配对。名义身份的修正支持可信的条件输入，仍需要终点观测、初始状态以及独立验证共同支撑课题数据可行性。
