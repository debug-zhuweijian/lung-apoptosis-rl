# GSE279184 处理条件与名义材料输入 v1

本包把已经核对的六个 RNA 样本连接到两项论文明确列出的产品：诱导历史使用 **MedKoo 201931 alisertib**，后续 RNA 处理使用 **MedChemExpress HY-15597 salinomycin**。结构来自这两个货号的官方页面及资料，不按药名替换盐形式。2026-10-10 核查范围为当前官方名义产品；未鉴定历史实验批次。

## 已核对的材料

| 名义材料 | 官方 CAS 与分子式 | 完整结构结论 | 指纹行 |
|---|---|---|---:|
| MedKoo 201931 / alisertib | 1028486-01-2；C27H20ClFN4O4 | 官网明确 free acid；单一中性片段，未表示独立水合分子；Cl/F 为共价原子。官网单列 sodium / sodium hydrate 的其他 CAS，未替换到这里。 | 0 |
| MCE HY-15597 / salinomycin | 53003-10-4；C42H70O11 | 官网完整 SMILES 表示游离羧酸；单一中性片段，无 Na 或独立水分子。保留全部厂商立体标记。 | 1 |

RDKit 2026.03.6 从上述原始 SMILES 计算得到：alisertib InChIKey `ZLHFILGSQDJULK-UHFFFAOYSA-N`，与 MedKoo 官方字段一致；salinomycin 的计算 InChIKey 为 `KQXDHUJYNAXLNZ-XQSDOZFQSA-N`（这是计算值，未伪称 MCE 给出此字段）。本版本检测到 alisertib 0 个、salinomycin 18 个潜在立体单元；未指定单元均为 0。此计数是 RDKit 对表示形式的检查，不是批次纯度检测。

没有脱盐、选母体、加反离子、重离子化、补水合物或改立体。没有从“potassium ionophore”的药理描述推断它是钾盐。`hydrate_in_vendor_structure=none` 只描述名义图式，不证明实物批次完全无水；批次溶剂/含水量仍未核实。

## 六个 RNA 样本如何连接

| CSV 列 | GEO 样本 | 共同初始诱导历史 | RNA 采集条件 |
|---|---|---|---|
| ut_r1 | GSM8563730 | alisertib 0.5 µM，作者方案 7 d | 诱导衰老的 UT 平行参考；独立时长未知 |
| ut_r2 | GSM8563731 | 同上 | 同上 |
| ut_r3 | GSM8563732 | 同上 | 同上 |
| tr_r1 | GSM8563733 | 同上 | salinomycin 2.5 µM，48 h |
| tr_r2 | GSM8563734 | 同上 | salinomycin 2.5 µM，48 h |
| tr_r3 | GSM8563735 | 同上 | salinomycin 2.5 µM，48 h |

原文 [PMC11670111 §s10、§s14、Fig. 2C 与 Fig. 3C](https://pmc.ncbi.nlm.nih.gov/articles/PMC11670111/) 支持产品货号、诱导方案及 RNA 处理条件。§s14 说明追加处理前 PBS 洗涤、胰酶消化和重新铺板。本包将这一步作为处理组的研究方案保留；UT 是否经历完全相同的操作没有独立记录，不补造执行轨迹或药物清除率。

UT 的 `duration_h=null`，不填 0 或 48；`dose_uM=0` 只继承源表“不加 salinomycin”的编码，不表示测量了某种 vehicle 浓度，vehicle 身份保留 null。三份 UT 是组级平行参考，不是处理组各自的 t0；相同 r1/r2/r3 后缀没有被用来创建个体配对。源标题的 `Saliomycin` 拼写保留。作者的中间 `R1` 字段不是这六列的生物重复编号；映射继承已冻结表达包核对的最终 biological replicate 后缀。

Fig. 3B 的死亡通路 Western blot 使用 **2 µM、48 h**；本 RNA 使用 **2.5 µM、48 h**。本包没有 WB、Annexin 或凋亡比例标签，没有创建 RNA—功能终点配对。材料结构一致也不能消除剂量、样本与终点的差别。

## 文件与读取

- `chemical_identities.json` / `.csv`：两项原始 SMILES、完整结构身份、材料形式、来源、hash 和指纹行。
- `morgan2048_chiral.npy`：`uint8[2,2048]`，Morgan radius=2，includeChirality=true；没有 UT 的虚构零向量。
- `fingerprint_metadata.json`：生成规则和行次。
- `condition_inputs.json`：六样本的完整嵌套记录；`sample_conditions.csv` 是扁平读取表。
- `source_evidence.json`：论文位置、官方产品 URL、名义材料依据及限制。
- `raw/MedKoo_201931_Alisertib.html`、两份官方产品 PDF：成功获取的原件。
- `MCE_official_web_chemical_excerpt.txt`：从成功读取的官方产品页提取的 Chemical Information，保留 SMILES。它是提取文本，不是原始 HTML。两次直接 HTML 请求的 403 响应也保留在 raw 并写入 acquisition，均不作为结构证据。
- `acquisition.json`：请求结果、URL、时间、字节数及 SHA256。
- `manifest.json` / `validation.json`：输入输出 hash、实际复算验证和准备状态。

成功的官方原件与链接：

- [MedKoo 产品 HTML](raw/MedKoo_201931_Alisertib.html)，[官方产品页](https://www.medkoo.com/products/5202)；[本地 PDF](raw/MedKoo_Alisertib_product_data.pdf)，[官方 PDF](https://www.medkoo.com/uploads/product/Alisertib/product_insert/ProductData-Alisertib-draft.pdf)。
- [MCE 本地 PDF](raw/MCE_HY15597_Salinomycin_datasheet.pdf)，[官方 PDF](https://file.medchemexpress.com/batch_PDF/HY-15597/Salinomycin-DataSheet-MedChemExpress.pdf)；[官方产品页](https://www.medchemexpress.com/Salinomycin.html)。

```python
import json
import numpy as np
from pathlib import Path
p = Path("08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round7/GSE279184_condition_inputs_v1")
conditions = json.loads((p / "condition_inputs.json").read_text(encoding="utf-8"))
fp = np.load(p / "morgan2048_chiral.npy", allow_pickle=False)
# 每个样本的 initial_induction_history.fingerprint_row 是历史诱导药。
# expression_collection_condition.fingerprint_row 是后续处理药；UT 为 None。
```

## 验证与边界

构建脚本为项目 `scripts/build_gse279184_condition_inputs.py`；包内保存当时脚本及 `drug_identity.py` 小快照。构建核对上游表达包 manifest、15 个输入和 9 个输出哈希，逐项核对六个 GSM/列次与条件；验证官方公式及 MedKoo InChIKey；从保存后的 canonical full SMILES 重新计算两个指纹并逐位回读比较。检查 UT 的时长和指纹引用仍为空，JSON 往返一致。`--verify-only` 可复核本包绑定文件。

本包可说 **表达条件的名义化学身份输入已补齐**，但 UT 独立时长仍未知，也未补出真实初始单细胞 RNA 或 RNA—Annexin 标签。读取的上游量纲仍是 `author_deposited_processed_count_table`；`raw_unmodified_count_matrix_verified=false`，`transform_applied=none`。没有把计数表接入 joint 单细胞 raw_counts 归一化；`single_cell_consumer_ready=false`、`Annexin_pairing_ready=false`、`formal_training_ready=false`。

本包只完成条件身份与追溯，不改变已冻结共同基因轴包、induced 原始资料、联合数据源或训练配置。
