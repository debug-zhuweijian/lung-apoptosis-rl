# 第三轮原文化学身份核验

核验日期：2026-10-10。范围仅限新纳入 `curated-20261010-r4` 的 1a、8h 和 6k。这里是新的审查包，没有改写任何冻结化学表、条件表、源码或训练结果。

**结果：1a 的名义结构核验通过；8h 的原文连接结构已核验，但立体构型未指定，严格特征入口保持 hold；6k 的 ChEMBL 漏 C-11 羰基已定位并生成候选修正，完整结构仍 hold。** 全部 `formal_training_ready=false`。化学图一致不代表历史试剂 lot、同实验样本、剂量计量基准或凋亡终点配对已证实。

| 原文化合物 | 实际核对的第一手证据 | 化学结果 | 默认是否可进入结构特征 |
|---|---|---|---|
| Li2012 1a / naphthol AS-E | 原稿 Chart 1 左侧 1a、Table 1 的 4-Cl、§2.1 Chemistry | N-(4-chlorophenyl)-3-hydroxy-2-naphthamide；21 个重原子逐连接核对；无立体中心；与 CHEMBL65653 一致 | 是，仅名义化学身份 |
| 8h | 原文 Scheme 2 中 8d–8j 共用骨架和 8h 的 R；正文 C6-morpholinyl 说明 | 31 个重原子连接一致；C6 通过 morpholine N 连接；产品中无 Cl⁻，无正电荷；C6 有 1 个未指定立体中心 | 否；另提供 `connectivity_only_eligible=true` 供显式协议判断 |
| 6k | 原文 Scheme 1、Table 1、§Chemistry、Compound 6k 表征段 | 原 ChEMBL C11 是 CH₂，原图明确为 C=O；候选加 O、减 2H 后式和 HRMS 一致；完整最终图和立体核验尚未闭合 | 否，不能使用旧 ChEMBL，也不能自动采用新候选 |

## 1a：可以解除名义结构待审

原 Chart 1 明确显示相邻的萘环 2-酰胺和 3-OH、未甲基化的 NH、酰胺 N 所连苯环的对位 Cl。右侧化合物 11 的 N-Me 与左侧 1a 的 N-H 清楚可区分。另以不同遍历顺序从原图转写 `Clc1ccc(NC(=O)c2cc3ccccc3cc2O)cc1`，RDKit 标准化后与保留的 ChEMBL 结构完全一致。

- 原文：[PMC3587690 Chart 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587690/#F6)、[Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587690/#T1)。表 ID 以原稿 XML 为准；Chart 1 为已核对的关键定位。
- 原始结构图：`raw/PMC3587690_chart1.jpg`；合成图：`raw/PMC3587690_scheme1.jpg`。
- 原稿 XML 输入：`../chembl_round2/raw/PMC3587690.xml`；它的 SHA256 记入 manifest。
- 完整 InChIKey：`OHAXNCGNVGGWSO-UHFFFAOYSA-N`；式 `C17H12ClNO2`。

这一步支持 1a 的名义结构编码。48 h 的终点、重复、培养条件以及数据划分仍由终点审查与总协议决定。

## 8h：连接已核验，立体与补充实验信息仍待核验

原 Scheme 2 的 8h 是血根碱衍生物：环 N 保留甲基，C6 由双键/正电荷形式转成连 morpholine N 的 sp³ 碳，两组 O–CH₂–O 保留。Cl⁻ 画在前体 1 一侧，不能移到产物 8h。采用不同遍历顺序转写的 SMILES 与 CHEMBL4751602 完全同构；这是与原图的比较，不是只计算 ChEMBL 自身的 InChIKey。

- 原文：[PMC7489057 Scheme 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489057/#sch2)。结构原图 `raw/PMC7489057_scheme2.jpg`。
- 完整但立体未指定的 InChIKey：`PCJWWMFUNQDHJU-UHFFFAOYSA-N`；从图计算式 `C24H22N2O5`，没有冒称此式已由补充表征读到。
- 在保留的 ChEMBL SMILES 的 0-based atom 21，存在一个未指定的 C6 四面体中心。原 Scheme 未标楔键/虚键，也未在已读正文中指定 R/S。
- **“未指定”不等于已证明外消旋，更不等于已知 50:50 比例。** 不生成单一 R 或 S 的 SMILES。

为了保持默认的完整身份门槛，`eligible_for_structural_features=false`。另设 `primary_reported_representation_verified=true`、`primary_connectivity_verified=true` 和 `connectivity_only_eligible=true`，说明这个不含 `@` 的图是原文同等精度的名义结构。如果主协议显式允许保留不确定立体结构的特征，必须带着该不确定性使用，不能改成“完整立体已证实”。培养、试剂、重复等 SI 信息的 hold 不因本次化学连接核验而解除。

## 6k：C-11 修正有原图依据，但不能自动放行

原 Scheme 1 中 AA 的 C1–C30 编号可定位 C11：它处于 C9 和 C12 之间，2→3 氧化后画有羰基，后续 4、5、6 仍有该羰基。Table 1 将 6k 指定为 2-F 和 OH 系列；C28 酰胺的 NH–CH₂–triazole-4-yl 连接和 triazole N1 连 2-fluorophenyl 可核对。**原 CHEMBL4527184 在此处是 CH₂，不能代表这份原文的 6k。**

原始 SMILES 中 0-based atom 29 对应 C11；其邻碳为 atom 28（C12）和 atom 30（C9）。`6k_atom_mapping.csv` 给出全部 C1–C30 的索引；两个 `.mol` 文件保留这一原始索引顺序。候选仅在 C11 增加双键 O（新 atom 48）；没有为了凑分子式改变其他部位。

| 检查项 | 原 ChEMBL | C11=O 候选 | 原文 Compound 6k |
|---|---|---|---|
| 分子式 | C39H55FN4O4 | C39H53FN4O5 | C39H53FN4O5 |
| C11 | CH₂ | C=O | Scheme 1 画 C=O；13C 201.81 ppm |
| [M+H]⁺ | 不匹配 | 计算值与 677.4073 相符 | 计算 677.4073、实测 677.4078 |

仍有两个具体缺口：

1. 当前完整候选沿用 ChEMBL 的所有手性标签。这里没有把继承的手性标签伪装成逐中心重新从原文确定的绝对构型，`primary_absolute_stereo_verified=false` 明记在逐原子表。
2. 原 Scheme 1 中 AA/2/3/4/5 的 C20 画有甲基，但最后的 generic 6 顶部没有可见的该甲基；6k 的表征段仍报告 CH₃-29、CH₃-30，分子式也要求 C39。因此主文的末端通式与前体及表征存在可见缺项，不能静默补全后声称最终图完全自洽。

需要有效 SI 或清晰且无歧义的最终 6k 完整结构及构型说明，才可解除该 hold。此审查包交付的 SMILES 是**待复核候选**，不是修改旧 registry 的授权或最终身份。

- 原文：[PMC6484948 Scheme 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sch1)、[Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#tab1)、[Compound 6k](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec37)。
- 原始图：`raw/PMC6484948_scheme1.jpg`。
- [RSC 出版页](https://pubs.rsc.org/en/content/articlelanding/2019/md/c8md00620b) 所列 SI 链接为 `https://www.rsc.org/suppdata/c8/md/c8md00620b/c8md00620b1.pdf`；本次返回 404 HTML。PMC SI HTTP 返回 403，普通浏览器进入验证码页面；没有操作验证码。没有取得有效 SI PDF。

## 文件与可复现性

- `review.json`：三份详细审查、来源定位、候选与旧结构、门槛及未解决问题。
- `chemical_candidates.csv`：供主代理读取的三行摘要；`source_id` 与 curated r4 原始 compound_id 对应，不能换成 ChEMBL ID 后丢失来源身份。
- `6k_atom_mapping.csv`、`6k_secondary_original_indexed.mol`、`6k_C11_oxo_candidate_indexed.mol`：局部差异与定位，不宣称手性已验证。
- `raw/`：从原文实际观察到的图 URL 下载的原图，以及失败回应；每个成功图的来源、状态与 SHA256 在对应 `.provenance.json` 中。
- `primary_read_notes.json`：第一手阅读范围和观察记录；`acquisition_failures.json` 明列失败，**扩展名为 PDF 的失败回应不是可用 PDF**。
- `validation.json`、`manifest.json`：计算检查、输入和输出内容哈希。

在项目根目录执行：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chemical_round3/build_review.py'
```

脚本不联网、不训练、不回写上游。它复现原图人工读取后的化学计算和门槛，不替代原图人工审查，也不以 RDKit 通过证明原论文结构完全无误。
