# Navitoclax / Nav-Gal：名义化学结构核验

日期：2026-10-10。**两种名义结构均已建立并实际生成 2048 位手性 Morgan 指纹。Navitoclax 与 Nav-Gal 必须使用不同的化学身份，不能作为同义名或盐型母体合并。** 本包只提供化学侧表，不修改已冻结的 challenge、数字化读数或正式训练表。

| 作者化合物名 | `challenge_treatment` | 完整 InChIKey | 分子式 | 已指定立体中心 |
|---|---|---|---|---:|
| Navitoclax | `navitoclax` | `JLYAXFNOILIKPP-KXQOOQHDSA-N` | C47H55ClF3N5O6S3 | 1 |
| Nav-Gal | `nav_gal` | `ASOTXTFRQRSGRQ-DPNZDFNUSA-N` | C61H73ClF3N5O15S3 | 6 |

这里的“核验”是论文所描述名义分子的核验，不是历史 lot、纯度或实际加药溶液的独立鉴定。Nav-Gal 分子式和精确质量是按原图与合成描述计算的结果，没有冒称已经从未取得的 S1 原谱中读到该精确分子式。

## 第一手依据

原论文为 González-Gualda et al. 2020，[DOI 10.1111/acel.13142](https://doi.org/10.1111/acel.13142)。读取了项目已有的完整 primary XML 和 Cambridge Apollo 提供的发表版 PDF。

- **PDF 第 3 页 Fig. 1b**：分别画出 Navitoclax 与 Nav-Gal 的完整结构。`Fig1_page3_Im1_original.jpg` 是直接抽出的 `/Im1` 原 JPEG 数据，2068×1850；没有重采样或重压缩。图中的名称/部分文字另在 PDF 文字层，判断以完整 PDF 第 3 页和图注为准。
- **Results §2.1**（XML `acel13142-sec-0003`）：明确给出 2,3,4,6-tetra-O-acetyl-α-D-galactopyranosyl bromide、作者描述的 SN2、连接到 bis(sulfonyl)aniline 的 N，以及纯化后糖仍保持乙酰化。
- **Methods §4.1**（`acel13142-sec-0011`，PDF 第 14 页）：Navitoclax 来自 Eurodiagnostico，糖溴化物来自 Sigma；记载合成和纯化步骤。
- **Methods §4.2**（`acel13142-sec-0012`）：作者报告 NMR/COSY、ATR、HRMS 表征。此审查未重新取得和独立指认 S1 的原始谱图，不能把作者“已经表征”的陈述升级为本次独立完成了谱图鉴定。

这些原文段落与 Fig. 1 图注从本地 XML 导出到 `primary_evidence_sections.json`，包含原 section ID、URL 和 XML 哈希。

## 连接关系：接在哪个 N，保留什么

Navitoclax 按 Fig. 1b 逐组核对：对氯苯基与二甲基环己烯骨架、苄基连接的 piperazine、对位苯甲酰基、NH–SO₂ 连接、芳环上的 SO₂–CF₃、苯胺 N 所连手性侧链、苯硫醚和 morpholine。官方 PubChem 与 MCE 提供的不同遍历 SMILES 标准化后完全一致，再与原图的各连接与取代位置对应。

Nav-Gal 在这一完整骨架上增加糖基。原图中的酰基磺酰胺 **NH 仍保留**；糖接的是另一个苯胺氮，而不是这个酰基磺酰胺氮。

- 在 `Navitoclax_indexed.mol` / PubChem 返回 SMILES 的索引下，糖连接氮为 **atom 32**，原来连接芳环 atom 29 和手性碳 atom 33。
- 酰基磺酰胺氮是 **atom 22**，在 Nav-Gal 中仍有一个 H，不能被错误糖基化。
- 新键为 **Nav atom 32 — 糖 C1（产物 atom 74）**。
- 糖的 **2、3、4、6 位四个 O-acetyl 全部保留**，没有默认去乙酰化，也没有用游离 galactose 替代。
- 整个 Nav-Gal 是一个共价连接的分子，不能拆成 Navitoclax 加糖的混合物。其 `parent_id` 是本项目定义的最大有机片段身份，仍属于完整 Nav-Gal，与 Navitoclax 的 `parent_id` 不同。作者描述的酶促释放关系单独记录，不能据此宣布完整结构相等。

`Nav_Gal_atom_mapping.csv` 保存全部 88 个产物重原子来自哪个参照分子及原索引，方便逐原子追溯；索引对应 `Nav_Gal_constructed_indexed.mol`，不是标准化 SMILES 重新解析后的新编号。

## 立体依据及重建方法

Navitoclax 的一个 R 中心直接从原 Fig. 1b 核对：先清除参照手性标签，再按原图中手性碳到 N 的虚楔、左下 CH₂S、右下 CH₂CH₂N 的方向重建，得到的完整立体结构与官方参照一致。Nav-Gal 原图保留该侧链立体关系，合成发生在 N，不在该手性碳。

糖的立体定义来自原文**明确命名的 α-D-galactopyranosyl bromide**和作者**明确描述的 SN2**，而不是根据药效、β-galactosidase 名称或期望结果猜测：C2–C5 沿明确的 D-galactose 前体保留；异头 C1 按作者的 SN2 描述反转，得到名义 β-N-糖苷。这也与 Fig. 1b 所示产物的糖基相对构型一致。

官方 Sigma A7562 和 PubChem CID 2734143 的前体结构具有不同 SMILES 遍历顺序。分别从两种顺序做上述取代和反转，得到**同一个完整 Nav-Gal 结构**。代码显式处理替换 Br 后的邻接顺序奇偶性，避免“直接翻转一个 @ 符号”产生错误。

| Nav-Gal 中心 | 产物索引 | CIP | 依据 |
|---|---:|---|---|
| 原 Nav 手性碳 | 33 | R | Fig. 1b 原药虚楔及产物保留 |
| 糖 C1 | 74 | R | α-D 前体 + 作者 SN2 描述所确定的异头反转 |
| 糖 C2 | 73 | R | 明确 D-galactose 前体，2-OAc 保留 |
| 糖 C3 | 72 | S | 同上，3-OAc 保留 |
| 糖 C4 | 71 | S | 同上，4-OAc 保留 |
| 糖 C5 | 70 | R | 明确 D-galactose 前体及 C6-OAc 支链 |

**异头反转不等于 R/S 字母必然改变。** Br 被 N 替换后 CIP 优先级也改变，所以本例 C1 前后同为 R，但空间配置发生了反转。每个中心的证据另存 `stereochemistry_evidence.csv`。本结论代表原文所描述的名义 β 产物，不额外保证实际产品的异头纯度。

## 外部参照及不确定范围

- [PubChem Navitoclax CID 24978538](https://pubchem.ncbi.nlm.nih.gov/compound/24978538) 与 [MCE HY-10087](https://www.medchemexpress.com/navitoclax.html) 的完整结构一致，CAS 923564-51-6。没有误用 dihydrochloride 的另一种完整形式。
- [Sigma A7562](https://www.sigmaaldrich.com/US/en/product/sigma/a7562) 与 [PubChem 前体 CID 2734143](https://pubchem.ncbi.nlm.nih.gov/compound/2734143) 支持原文所命名前体的名义完整结构，CAS 3068-32-4。论文没有写 A7562 或 lot，不能反推历史采购批次；当前目录提及的 CaCO₃ 稳定剂没有被错误加入纯化产物结构。
- 官方 PubChem 对名称 `Nav-Gal` 的此次查询返回 404。该结果仅表示这次名称查询没有找到 CID，**没有为 Nav-Gal 编造 CID**。其身份由原图及合成定义重建，完整 SMILES / InChIKey / 项目 identity_id 均已计算。
- `identity-v3` 现有结构表中，两者均没有完全相同的结构身份，因此本次没有复用或覆盖旧登记。
- 原图按中性单组分结构绘制；历史加药材料的精确盐型/计量基准未独立鉴定。这里不转换原作者报告的 µM，也不把化学身份核验当作样本配对或训练可行性的全部证明。

## 输出与实际检查

供主代理连接的键保持明确：

| `source_id` | `challenge_treatment` | 指纹行 |
|---|---|---:|
| `GonzalezGualda2020:navitoclax` | `navitoclax` | 0 |
| `GonzalezGualda2020:nav_gal` | `nav_gal` | 1 |

- `chemical_identity.csv` / `.parquet`：完整 SMILES、完整 identity_id/parent_id、CID、来源、警示与是否可用。
- `morgan2048_chiral.npy`：实际计算的 `(2,2048)` uint8 数组，Morgan radius=2、includeChirality=true；算法版本和行顺序见 `fingerprint_metadata.json`。
- `review.json`：完整化学审查、合成映射与既有 registry 比较。
- `validation.json`：实际检查结果。两药指纹差 **40 位**，Tanimoto 约 **0.689922**；这个相似度仅是结构指纹相似，不代表生物等效。
- `raw/`：PubChem 原始响应、来源和哈希；MCE 的失败 HTTP 存档明确标 403，Sigma HTTP 读取超时不冒充成功原始页面。官方结构字段通过网页工具读取的来源另记在 `vendor_reference_properties.json`。

已通过的反证检查包括：接到错误的 atom 22 会产生不同结构（尽管分子式相同）；不反转异头位会产生不同立体身份和手性指纹；去除新连接后 Navitoclax 片段精确恢复；四个乙酰基和六个已指定中心保持完整。`.mol` 和 `.smi` 可用于下游独立复核。

两药 `eligible_for_structural_features=true`，**全部 `formal_training_ready=false`**。是否用于后续 challenge 或模型由主任务决定，当前没有改动 `digitized-v1`、`challenge-v1` 或启动训练。

从项目根复算：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round4/nav_gal/chemical_review/build_chemical_review.py'
```
