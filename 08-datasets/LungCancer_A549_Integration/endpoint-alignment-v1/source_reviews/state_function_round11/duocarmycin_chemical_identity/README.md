# Vielhauer2013 两种实验材料的名义化学身份

日期：2026-10-10。核验对象：[Vielhauer et al. 2013，10.4161/cbt.24348](https://doi.org/10.4161/cbt.24348) 中 BocNHO 与 CBI-indole2。仅作公开结构、立体和材料身份核验；没有生成任何化合物制备或活化操作指导。

## 已完成的结论

**两种材料的来源与图示完整结构均可追溯，两个结构指纹已实际生成并验证。** 结论是论文图示的名义分子身份，不是对历史实验批次的成分、盐含量或光学纯度实测。

| 2013 图中名称 | 本文明确全名 | 精确结构对应的官方 PubChem CID | 分子式 | 图示立体 | 指纹行/置位 |
|---|---|---:|---|---|---|
| CBI-indole2 | seco-CBI-indole2 | [11016947](https://pubchem.ncbi.nlm.nih.gov/compound/11016947) | C31H23ClN4O3 | CBI 位点 1S | 0 / 60 |
| BocNHO | BocNHO-CBI-indole2 | [24755111](https://pubchem.ncbi.nlm.nih.gov/compound/24755111) | C36H32ClN5O5 | CBI 位点 1S | 1 / 74 |

InChIKey 分别为：

- seco-CBI-indole2：`OMTPLEBPJORDFA-LJQANCHMSA-N`
- BocNHO-CBI-indole2：`XTLVCBPHMQCFAN-JOCHJYFZSA-N`

PubChem 的 CID11016947 页面标题写 `Seco-cbi-indole`，而论文简称有 `indole2`。本次根据**完整连接图、两个 indole 单元、立体和分子式**核对，不凭相似名称合并。BocNHO 的四个简称查询未命中，但原图转录的精确 InChIKey 命中 CID24755111；官方结构解析后与原图转录的 canonical isomeric SMILES 完全相等。

本论文 `CBI-indole2` 明确定义成 **seco** 形式；没有用名字相似的环丙烷结构替换它，也没有合并前药和对照材料。

## 原始依据与阅读范围

1. **2013 原文**：摘要、Introduction、Fig1D、Glossary 与 references24/25 已亲读。Glossary 明确两组别名。Fig1D 是两个不同完整分子：上方为 seco 结构，下方为 BocNHO；二者都画有相同方向的虚楔立体键。原图保存 `Vielhauer2013_Fig1.jpg`，逐字节复制自同轮 `exposure_duration/cbt-14-527-g1.jpg`，SHA256 `ebdaaba83c066b28f8b038984a16c1705b782c553832863ca53335d878b4112f`。
2. **2010 原引用**：[Lajiness et al., 10.1021/jm1010397](https://doi.org/10.1021/jm1010397)，官方 PMC Cloud author manuscript `PMC2974002.1.xml`。只围绕 sec S13 的 compound5 系统名称、分子式和 1S/1R 身份读取。该节将所画 Boc 结构标为 tert-Butylcarbamate Prodrug5，并明确 natural=1S、unnatural=1R；分子式 C36H32ClN5O5 与本轮转录一致。身份摘录未导出制备步骤。
3. **2007 原引用**：2013 references24 中列 Jin et al., DOI10.1021/ja075398e。其全文本轮未取得；不以它支撑未亲读细节。2010 已读材料身份与 2013 原图足以完成当前结构核对。
4. **官方 PubChem PUG REST**：两份 `PubChem_<CID>_properties.json` 实际保存原响应，含 isomeric SMILES、connectivity SMILES、InChI、InChIKey、formula、IUPAC 和 charge。完整来源 URL 与访问记录见 `access_log.md`。

`primary_identity_readback.md` 是便于主代理逐项复读的定位材料；2013 原文只读路径和 hash 也在其中。`nominal_structures_render.png` 是 RDKit 对实际写入 SMILES 的显示，已与原 Fig1D 亲视比较。图中布局与原图不同，不代表不同分子。

## 立体、盐与实验材料边界

- 两结构均有 **1 个明确定义的四面体 S 中心、0 个未指定原子立体中心**。本次没有从“天然”一词自行补上缺失立体：2013 原图画出了立体键，2010 compound5 标签及 PubChem 全结构用于交叉检查。
- 每个图示结构为 **1 个共价连通分量、净形式电荷 0**。图中的 Cl 与碳共价相连，不是氯离子对离子；没有人为增加 hydrochloride、脱盐或替换成母体。
- 历史批号、实验供货号、批次实际盐/溶剂化组成及 ee/纯度没有在本次读取的 2013 材料说明中核定。`none_depicted_no_salt_added` 表示编码忠实于画出的分子，**不表示已测得历史样品不含溶剂或杂质**。
- 此身份映射限于 Vielhauer2013 的明确命名和原图；其他论文出现 CBI-indole2 仍要重新核对。

## 实际产物与验证

| 文件 | 内容 |
|---|---|
| `nominal_structures.csv` / `.json` | 两个分子的稳定 ID、原文简称、全名、官方 CID、canonical full isomeric SMILES、InChIKey、分子式、材料边界和行号。 |
| `fingerprints.npy` | `uint8[2,2048]`，Morgan radius2、includeChirality=true。 |
| `compound_0.mol` / `compound_1.mol` | 从相同完整图生成的二维 MOL，可保存和回读立体。 |
| `nominal_structures_render.png` | 对实际模型结构的科学结构图显示，便于与原文复核。 |
| `build_nominal_structures.py` | 本目录独立构建与验证脚本，不修改公共 builder。 |
| `validation.json` | 实际执行和逐位回读结果。 |
| `manifest.json` | 本目录原件、结构与指纹的 hash，以及只读上游原图/原文引用。 |

实际验证已通过：

- 官方 PubChem 结构与人工图转录经 RDKit 解析后完全一致；两种分子没有被合并。
- 保存后的 canonical SMILES 重新解析，再生成指纹，2048 位逐位相等。
- 使用项目的 `rdFingerprintGenerator.GetMorganGenerator` 和独立 legacy Morgan 实现交叉核对，逐位相等。legacy deprecation 提示不影响本次一致性验证。
- MOL 保存—回读后 InChIKey 和立体身份不变。
- 将 S 改成 R 的负对照，两个指纹分别改变 10 和 9 位，验证本次没有丢弃立体。
- Boc 图转录的分子式与 2010 原文一致；计算身份质量信息与其明确列出的分子离子质量一致，仅用于结构校验。

RDKit 版本：2026.03.6。指纹文件 SHA256：`b0bace7ff67f76dfd58361fa4f574e551fdfdf8c98c5669726c9f008510a1af5`。

复算：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round11/duocarmycin_chemical_identity/build_nominal_structures.py'
```

最终实际运行 exit0，两个结构均通过。初次 Windows 中文路径的 RDKit C++ 文件 IO 错误改用 Python Path 读写 MOL block；立体负对照显式刷新 CIP 状态后验证通过。没有安装库、训练或模型推理。

**对源数据的影响为 0 行修改。** exposure_duration 的 20 个条件由原代理维护，结构核验或某个消费者是否支持时长都不取消这些实验记录。这里完成的是两个名义化学条件输入；不据此宣称原模型已具备时长动作、生成了表达特征或验证了策略收益。
