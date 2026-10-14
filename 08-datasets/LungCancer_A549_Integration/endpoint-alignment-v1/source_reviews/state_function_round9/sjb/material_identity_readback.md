# SJB2-043 材料定位补核

日期：2026-10-10。此补核不修改功能源表。

**可以建立三个 24 h 剂量的“官方名义材料—实测条件”连接候选；证据等级是论文明确厂家/名称与当前官方产品记录相连，不是原实验 lot 或历史采购货号已核实。**

## 原论文直接证据

§2.1 的内容只有 SJB2-043 及 MedChemExpress（Monmouth Junction, NJ, USA）。准确位置为 **PDF 第 2 页**，`cimb-47-00155.md` **117–118 行**，XML `sec2dot1-cimb-47-00155`。此前向主代理口头报出的“第3页、约128行”已更正；厂家事实不变。

正文没有 catalog number、CAS、lot/批号、该批纯度或采购日期。不能把当前网页的货号/纯度写成论文原文提供。

## 官方名义产品记录

已亲读 [MCE 官方产品页](https://www.medchemexpress.com/SJB2-043.html) 的商品标题、Chemical Information 和 Publications，以及它链接的 [官方 Data Sheet](https://file.medchemexpress.com/batch_PDF/HY-15757/SJB2-043-DataSheet-MedChemExpress.pdf)。

| 当前官方字段 | 核定值 |
|---|---|
| 名称 | SJB2-043 |
| 厂家货号 | HY-15757 |
| CAS | 63388-44-3 |
| 分子式 | C17H9NO3 |
| 分子量 | 275.26 |
| 产品页引用本论文 | 是，PMID 40136409，2025 Feb 27;47(3):155 |

官方网页 Chemical Information 给出的 SMILES：

```text
O=C1C2=C(N=C(C3=CC=CC=C3)O2)C(C4=CC=CC=C41)=O
```

`material_smiles_validation.json` 保存原字符串和 RDKit 解析结果：1 个连通分子、形式电荷 0、计算分子式 C17H9NO3，计算分子量约 275.263；与厂家列值一致。规范异构 SMILES 为 `O=C1c2ccccc2C(=O)c2oc(-c3ccccc3)nc21`，InChIKey 为 `CMYQQADDUUDCCA-UHFFFAOYSA-N`。这些是对当前名义结构的计算核验，不是历史试剂的化学实测。

官方 PDF 已下载为 `SJB2-043-DataSheet-MedChemExpress.pdf`，第1页列产品标识，第2页 Customer Validation 列本篇论文。产品网页可经 Web 工具读取，直接 GET 正常地址及英语区域地址均返回403；因此仅保留 `MCE_web_tool_evidence.md` 为网页工具读回证据，没有伪称保存了 HTML 原件。失败与成功下载分别记录在 `MCE_product_source.json`、`MCE_material_download_manifest.json`。

## 三个条件输入的合法范围

`nominal_material_condition_links.json` 单独列出三条可供主代理接入核验的条件：

| 材料 | 论文浓度 | 换算浓度 | 实际时长 |
|---|---:|---:|---:|
| 论文 SJB2-043 / MCE → 当前 HY-15757 名义结构 | 0.625 µM | 625 nM | 24 h |
| 同上 | 2.5 µM | 2500 nM | 24 h |
| 同上 | 10 µM | 10000 nM | 24 h |

浓度和 24 h 均由论文 Fig4/§2.8 支持；厂家页面用于材料结构定位。厂家的 K562/24–72 h 示例协议不用于给本论文补时长。0 µM 对照不编码成一个有药物结构的给药条件。

该连接能够支撑名义小分子、剂量和时间特征的构造；不代表项目动作库的化学核验、指纹或正式接入已完成。原实验批号、历史货号、该批纯度继续保持未知；RNA 文件缺失的判断不因材料定位而变化。
