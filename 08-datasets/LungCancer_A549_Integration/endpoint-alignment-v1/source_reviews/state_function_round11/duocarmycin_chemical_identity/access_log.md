# 化学身份定向访问记录

日期：2026-10-10。仅核对 Vielhauer2013 已命名的两种材料；不检索类似物库，不记录制备/活化操作。

## 本地已读来源

- `state_function_round11/exposure_duration/Vielhauer2013_PMC_fulltext.md`：2013 正文/缩写/引用。
- `state_function_round11/exposure_duration/cbt-14-527-g1.jpg`：原 Fig1D 已亲视；复制为本包 `Vielhauer2013_Fig1.jpg` 并核同 hash。
- 既有 `scripts/build_endpoint_condition_inputs.py` 第132行附近：只读确认 radius2、2048 bits、includeChirality=true、uint8；没有改该脚本。
- 采用 Datamol skill 的分子解析和立体检查思路；复用项目已有 RDKit 环境，没有按 skill 示例安装或做脱盐/标准化。

## 官方 PubChem

精确名称查询分别为 `BocNHO-CBI-indole2`、`BocNHO`、`seco-CBI-indole2`、`CBI-indole2`，PUG REST `/compound/name/<name>/cids/JSON` 均返回 HTTP404 `PUGREST.NotFound`。记录在 `pubchem_name_queries.json`；仅表示这些别名未解析，不表示结构不在 PubChem。

随后网页定向查询 `site:pubchem.ncbi.nlm.nih.gov "CBI-indole"` 定位官方 CID11016947，并实际获取：

`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/11016947/property/CanonicalSMILES,IsomericSMILES,InChIKey,InChI,MolecularFormula,IUPACName,Charge/JSON`

从2013 Fig1D独立转录 Boc 图、核对2010系统名/分子式/立体后，以**精确结构键**查询：

`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/XTLVCBPHMQCFAN-JOCHJYFZSA-N/cids/JSON`

该公开接口返回 CID24755111，随后实际获取：

`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/24755111/property/CanonicalSMILES,IsomericSMILES,InChIKey,InChI,MolecularFormula,IUPACName,Charge/JSON`

两个结构属性 JSON 保持原响应；Boc 的 PubChem canonical isomeric SMILES 经 RDKit 与先前原图转录逐图一致。没有按名称直接合并。

## 原材料引用

- 确认2013引用24：Jin2007 DOI10.1021/ja075398e，PMID18020335，PMC2519901。
- 确认2013引用25：Lajiness2010 DOI10.1021/jm1010397，PMC2974002。
- 两个 PMC 网页入口出现浏览器验证页，未绕过。Europe PMC `fullTextXML` 两次独立请求分别返回 HTTP500，未反复尝试同一失败点。
- 官方 PMC Cloud 前缀 `PMC2519901.` 返回0项，`PMC2974002.` 返回 `PMC2974002.1/`；原始列表保存。前者不代表论文不存在或无任何合法全文。
- 2010官方metadata：`https://pmc-oa-opendata.s3.amazonaws.com/PMC2974002.1/PMC2974002.1.json`；其给出的XML下载成功，MD5=`400bed1d4bb45f27ed0c4f2471f10486`。metadata标记author manuscript、TDM；没有把它说成出版社原始PDF。
- 只提取该XML `sec S13` 的系统名称、式与1S/1R身份描述用于本任务；没有生成任何实验操作指南。

## 复算记录

脚本先保存完整canonical SMILES、MOL与指纹，再从保存文件读回；同时采用两个RDKit Morgan入口、MOL立体回读和反转立体的负对照。最终exit0；详细结果在 `validation.json`。所有输出仅本目录；无 Git、Zotero、Memory 或外部写入。
