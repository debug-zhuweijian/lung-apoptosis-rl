# 第十一轮其他来源的主代理读回

日期：2026-10-10。下列结论只记录已亲读的具体来源，不代表完整领域检索。

## 肺癌代谢表型研究

*Liquid biopsy-based single-cell metabolic phenotyping of lung cancer patients for informative diagnostics*，DOI [10.1038/s41467-019-11808-3](https://doi.org/10.1038/s41467-019-11808-3)。主代理读取[公开作者PDF](https://repositories.cdlib.org/content/qt1488w471/qt1488w471_noSplash_df87d399b5c12e62357beff77e1a548c.pdf)第12–14页方法/数据说明。

- 12h代谢抑制剂处理后Annexin成像方法明确用于Patient30胸腔积液样本；不能因为下一段使用A549做Seahorse ECAR/OCR就把该Annexin结果配给A549。
- RNA测序部分研究患者肿瘤细胞；数据声明为BioProject PRJNA554445。本轮没有据此下载或登记A549转录组—凋亡配对。
- 它是可能有价值的肺癌患者样本方法参考，但当前主线先验证A549，不把患者实验扩大为当前已完成的验证对象。

## TRA-1-81研究的初筛定位

*A targetable pathway to eliminate TRA-1-60+/TRA-1-81+ chemoresistant cancer cells*，DOI [10.1093/jmcb/mjad039](https://doi.org/10.1093/jmcb/mjad039)。主代理亲读[正式版全文](https://academic.oup.com/jmcb/article/15/6/mjad039/7199663)的Fig2、Fig3图注及相关Results。

Fig2J确有A549的早期Annexin+/DAPI−结局；随后RNA-seq比较明确使用SW1990细胞。不能将SW1990 RNA配到A549。药前是否分选和暴露时长须由Annexin实验自身及适用通用方法核实，不直接借Fig2B的活力实验设置。更详细原图与协议审查由本轮`clonal_states/`单列保存。

## 初筛而非已核全文的返回

本轮SciSpace返回10项元数据，含sci-Plex、SCCP及其预印本、DREEP/scDrugAtlas、旧耐药表达研究、dacomitinib研究等。sci-Plex/SCCP沿已有资格结论去重；仅凭检索摘要没有新增任何功能训练条件。发现记录见`initial_searches.json`。
