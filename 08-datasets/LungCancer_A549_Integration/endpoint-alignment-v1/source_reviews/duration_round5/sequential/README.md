# Lee 2012：A549时序处理的原文与表达库核对

日期：2026-10-10。来源为Lee et al., *Sequential Application of Anticancer Drugs Enhances Cell Death by Rewiring Apoptotic Signaling Networks*，[DOI 10.1016/j.cell.2012.03.031](https://doi.org/10.1016/j.cell.2012.03.031)。已取得[作者机构保存的发表版PDF](Lee2012_Sequential_Apoptotic_Networks_author.pdf)，15页；获取URL、SHA256和失败记录见[source_acquisition.json](source_acquisition.json)。

主代理已亲读PDF第3页Fig1图注、第12页相关结果和Fig7图注、第13页实验方法；查看[Fig7原嵌入图](Lee2012_page11_image0_Im0.jpg)和[第3页完整渲染](Lee2012_Figure1_page3_render.png)。

- Fig7F确为A549，包含control RNA与CASP8 siRNA两个背景，各列DMSO、ERL、DOX、同时处理、ERL→DOX及DOX→ERL。
- 检测为cleaved caspase-3与cleaved PARP双阳性流式，不是Annexin。主文Fig1规定ERL/DOX各10 µM，涉及DOX的方案在加入DOX后8h检测；文本抽取会丢失µ符号，剂量已回到渲染页核对。
- Fig1的常规时序定义为ERL领先24h或DOX领先4h；Fig7引用该方法。Fig7F的具体给药保留/移除、single-agent ERL时间以及重复合并方式仍需补充方法进一步核实，不能仅按箭头补造完整暴露窗口。

**没有把本文转录组配给A549。** 已从官方GEO取得[GSE30516 SOFT](GSE30516_family.soft.gz)，逐样本提取到[GSE30516_sample_metadata.json](GSE30516_sample_metadata.json)：25个样本分为BT20 13个、MDA-MB-453 6个、MCF 6个，未包含A549。论文同时出现A549功能实验和GEO编号，不足以建立它们之间的RNA配对。

本轮把它保留为有限的A549遗传状态×给药顺序功能候选，没有提取柱高或写入正式训练表。需先核清Fig7F完整条件与补充材料；即使取得发表均值，也仅按实际双阳性终点评价。不得拿乳腺细胞RNA代替A549的初始表达。

Europe PMC fullTextXML本次返回HTTP500，已记录；通过作者机构PDF完成主文审查。没有绕过PMC验证码，也没有联系作者。
