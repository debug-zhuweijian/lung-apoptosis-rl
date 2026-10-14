# Pogacar 2022：已核实34个GSM，状态表达与克隆终点仍需分开

核验日期2026-10-10。[原论文](https://doi.org/10.1371/journal.pone.0273182)，[PMC9447877](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447877/)。Cloud完成正文、方法和补充S1文字层审查；主代理进一步取得官方全文XML及三个系列/全部样本SOFT，亲读Fig2K、Fig3E/G、Fig5F/H和RNA/流式/克隆实验方法。

Cloud端GEO获取失败，Windows本地GET成功；因此将元数据状态更新为“官方记录已取得并读取”，不是因为一次失败就判数据不存在。Cloud原报告保持原样，见 [Cloud交付](../../../../../../experiments/20261010-cloud-dot-coordination/cloud_pogacar_delivery.md)。

| 系列/样本 | 实际材料和处理 | 时间与重复 | 资格 |
|---|---|---|---|
| GSE197600；GSM5923408–11，共4个 | A549 CDK2 KO，untreated或palbociclib 2µM；具体克隆编号未核实 | 处理10天；Series明确technical duplicates | 基因型背景的状态RNA，不改成亲本A549；A3/A4不能凭标题当作独立克隆 |
| GSE197602；GSM5923416–31，共16个 | A549/SUM159各8个；UT、palbociclib 2µM、indisulam 3µM、组合各2个 | 16h；Series明确technical duplicates | 短时剪接实验，不能混入10天诱导初态 |
| GSE197603；GSM5923432–39，共8个 | A549的UT、palbociclib 2µM、indisulam 0.5µM、组合 | 处理10天；Series明确technical duplicates | 可作来源内多群体状态参考 |
| GSE197603；GSM5923440–41，共2个 | A549，PF-06873600 0.5µM | 处理7天；technical duplicates | 主代理已补核Cloud尚未取得的剂量/时间；不能写为10天 |
| GSE197603；GSM5923442–45，共4个 | SUM159，UT或PF-06873600 0.5µM | 活性臂7天；technical duplicates | 不移作A549 |

共34个GSM，22个A549、12个SUM159。[sample_metadata.csv](sample_metadata.csv)保存官方标题、材料、处理字段和34个逐样本genecounts文件位置。**本轮未下载或读取这些表达矩阵值**，不称共同基因轴对齐完成。

Series/GSM的 treatment_protocol 是可在对照与不同实验臂中重复出现的通用说明，必须结合逐样本 characteristics 中的 treatment。例如GSE197603的所有记录含10天和7天两种说明，不能给每列同时标两种暴露。UT参考的具体对应采样日若未由逐样本记录唯一确定，仍须保留不确定性。

## 与真实功能的连接缺口

Fig3G 的表达为10天诱导；Fig3E 则先用palbociclib 2µM＋indisulam 0.5µM预处理A549两周，再用ABT-263 0.5或2µM一周，读出是长期**克隆形成**。它混合了增殖、存活和再生能力，不能改称Annexin比例。两处状态相差4天，在已读内容中没有高维状态稳定性的直接验证。

Fig5H 使用70%乙醇固定和DAPI DNA含量，测的是细胞周期，不因为用了流式仪就变成凋亡检测。Fig5F的indisulam 3µM/16h，也不能与Fig3G的0.5µM/10天互配。克隆实验方法说定期换培养基和药物，不足以确认Fig3E第二阶段是否完整撤去前药。

本源可保留为状态表达/增殖相关的辅助资料。未新增可信A549高维初态—定量凋亡配对；没有把新增GSM数算作独立生物重复、凋亡标签或完整可行性通过。

## 文件与复核

`download_manifest.json`记录3个样本SOFT与官方XML；`series_download_manifest.json`记录3个Series SOFT。所有源URL和SHA256保留。本文判断不依赖二级摘要替代官方元数据；Cloud补充S1仅文字层已读的覆盖范围如实保留，不宣称本地主代理已审全部附件。
