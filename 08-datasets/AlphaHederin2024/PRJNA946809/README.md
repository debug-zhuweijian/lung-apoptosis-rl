# α-hederin/sEV 2024 / PRJNA946809

来源：[DOI10.3389/fphar.2024.1257941](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2024.1257941/full)，[ENA PRJNA946809](https://www.ebi.ac.uk/ena/browser/view/PRJNA946809)。核验2026-10-11。本目录是正式执行数据副本，Round20原件归档保留在 `experiments/20261011-data-alignment-round20/a549_function_bridge_sources/`。

实际数据：官方Table1.XLSX、Table2.XLSX保持源名称。Table1两个sheet共737唯一差异基因（463+274）；H/I/J/K列为N1/N2/T1/T2 raw counts。派生 `alpha_published_selected_gene_raw_counts.tsv` 保存737×4=2,948原值，44,190bytes，SHA256 `CADD3AD812EC5B0A347CE3DC82747CB76DE8FDE54ECAEF7F56C04854AB89E212`。原counts不是metadata，也不是全转录组；由全样本差异筛选的基因表不能当无泄漏完整输入。另normalized columns未混入。

已核验ENA run/BioSample：T1=SAMN33832283/SRR23923785，T2=SAMN33832284/SRR23936189，N1=SAMN36425337/SRR25270067，N2=SAMN36468537/SRR25299233。官方alias连接表列，T/N生物语义由论文Fig3（sEV-treated/untreated）和原表确认；项目明示A549T来源sEV培养A54948h。单样本未写全部处理浓度；论文RNA段是10^9 particles/mL sEV48h，无PTX/α-hederin。

`alpha_fig2.jpg`/`PMC10867254.xml`保存真实AnnexinV/PI功能出处。Fig2D为A549+sEV同时与PTX20ng/mL、α-hederin5µg/mL或组合48h；未确认“先sEV48h、再药物挑战”。Fig2H为另一A549T耐药背景，不能拿A549+sEV RNA作其实测RNA。Viability24/48h与PTX10/20ng/mL曲线不补成Annexin多剂量时长；单位保持原样，图中D/H坐标方向不同。

可作来源明确sEV状态参考及限定共同48h候选功能挑战，需显式参考时钟、batch与protocol迁移假设；未取得直接高维状态×多候选功能全覆盖。公开小表/元数据读取通过，未下载FASTQ、未训练/推理。复制hash逐文件一致台账和完整原文定位见Round20归档；数据本体只本地保存，不入Git/LFS。
