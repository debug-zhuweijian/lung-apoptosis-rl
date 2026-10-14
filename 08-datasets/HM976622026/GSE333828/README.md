# HM97662 2026 / GSE333828

功能表已恢复为[两条近似发表均值](processed/annexin-published-means-v1/README.md)：A549 NT约15.9%、HM1µM144h约14.0%，原图坐标/单位/RNA group关系独立复核。读图包络非生物CI，不恢复SD/rawrep；仅2Y，不复制六RNA行，角色unassigned。ns不是零效应或降凋亡证明；尚未取得精确补表，完整flow协议等价与模型资格仍未成立。

实际共同轴对齐已完成并独立复核：见[processed/joint-axis-v1](processed/joint-axis-v1/README.md)。六列浮点bulk表达按exact Ensembl对应30,268／32,461，2,193缺失为NaN/mask；不做归一化、整数化、初态或逐重复配对。全部已对应值与原表独立逐值相等。该版本只有表达与条件连接，不含Annexin数值、学习角色或模型资格。

来源：[论文 DOI10.1016/j.neo.2026.101349](https://doi.org/10.1016/j.neo.2026.101349)，[GEO GSE333828](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE333828)。核验2026-10-11。

本目录是正式执行数据副本；Round20原归档保留在 `experiments/20261011-data-alignment-round20/a549_function_bridge_sources/`，没有移动或删除。源文件 `GSE333828_raw_counts.tsv.gz` 保留官方名称，1,401,587bytes，SHA256 `28DD1A5341EB861CFABB2CF33EF5972D35437CE8122F63053B960777C9204F00`。派生 `GSE333828_A549_six_samples.tsv` 为57,905行×6样本数值及原gene_id/gene_name，不归一化、不四舍五入、不改原值。原18样本矩阵包括H841/SBC5，不能计为A549内多状态。

精确样本：GSM9774923–25为A549_Notreatment_rep1–3，GSM9774926–28为A549_HM97662_rep1–3。RNA为1µM/6天，平行未处理亦6天；原文RNA段每3天换液，不是t0初始测量。源raw_counts含非整数值，六A549列中41,656/347,430值非整数；保留source scale，未确认可用当前整数count decoder。

功能原件 `HM97662_fig3.jpg`、`PMC13476540.xml`：AnnexinV-FITC/PI、1µM/6天，作者定义AnnexinV-positive，3独立实验mean±SD。A549为未检出显著诱导(ns)，不证明效应零。RNA与功能source/background/dose/clock相容，无同culture/well/batch/逐重复对应；flow换液、浮游细胞收获未知，不能把RNA完整物理协议直接填到功能。HM由Hanmi提供、DMSO溶解，对照载体比例未说明。

可作限定同条件RNA/真实Annexin功能桥和负例挑战；不能单独验证A549内高维状态增量或多drug/dose/duration联合选择。完整源审查、定位与复制hash台账见上述Round20归档README、evidence_excerpts.json、execution_copy_manifest.json。未下载原始reads、未训练/推理，数据本体保持本地不入Git/LFS。主代理另维护processed/joint-axis-v1，不由本专项覆盖。
