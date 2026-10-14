# TIDYI 10x：E-MTAB-16351 的 A549 元数据核查

日期：2026-10-09。来源：[EBI官方研究](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-16351)。

本轮仅取得研究JSON及两个A549逐细胞元数据CSV，**未下载并验证该研究完整表达矩阵**。

| 文件 | 实读行数 | sample / celltype |
|---|---:|---|
| Chromium10x_1001_A549_metadata.csv | 1208 | 1001 / A549 |
| Chromium10x_1002_A549_metadata.csv | 1251 | 1002 / A549 |

共2459条元数据，字段有sample、batch、celltype、计数/QC等，无逐细胞Annexin状态标签。文件URL、字节数及SHA256见`source_verification_20261009.json`。

原作者`classifier_data.tar.gz`归档保存在邻近E-MTAB-16354目录，其中human_10x对象存在，但本轮未完整解压。归档内rate小表已只读读取：A549仅一行汇总参数；作者程序按celltype把同一rate配给不同sample，不能当成多个独立药物终点。

该来源可补查平台迁移及有限群体率连接，不能单独验证多药凋亡排序。原始时间死亡率不能直接当作某时点Annexin阳性百分比。未接入正式训练。

[研究说明](G:/研究生科研文件/paperstudy/02-notes/LungCancer_Data_Gap_Resolution_Study_2026-10-09.md)。数据保持本地，不修改已冻结743细胞分选状态资料。
