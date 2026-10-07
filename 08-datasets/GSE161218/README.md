# GSE161218／Macaya 2023 表达—凋亡连接核验

> 核验日期：2026-09-18。证据状态：官方元数据、9 个 gene-count 文件、补充 PDF 和 Source Data 已下载并实际读取；尚未完成正式表达质控、模型训练或校准。

## 本次结论

**该研究确实提供可读取的肺癌药物表达数据和 Annexin V 阳性百分比原始数值。**表达和终点可以在 H1792 的部分处理条件上连接，但属于不同时间、不同实验的条件级对应，不能直接称为同一样本配对数据，也不是序贯用药数据。

- RNA：H1792，DMSO 对照、Lestaurtinib 0.625 μM、Trametinib 0.5 μM，各 3 个样本，处理 16 h；没有组合处理 RNA。
- Supplementary Fig. 2C：6 个肺癌细胞系，Control／Trametinib／Lestaurtinib／同时组合，24 h，每组 3 个独立实验；实际读取 **72 个 Annexin V 阳性百分比值**。
- Supplementary Fig. 3D/E：同 6 个细胞系的 Control／Trametinib／Midostaurin／同时组合，24 h；72 个预期单元格中 **70 个有数值**。A549 的 Control Exp 3 与 Tram Exp 3 为空，按缺失保留。
- 两个凋亡数据块分别保存。重复的条件名称和 Exp 编号不意味着同一样本，也不自动形成更多独立重复；两图共同非空对照位置的数值不相同。

逐样本、逐条件和原始数值见 [CONDITION_MATCHING.md](CONDITION_MATCHING.md)；Midostaurin 扩展见 [MIDOSTAURIN_ENDPOINTS.md](MIDOSTAURIN_ENDPOINTS.md)。

## 官方来源与版本

| 对象 | 来源与定位 | 本次读取范围 |
|---|---|---|
| GEO series | [GSE161218](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE161218)，记录最后更新 2026-01-27 | Series SOFT、GSM4892080–GSM4892088 SOFT、9 个原始命名的 gene-count 附件 |
| 论文 | [Macaya 等，Nature Communications，2023，DOI 10.1038/s41467-023-41828-z](https://www.nature.com/articles/s41467-023-41828-z) | 正文 Fig. 1F/G 单药 RNA、Methods 的 Apoptosis、Data availability；HTML 原件留存 |
| 补充信息 | `41467_2023_41828_MOESM1_ESM.pdf` | 第 5 页 Fig. 2C/D 图注；第 7 页 Fig. 3D/E 图注 |
| 官方源数据 | `41467_2023_41828_MOESM7_ESM.xlsx` | `Fig Suppl 2`、`Fig Suppl 3` 的 Annexin V 百分比原值，保留来源单元格 |

完整 URL、字节数、SHA-256 和获取/复核 UTC 时间见 [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md)。22 个源文件合计 **58,504,599 字节**。没有下载 FASTQ 或整批 SRA reads。

## 本地文件与读取方式

```text
GSE161218/
├── README.md
├── CONDITION_MATCHING.md
├── MIDOSTAURIN_ENDPOINTS.md
├── SOURCE_MANIFEST.md
└── GSE161218/
    ├── sources/                         官方 SOFT、HTML、PDF、XLSX、counts.gz
    ├── source_manifest.json             机器可读来源和校验值
    ├── sample_records.json              GSM 原字段
    ├── rna_sample_conditions.csv        9 个 RNA 样本的条件解释
    ├── counts_read_validation.csv       逐文件读取与数值检查
    ├── annexin_v_fig_s2c_long.csv        72 个观测值及源单元格
    ├── annexin_v_fig_s3de_midostaurin_long.csv  72 个位置，70 个观测值
    ├── article_extracted.md             正文 HTML 的派生全文，保留本地
    └── supplement_extracted.md          补充 PDF 的派生全文，保留本地
```

来源全文、下载文件、CSV/JSON 均是本地数据产物；只有治理 Markdown 和局部 `.gitignore` 进入 Git。全文派生 Markdown 已额外设为忽略，避免将其误当治理说明提交。

复核入口为项目根目录的 `scripts/prepare_gse161218_bridge.py`，依赖当前环境已有的 requests、beautifulsoup4、pandas、openpyxl 和 PyMuPDF：

```powershell
python scripts/prepare_gse161218_bridge.py
```

脚本复用已有源文件，重新计算 hash、读取并验证数据；每个远程源文件限定不超过 20 MB。重新读取本地文件不表示重新核验服务器上的最新版本。原始 GEO 附件仍保留 `GSM4892080_CTL_1.counts.txt.gz` 等官方文件名。

## 实际检查结果

9 个 count 文件均完整解压、读取成功，均含 **58,381 个基因行**；基因标识唯一、各文件基因顺序一致、计数非负且为整数。GEO 声明其生成流程为 STAR 对 hg38 比对、featureCounts/Gencode v28 生成 gene counts；本次没有重跑比对。

| 样本 | 检出计数大于 0 的基因行数 | gene counts 总和 |
|---|---:|---:|
| CTL_1 | 13,106 | 7,502,053 |
| CTL_2 | 19,698 | 7,342,213 |
| CTL_3 | 20,158 | 7,597,932 |
| Lest_1 | 20,377 | 6,963,584 |
| Lest_2 | 20,411 | 8,159,233 |
| Lest_3 | 20,169 | 7,788,159 |
| Tram_1 | 18,959 | 5,386,708 |
| Tram_2 | 19,681 | 6,412,478 |
| Tram_3 | 19,106 | 5,494,442 |

CTL_1 的检出基因数明显低于其他对照，应进入后续表达分布和样本相关性审查；本次不据此自动剔除。表中是 gene-count 总和，不能当作 FASTQ 原始测序 read 总数。

## 对本项目的意义与限制

1. **有真实药物结局标签。**Annexin V 百分比不是由表达打分生成的伪标签；本次读取的值来自论文源表，不是图像目测取值。72 h 细胞活力、cleaved PARP Western blot、肿瘤体积均未混入本标签表。
2. **H1792 两个单药的药物和剂量能对应。**表达是 16 h，Annexin 终点是 24 h；这允许提出早期表达与后续条件级药效关联的探索任务，但不能将 RNA 和结局实验编号强行一一配对。
3. **A549 有实测凋亡终点，GSE161218 没有其 RNA。**可与 Tahoe A549 条件继续核对；同细胞和同药物仍不足，剂量、时长、培养、平台和对照定义要单列。
4. **对照存在描述差异。**RNA 正文图注为 DMSO 对照，Annexin 图注写 untreated；未取得更精确信息前，不能默认两者完全相同。
5. **百分比的分母受实验门控约束。**补充 Fig. 2D 图注写先依 FSC/SSC 选择 alive 与 single cells，再做 7AAD／Annexin V 分析。这是该采样和门控协议下的操作性终点，不能未经检查就解释为最初培养细胞的总死亡比例。
6. **目前只有很少的独立处理条件。**不能把每个 RNA 重复配到每个终点重复来伪增样本量，也不能把三种名义条件当作充分的高维终点评价器训练和独立校准集。
7. **两种 Combo 均为同时组合。**没有给药先后、间隔、洗脱等序贯设计，不能替代真实序贯验证。

结论范围：**通过数据可访问性与条件级连接核验；可作为限定条件的外部终点和辅助验证来源。通用终点评价器校准、跨来源可靠性以及序贯决策验证尚未通过。**
