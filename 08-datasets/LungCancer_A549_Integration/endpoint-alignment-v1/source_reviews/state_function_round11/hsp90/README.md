# HSP90/AXL/eIF4E：A549克隆表达与功能连接的限定核验

日期：2026-10-10。原文 Yang et al., *HSP90/AXL/eIF4E-regulated unfolded protein response as an acquired vulnerability in drug-resistant KRAS-mutant lung cancer*，Oncogenesis 8:45 (2019)，DOI [10.1038/s41389-019-0158-7](https://doi.org/10.1038/s41389-019-0158-7)，PMCID PMC6702198。

**本轮结论：存在A549 holo/para克隆RNA-seq的原文描述和多药生长反应图，但本轮尚未取得其RNA矩阵、样本映射或同克隆的定量Annexin数据，不能升级为已补齐的表达—凋亡配对。**

## 亲读范围与具体连接

| 原始证据 | 能确认什么 | 不能转移到什么 |
|---|---|---|
| 正文Results，Fig1D–G | A549/A549R的onalespib反应；Fig1G原图0.5 µM、0/1/6/12/24h的cleaved PARP/caspase7等WB | WB条带不是Annexin比例；A549R不能自动替换holo/para克隆 |
| Fig3K及正文 | A549_holo与A549_para进行RNA-seq差异比较 | 本地没有获得矩阵、单细胞分布、样本ID或测量时点 |
| Fig3L–N | holo/para对多种药物的浓度反应，原图纵轴为% control | 该纵轴不是凋亡率；不能把生长抑制反转直接称为Annexin选择收益 |
| Fig4C及图注 | A549/H358接受24h药物处理后洗脱，再无药培养10天 | 结局是长期克隆生长，不是共同终点的两种暴露时长Annexin对照 |
| 主文Methods的公共数据分析 | GSE15126、GSE65185用于所说明的外部基因集分析 | 不把这两个accession写成本文holo/para RNA原始数据 |

主代理已亲视 `41389_2019_158_Fig1_HTML.jpg` 与 `41389_2019_158_Fig3_HTML.jpg`，读取正文相关Results、Methods和18页补充文件的可提取文字。补充PDF包含FigS1–S8及试剂表，未在所读内容中发现holo/para RNA原始矩阵或明确下载accession。这里是获取范围内的缺项，不声称作者数据不存在。

## 原始文件和访问状态

- `PMC6702198.xml`：Europe PMC官方全文XML，104,806 bytes；SHA256 `d03e362cfa3e8ea34b8953fb07761e346d0c3773ffe45ad0e55888da59f89ae7`。
- `PMC6702198.md`：XML文本定位副本；部分嵌套图注重复，定位时以XML原结构/原图为准，不把重复文本算作独立证据。
- `41389_2019_158_MOESM1_ESM.pdf`：官方PMC Cloud补充，4,279,518 bytes；SHA256 `1d0e1937963632fc1384b65366a8389e94d7aedf7120d98dc4c0fe5b9dbd3a6c`。
- `supplement_text.md`：逐物理页文字提取，图示与特殊符号应回原件。
- `cloud_listing.xml`、`PMC6702198.1.json`、`asset_downloads.json`：官方公开对象清单、版本与4个实际下载的hash/大小。
- PMC网页出现验证码；Nature补充点击发生cookie重定向错误，没有进行验证绕过。资料来自另行公开的Europe PMC API和NCBI官方Cloud对象，均正常读取。

未下载整库、生成化学指纹、训练或推理；没有从WB强度推Annexin百分比，也没有向作者索取数据。当前用途为“状态差异及替代终点方法线索”，不是可训练的A549表达—Annexin数据。
