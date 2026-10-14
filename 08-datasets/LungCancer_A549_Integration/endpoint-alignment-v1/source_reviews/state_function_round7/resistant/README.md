# 第七轮：A549 亲本/耐药公开数据的条件对应复核

日期：2026-10-10。主代理完成原文与官方元数据核查。范围为两个新的重点来源；本轮没有取得可新增到当前核心表的完整状态表达—真实凋亡配对。**这两项结果分别定位了读出不一致和细胞状态不一致，不代表所有公开来源都不可行。**

| 来源 | 实际核验的公开数据 | 功能结果实际对应什么 | 当前资格 |
|---|---|---|---|
| Fang 2017，[DOI 10.1371/journal.pone.0170609](https://doi.org/10.1371/journal.pone.0170609) | [主文 XML](Fang2017_PMC5256872.xml)；[SRP087614 的两个 RNA-seq run](SRP087614_runs.tsv)，分别标为 A549 和 A549/DDP | Methods“Cytotoxicity assay”及 Fig2 使用 CCK-8，24h 剂量响应；主图1–10没有给出可配套的 Annexin 或 caspase 定量 | 可作为亲本/耐药表达和活力来源线索，不能把 CCK-8 改为凋亡标签。未下载测序 reads 或声称表达重建完成 |
| Yang 2013，[DOI 10.1371/journal.pone.0065309](https://doi.org/10.1371/journal.pone.0065309) | [主文 XML](Yang2013_PMC3669360.xml)；[GSE43494 的12个GSM元数据](GSE43494_samples.soft.txt)，两类芯片分别覆盖亲本/稳定耐药状态、各3个样本 | §Flow Cytometry、Fig5C 为 A549 的对照及 AK126698 处理背景下，cisplatin 10mg/L、24h 的 Annexin V阳性/PI阴性结果；不是对RNA芯片中的稳定耐药株直接重复测量 | 有真实早期Annexin类别挑战线索；高维RNA与其变更背景没有直接对应。Fig1剂量曲线为48h MTS，同样不能改为凋亡 |

Yang 的 GSE43494 中，GSM1059188–1059193 为 microRNA 平台，GSM1064138–1064143 为 lncRNA/mRNA 平台。这里的12个样本不等于12种不同生物状态；元数据没有对应Fig5的AK126698处理样本。亲本行的存在不能补造缺失的处理状态表达。Fig5A的少量qPCR是另一层证据，不能称为该处理背景的全转录组。

本轮通过 Consensus 语义发现、PubMed/出版原文定位、Europe PMC 全文和 GEO/ENA 元数据交叉核查。Consensus的两个重点条目已按要求fetch；最终判断依赖原文和官方元数据。[discovery.json](discovery.json)保留检索与选取记录，[primary_evidence.json](primary_evidence.json)保存可定位的原文节点。[Zotero核对](zotero_check.json)读取完整60个顶层条目，两DOI均未命中，未写回。GSE关键词检索中包含把GSE理解为化合物/提取物缩写的无关结果，已排除，未计为支持证据。

最初的委派检索因平台内容过滤中断；主代理随后直接完成上述公共文献和数据字段复核，未执行生物实验或联系作者。两个重点来源均未进入当前推荐训练表，没有把获取元数据写成表达矩阵已读取。来源文件、获取状态和内容哈希见各 acquisition 文件与 manifest.json。
