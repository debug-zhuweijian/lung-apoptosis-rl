# Hu 2025：肺癌真实序贯给药候选的核验记录

核验日期：2026-09-18。状态：**原文 XML、公开补充附件和 Fig. 3 已读取；尚无可直接训练的公开逐重复凋亡数值表或配套转录组矩阵。**本次没有从柱状图估算数值，也没有联系作者索取资料。

## 来源与读取范围

- Hu 等，2025，*Pharmacodynamic-Driven Sequence-Dependent Synergy Effects in Pemetrexed-Osimertinib Combination Against Non-Small Cell Lung Cancer (NSCLC): Optimizing Synergy Through Sequential Interval*，Pharmaceutics 17(8):1044，[DOI](https://doi.org/10.3390/pharmaceutics17081044)，[PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12389513/)。
- 亲读 Methods 2.2、2.8，Results 3.2–3.3，Fig. 2–3 图注、Fig. 3A 给药示意及 Data Availability Statement。
- 通过 [Europe PMC 全文接口](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12389513/fullTextXML) 保存 `PMC12389513.xml`；通过 supplementaryFiles 接口取得原始附件 `pharmaceutics-17-01044-s001.zip`，内部为 25 页 `pharmaceutics-3727392-supplementary.pdf`。
- 附件目录及 PDF 文本检查显示补充表、图、Western blot 与 STR 材料；没有找到可下载的 RNA 表达矩阵或逐重复 Annexin 数值文件。论文说明支持结论的数据需向通讯作者合理请求，不能据此称其已公开可取得。
- 文件与 SHA256 见本目录 `source_manifest.json`。本记录只核对这一候选，不代表已穷尽所有公开序贯数据。

## 已核实的真正序贯设计

| 项目 | 原文支持的内容 | 定位 |
|---|---|---|
| 药物 | Pemetrexed（PEM）与 Osimertinib（OSI） | Fig. 2–3 |
| 凋亡细胞背景 | HCC827、NCI-H1975 | Fig. 3B/C |
| HCC827 浓度 | PEM 1 μM、OSI 20 nM；仅指 Annexin 的 B 面板，不能套用 E/F 面板的 50 nM | Fig. 3B/E/F 图注 |
| H1975 浓度 | PEM 1 μM、OSI 50 nM | Fig. 3C |
| 顺序与间隔 | PEM 从 Day 0 到 Day 2，随后 OSI；图中另设 PEM→BLK、BLK→OSI 和持续 OSI | Fig. 3A |
| 终点 | HCC827 Day 3 与 Day 4；H1975 Day 4，n=3 | Fig. 3B/C |
| 检测 | Annexin V-FITC/PI 流式；早期与晚期分开显示 | Methods 2.8、Fig. 3 |
| 撤药 | 图注明确存在药物移除与更换阶段；精确洗脱次数、药物残留量未在所读流程中量化 | Fig. 1B、Fig. 3A |

上述是本文实测设计，不是本项目构造的模拟序列。Fig. 2 包含正向、反向及同时组合的增殖抑制比较；**不能将其 MTT/抑制率和 Bliss 表替换为 Fig. 3 的凋亡标签**。Fig. 3 的 Annexin 实验也不能被扩写成所有正反顺序均已测量。

## 能支持什么、还缺什么

可以确认：肺癌真实序贯处理与凋亡终点实验已存在，尤其提供了两段给药、48 小时间隔及统一 Day 4 比较的设计参考。

尚不能完成本项目的独立数值评测：目前只有论文图及有限补充表，未取得逐重复凋亡原数值；没有找到相应的单细胞／bulk 转录组输入。A549 双数据原型与这里的 HCC827/H1975 也不能直接视为同一背景。

如果后续取得匹配数据，独立测试应冻结为图示已测方案集合及同一终点，并按实验定义分别评价早期、晚期或预先指定的总 Annexin 阳性比例。数据不包含的顺序、间隔或连续剂量不能获得伪造的真实标签。当前仅为候选核验，不代表序贯验证已通过。
