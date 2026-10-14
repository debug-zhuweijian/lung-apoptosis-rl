# Nav-Gal / Navitoclax：Fig4b 公开数字与原图核验

日期：2026-10-10。只检查 González-Gualda et al. 2020，[DOI 10.1111/acel.13142](https://doi.org/10.1111/acel.13142)，没有扩展到其它药物论文、联系作者或开始训练。

## 1. 已取得的主要证据

**Fig4b 的均值可以辨认，不应判为“没有数据”。目前已取得发表版的原嵌入图，尚未取得逐生物学重复的作者数值表。** 这应归类为“公开发表的汇总图数据，可以进行带误差的数字化”，而不是“已下载原始读数”。

| 文件/来源 | 当前可读程度 |
|---|---|
| 先前的 `state_function_round3/PMC7189993.xml`，161,676 B | 论文 primary XML；Fig4 id=`acel13142-fig-0004`，图注、全部7附件清单与 data availability 已读 |
| `GonzalezGualda2020_NavGal_published.pdf`，2,703,021 B、19页 | 来自作者机构 Cambridge Apollo 的公开发表版；Fig4 在 PDF第8页、完整图注在第9页 |
| `Fig4_embedded_original.jpg`，1880×2792、538,992 B | 直接抽出 PDF 第8页 `/Im1` 的原 JPEG 数据；没有放大、重采样或重压缩。这是本轮取得的最佳可用原图，并不声称已穷尽出版商所有分辨率版本 |
| `Fig4b_geometry_and_readability.json` | 原图像素坐标、各 panel 坐标轴初值、图例与可读时间标签；不是响应数字表 |
| `Cambridge_article_bundles.json`、`Cambridge_original_bitstreams.json` | 机构库文件清单。ORIGINAL 只有发表版和接受稿，另有 LICENSE/TEXT/THUMBNAIL bundles；未列作者原始实验数值表 |

机构原始记录：[Cambridge Apollo](https://www.repository.cam.ac.uk/items/7ed45d55-2eb1-4e97-938d-393f3d594781)，仓库 DOI [10.17863/CAM.50786](https://doi.org/10.17863/CAM.50786)。只下载标为 Published version 的公开文件；页面标接受稿 embargo 至 2100，本轮未请求该受限文件内容。

## 2. Fig4b 实际是什么数据

Fig4b 不是散点原始重复，也不是每细胞轨迹；它是 **3 个处理 panel × 2 个群体状态的均值时序柱图**：

- 处理 panel 自上而下：vehicle、Navitoclax **10 µM**、Nav-Gal **10 µM**。
- 每个 panel：Control 黑色，Senescent 红色；此处两者均为 A549 体系。
- x轴完整可辨：**0、4、8、12、16、20、24、28、32、36 h**。
- y轴：**Annexin V-Positive cells (%)**，0–100，每20个百分点有刻度。
- 图注：**mean±SD，n=3 生物学重复；每个生物学重复每条件另有3个技术重复**。

所以可数字化的汇总量为 **6组序列、60根均值柱及其显示的 SD**，不是60个独立实验，也不是9个独立生物学重复。连续时间点重复观察同一个条件下的实验过程，必须保留时间关联。数字化图均值以后，不能倒推出3个原始生物学观测值。

正文 Results §2.3 另外明确给出 36 h 的粗粒度概述：非衰老 A549 的 Navitoclax 组超过70%有 Annexin 信号，而同浓度 Nav-Gal 组约30%。这可用于独立核对数字化方向；它不是带完整精度的逐重复表值。

本次没有把 Fig4c–g 的克隆形成、归一化OD或药物相互作用数字混入 Annexin 端点。它们在同一张大图内，但不是同一种响应标签。

## 3. 坐标和可辨程度

坐标参考 `Fig4_embedded_original.jpg` 原生 **1880×2792**，原点左上，x向右、y向下。下表是从连续轴线识别并与已视读原图核对的**数字化校准初值**；实施数字化时应重新确认刻度中心及线宽，不能把它们当实验值。

| panel | y轴x像素初值 | 100% y像素初值 | 0% y像素初值 | 大致绘图区 |
|---|---:|---:|---:|---|
| vehicle | 1257.5 | 68 | 348 | x1238–1680，y62–356 |
| Navitoclax | 1254.5 | 457 | 746 | x1238–1690，y451–754 |
| Nav-Gal | 1255 | 849 | 1133 | x1238–1690，y843–1141 |

各 panel 的10根红色均值柱的横向中心也在 JSON 中登记，目的是帮助逐根定位；未由其高度计算响应百分比。

- 时间标签、处理名称、颜色图例与 0–100%刻度均清楚。
- 主要均值柱可定位。当前绘图区高约280–290像素，1像素约0.35个百分点；这是分辨率比例，不是最终数字化不确定度。
- vehicle和早期低值柱短，SD较小；高值衰老组接近100%上界时，均值、误差条与轴界可能接近，不能声称可无误差复原。
- 若后续抽取，需将 `value_origin` 标为 `digitized_published_mean`；分别记录SD是数字化、不可辨还是未抽取。保留图文件hash、panel、时间、像素坐标及误差界限，不填写伪生物重复。

## 4. 全部补充及公开记录的检查范围

Wiley 正文及 primary XML 均列 **7个PDF**，对应关系已保存 `supplement_manifest.csv`：Figure S1–S6，及 Legends。未列独立 XLSX/CSV、Prism 工程或 FCS 文件。

正文明确把 A549 补充 apoptosis 图指向 **S3A**，其余 S3B–D 为 melanoma 模型。S3E/F 是序贯处理的克隆形成，不能替代 Annexin。实验条件的进一步判定由本轮另一个只读原文核验任务负责，本文件不替其填补未知培养细节。

附件读取受限，不能声称已经看过全部附件内容：

- Europe PMC `supplementaryFiles` 整包第一次读取超时，第二次返回HTTP500，未取得有效ZIP。
- Wiley 正文可经网页检索工具读取，但直接文件GET返回403；S3与Legends下载入口也返回403。
- Europe PMC 的 S3/Legends 独立文件URL返回403。
- NCBI OA package接口旧域名路径返回404、新PMC路径返回403；未取得其tar包。
- 最后对 PMC `articles/instance/7189993/bin/` 的 S3 和 Legends 直接请求亦均为403（134字节拒绝响应），记录在 `PMC_direct_attachment_attempts.json`；没有将其保存为PDF。
- Wiley Fig4 PowerPoint入口经网页工具显示二进制类型但无法返回文件，直接GET返回403。未将失败的HTML存为假PPT/PDF。
- 剩余 S1/S2/S4/S5/S6 已核对清单；因整包与相同下载入口持续失败，没有重复对每个文件进行同样的失败请求。不能据此判断这些PDF内部绝对不含数字。

公开记录定向检索：

| 入口 | 结果 | 结论边界 |
|---|---|---|
| DataCite，related identifier精确查询 `10.1111/acel.13142` | API返回0条，响应已保存 | 只说明该查询未返回记录，不等于所有仓库都没有数据 |
| Zenodo，精确DOI全文查询 | API返回0条，响应已保存 | 同上 |
| Cambridge Apollo 原始bundle/bitstream清单 | 发表版、接受稿两个PDF | 本条仓库记录未列实验数据文件 |
| DOI/完整题目加 source data、dataset、figshare、repository 等定向搜索 | 定位到论文、机构文章副本及引用它的其它论文 | 未把他文source data、同名药物的其它细胞模型混作本文数据 |
| 原文 Data availability | 数据向通讯作者索取 | 本轮未联系作者；这个声明不抹去已经公开在图中的均值信息 |

## 5. 当前交接判断

**已获得足以进行定量数字化的官方发表版汇总图来源；尚未获得作者逐重复原始读数/原始数值表。** 两者应分开记录。可将图均值纳入有清晰误差与来源等级的有限比较，不能用“原始表未取得”直接否决整个实验数据；也不能将数字化均值升级成原始逐孔记录。

本任务按父代理要求先交原图、坐标、来源层级与访问边界，未提前生成60个估计响应值，未加入训练表。复算图提取与几何登记可运行 `extract_figure4_source.py`。
