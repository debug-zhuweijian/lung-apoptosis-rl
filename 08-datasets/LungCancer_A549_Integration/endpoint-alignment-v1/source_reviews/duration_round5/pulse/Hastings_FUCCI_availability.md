# Hastings 2020 FUCCI 同细胞状态—命运：公开载体窄核验

日期：2026-10-10。仅核查已下载的官方文章/API附件目录与其 Figure5 图组；不扩大文献搜索，不把它替代 RNA 状态增量问题。

**结论：实验设计确有真实给药前 FUCCI 状态与后续同一细胞命运的纵向关系。当前取得的是作者发表的逐细胞彩色轨迹图及汇总图，没有在官方附件目录中取得逐细胞原始表或原始影像电影。**

原文：[DOI 10.7554/eLife.53367](https://doi.org/10.7554/eLife.53367)。本地原件：[PMC7282820.xml](PMC7282820.xml)、[官方API记录](eLife53367_api_record.json)。

## 1. 方法亲读定位

Methods XML `s4-10`（FUCCI live-cell imaging）说明：

- 给药之前先拍摄一张图，用于标注初始细胞周期。
- 每20分钟拍摄一次，连续72小时；人工逐细胞跟踪。
- 用核颜色标记 G1、G1/S、S/G2，另标注核膜破裂、后期分裂等事件。
- 分裂后只追踪其中一个子细胞，而不是所有子代分支。

因此能说“该实验观察了同一条细胞/选定子代跟踪线的初始周期与后续命运”，但不能说“完整谱系中所有子代均有记录”。FUCCI 是荧光报告器而非 RNA 测序；作者没有为这些逐细胞轨迹提供匹配 RNA 的证据。

## 2. 已获取的实际载体

| 原件 | 实际内容 | A549及条件 | 数据层级 |
|---|---|---|---|
| [Figure5 主图](elife-53367-fig5-v1.jpg)，3577×5502 | A逐细胞彩色周期轨迹；B各周期时间散点/分布；C命运分类百分比；D存活曲线 | A549位于左列；control n=50；cisplatin、dactolisib、二者组合各n=100 | 已发表逐细胞轨迹图与汇总图，不是可下载原始数表 |
| [Figure5-supp1](elife-53367-fig5-figsupp1-v1.jpg)，3496×2120 | aberrant mitosis、G2 exit、DBM、DAM的代表性细胞图片 | 图注为cisplatin pulse后示例 | 代表性图片，不是全部细胞原始影片 |
| [Figure5-supp2](elife-53367-fig5-figsupp2-v1.jpg)，3649×3984 | A p53 knockdown WB；B逐细胞轨迹；C周期时长；D命运分类；E存活曲线 | A549；control siRNA、p53 siRNA #1/#2，各untreated n=25、cisplatin n=50；siRNA预处理48h | 具有明确群体预处理状态的已发表轨迹图；没有对应RNA矩阵 |
| Figure5-supp3 | 官方API/XML列明P70S6K knockdown轨迹图 | NCI-H358 | 非A549，本次只记录目录与图注，未为A549问题重复下载 |

来源与 SHA256 见 [Hastings_FUCCI_download_log.json](Hastings_FUCCI_download_log.json)。主 Figure5、supp2 已直接目视；supp1 按官方目录/图注登记用途，未据其代表图推断全体命运。

## 3. 动作及结局的已知/未知

- Fig5图注：cisplatin 5 µg/mL、2h pulse，dactolisib 1 µM，含单药和联合条件。**dactolisib 是否随cisplatin一起撤除及其完整维持窗口，在本次窄核验中未解决**；不能自动写成另一种确定的时序方案。
- Fig5C 的死亡类别为 DBM（分裂前死亡）、DAM（分裂后死亡）；另有周期停滞、G2 exit、正常等类别。
- Fig5D 表示 viable-cell proportion 随时间变化；不能把所有时间点视为独立实验重复。
- 图上 n 为跟踪细胞数，不是独立生物重复数。透明报告把 Figure5 单细胞分析列在一般 biological-triplicate 声明的例外内。
- 彩色轨迹的白色尾段没有在现有图例中明确逐条解释为死亡、丢失跟踪还是其他终止。未经额外验证不能把每条截断轨迹都标成死亡，并自行恢复“原始”生存标签。
- 不同颜色的左端允许有限读取初始 FUCCI 类别；早G1/晚G1等更细区分不应只凭红色同一颜色自动恢复。正文的组间机制解释不能替代缺失的逐细胞数表。
- 作者在正文 `s2-6` 使用 apoptosis 描述部分事件，但这批图的逐细胞指标不是 Annexin，也没有每个细胞配套 caspase assay；保守标为作者判定的命运/死亡辅助终点。

## 4. 附件可得性核验方法与边界

已读取全部 PMC XML `media`/`supplementary-material` 项及 eLife API source-data/additionalFiles。可枚举的数值附件是 Figure1D 的 XLSX，另有 supplementary1–5 DOCX 与透明报告；API/XML未列出 Figure5逐细胞CSV、XLSX、FCS、影片或压缩包。透明报告最后一节仅登记 Figure1D source data。

该结论限定于本次实际核查的官方发表载体，不断言作者手中不存在原始轨迹，不把公开图否定为完全没有数据。图像有可能支持未来明确标为数字化的有限逐细胞分析，但需先解决轨迹行分割、时间像素标定、终止含义和人工复核，不凭图像表象宣布已经得到可用原始训练集。

## 对主问题的贡献

它直接证明在本研究体系中“某个细胞给药前可观察状态→该细胞后续命运”的实验连接可以真实建立；A549 p53敲低图还提供有来源的群体预处理差异。当前公开形态支持辅助挑战与方法边界分析，**仍不足以证明现有 RNA 表征能预测这些命运、RNA状态增量成立或RL具有独立生物学收益**。
