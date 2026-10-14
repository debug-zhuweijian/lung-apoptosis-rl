# HM97662 A549 Fig3A 发表组均值（近似读图）

2026-10-11，从现存官方原图和论文XML实际恢复两个发表组观测。没有精确数表或矢量PDF；数值均标明近似，不恢复SD或三个原始重复。

| 源条件 | Annexin V阳性均值（% total，近似） | 像素读图区间（%） | RNA条件成员 |
|---|---:|---:|---|
| A549 NT，同期144h | 15.9 | 14.7–17.1 | GSM9774923、24、25 |
| A549 HM97662 1µM/144h | 14.0 | 12.8–15.2 | GSM9774926、27、28 |

出处：[论文](https://doi.org/10.1016/j.neo.2026.101349)，PMC13476540 Fig3A右上量化柱图中第二组A549；para0016规定flow分析前6天处理，para0019规定RNA的1µM/6天及每3天换液。Fig3caption定义Annexin V-positive，3独立实验mean±SD；A549比较为ns。ns不证明零效应，也不根据近似均值宣称药物降低凋亡。左侧代表性FACS象限例图未作为组均值。

原图731×427像素，未修改。零基原尺寸坐标：纵轴(144,0)、(123,10)、(103,20)、(82,30)、(61,40)、(40,50)，最小二乘标定；NT柱顶y111、HM柱顶y115。人工定位、宽8/10暗像素水平线检测、避开中心dots/errorbar的双边缘条检查，三者一致。采用柱顶±1.5px与轴标定±1px相加的保守读图包络；此区间仅表示图像分辨率/读数不确定性，不是生物CI、SD或显著性区间。点未数字化为原始重复。不同检测检查共享同一图，不能消除源图低分辨率误差。

`source_observations.json/tsv`仅2行、每条件weight=1；`candidate_to_RNA_group_mapping.json`连接两个RNA组三成员，未复制Y到六RNA行。GSM/source_column实际对RNA样本清单、各GSM元数据及六列表头核对。属于同论文/细胞/剂量/终点时钟的条件桥，没有Annexin exp1→RNA rep1连接，也没有逐细胞配对。

未知字段保持JSON null：flow换液、浮游细胞收集、well、batch、逐重复关系、SD、原始重复。RNA对照是同期NT而非t0。材料Hanmi、DMSO溶解（原文材料段）；载体比例未知。RNA每72h换液不能填为flow协议；完整物理协议等价未验证。

只有一个活性drug/dose/duration组合，不能验证状态增量或完整候选排序。learning_role=unassigned；model/clinical/full_A_B_C_D资格均false；未进入旧fit/selection/final、未训练/推理、未修改任何冻结角色或核心Y。

定向精确表/矢量来源检查：XML列mmc2.xlsx(10,965bytes)、main.pdf(6,940,629bytes)、mmc1.docx(75,157,845bytes)。官方CDN两小文件404，PMC instance两链接403，EuropePMC PDF403，见实验access_log.json；未获得补表内容，不能断言补表没有均值。75MB DOCX未下载。最终使用已在本地的官方Fig3 JPG；精确表/矢量恢复仍未解决。

重跑：`F:/anaconda/envs/study/python.exe experiments/20261011-data-alignment-round20/hm97662_annexin/extract_annexin.py`。实际10检查通过；validation.json记录核查，manifest.json记录源SHA256和派生文件SHA256，digitization_calibration.json保存全部坐标。原文件保留，数据不入Git/LFS。
