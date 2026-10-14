# 化学身份第三轮：修订 2

日期：2026-10-10。本目录替代上一版对 6k 的相关判断；旧包原样保留，便于追溯。**明确撤回“最终 generic 6 缺少 C20 甲基”的结论：那是审图时漏看虚楔，不是原文或数据的缺陷。**

## 更正与原图定位

在 784×630 的原始 `../raw/PMC6484948_scheme1.jpg` 中，最终 6 的 C20 顶点位于约 `(636,484)`，其上方 `(634…639,462…483)` 清楚存在向上的虚楔短线，表示 C20→CH₃-29。像素行定位可在 `C20_hashed_wedge_pixel_rows.csv` 复核：y=462、466、470/471、475、479、483 均有短横线。

先亲读原图确认化学符号，再用像素记录固定位置；像素统计本身不承担化学图识别。**旧报告中由“甲基缺项”推出的图式矛盾、结构不完整和阻塞理由全部撤回。**

同一处立体关系也作了独立局部检查：C20 的 C19 键朝左下、C21 键朝右下，CH₃-29 以虚楔向上。清除候选原有手性标签后，以该原图二维方向重建 C20 的虚楔，RDKit 得到 C20=R，与候选吻合。计算只验证这一个中心，没有把其余手性标签的继承包装成独立核验。

## 修订后的实际状态

| 化合物 | 现在确认的内容 | 默认结构特征入口 |
|---|---|---|
| Li2012 1a | 原 Chart 1 与独立转写一致，无立体中心；`OHAXNCGNVGGWSO-UHFFFAOYSA-N` | 可用，限名义身份 |
| 8h | 原 Scheme 2 连接一致；C6 在原文未指定立体；`PCJWWMFUNQDHJU-UHFFFAOYSA-N` | 严格入口 hold；可由显式连接结构协议另行判断 |
| 6k | 原 Scheme/Table 支持完整连接；C11 必须是 C=O；C20→CH₃-29 存在且 C20=R；候选式和 HRMS 吻合 | 完整手性复核尚未完成，暂时 hold |

6k 候选没有再次变动：`C39H53FN4O5`，InChIKey **`RAQBGKSMFIKFIP-DGKPIFQKSA-N`**，计算 `[M+H]⁺=677.40727548`。旧 ChEMBL 少 C11 羰基的错误仍然成立；此更正不恢复那个错误结构。

当前 hold 的准确含义是：**12 个候选手性中心中，C20 已独立按原图确认；其余 11 个中心尚未在这次窄复核中逐一完成独立转写。** 这是剩余审查工作，不能表述成数据不可得、原图有缺陷或研究本质不可行。主文原图可能足以完成这些检查；SI 访问失败本身不作为阻塞理由。逐位状态见 `6k_stereo_audit.csv`。

## 文件与复核

- `correction.json`：撤回内容、证据坐标、原因与追溯信息。
- `review.json`、`chemical_candidates.csv`：修订后的三化合物审查与下游摘要。
- `6k_stereo_audit.csv`：12 个中心的候选 CIP 与本次独立核验范围。
- `primary_read_notes.json`：删除错误观察并写入更正后的阅读记录。
- `validation.json`、`manifest.json`：修订版检查和哈希；原报告 SHA256 为 `975d79e00612817b25440cf26506592f76f28ac4308f003b300c8a3add417fd4`。

从项目根运行 `F:/anaconda/envs/study/python.exe -X utf8 08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chemical_round3/revision-2/build_revision.py` 可复现修订计算。脚本只写本修订目录，不改原包、上游 identity/condition 表或源码，不启动训练。

原始来源：[PMC6484948 Scheme 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sch1)、[Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#tab1)、[Compound 6k 表征](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec37)。
