# 6k 完整立体核验：修订 3

日期：2026-10-10。**12 个中心已全部审查，11 个可唯一核对，6k 因 C4 图式歧义继续 hold。** 反证校验发现 C4 的两条相对楔键不能同时给出唯一配置：只用向上的 C24 实楔得到 R，只用向下的 C23 虚楔得到 S，两条同时输入 RDKit 被判歧义。原图文摘要中的具体 6k 也保留这对冲突符号。先前草稿的“12/12 通过、可放行”已在冻结前撤回。C17 则已由新取得的原图文摘要中的产物实楔直接确认，不再只依赖 AA 连续性推断。没有修改上游 identity、condition 或训练表。

旧 ChEMBL 缺 C11 羰基的结构仍被拒绝。之前“C20 甲基缺项”的误判已在修订 2 撤回，本版继续保留该更正。

## 每个中心的实际证据

索引为原 `6k_C11_oxo_candidate_indexed.mol` 的 0-based 原子索引。方向指从中心指向所列取代基；实楔为朝向观察者，虚楔为背向观察者。R/S 是在含 C11 羰基的最终候选连接图上重新计算的 CIP 结果。

| 中心 | 原子索引 | 原图取代基 | 楔键与平面方向 | 来源 | 重建 CIP |
|---|---:|---|---|---|---|
| C2 | 34 | OH | 虚楔、左上 | 最终 6 | R |
| C3 | 36 | OH | 实楔、左下 | 最终 6 | R |
| C4 | 38 | CH₃-24；CH₂OH-23 | 向上的实楔；向下的虚楔 | 最终 6，两条分别得到相反构型；同时读入报歧义 | **R / S 冲突** |
| C5 | 42 | H | 虚楔、向下 | 最终 6 | R |
| C8 | 45 | CH₃-26 | 实楔、向上 | 最终 6；AA 编号定位 | R |
| C9 | 30 | H | 虚楔、向下 | 最终 6 | R |
| C10 | 31 | CH₃-25 | 实楔、向上 | 最终 6；AA 编号定位 | S |
| C14 | 25 | CH₃-27 | 虚楔、向下 | 最终 6；AA 编号定位 | S |
| C17 | 6 | C28 羰基碳 | 实楔、右下 | **原图文摘要中标为 6k 的产物图**；AA 相同取向提供交叉支持 | S |
| C18 | 47 | H | 实楔、向下 | 最终 6 | S |
| C19 | 1 | CH₃-30 | 实楔、左上 | 最终 6 | S |
| C20 | 2 | CH₃-29 | 虚楔、向上 | 最终 6 | R |

`6k_stereo_audit.csv` 另含楔键终点索引、各相邻键的二维方向、候选 CIP、重建 CIP 和逐中心是否一致。`stereo_source_annotations.json` 保存人工读图的输入，脚本中没有把预期 R/S 作为重建输入。

## C17 已取得产物图直接证据

Scheme 1 中最终 generic 6 的 C17 四条键没有单独标楔，编号 AA 图显示 C17→C28 的右下实楔。因此最初需要参考合成连续性。随后从原 PMC 摘要取得 `raw/PMC6484948_graphical_abstract.jpg`：其中的化学图明确标为 **6k**，且直接画出 C17→C28 的右下实楔。本版据此完成 C17 的直接产物图核验。

原文 Scheme 1、§Synthesis `sec17` 和 `sec26` 给出从 AA 到 6 的路线。合成上下文作为交叉支持保留，完整的直接英文合成段落导出在 `primary_synthesis_sections.md`。它没有加入审查者的机制解释，连原文显示的异常用量 “150 g” 也没有擅自改写。

C17 的后续直接证据不能替 C4 消除歧义。名义结构核验也不等于新做了手性纯度实验，不证明历史样品的 lot、纯度、对映体含量或现场材料。

## C4 的具体未解决点

Scheme 1 和图文摘要中，C4 的两条普通环键位于左上与右上，实楔 C24 位于二者之间，虚楔 C23 位于下方。它们在中心周围形成普通键与楔键交错的排列。单独读取不同楔键会得到相反的 R/S；同时读取时 RDKit 报 `opposing bonds have opposite wedging` 并不指定该中心。

这个判断另有图形规范支持：[IUPAC 2006《Graphical representation of stereochemical configuration》ST-1.1.10](https://iupac.qmul.ac.uk/drawing/stereo.html) 将这种两普通键、一实楔、一虚楔且普通键不相邻的排列列为形式歧义；本地 `raw/ST1110b.gif` 可与原 C4 图直接比较。故不能根据哪一条恰好匹配 ChEMBL 来选择配置。

主文 General (`sec16`) 对起始 AA 的供应方仅写 Biological Technology of Wuhan, China，没有 catalog/CAS/lot，无法从该文字唯一追溯用于解除 C4 歧义的原厂结构记录。当前具体缺口是 **C4 的无歧义第一手结构说明或可追溯起始物构型依据**；它不影响已经确定的 C11 羰基修正、其他 11 个中心，也不能放大成整套研究数据不可行。

## 重建与验证

1. 从已经逐连接核对并修正 C11=O 的分子图开始，移除全部原手性标签、楔键信息和坐标；断言没有留下任何已指定中心。
2. 对每个中心分别建立仅含其局部平面方向和一条原图楔键的二维邻域，由 RDKit 赋予手性；C5/C9/C18 的画出 H 显式保留。C19 的键方向从中心重新排序，避免受旧 SMILES 遍历影响。
3. 将各中心独立得到的标签合入同一个去手性骨架，移除辅助显式 H，计算完整 isomeric SMILES、InChIKey、分子式和精确质量。
4. C4 的另一条虚楔独立给出相反结果；同时输入两条时 RDKit 不赋值。不能挑选与候选匹配的一条。下面的完整 SMILES 仅是采用 C24 实楔分支的结果，不能视为唯一被原图确定的结构。其余中心仍可单独核对；C17 必须保留合成连续性证据的性质。

暂定分支 A（只使用 C24 实楔；**尚未通过**）：

```text
C[C@H]1[C@H](C)CC[C@]2(C(=O)NCc3cn(-c4ccccc4F)nn3)CC[C@]3(C)C(=CC(=O)[C@@H]4[C@@]5(C)C[C@@H](O)[C@H](O)[C@@](C)(CO)[C@@H]5CC[C@]43C)[C@H]12
```

- 完整 InChIKey：`RAQBGKSMFIKFIP-DGKPIFQKSA-N`
- 分子式：`C39H53FN4O5`
- 计算 `[M+H]⁺`：`677.40727548`；原文计算 `677.4073`，实测 `677.4078`
- C4 存在具体图式歧义；其余 10 个 Scheme 最终产物直接中心 + 1 个图文摘要产物中心可核对。
- `eligible_for_structural_features=false`，`formal_training_ready=false`。本版作为完成的审查记录替代修订 2 的“尚未逐位审查”，不放行 6k。

“骨架取向保留”不能理解为反应前后的 R/S 字母必定相同。CIP 在修正 C11 羰基之后按当前连接图重新计算。

## 下游文件与范围

- `chemical_candidates.csv`：继续保留 3 行及修订 2 的同一组列；仅 1a 可用，6k 和 8h 均 hold。
- `review.json`：明确记录 C4 的图式歧义，不以单一匹配分支放行。
- `6k_branch_A_C4_R`、`6k_branch_B_C4_S` 的 `.mol/.smi`：两条楔键分别导出的完整结构分支；A 匹配修正候选仍不构成唯一性证明。
- `6k_primary_C4_unspecified.mol/.smi`：保守保留 C4 未指定的结构表示，其他 11 个中心保持已核配置。
- `6k_stereo_audit.csv`、`stereo_source_annotations.json`：逐中心原图依据及可复算输入。
- `primary_stereo_viewer.html`：放大显示同一份原始图片；没有修改原图像素。
- `primary_synthesis_sections.md`：`sec17/sec26` 的完整直接英文合成段落，附来源和日期。
- `validation.json`、`manifest.json`：执行结果和内容哈希。manifest 采用 `input_base/inputs_sha256/outputs_sha256`，直接绑定 1a Chart 1、8h Scheme 2、6k Scheme 1、6k 图文摘要、IUPAC 规范及合成段落导出。

从项目根运行：

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chemical_round3/revision-3/reconstruct_primary_stereo.py'
```

脚本只写本修订目录，不训练，不修改主表。不同来源的化学身份对应不等于表达样本与凋亡样本的一一配对。

原文定位：[Scheme 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sch1)、[Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#tab1)、[合成步骤 1→5](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec17)、[5→6](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec26)、[6k 表征](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#sec37)。
