# r4 终点扩展与统计口径独立复审

日期：2026-10-10  
对象：extend_endpoint_alignment_round2.py、test_endpoint_round2.py、curated-20261010-r4 相对 r3、condition-inputs-v4、coverage-v3。  
执行边界：其他文件只读；只新增本审查文档。未重新构建数据，未改冻结 manifest、coverage 脚本或已有审查，未训练。

## Technical Review

**结论：新增 12 个来源条件的数值转录、真实处理时长、单面板重复口径和排除状态核查通过；其中仅 8 个进入可用条件汇总。38 个可用 Annexin 条件与 14 个当前 24 h 化学输入候选的计数均正确。**

本轮发现一项非阻断的适用域标注缺口：6k 的 10/20 μM 条件尚未明确标记为相对现有 Tahoe 剂量范围的外推。它们目前已被化学身份未解决这一条件挡住，没有错误进入 14 个候选。该标注建议另版补充或写入治理说明，不覆盖 r4。

### Errors (incorrect content)

**未发现改变本轮数值对齐结论的错误。**

- [extend_endpoint_alignment_round2.py:22](../scripts/extend_endpoint_alignment_round2.py) 指定四象限顺序为 double negative、dye-only、Annexin-only、double positive，并在第 23 行仅相加右侧 Annexin 阳性两象限。这与亲看的三张原图横纵轴一致。
- [同文件:37](../scripts/extend_endpoint_alignment_round2.py) 将 Li2012 AS-E 记录为 48 h；[同文件:47](../scripts/extend_endpoint_alignment_round2.py) 将名为“8h”的化合物记录为 48 h，而不是将化合物编号当成暴露时长。
- [同文件:52](../scripts/extend_endpoint_alignment_round2.py) 的 supplement_assay_method_not_verified 已实际进入 8h 的四条 observation.exclusion_reason，没有只写在注释里。
- [同文件:88](../scripts/extend_endpoint_alignment_round2.py) 覆盖 builder 默认重复类型，实际保存为 published_value_not_raw_replicate；新增可用汇总的 measured_replicates 全为 0，表示未取得原始独立重复序列，不表示实验没有做过重复。

### Rigor Issues (needs strengthening)

**[非阻断] [extend_endpoint_alignment_round2.py:82](../scripts/extend_endpoint_alignment_round2.py)：6k 的剂量外推身份尚未显式落入请求表。**

本轮亲读现有 [training-v1/conditions.parquet](../08-datasets/LungCancer_A549_Integration/training-v1/conditions.parquet)，其 A549 dose_um 只有 0.05、0.5、5 μM；新 6k 条件包含 5、10、20 μM。r4 和 condition-inputs-v4 中，6k 的 response_domain 均为 NaN；当前请求阻塞仅记录 unresolved_chemical_identity。10/20 μM 因而缺少与 Mphahlele 高剂量条件相同意义的外推标记。

建议在后续版本或补充治理说明中明确 10/20 μM 属于相对当前 P 训练剂量域的外推，5 μM 仅在数值范围内，不代表该新化合物响应已经有监督。**这不是要求将所有外推研究一律排除，而是要求把待验证的外推与范围内预测分开报告。** 当前化学身份阻塞仍有效，故不影响 r4 数值入表或 coverage-v3 的 14 候选计数。

另外必须保留以下解释限制：

- 一个发表面板的数值不能作为独立生物重复序列，也不能通过复制细胞或生成样本扩充独立 n。
- Annexin 阳性两象限的合计是所报告流式门内的检测比例，不是相对初始接种细胞数的死亡概率。6k 原文对右上象限包含晚期凋亡或坏死的解释，不能在汇报时改写成单一机制的确定比例。
- method_complete_for_intake=true 表示当前阶段足以登记该观测，并非证明完整培养/收集/门控细节均已取得。6k 与 AS-E 的完整前置门控及所有重复设计仍有限。
- 38 不是“38 个可训练处理样本”；它含 9 个对照、29 个非对照条件，横跨检测定义及时间范围。

### Notation Issues

建议在面向用户的汇报中持续区分：

- “化合物 8h”与“8 小时”；本记录 Annexin 为 48 h。
- “early Annexin+/PI−”与“total Annexin-positive”；本次 G 标签采用后者。
- “可用条件汇总”“当前化学输入候选”“完整训练资格”；三者不能都简称为训练数据量。
- measured_replicates=0 的含义是独立实验原值未取得，不是原文明确报告零次实验。

### Citation Gaps

数值已回到原图核对；时间与方法分别核对 Li2012 的原始 XML，以及普通 Edge 浏览器中成功显示的 6k、8h 原文。8h 的补充方法本轮仍未核查，因此保留 hold，未将主文图注扩写成完整方法已验证。

直接网页提取工具对两个 PMC 页面返回检查页、出版商页面返回 403；随后通过普通浏览器读取到公开主文。没有绕过 CAPTCHA 或下载/改写原始附件。

## 1. 原图与数值复算

下表使用图中打印值，均为百分比；保存时除以 100 得 fraction。这里只报告检测定义，未增加生物学机制结论。

| 来源/时间 | 剂量 μM | Annexin-only | 双阳性 | Annexin 两象限合计 % | r4 状态 |
|---|---:|---:|---:|---:|---|
| 6k，24 h | 对照 | 1.57 | 0.010 | 1.580 | 可用面板 |
| 6k，24 h | 5 | 9.25 | 2.55 | 11.80 | 可用面板；化学身份待审 |
| 6k，24 h | 10 | 26.5 | 2.92 | 29.42 | 可用面板；化学待审，剂量外推待标注 |
| 6k，24 h | 20 | 36.0 | 4.38 | 40.38 | 同上 |
| AS-E / 1a，48 h | DMSO | 3.9 | 2.98 | 6.88 | 可用面板 |
| AS-E / 1a，48 h | 5 | 6.5 | 3.06 | 9.56 | 可用面板；非当前 24 h 生成域 |
| AS-E / 1a，48 h | 7.5 | 10.1 | 2.9 | 13.00 | 同上 |
| AS-E / 1a，48 h | 10 | 19.4 | 4.2 | 23.60 | 同上 |
| 8h，48 h，A549 | 对照 | 3.26 | 2.94 | 6.20 | hold |
| 8h，48 h，A549 | 0.3 | 4.92 | 6.92 | 11.84 | hold |
| 8h，48 h，A549 | 1 | 8.03 | 9.40 | 17.43 | hold |
| 8h，48 h，A549 | 3 | 3.91 | 93.73 | 97.64 | hold |

亲看原图：[6k Fig. 6](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chembl_round2/raw/PMC6484948_fig6.jpg)、[AS-E Fig. 3](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chembl_round2/raw/PMC3587690_fig3.jpg)、[8h Fig. 4](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chembl_round2/raw/PMC7489057_fig4.jpg)。

四象限完整打印值逐条与脚本常量匹配。6k 对照的四象限总和为 100.040%，允许的打印舍入上界为 0.0605 个百分点，当前校验接受它合理；代码没有强行将四象限归一化到 100%。这里的舍入容差不是生物实验置信区间。

### 6k

原文 Results “Compound 6k induces apoptosis in A549 cells”、Fig. 6 图注、Methods “Apoptosis analysis”均明确 24 h，且正文总值 1.58%、11.80%、29.42%、40.38% 与图中右侧两象限一致。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6484948/#fig6)

源码保留原文分子式与二次结构不一致的化学限制。本轮直接读到原文 HRMS 的 C39H53FN4O5、677.4073/677.4078，但未重建完整结构或批准指纹；仅确认当前 unresolved_chemical_identity 没有被绕过。

### AS-E / 1a

原文 §2.3.2 讨论的 19.4% 是 Annexin+/PI− 早期部分；Fig. 3 还报告右上 4.2%。因此本次计算 total Annexin-positive=23.6% 正确，不应把正文 early 数值直接当总值。

原文 §4.2.3 Flow cytometry 与 Fig. 3 图注均为 48 h；图内对照标明 DMSO。读取器只保存绝对门内比例，本轮没有做对照扣除或把 DMSO 复制成药物标签。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3587690/#F3)、[本地 XML](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/chembl_round2/raw/PMC3587690.xml)

### 8h

Fig. 4A 的上排是 A549，下排是 H1975，本次只提取上排。亲读原文 Fig. 4 图注确认：A 的 Annexin V-FITC/7-AAD 为 48 h；B 的 Western 为 24 h。不能把 B 的时间移给 A。

四条观测数值本身已登记，但因 supplement_assay_method_not_verified 被排除。实际 condition_units 无这四条件，输入请求同时保留 no_eligible_measured_endpoint 和 time_not_supported_by_current_Tahoe24h_generator；处理条件另有化学身份未解决标记。[原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7489057/#fig4)

## 2. r4 相对 r3 的实际增量

| 统计对象 | r3 | r4 | 增量/解释 |
|---|---:|---:|---|
| 条件台账 | 292 | 304 | +12，三来源各四条件，含各自对照 |
| 观测源位置 | 830 | 842 | +12，每个新增条件一个发表面板 |
| 有数值观测 | 828 | 840 | +12；数值存在不等于可用 |
| 排除/缺失标记记录 | 12 | 16 | +4，8h 的补充方法 hold |
| 可用全部终点条件汇总 | 288 | 296 | +8 |
| 可用 Annexin 条件汇总 | 30 | 38 | +8=6k 四条件+AS-E 四条件；其中新增六处理、两对照 |
| 当前单药 24 h Annexin 化学输入候选 | 14 | 14 | 三项新增药物均未完成可用结构连接；AS-E/8h 另为 48 h |
| 当前候选结构身份组 | 13 | 13 | 未变 |
| 条件输入中的来源化合物条目 | 279 | 282 | +3 来源化合物 ID |
| 可用结构条目 | 50 | 50 | 未变 |
| 有化学输入的全部条件 | 53 | 53 | 未变 |

38 条 Annexin 条件具体为：24 h 非对照 25、48 h 非对照 4、24 h 对照 7、48 h 对照 2。它们不能组成一个默认可比较的统一排行榜，也不等于完成了相应模型特征生成。

当前观察初态表达条件仍为 0，已完成表达特征仍为 0；formal_endpoint_training_ready 和 full_research_data_feasibility_verified 均保留 false。计数扩展没有补造 RNA 配对或初态监督。

已直接以 coverage-v3/condition_inventory.json 调用 summarize，结果与 coverage.json 完全一致。[覆盖报告](../08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/coverage-v3/coverage.json)

## 3. 重复、来源绑定及旧版保留

本轮通过以下实际检查：

1. 分别调用 source-bound load_alignment 读取 r3、r4；表哈希、来源哈希及行级来源绑定均通过。r4 登记的 21 个来源文件全部核验。
2. 按 condition_id 对齐比较，r3 的 292 条条件全部在 r4 原值保留；按 observation_id 对齐，830 条旧观测原值保留。
3. r3 的 287 条干预记录和 8 个旧协议完整保留。新增打印象限字段没有改写旧值。
4. condition-inputs-v4、coverage-v3 的产物及输入依赖哈希均无不一致。
5. 279 个既有来源化合物的指纹按 compound_id 对齐后，数值逐项一致；新表增加三行不造成既有 fingerprint_row 错配。
6. 新增八个可用条件均为 available_numeric_records=1、measured_replicates=0、sd_fraction=NaN；没有补造 SD 或生物重复。
7. write_alignment 在既有 manifest 存在时拒绝覆盖；扩展脚本将 r3 manifest 作为哈希登记来源，已有来源再次进入时会拒绝。

快照哈希：

- r3 manifest：90f3df2d881aef5704a95eafe050ab5f51c99430da17be8729ba04ab7c5f3469。
- r4 manifest：2f7fddb69d8e5ffc123118db2c6cc2f8a70b6392b6baf967acf891be302145d1。
- condition-inputs-v4 manifest：9c59a22ba2c4da20f49759956e0f7284aab68f2d8b52fd8df6430fac96855bb8。

本轮只读调用 [test_endpoint_round2.py](../tests/test_endpoint_round2.py) 的三个测试函数，全部通过；没有通过重新执行扩展脚本来覆盖或生成另一套数据。测试结果用于验证早期/总阳性区别、打印精度校验与 48 h/hold 行为，不能替代上面的原图亲读。

## 4. 科学设计判定不随计数增加自动改变

应当表述为：

> r4 增加了两个化合物各三个剂量条件及各自对照的可用发表 Annexin 数值；另保留一个 48 h 来源的数值，但补充方法未核验，仍排除。当前 24 h 化学输入候选数量没有增加，完整研究的数据可行性尚未通过。

不能表述为“新增 8 个独立药物”“已有 38 个合格训练样本”或“只因四分区凑不齐，所以任何研究都不可行”。

固定 T/V/C/E、单一观测定义，以及来源/发表图块/化学实体联合连通块，是现行接口的一套保守契约。现有连通块阻塞可说明该接口还不能生成合格训练 manifest；不构成所有来源内剂量研究、冻结模型案例检验或嵌套留组设计均不成立的证明。后者应另定协议并如实报告目标和数据限制，不能通过更换划分把缺失的初态、功能配对或独立实验信息变出来。

本轮新增同一药物的多个剂量，确实增加了研究该来源内剂量方向的资料。但每条件只有公开面板、结构仍待核验、部分剂量/时间位于当前 P 支持范围之外；仍不能证明状态信息的增量价值、跨药/跨来源泛化、联合时长选择或序贯收益。完整设计说明继续见 [科学设计复审](DATA_FEASIBILITY_DESIGN_REVIEW_2026-10-10.md)。

## 5. 最终审查范围

本轮没有发现需要撤回 r4 新增数值或改动“38 / 14”计数的证据。唯一新增意见是 6k 10/20 μM 的外推身份应显式补充。8h 的补充方法、三项新药完整结构连接、正式表达特征生成和功能验证仍未完成，均未在本报告中宣布通过。

本报告独立于冻结数据版本；如后续补充这些证据，应另建版本并再次检查对应消费者，不覆盖本次保留的 r3/r4。
