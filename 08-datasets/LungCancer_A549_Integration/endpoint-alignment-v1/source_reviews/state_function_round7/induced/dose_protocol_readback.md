# Schepers 2021 Fig4C 剂量与时长窄复核

日期：2026-10-10。来源：PMC7611779，DOI 10.1158/1541-7786.MCR-21-0146。已亲读本目录 `PMC7611779.xml` 的全部 Methods（`sec2`），Fig4 图注及官方原图 `1613fig4.jpg`。

| 字段 | 依据 | 登记判定 |
|---|---|---|
| Fig4C ABT263 后时长 24 h | `fig4` caption C 直接说明 | direct_caption，可用 |
| Fig4C SBI0206965 5 µM | C 第三列标题直接印值 | direct_panel，可用 |
| Fig4C ABT263 浓度 | C 未印；完整 Methods 未指定通用浓度 | unknown_in_read_maintext，留空 |
| Fig4B ABT263 2 µM | B 第二列标题 | direct_panel，只属于 B |
| Fig4B SBI 2.5 µM × 4 d | B 行标题与 B 图注 | direct_panel_and_caption，只属于 B |
| Fig4C 诱导天数 | C 图注仅说 pretreated；未直接给天数 | unknown_in_read_maintext，留空 |
| Fig4C 撤药/维持处理 | 已读主文没有细化 | unknown，不能从 “+” 号补全 |

Methods `sec2-5` 是适用于 IncuCyte 的通用方法：按 indicated concentrations 加药，每 4 h 成像，Caspase-3/7 reagent #4440，百分比基于 biological triplicates 图像。它没有为 ABT263 指派一个统一剂量。全部 `sec2` 内 ABT263 唯一出现位置是 `sec2-10` 的供应商条目；AP20187/CID 的 10 nM 是另一种试剂，不能移用于 ABT263。

因此 **2 µM 暂不登记成 Fig4C protocol_inferred dose**。这不是“图未印数值便一律拒绝”的规则；若以后找到明确适用于 C 的通用 senolytic 协议，可以据协议新增推定层级。本轮出版商 Supplementary Figures 1–5 PDF 未取得，保留这一检索盲区。Europe PMC 附件 ZIP 仅含主图 JPG/GIF，不能当完整补充材料。

图注描述 before and 24 hours after addition。Control/SBI 两张图的 24 h vehicle 身份未确认；印值可登记，但不能未经确认组成严格时间匹配的完整 2×2 处理效应差分。方法中的 biological triplicates 也不说明四个印值各为平均值或某一代表实验，故不恢复重复或置信区间。
