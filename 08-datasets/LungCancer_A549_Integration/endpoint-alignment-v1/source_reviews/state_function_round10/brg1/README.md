# BRG1 Fig6 定向原始资料核查

本轮仅核查 BRG1 预印本 `10.1101/2025.05.01.651609`，版本为 **2025-05-07**。作者公开 PDF 已由网页工具成功解析为 39 页，正文和图注已重读；本机未下载到 PDF 字节，当前代理未收到可见原图像素。没有把网页文本层当作原图视觉复核。

正式版 `10.1016/j.jpet.2025.103772` 的全文未读取，本目录结论不冒充正式版核验。

公开作者 PDF：

[BRG1 preprint author copy](https://www.researchgate.net/publication/391552844_BRG1_targeting_overcomes_ABCC-based_multidrug_resistance_induced_by_paclitaxel/fulltext/681c70b2d1054b0207ebfe29/BRG1-targeting-overcomes-ABCC-based-multidrug-resistance-induced-by-paclitaxel.pdf)

## 实际取得的证据

- `researchgate_author_text_exa.json`：作者公开页面的连接器原始响应。
- `researchgate_preprint_extracted.md`：108,186 字符的解析文本；§2.2 行 448，§2.13 行 814，§2.15 行 868，§3.6 行 1314，Fig6 图注行 2532。
- `author_pdf_web_open.json`：正确 PDF URL 的成功解析响应。PDF 第 6 页为处理方法，第 10–11 页为 Annexin 方法，第 31–32 页为 Fig6 图注，第 39 页为 Fig6 图像。
- `biorxiv_supplementary_listing_exa.json`：官方补充入口实际返回的条目，列有 `supplements/651609_file09.pdf`。这证明附件被列出，不证明本轮已取得附件内容。
- `protocol_evidence.json`：逐项已知值、缺失项及证据边界；不是可训练终点表。

## Fig6 的可确认部分与缺口

| 面板 | 已从正文/图注核定 | 尚未核定 |
| --- | --- | --- |
| Fig6A–C | 耐药球体培养 3 周；抑制剂 72 h 后加化疗 48 h；Annexin V-FITC/PI 共聚焦强度 | 面板属于 A549 还是 MDA、化疗剂量、重复单位/n、具体归一；文本层轴名与图注单位冲突尚需原图与补表 |
| Fig6D | 明确 A549-PTX、2D；PFI3 2.5 µM × 72 h；列有 doxorubicin 和 paclitaxel；材料列 cleaved-caspase-3 Asp175 抗体 #9664 | 化疗剂量、D 面板究竟用 24 还是 48 h、共同终点、定量均值/分母/n、洗脱或抑制剂保留 |

方法通则中的 24/48 h 不能给 Fig6D 任选一个值。Fig6E/F 的 resazurin 口径和时钟也不能填入 Fig6D。Fig6A–C 的 72+48=120 h 只是依图注做的名义累计时长计算，不证明它与 D 共享同一终点。

目前没有取得 Fig6D 的定量 cleaved-caspase-3 Y，不能从图示存在推导出凋亡比例，也不能把 ACBI1 复制成 PFI3 的 caspase 结果。Fig6B/C 的强度不按总细胞比例处理。

## 表达对应边界

已有四个 A549 run 的身份保持为 parental、paclitaxel-resistant、siCTRL、siBRG1，本轮没有重新查询 run 元数据，也没有将其改名为 PFI3 或 siHIF1A。若 t0 定义在 PFI3 72 h 之后，现有高维 RNA 未覆盖这一初态；同样本 RNA—功能配对未核定。该限制与原文是否具有局部功能证据分别记录。

## 访问失败与停止位置

本机普通公开请求访问作者页面和 bioRxiv 入口出现 TLS EOF；正确作者 PDF 的一次本机下载返回 403，之后未重试。网页工具的原始 PDF 点击曾重复拼接 `/publication/publication/`；正确解析已观察到的链接后，PDF 文本可读。网页截图仅返回引用，当前代理未收到可视像素；IAB 浏览器不可用。

官方正文和补充入口后续出现连接器超时；公开 API 不可访问，Sider 原生入口要求重新认证。没有改变认证、关闭 TLS 校验、绕过付费墙、猜测循环下载 URL 或联系作者。`download_attempts.jsonl` 和 `access_audit.json` 保留结果。

因此，`651609_file09.pdf` 与文中声明的 Supplementary Data 本轮仍未取得内容；不能声称它们不存在，也不能声称其中没有 Fig6 数据。下一步应只补原图像素、附件与面板剂量/时钟/量化/n；本轮不扩展检索范围。

## 复核

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 -B '08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/state_function_round10/brg1/verify_audit.py'
```

本目录未生成可训练 Y、模型预测或实验收益，没有修改核心数据、Kotani 或旧来源。
