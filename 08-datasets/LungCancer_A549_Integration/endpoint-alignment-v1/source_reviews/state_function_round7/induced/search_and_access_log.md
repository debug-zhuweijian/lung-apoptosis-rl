# Round 7 induced 检索与访问记录

日期：2026-10-10；任务：补充诱导 A549 状态与后续凋亡功能连接。优先两个新候选，针对明确缺口补一轮，不作系统综述或“全网不存在”判断。

## 入口与执行情况

1. SciSpace 自然语言语义检索已成功：

   `Which primary studies induce senescence or drug-tolerant persister states in A549 lung adenocarcinoma cells, publicly deposit RNA sequencing or transcriptomic profiles of those states, and then compare at least two small-molecule treatments or doses using Annexin V, cleaved caspase, or cleaved PARP after rechallenge?`

   工具返回 10 条，含一个 DOI 重复。该结果仅用于发现；本轮没有把搜索摘要当成协议证据。原始完整响应未落盘，不伪造精确去重审查总数。

2. Sider Scholar PubMed 查询尝试：接口返回 UNAUTHORIZED/需要重新认证，未取得文献结果。错误原响应保留为 `r7discovery0.json`。不把这个失败当作 PubMed 无论文，也未要求用户为此认证。

3. Web 定向检索 + Europe PMC 官方全文/补充 + NCBI GEO 官方元数据构成互补入口。Web 返回原文本保留为 `r7discovery1.json`、`r7candidate_search.json`、`r7gapsearch.json`；这些是返回记录，不是结构化结果数据库。精读判断以实际 XML、原图、补充 PDF 与 SOFT/源表为准。

4. 缺口定向补查：Schepers 的 SBI0206965/ABT263 图4、RNA accession 与补充文件；Raffi 的 GSE279184 六个样本、Fig3B 原图、补图 S3–S5。官方 GDS ESearch 请求及结果见下表。

5. 最后只核对一条指定原文引文：Wang2022 cFLIP/DR5。经 DOI、题名和已有源表比对为第一轮已审论文，无新增功能块，不继续展开引文树。

## 官方 GEO 窄查询

| 查询 | 保存响应 | 实际结果 | 范围限制 |
|---|---|---|---|
| SBI0206965 AND A549 | GEO_search_1.json | count=0 | 未命中此术语组合，不是不存在记录 |
| 34158393[PMID] | GEO_search_2.json | count=0 | GDS 查询转换为 All Fields，不能当完整 PMID 交叉库映射 |
| MCR-21-0146 | GEO_search_3.json | count=0 | DOI 后缀窄查询 |
| ATG9A AND A549 | GEO_search_4.json | count=0 | 未命中此术语组合 |

已实际取得 GSE279184 的 series self SOFT、GSM8563730–35 六份 SOFT、官方 Gene_counts gzip；完整内容通过读取验证。没有下载 FASTQ/整库。

## 官方文件访问

| 文件/端点 | 结果 |
|---|---|
| Europe PMC PMC7611779/fullTextXML | 成功，本地 PMC7611779.xml 与 source.json |
| Europe PMC PMC7611779/supplementaryFiles | 成功，715,008 字节；8 个主图 JPG/GIF；没有 publisher S1–S5 PDF |
| AACR 出版商主文/补充入口 | 页面搜索索引可读；直接访问受限，S1–S5 PDF 未取得。没有声称读完 |
| Europe PMC PMC11670111/fullTextXML | 成功，本地 PMC11670111.xml 与 source.json |
| Europe PMC PMC11670111/supplementaryFiles | 成功，10,104,426 字节；提取 pnas.2417724121.sapp.pdf 与 Fig3 官方图 |
| Amsterdam UMC 的 PNAS PDF | HTTP403，未保存目标 PDF；由 PMC XML、官方补充和原图完成本轮相关核验 |
| GEO GSE279184 官方 FTP Gene_counts gzip | 成功，947,655 字节；16,670×17 源表实际读取 |

失败记录见 `download_log_initial.json`；不是所有链接均成功。下载使用公开 GET，没有联系作者或绕过认证。

## 旧候选去重及不升级原因

- Nav-Gal：已在 round4 核验，功能数字化由主代理保存；本轮不重做。
- Sarin/GSE108214：已在 round2 核验，本轮不重新算作新状态功能来源。
- Nature Aging donor/recipient、SCAT7 等：已读 round5 结论，避免把不同细胞/不同处理表达强配为功能初态。
- Deryabin2021/PMC8629786：第一轮已审；etoposide 诱导/恢复与借用 RNA 协议不一致，不能由重复检索转为匹配。
- PMC11889205 mitochondrial priming：此前已审，alamarBlue 与 BH3/JC1 不能升级为后续 Annexin 标签。
- Wang2022 cFLIP/DR5：原始源表及公开状态表达已取得；本轮最后引文核对仍为同 DOI，保留蛋白/抗体动作与 RNA 协议差异。
- BET/DTP 的 Cell Reports Medicine 2024 DOI 10.1016/j.xcrm.2024.101471 在发现中出现，但本轮未确认 A549 及完整协议，未列入已通过候选。

## 当前停止界限

本轮正式精读两个新候选，补查其核心公开材料，并完成一个指定引文的去重核验。没有获得新的、可直接支持多状态多小分子排序与 RNA 增量检验的完整定量块。保留局部可用证据，不以“完整块仍缺”否定已取得的单药状态反应和参考表达；也不以这些局部证据宣称主目标的数据对齐已经解决。
