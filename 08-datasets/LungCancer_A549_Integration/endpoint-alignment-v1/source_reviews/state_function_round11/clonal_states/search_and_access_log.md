# Round11 查询与访问记录

日期：2026-10-10。只读公开资料；写入仅 `state_function_round11/clonal_states/`。

## 初筛与去重

- 本地：按 `sister|clon|sort|CD133|CD44|ALDH|ALDEFLUOR|ABC.?G2|Sharma|Shaffer|Spencer|Emert|Flusberg|persister` 搜索 source_reviews 的 README/review/search 文档，读取 Round9/10 discovery 去重记录。
- memory registry 的 `CellCage|Pogacar|clonal|sister cell` 快速查询无匹配，没有使用 memory 作为事实证据，也没有写入 memory。
- SciSpace 问题和 10 条返回 metadata 见 `scispace_discovery.json`。摘要只作发现，SCCP 正式/预印本去重并与旧包核对。
- 网页查询：
  1. `A549 sorted subpopulations transcriptome Annexin apoptosis cisplatin ALDH`
  2. `A549 clones single cell RNA sequencing caspase apoptosis drug`
  3. `A549 sister cells transcriptome apoptosis baseline drug`
  4. `"A549" "sorted" "Annexin" "microarray"`
  5. `"A549" "subclones" "transcriptome" apoptosis`
  6. `"A549" "baseline" "ALDH" "RNA-seq" apoptosis`
  7. 精确题名 `"A549 subclones demonstrate" "G4S" pdf` 及 `"A549 subclones demonstrate" expression microarray`
  8. 主代理提供后，精确检索 `"10.1093/jmcb/mjad039"`；不重复其 HSP90 路线。

## 官方原件及读回

采用并读取 NCBI PMC skill：
`C:/Users/Windows11/.codex/plugins/cache/openai-curated-remote/life-sciences-literature/0.1.5/skills/ncbi-pmc-skill/SKILL.md`。

- PMC4632318 metadata 通过该 skill 的 `scripts/ncbi_pmc.py` 成功读取，保存 `PMC4632318_metadata_raw.json`；其官方资源支持精确 DOI/版本/URL，而不单独支持科学结论。
- 正式 XML 与 Fig1、两份 DOCX 通过 metadata 指定的 PMC Cloud HTTPS URI 成功下载。MD5 已与官方登记值比对。
- 随后 Python requests 对公开 S3 列表与 FigS2 TIFF 出现 TLS `UNEXPECTED_EOF_WHILE_READING`，ResearchGate 普通 GET 同样失败。未关闭 SSL 检查、未改认证、未绕过验证码。
- 改用 PowerShell `Invoke-WebRequest` 对官方原 URL 获取 FigS2 TIFF 成功。TIFF 为 6,474,936 bytes，官方 MD5 `bd2588202e3c9db57b3c9ff99bad1730` 比对通过。PNG 仅为同像素显示副本；源 TIFF 保留。
- PMC10847630 的 skill 调用返回 `network_error / PMC Cloud version lookup failed (SSLError)`。记录失败后，按同一官方服务文档用 PowerShell 做有限回退：PMCID 前缀版本列表只返回 `PMC10847630.1/`，再读对应 JSON；没有假定版本即结论。列表与 metadata 均保存。
- `PMC10847630.xml` 通过 Europe PMC 官方 `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10847630/fullTextXML` 获取；Fig2 与 Supplemental PDF 通过官方 metadata 精确 URL 获取，MD5 均匹配。
- Tan2023 原图已 `view_image` 亲视；补充 PDF 使用本地已有 pypdf 读取全部 10 页。正文 Results、Flow cytometry、Fig2 caption 和补充 viability/sphere/colony 方法已核对；没有从邻近 assay 借 Annexin 时间。

## Watanabe2002 访问失败边界

- `https://journals.physiology.org/doi/full/10.1152/ajplung.00025.2002` 与 `/doi/pdf/10.1152/ajplung.00025.2002`：网页工具返回内部访问错误；后者 PowerShell GET 明确 HTTP 403，停止此入口。
- 作者 ResearchGate 页面 `https://www.researchgate.net/publication/11163589_A549_subclones_demonstrate_heterogeneity_in_toxicological_sensitivity_and_antioxidant_profile`：网页工具可读公开全文文字，页面注明作者 Karen Iles 上传；普通 requests TLS 失败，PowerShell GET 403。
- 网页工具点击“Download full-text PDF”解析到含重复 `publication/publication/` 的地址，403。该失败地址没有当成有效原件，不继续猜测私有文件路径。
- 未取得其 PDF 二进制、Fig3 像素或 raw source data。已读范围明确列在 README；未标记已定量复现。

## 数据层级验证

- Seah FigS2B 9 个 R5 印值和 Tan Fig2J 6 个 early Annexin 印值逐个与原图亲视核对。
- 检查 CSV condition keys 唯一、单位和 time 字段正确：TRAIL=ng/mL、4 h；GEM=µM、暴露未知。未将 3 个独立实验声明展开为 3 份重复值。
- Seah 10,000 acquired events 是方法采集量，不用百分比反推原始计数；未补 R6/R7，也未总和归一化。
- Tan 的 5 d viability 与 Annexin 不混同；RNA 的 SW1990 与 A549 不混同；给药前分选/药后 marker assignment 未解决时显式标记。
- 官方 XML/图片/PDF/DOCX 原字节保留。网页 signed CDN URL 的签名参数没有写入文件日志；正式引用使用 DOI/PMC/出版社页面。

未联系作者、未做 wet-lab 设计、未安装依赖、未下载测序整库、未训练/推理、未提交/推送。
