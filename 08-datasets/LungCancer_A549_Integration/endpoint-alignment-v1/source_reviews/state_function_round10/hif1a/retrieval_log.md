# HIF1A Round10 获取与读取台账

日期：2026-10-10。范围只限 DOI 10.1016/j.omton.2025.201049 / PMC12493206。本轮未做广泛检索。

## 获取顺序及失败解释

1. 读取 Round9 `hif1a/` 的原 XML、README、下载失败记录和 ENA 元数据。保留原件不改；不重复 Europe PMC `supplementaryFiles` 整包超时或猜测 `.../articles/PMC12493206/bin/mmc3.xlsx` 的 404。
2. `https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC12493206`：web 工具不可访问；普通 requests GET 返回 HTTP 404。保存为 `oa_api_response.xml`（实际内容是 HTML 404，扩展名不能当成功证据）和 `oa_api_fetch.json`。
3. 检索并亲读 [官方 OA Service 页面](https://pmc.ncbi.nlm.nih.gov/tools/oa-service/) 与 [官方 PMC Cloud 文档](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)：2026 年 8 月 legacy OA API/FTP 已停止；新服务支持匿名 HTTPS/S3 逐文件读取。这解释本轮旧 API 404。没有继续尝试其他旧 API 主机别名。
4. 初始尝试的 `https://www.sciencedirect.com/science/article/pii/S2372770525001751` 被 web 工具返回 403。该 PII 当时未得到原始元数据确认，**不作本论文出版社身份或任何结论的证据**；未继续猜测出版社资产。
5. `https://pmc.ncbi.nlm.nih.gov/articles/PMC12493206/` web open 返回浏览器验证页，未绕过、未重试验证路径。
6. 按新官方文档读取 `https://pmc-oa-opendata.s3.amazonaws.com/README.txt`（HTTP 200）与 `https://pmc-oa-opendata.s3.amazonaws.com/?list-type=2&prefix=PMC12493206&max-keys=100`（HTTP 200）。仅 16 个对象，非整库。原始响应分别保存 `cloud_README.txt`、`cloud_article_listing.xml`。
7. 实际读取 `https://pmc-oa-opendata.s3.amazonaws.com/PMC12493206.1/PMC12493206.1.json`（HTTP 200）。JSON 中 URI 为 `s3://`。第一次把该 URI 直接传给 requests 导致本地 `InvalidSchema`，尚未发出 HTTP 请求；按官方文档改成对应 HTTPS 后成功，原尝试保留在台账，不当远端失败。
8. 从 metadata URI 精确获取 `PMC12493206.1.xml`、`gr2.jpg`、`mmc1.pdf`、`mmc3.xlsx`、主文 `PMC12493206.1.pdf`；均 HTTP 200，全部官方 MD5 相等并计算 SHA256。完整 URL/字节数见 `official_asset_downloads.json`。主文 8.7 MB 用于核对页码和完整方法，没有取 24.5 MB 合并包。
9. web 工具打开新官方 PDF URL 返回不可访问；本地 requests 已成功取得同一公开文件，因此此 web 工具失败不影响原件可用性。未用其结果推断论文内容。

## 本地实际读取与回退

- 用 `view_image` 亲读官方 `gr2.jpg`（764 × 1042）。下图 A549 的横轴为 Dox 0.25 µM；上图 MDA-MB-231 的横轴为 PTX 0.25 µM。
- study Python 自身没有 openpyxl；使用机器已存在的 Codex dependency 目录加载 openpyxl，未安装依赖。所有 8 个 sheet 都列举；`Cell viability` 全表、qPCR relevant block 及全部文本/注释检查已读。Fig2K 80 个目标单元格均为数值常量，不是公式。
- 使用已存在的 pypdf 读取主文和 mmc1。主文相关页及 mmc1 的 8 页文字已检查；没有用 PDF 提取文字猜图中数值。
- 本地额外 PDF 高分辨率渲染尝试缺少 fitz，停止该路线；没有声称产生渲染截图。已保存官方原图与主 PDF 供复读。
- 独立复审提供具体定位后，用 pypdf 从已取得主 PDF 第 5 页提取原生嵌入图 `X29.png`，保存 `Fig2K_PDF5_native_X29.png`；这不是 fitz 渲染或重采样图片。随后亲读该 A549 K 面板，确认末组图示误差不能由 10 个表格数值的 sample SD 重现，细节见 README。原数值表、均值与提取程序未修改。
- Caspase 数表已有精确值，未做读图数字化、未反造误差或重复。8 条条件摘要由每组 10 个源值算术重算；六条作者 posthoc mean difference 用来验证分组和映射。

## 未覆盖/仍未知

- 本轮没有取 `mmc2.pdf`、`mmc4.xlsx`、`mmc5.pdf`，没有下载 FASTQ/高维表达矩阵。
- Figure2K 的 Dox 暴露时间、MK571 窗口、洗脱/持续处理、采样时点、重复独立层级仍未从已读证据确认；不能把未写清当成零时长、持续给药或缺乏实验。
- 只作官方 metadata 当前撤稿标志读取，未另做 PubMed/Scite 勘误/撤稿专题检索。
- 本轮只有原件与源值审查输出，没有改主项目训练输入或全局可行性结论。
