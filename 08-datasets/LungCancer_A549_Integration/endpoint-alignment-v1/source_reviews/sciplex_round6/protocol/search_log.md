# sci-Plex3 round6 定向核验记录

日期：2026-10-10。仅核验原sci-Plex3时间协议、对照键及64,440/67,443差异，没有开展新候选数据大范围搜索。

## 路径和入口

读取原数据包README、audit/paper_code_sources.json，亲读原PMC XML和Notebook1/3/8，验SHA256后复制到本目录。读取当前审计脚本的pData读法及hash条件，没有运行其表达矩阵步骤。

从作者GitHub固定commit下载tree、Notebook2、viability.R、dose_response.R、README、valid_cells.tsv；读取GEO GSM4150378 SOFT。另查作者官网论文页和相关两文件提交历史。作者valid_cells历史只返回2019-11-07初次上传；Notebook8返回2019-11-06与11-16记录，未由此确认额外QC。

## Supplemental Methods获取结果

原XML列出精确文件`NIHMS1595611-supplement-Supplemental_Materials.pdf`。PMC附件GET403，Science补充端点尝试403；Europe PMC两次均以XML报错称文章不属于其开放附件服务，未把HTTP200误记为成功ZIP。相关错误及下载日志已保存。

PMC OA S3目录只列正文json/txt/xml三个对象，没有补充；NCBI旧OA工具路径404。作者主页只列主文PDF，作者仓库tree没有补充PDF。停止同一失败点重复请求，登记“已确认文件目录存在，本轮未读到原件”，不以其它实验的Methods替代。

## 网络查询式

搜索结果仅作入口发现，其他sci-Plex变体、综述及专利未作为当前实验的协议证据。

- `sci-Plex Srivatsan 2020 aay2445 supplementary 72 hours 200 500 quality cells`：首个查询误含候选DOI词，依据原XML/官方记录纠正为science.aax6234；错误查询未进入结论。
- `"science.aax6234" "supplementary" pdf`
- `"NIHMS1595611-supplement-Supplemental_Materials.pdf"`
- `"sci-Plex" "72 hours" "medium"`
- `"aax6234" "Materials and Methods"`
- `"sci-Plex" "72" "Cells were"`
- `"Massively multiplex chemical transcriptomics" supplementary pdf`
- `"sci-Plex" "washed" "24"`
- `"aax6234_srivatsan_sm.pdf"`
- `"aax6234" "supplement" site:science.org`
- `"sci-Plex" "72 hours" "25,000"`
- `"sci-Plex" "500" "UMIs"`
- `"Massively multiplex chemical transcriptomics" "biorxiv"`
- `"sci-plex" "Supplemental Materials" "25,000"`
- `"sci-plex" "72 hours" "medium replaced"`

## 实际改变判断的证据

作者valid_cells与当前A549 hash集合完全一致：24 h143,015、72 h67,443。这排除了“当前提取漏执行已公开hash规则”对3,003差额的解释。72 hVehicle1,633亦不解释差额；n.umi下限500已在保留细胞中满足。未为凑数挑新阈值。

20个A549培养板都有同时间/重复的2个Vehicle孔，提供参考键的实数核验。3,003成因仍未知，不写成笔误或额外QC已解决；maintenance/washout等未知协议继续保留null。
