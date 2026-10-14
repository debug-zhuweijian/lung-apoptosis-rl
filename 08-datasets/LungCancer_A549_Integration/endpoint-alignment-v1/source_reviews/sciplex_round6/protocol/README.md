# sci-Plex3 第六轮：24/72 h协议、对照键与QC差异

核验日期：2026-10-10。对象：GSE139944 / GSM4150378，A549。作者代码commit：`079639c50811dd43a206a779ab2f0199a147c98f`。只读原提取包与元数据，没有加载或重下表达矩阵，没有运行训练；新增文件限于本目录。

## 结论

1. **24/72 h是作者报告的处理到终点的时长，分别对应破坏性单核RNA取样群体。**不是同一细胞24→72 h的纵向配对，也不是共同72 h终点的“24 h给药+48 h无药恢复”与“72 h持续给药”比较。
2. **当前A549 hash筛选与作者公开保留名单逐细胞完全一致。**24 h为143,015，72 h为67,443；差异ID均为0。正文64,440与67,443的3,003差额尚未解释，不能归咎于漏做已公开hash规则，也不能为凑数增加没有出处的阈值。
3. **同时间、同重复、同培养板的Vehicle参考可构造。**20个A549培养板每板都有2个Vehicle孔；同板Vehicle细胞数136–481。它们是该时点的Vehicle群体，不是逐个treated细胞的实测t0。
4. **连续暴露、换液、补药及洗脱细节仍未知。**主文和GEO提供处理/采样时间；补充Methods已在官方XML确认存在，但本轮下载受阻，未亲读，不能称完整协议闭合。

机器可读语义：[time_control_contract.json](time_control_contract.json)。复算摘要：[qc_protocol_audit.json](qc_protocol_audit.json)。

## 1. 原始证据与版本

- Srivatsan et al., Massively multiplex chemical transcriptomics at single-cell resolution，Science 367:45–51，DOI [10.1126/science.aax6234](https://doi.org/10.1126/science.aax6234)。[本地PMC7289078 XML](Srivatsan2020_sciPlex_PMC7289078.xml) 与原数据包登记SHA256一致。
- [GSM4150378官方记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150378)，实际下载的 [SOFT文本](GSM4150378.soft.txt)。
- [作者固定commit仓库](https://github.com/cole-trapnell-lab/sci-plex/tree/079639c50811dd43a206a779ab2f0199a147c98f)，[tree记录](author_repository_tree_079639c.json)，[作者README副本](author_README_079639c.md)。
- 原包 `audit/paper_code_sources.json` 指向的XML与Notebook1/3/8复制后逐文件验SHA256，见 [local_source_copies.json](local_source_copies.json)。
- 新取得作者 [valid_cells.tsv](author_valid_cells_079639c.tsv)、Notebook2、dose_response.R、viability.R，见 [代码下载记录](download_log_code.json)、[额外下载记录](download_log_round6.json)。

本轮采用 `data-validation` 技能的原始数据/筛选结果分层核验方法，没有安装新框架或套用通用QC阈值。

## 2. 时间语义

| 证据定位 | 可确认内容 |
|---|---|
| 主文Figure3A及图注 | 24 h主屏：188小分子、四剂量、三细胞系、生物重复；处理24 h后分析 |
| 主文HDAC时间比较段，检索64,440 | A549接受48 compounds、72 h处理后的转录组；与24 h比较嵌入 |
| GEO treatment_protocol | 96孔板、25,000 cells/well，接种24 h后加药或DMSO Vehicle |
| GEO extract_protocol | 加药24 h后孔内提核、固定条码化，然后合并 |
| GEO提取段与pData对照 | 泛称只写24 h，但同一样本元数据含72 h；不能用泛称排除72 h，也不能从它补齐72 h专属操作 |
| [Notebook8](Notebook8_align_24hrs_72hrs_079639c.R) L89–116 | 以time_point==72定义72 h出现候选；24与72分别进入回收计数/剂量拟合 |

给药期间是否换培养基、重新补药、撤药及恢复，在本轮亲读来源中尚未明确核定。机器接口将`continuous_exposure_explicitly_verified`、`media_change_schedule`、`redosing_schedule`、`washout_schedule`置null；**null不是“无该操作”**。

当前可以建立药物、剂量、作者报告终点时间条件化的表达分布预测。24 h和72 h是各自的观察终点，把两者的表达差异解释为共同终点的时长优化收益，超出直接证据。不存在同一细胞在24/72 h的实测配对；按分布构造的连接不能改称纵向轨迹。

### 补充Methods访问边界

原XML列出`NIHMS1595611-supplement-Supplemental_Materials.pdf`，PMC目录显示20.6 MB。PMC附件GET返回403，Science补充端点尝试返回403；Europe PMC明确返回该文章不属于其开放附件服务，见 [错误响应](EuropePMC_supplement_response.xml)。作者官网只列主文PDF；作者仓库tree没有该补充PDF。没有将其他sci-Plex变体论文的流程替代本实验，也没有把获取失败写成补充不存在。

## 3. QC：3,003差额目前不能用额外筛选解释

[Notebook3 L49–66](https://github.com/cole-trapnell-lab/sci-plex/blob/079639c50811dd43a206a779ab2f0199a147c98f/large_screen/Notebook3_preprocess_data.R#L49-L66)、Notebook8 L68–86的规则为：两种top_oligo非缺失，两种hash_umis均≥5，两种首/次hash比均≥5，rt_well不等于1。取人基因轴是基因操作，不能当作额外删细胞步骤。

[Notebook2](large_screen__Notebook2_summarize_hashes.R) 中写出valid_cells的段落直接将上述规则得到的ID保存到`bin/valid_cells.tsv`。本轮实际下载该名单并逐ID对齐GEO pData。

| A549时点 | pData筛选前 | 复算hash保留 | 作者名单命中 | 两集合差异ID |
|---|---:|---:|---:|---:|
| 24 h | 162,171 | 143,015 | 143,015 | 0 |
| 72 h | 82,110 | 67,443 | 67,443 | 0 |

- 作者名单649,340个ID全部可在当前pData找到。全细胞系当前hash规则得到649,341，另有1个非A549差异，见 [全系差异行](all_cell_types_hash_author_list_disagreement.csv)；不声称全细胞系完全相同。
- 72 h Vehicle为1,633，非Vehicle为65,810，不能解释64,440。
- 当前A549 hash合格细胞的n.umi最小为500，另加`n.umi>=500`不会减少3,003个细胞。没有为接近论文数字反推并采用新阈值。
- Notebook8可见的细胞选择包括hash、A549、72 h出现的候选集合；表达比例>5%筛选的是基因。后续PCA、MNN和绘图坐标窗口不能直接当成已确认的3,003个细胞QC来源。
- Notebook8 L405–411将log(n.umi)、replicate、viability、proliferation_index用于回归；L575–599按时间进行PCA空间MNN并重排到原列名。原始计数和这些作者降维表示属于不同数据层级。这里只静态核查R代码，未运行作者R环境。

**判定：**当前A549选择有与作者公开名单一致的可复算依据；正文64,440的具体历史筛选来源仍未知。保留两条不同的来源记录，不把正文宣布为笔误，不把尚未核实的额外QC宣布为已解决。

## 4. 对照键与实验单位

[Notebook1](Notebook1_preprocess_data_079639c.R) L34–66给出`culture_plate=top_oligo_P`，`Combo=top_oligo_W+top_oligo_P`，hash表连接cell_type、replicate、time_point、drug_dose。

对所有A549合格行实际核验：`culture_plate==plate_oligo`、`Combo==well_oligo+plate_oligo`。20个culture_plate均分别只属于一个time和一个replicate。culture_plate不能与RT、ligation、PCR板混名。

| 时间和重复 | 培养板数 | 有合格细胞的孔数（含Vehicle） | 合格细胞数 |
|---|---:|---:|---:|
| 24 h rep1 | 8 | 768 | 59,393 |
| 24 h rep2 | 8 | 768 | 83,622 |
| 72 h rep1 | 2 | 192 | 35,542 |
| 72 h rep2 | 2 | 191 | 31,901 |

A549共计划1,920孔；其中一个72 h rep2孔没有hash合格细胞。[计划孔表](a549_planned_wells_with_hash_counts.csv)保留原设计及零回收，不能从条件池中悄悄删去。

**本课题推荐参考键：**`cell_type + time_point + replicate + culture_plate`。每板2个Vehicle孔，全部20个分层都可构造参考；[分层统计表](a549_control_strata_by_time_replicate_plate.csv)给出细胞/孔数。保留孔ID，不把孔内细胞数当生物重复数。Vehicle是终点群体参考，名为`time_matched_vehicle_reference`，不是observed_t0。跨24/72 h不应默认共用一个Vehicle池。

### 作者viability处理与本课题参考键不同

[dose_response.R](bin__dose_response.R) L8–17先按细胞系、sample、replicate、培养板、well、treatment、dose、vehicle计数；L36–94按指定协变量校正孔回收数；L100–126提取输入中的全部Vehicle行，并分别复制到每个药物的剂量拟合。

- Notebook3 L106–116显式使用cell_type、culture_plate、replicate作为计数校正项。
- Notebook8 L110–116先按时间拆开，再用默认cell_type、culture_plate；培养板嵌套重复不等于源码显式含replicate。
- 所以作者viability拟合不是每个treated孔只配某个同板Vehicle的逐孔一一配对。
- [viability.R](bin__viability.R)计算拟合预测回收数的比值，不是Annexin/Caspase或实测凋亡率，不能补真实凋亡标签。

## 5. 复算与交付

```powershell
& 'F:/anaconda/envs/study/python.exe' -X utf8 'I:/科研/肺癌细胞凋亡-强化学习/08-datasets/LungCancer_A549_Integration/endpoint-alignment-v1/source_reviews/sciplex_round6/protocol/review_sciplex_metadata.py'
```

脚本分块读84.7 MB压缩pData、约65 KB压缩hash表、18.5 MB作者名单，不访问3.1 GB表达计数文件。只写本目录。验证覆盖A549集合一致性、rowname/cell一致、分层键唯一性、Vehicle覆盖、计划孔零计数保留。

- [qc_protocol_audit.json](qc_protocol_audit.json)：SHA256和QC摘要。
- [a549_metadata_qc_review.csv.gz](a549_metadata_qc_review.csv.gz)：所需元数据与两个筛选标记。
- [a549_control_strata_by_time_replicate_plate.csv](a549_control_strata_by_time_replicate_plate.csv)：20个参考层。
- [a549_planned_wells_with_hash_counts.csv](a549_planned_wells_with_hash_counts.csv)：全部计划孔。
- [time_control_contract.json](time_control_contract.json)：时间/参考状态语义与未知项。
- [search_log.md](search_log.md)：查询、失败回退和范围。

完成：时间语义的已知部分、对照键、当前hash版本的原始证据核验。未闭合：补充Methods的维护/洗脱操作、正文64,440的具体历史筛选来源。未修改原提取包、化学身份表、训练版本或Zotero。
