# Bothe2021 / E-MTAB-9923：A549 撤药再处理 RNA

2026-10-11最新下载恢复：先导28793已exit1，R1达到1,507,908,000bytes但原MD5不符，追加gzip容器检查CRC亦失败，未发布正式文件/receipt或进入计数。错误副本同目录保留rename；原登记路径使用原脚本/MD5从零重试54677。21:55:50 UTC root工具核54677/5491/10921非终止；0296进入R2，计数10921仍等pilot完整双端和固定回执，真实counts0。详见[完整性故障记录](../../../experiments/20261011-data-alignment-round20/pilot_md5_diagnosis/README.md)；下方历史下载快照不覆盖本条状态。

第二十轮补齐[官方GeneID→joint Ensembl桥](reference/gene-axis-bridge-v1/README.md)：17,107可测基因取得唯一对应，歧义/不可测/缺失保留NA及台账；当前官方snapshot不证明作者注释年代一致。新固定`project_production_counts(version)`消费已验收生产数据并保留整数0/None/mask，30检查与独立复审通过；真实counts/投影当前仍未执行，不将本桥视为模型就绪。

2026-10-11 第二十轮新增：真实FASTQ全文验证及受控Subjunc／strict filter／featureCounts／12列count生产入口已实现，101项联测和独立复审通过。root计数session10921已启动，固定版本`processed/registered-rna-counts-v1`顺序处理全部12run，当前等待pilot下载正式回执及双完整gzip；尚未开始真实RNA native stage，真实counts仍0。等待不授资格、不读`.part`，模型训练停止。执行与接续见[第二十轮报告](../../../docs/DATA_ALIGNMENT_ROUND20_2026-10-11.md)和[当前root句柄记录](../../../experiments/20261011-data-alignment-round20/CONTINUATION_HANDOFF_2026-10-11.md)。以下第十九轮为历史准备快照。

2026-10-11，状态：**真实来源、注释及完整human genome索引已核验，十对参考DNA人工reads实际比对/过滤/count通过；12个原始RNA run获取中，尚无恢复后的真实样本counts，不登记训练就绪。** 数据只保存本地，不进入 Git/Git LFS。最新说明见[第十九轮](../../../docs/DATA_ALIGNMENT_ROUND19_2026-10-11.md)。

原论文 DOI [10.26508/lsa.202101080](https://doi.org/10.26508/lsa.202101080)，原件、SDRF、ENA12-run对应及三个官方DESeq2对比表在[来源核查目录](../../../experiments/20261011-data-alignment-round17/bothe_protocol_source/)。RNA accession E-MTAB-9923 / ERP125910。四历史各三重复，12个样本、24个FASTQ；双端不当作两个重复。

Fig3方案：Dex100nM或EtOH0.1%处理4h，PBS洗两次，在含5%FBS且无新增激素培养基恢复24h，再Dex/EtOH4h至共同T32。vehicle百分比的v/v基准未确认，完整输入张量保留unknown。T32平行vehicle不是实测t0/t28，同细胞轨迹未取得。此来源没有凋亡标签，不替代Viel6+18h/T24 RNA。

## 本地文件及真实状态

| 目录 | 用途与证据状态 |
|---|---|
| `reference/current-ucsc-hg19-20261011/` | 官方当前下载的hg19 genePred/RefSeqLink表、SQL schema、源hash、chrom.sizes和genome declaration；SQL仅保存为schema，不执行。完整hg19.fa.gz已取得，948,731,419bytes、publisher MD5核验通过 |
| `reference/unique-introns-v1/` | 新版primary/both-strands保守unique intron注释；99,517源行，29,634选择gene、23,125有可测feature的gene，207,555 SAF区间；完整独立区间核验通过 |
| `reference/subread-hg19-index-v1/` | 新portable Subread2.1.1实际索引已完成：exit0，6文件共6,031,893,097bytes大小/SHA256独立核验通过；完整index上10对参考DNA人工read比对/count oracle通过，真实RNA比对未执行 |
| `tools/subread-2.1.1/` | 作者SourceForge Windows ZIP及解压工具；版本/help、gzip合成read及中文路径已实际验证，无全局安装/WSL/Docker |
| `raw/ERR5000305/` | 首个恢复先导：EtOH→EtOH rep2双端原始RNA，声明总3,301,778,336bytes；未完成文件使用`.part`，不当作完整FASTQ或counts |
| `raw/<其他ERR>/` | 剩余11run由两worker依次下载，声明共56,754,947,648bytes；固定对应表、sourcehash与每run.log/completed receipt保留，不占用pilot目录 |

参考genome采用UCSC初始hg19，而不是带p13/新MT补丁的`latest`别名；publisher MD5为`806c02398f5ac5da8ffd6da2d1d5d1a9`，实际SHA256为`ac73947d38df63ccb00724520a5c31d880c1ca423702ca7ccb7e6c2182a362d9`。基因组验证和annotation源验证分别记录，不借一个manifest认证另一项未登记输入。

## 新处理版本的边界

原作者为STAR2.7.0a/hg19及其历史RefSeq最长转录本intronic计数；精确历史注释、strandness和部分参数未知。本课题新版本使用已冻结的当前下载RefSeq、stable NCBI GeneID、最长spliced长度确定性选择、primary scope、both-strands去除全transcript exons和多gene intronic交叠。primary跨染色体/strand或不同locus冲突隔离，非primary记录保存台账。它不是原作者精确复现。

SAF是1-based闭区间，源genePred是0-based半开区间；featureID与geneID通过映射表对应。gene_table的`retained`共23,125个，其余无可测feature的gene必须保留不可测身份，后续不能把它们填成实测零。第十九轮严格pair及`featureCounts -f`后按映射汇总gene已在合成链实测，真实executor和全样本QC仍待完成。

已实测的工具陷阱：`-Q10`接纳至少一端MAPQ≥10的pair；逐read丢弃低MAPQ后，`-B`仍可能凭mate flags接纳孤儿。第十九轮已实现严格QNAME/R1/R2+双端MAPQ/互逆坐标联合过滤，secondary/supplementary、NH、duplicate、proper-pair及unknownstrandness分别声明，35tests与独立复审通过；规则属于新processing选择。count接口保持缺sample/不可测gene为空、合法实测count0为整数0；native CP936注释先bytes跳过、data严格解码，31tests/独立复验通过。详情见[过滤与矩阵报告](../../../docs/DATA_ALIGNMENT_ROUND19_2026-10-11.md)。

名义Dex CID5743/fullkey及四事件指纹在独立[协议v3](../../../experiments/20261011-data-alignment-round19/full-protocol-inputs-v3/manifest.json)接入；stock/material认证与EtOH百分比basis仍未知，旧v2保留可读。名义药物指纹不补出state/真实counts或凋亡配对。

## 接续入口

- `scripts/download_bothe_inputs.py`：仅指定官方genome或登记run，exact bytes/publisher MD5检查；支持严格Content-Range续传，已有完整文件先验证，独立新receipt。
- `scripts/download_bothe_remaining.py`：两个worker取剩余11run，排除live pilot、拒已有run目录；失败partial保留，不由协调器反复重启。
- `scripts/verify_bothe_downloads.py`：固定12run独立重核每端回执字段和真实完整size/MD5/SHA；协调器complete摘要不能单独认证完成，metadata-only不授实际文件资格。首轮0verified/12pending/0reject，之后另建验收JSON。
- `scripts/build_bothe_intron_annotation.py`：独立新annotation版本，拒绝覆盖非空输出。
- `scripts/build_bothe_genome_index.py`：原生data index任务，当前prefix拒绝覆盖；不运行比对/count或模型。长任务先核实际工具handle，再决定接续，不只看run.json。
- [最新运行/接续说明](../../../experiments/20261011-data-alignment-round19/CONTINUATION_HANDOFF_2026-10-11.md)：索引及合成probe已完成，两个raw handle需重新poll；[索引完成验收](../../../experiments/20261011-data-alignment-round18/index_completion_validation.json)独立重算全部6文件，[实际合成result](../../../experiments/20261011-data-alignment-round19/reference_index_probe/result.json)核坐标与count。

尚未完成：12run双端raw完整MD5/FASTQ验证、专用真实RNA比对/count executor、12列矩阵/QC、common axis桥和source decoder。正式训练/真实P/G/Q仍未执行。
