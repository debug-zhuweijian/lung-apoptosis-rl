# Bothe A549 unique introns v1

2026-10-11：仅完成注释生成与完整独立读回，**未下载FASTQ、比对、计数、训练或模型推理；RNA counts仍未恢复**。

这是本课题冻结的新处理版本。Bothe et al. PMC8403771 Methods “RNA-seq”规定每gene最长转录本、剔除全部转录本exons和多gene归属intron、featureCounts以独立intron feature计数后按gene汇总。作者历史RefSeq版本、最长定义细节和strandness未知，不能声称exact作者复现。这里使用2026-10-11下载的UCSC hg19 ncbiRefSeq/ncbiRefSeqLink快照；源SQL中的dump日期为2024-09-22，不将下载日期当注释生物学发布日期。

冻结定义：

- GeneID来自ncbiRefSeqLink.id与RefSeq accession精确连接后的locusLinkId，不以symbol作基因键。
- 仅chr1–22/X/Y/M用于选择；9,073非primary transcript行留在scope台账。实际1,842个gene同时有primary/其他contig行，其他contig不参加最长选择。
- 同一gene在primary不同chrom/strand的27个gene（187行）隔离；primary同chrom/strand但转录本跨度不相交的3个gene（17行）隔离，不拼成一个gene。跨度按重叠连通分量判定，端点相接不视为重叠。
- 最长定义为sum(exonEnd−exonStart)，不是基因组跨度；并列按RefSeq accession、start、end、exon坐标tuple字典序选最小者。
- 输入坐标0-based half-open，先取选择转录本相邻exons间的非空区间。剔除**所有源primary转录本**的exons，包括未选转录本、被隔离gene、缺失映射的转录本和反义链；再剔除多个选择gene归属的片段。该保守both-strands规则是本课题新定义，未知strandness仍需raw诊断；注释本身不解决链特异计数设置。
- 输出SAF为1-based closed：Start=start+1，End=end。每feature一个独立UIV1 ID；后续featureCounts需`-f`，然后按feature_gene_mapping.tsv的ncbi_gene_id汇总，不能把SAF GeneID当stable geneID。

真实构建：99,517源转录本行、90,529源唯一accession；29,634选择gene、23,125保留gene、207,555独立片段、1,206,619,076 retained bp。4,416选择gene原本无非空intron；2,093选择gene全部intronic bases被剔除。候选intron union 1,261,485,654 bp，其中11,342,721 bp被exons剔除，43,523,857 bp因多gene归属剔除。所有计数是注释轴，不是RNA样本count。

文件：unique_introns.saf；feature_gene_mapping.tsv；gene_table.tsv（含无保留feature的gene及原因）；transcript_scope_ledger.tsv；manifest.json（源、生成器和派生hash、轴、过滤理由、验证结果）。README为治理说明，不参与四个派生表的内容hash。

运行：`F:/anaconda/envs/study/python.exe scripts/build_bothe_intron_annotation.py`。入口拒绝覆盖非空注释版本；若需要不同规则，另建版本目录。

验证：`F:/anaconda/envs/study/python.exe -m pytest tests/test_intron_annotation.py -q`，21项通过。`experiments/20261011-data-alignment-round18/bothe_annotation/verify_serialized_annotation.py`独立解析真实源、重建exon union、查验序列化SAF owner归属、feature/gene轴及汇总长度，完整207,555 features通过；读回不载入模型。

该产物只冻结annotation；hg19 FASTA/index、reads完整性、链特异诊断、比对/QC/count及P观测域映射仍是后续独立步骤。无凋亡功能标签、无readyRNA资格、不改变毕业论文既定响应预测与干预筛选架构。
