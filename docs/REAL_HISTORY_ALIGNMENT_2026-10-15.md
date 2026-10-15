# A549 真实历史表达对齐与全架构数据链验收接续

**结论：本轮完成真实源值恢复、GSE96779 新的基因身份桥及多来源逐值复算；整个课题的数据与科研可行性仍未通过。** 这里的“已核验投影”仅指明确来源、样本顺序、基因身份和原尺度数值的连接，不指共享潜在表示、表达—凋亡因果联系、模型效果或序贯治疗收益。

日期：2026-10-15（北京时间）。本轮由当前 Chat 的执行者本人检索、实现、复算与自审，没有分发给其他代理。公开仓库中已有其他任务提交，本轮保留并利用其可验证的数据产物，不将其工作冒称为本轮新完成。

## 1. 基线、环境与证据等级

原交接：`4a78560bdeb2e0d98332d5a9520764a120abf612`。本轮起点：`ec5be211874c6ccc64ce81b6238a689fc56fad12`，相对交接领先六个提交；本地独立分支为 `codex/protocol-source-repair-20261015`。没有直接修改远端 main、合并 PR、改写历史、修改冻结划分或奖励，也未恢复本机训练/数据作业。

本轮下载了既有 [Actions run 38611746440](https://github.com/debug-zhuweijian/lung-apoptosis-rl/actions/runs/38611746440) 的 artifact `11918331571`（`source-bridge-verified-data`），其获取过程属于该既有任务。下载后 ZIP 实测 SHA256：

```
6c7a9fc8b3dfc37abb82fa0be10b488faf09e6b7dba3cae1dc122ba0bf53b5ba
```

包内源码 bundle 对应 `68596ecd89b58dc93ec4d9fea38ed33ef8e68d2e`，并含上面的最新 main 引用。本轮由 bundle 恢复完整源码，再基于最新 main 开始修改，而不是把旧分支覆盖到 main。包中源文件经其回执和输出 manifest 检查，共完成 50 项输入/输出/bundle 哈希检查（包含重复引用，不是 50 个独立数据集）。旧 SHA256SUMS 使用 `cloud-receipt/` 相对前缀；解压后按已知路径布局核验同一 bundle，没有改写原回执。第一次直接运行 sha256sum 的路径失败也保存在日志中。

| 等级 | 本轮对应内容 |
|---|---|
| 文档记载 | Round22 的旧 743 细胞、Bothe 4/12 计数、完整冻结共同轴及模型状态；未把它们升级为本轮原件验收 |
| 源码核实 | 现有源恢复解析器、外部基因映射、Sarin 注释映射；新增历史基因桥与实际 CLI；Bothe 观察包现有协议约束 |
| 云端实际执行 | 下载包哈希、上述 50 项哈希；四组源表达恢复；三组已有投影重新计算；GSE96779 新投影；TIDYI 全 CSV；Bothe SDRF/DE 表；测试与修复 |
| 尚待其他有资产环境验证 | 真正冻结轴/角色/检查点一致性、缺失板、Bothe 12 样本计数、来源模型、独立最终功能与序贯效果 |

实测环境：Python 3.13.5、pytest 9.0.2、PyTorch 2.10.0+cpu、NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.0、RDKit 2025.09.4；CUDA 不可用，PyArrow 未安装。终端 DNS/依赖下载不可用，GitHub 连接器下载 artifact 可用。当前暴露的 GitHub 动作未提供创建/修改分支或 PR 的入口，所以本轮交付绑定基线的补丁，不声称已经更新旧 Draft PR。历史回合曾成功写入，不构成本轮写入能力证明。

## 2. 本轮真正补上的数据连接

### 2.1 GSE96779：重复基因符号不再阻断全部共同轴投影

[原论文 Kitajima et al., Cancer Cell 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6422029/)，DOI `10.1016/j.ccell.2018.08.009`；[GEO GSE96779](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96779)。实际读取 SOFT 与 FPKM，核对物种/总 RNA、系列、培养背景、处理记录、hg19、UCSC RefGene Feb2016、Cufflinks 2.2.1。GEO 列名 `gene_id` 实际是符号，不是 ENSG；原脚本保留了重复符号及其 locus，但尚无该来源的可核验共同轴投影。

新增 `src/lung/data/history_gene_axis.py`，由实际执行入口 `scripts/recover_history_gene_bridge.py` 调用。不是给重复符号随便选第一行，也不把 FPKM 当成 counts。

规则是：**完整 GRCh37 Ensembl75 注释中的同染色体区间 + GTF 名称或 HGNC 正式/历史名称支持 + 唯一基因候选**。先在完整注释中处理歧义，再检查是否在目标轴；不能因另一个候选没进共同轴就把歧义伪装成唯一。多个原始行指向一个目标时整组排除，不求和、平均或按表达强度择优。非标准染色体/不能核验的 locus 保留排除状态。

GTF 的基因组声明从文件头读取，不能由文件名猜测。GTF 为 1-based；Cufflinks 文档把 locus 定义为浏览区间，本轮不擅自确定该导出起点约定，而只接受按两种起点解释候选集合都一致的行。边界敏感行保留缺失。这是坐标与身份支持，不证明 RefGene 和 Ensembl 的转录本量化完全等价。[Ensembl 格式说明](https://www.ensembl.org/info/website/upload/gff.html)；[Cufflinks 原文档](https://cole-trapnell-lab.github.io/cufflinks/file_formats/)。

设原矩阵为样本×来源特征的 \(V\)，唯一映射为 \(j(g)\)。输出仅执行

\[
X_{ig}=V_{i,j(g)},\quad M_{ig}=1;
\qquad \text{无可靠映射时 }X_{ig}=\mathrm{NaN}, M_{ig}=0.
\]

因此观测零仍是 `0, mask=True`，没有测量支持的基因是 `NaN, mask=False`；既无表达插补，也无归一化或参数拟合。

**最终真实执行 v3 的结果：**

| 项目 | 实测结果 |
|---|---:|
| 原始样本 / 来源特征 | 12 / 27,427 |
| 原尺度 FPKM 值 | 329,124 |
| 公共目标轴 | 32,461 |
| 唯一支持的目标基因 | 17,978 |
| 未观测目标基因 | 14,483 |
| 投影后与原件逐值一致的数值 | 215,736 |
| 其中实测零 | 31,339 |
| 排除的来源行 | 9,449 |

排除原因：不在目标轴 4,841 行、符号与坐标不能共同支持 3,142 行、暂不支持的 locus 1,292 行、多候选歧义 143 行、多源行指向同目标 31 行。所有行均保留 ledger，不把“排除来源行”误解为丢失的独立实验。

输出有原尺度 NPZ、带 mask 的公共轴 NPZ、样本表、逐行映射表、源特征表、公共轴顺序表、完整源元数据、条件连接、输入/代码/输出哈希及完成 manifest。发布采用独立临时目录，所有检查通过后才公开输出目录；第一次慢速读回超时产生的 partial 目录未宣布完成，也未覆盖。重复解压 NPZ 的读回实现已改为一次读取并向量化逐值比较。最终 v3 实测 7.87 秒，最大 RSS 374,584 KiB；这是该云端数据处理命令，不是 RTX 3080 训练成本。

### 2.2 四个历史状态不是任意序贯训练轨迹

| 真实条件 | 样本数 | 已知历史及当前处理 | 未补造的内容 |
|---|---:|---|---|
| Parental | 3 | A549，DMSO | 与其他样本同细胞 t0、完整批次身份 |
| MSR | 3 | 两个月交替筛选后，持续 momelotinib 5 µM + selumetinib 1 µM | 精确维护总时长、独立筛选谱系数 |
| Withdrawn | 3 | 上述来源撤去两药 10 天，即 240 h | 与 MSR 的逐重复纵向配对、未报告洗涤细节 |
| JQ1 | 3 | 保持两药，增加 JQ1 200 nM，即 0.2 µM，24 h | JQ1 单药、药物全排列、配套 Annexin 数值 |

新入口保留联合处理，而非把 JQ1 行解释成 0.2 µM JQ1 单药。`condition_links.json` 只建立条件—成员关系及三个群体对比，不把 3×3 的成员组合扩成 9 个观测。`replicate_label` 保留原标签，不宣称这些标签已证明独立培养谱系。

论文 Figure 2D / STAR Methods 给出筛选与撤药背景；Figure 4G 的功能是不同 JQ1 剂量和时长的生长实验，Figure 5I 的共处理与 RNA 条件亦不同，Figure 6A 是含 trametinib 的间歇方案及 crystal-violet 生长终点。它们可提供另外的、协议明确的挑战，**不能变成这里 12 个 RNA 样本的配套 Annexin 标签**。原论文的间歇方案存在，也不能被说成已恢复其全转录组轨迹或本课题已经验证序贯收益。

本包仅保留药物名义名称与原剂量；三个药物的来源材料/完整结构映射本轮没有完成。该不确定性是显式字段，不用假指纹填补。论文材料表中个别字符串需要进一步核对，也没有自动纠正为实验试剂身份证明。

### 2.3 三个其他来源：从原件重新运行，而非复制“成功”标志

在新目录运行现有 `recover_public_expression.py`，再用现有映射函数重新计算公共轴，与既有 artifact 中的整个数组、样本 ID、轴顺序、mask 逐值比较：

| 来源 | 实读样本 | 来源特征 | 原值数 | 公共轴观测 / 缺失基因 | 尺度与限制 |
|---|---:|---:|---:|---:|---|
| Sarin / GSE108214 | 22，均为 A549 谱系 | 29,729 探针 | 654,038 | 12,070 / 20,391 | 微阵列原 VALUE，负值合法；不是 RNA-seq counts |
| HM97662 / GSE333828 | 18，其中 A549 仅 6 | 57,905 | 1,042,290 | 30,268 / 2,193 | Salmon 估计计数，129,069 个非整数原值保留；其他 12 样本不冒充 A549 |
| GSE161800 | 12，A549 | 58,735 | 704,820 | 32,461 / 0 | 原整数 counts；原论文 5 µg/mL 与 GEO 5 mg/mL 冲突按现有明确来源规则保留记录 |

三个重新计算的投影均与既有 artifact 数组一致；这证明当前执行可重复，**不是独立生物来源验证或独立第三方审查**。Sarin 映射保持“一个可靠探针对一个基因”，不平均多探针；空的显示用符号字段不参与选择，所有选择来自显式 Entrez/ENST/ENSG 证据。

GSE161800 的原文 RNA-seq 方法和源计数已实读，其任务是脉冲处理后特定状态群体，不自动构成均匀未分选细胞群体或任意两药顺序实验。[原论文](https://elifesciences.org/articles/65234)，DOI `10.7554/eLife.65234`。

### 2.4 实测 Annexin 原数值：合法条件连接，不制造配对

从 [Sarin 2017 原始 S8 表](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181081) 重新读取 10 个 Annexin 诱导倍数，空单元格不补零；用已有代码与 2018/GSE108214 条件对应。保留源表行列位置，重复序号不跨实验配对。

| 条件，均 24 h | 药后 RNA / 同期对照 RNA | 原功能数值数 | 本轮复算平均 Annexin 倍数 |
|---|---:|---:|---:|
| A549，cisplatin 11 µM | 3 / 4 | 3 | 5.855 |
| A549rCDDP2000，11 µM | 5 / 5 | 4 | 1.9495 |
| A549rCDDP2000，34 µM | 5 / 同一组 5 | 3 | 3.7136667 |

这是相对各自未处理对照的倍数，不是 fraction、绝对凋亡率或概率。原方法亲本与耐药细胞的接种数量不同；该混杂必须保留。两种耐药剂量共享同一参考组，三个条件也不等于三个独立研究。SEM 在机器表中仅为这几个发表数字的描述性汇总，不能覆盖跨研究批次或配对不确定性。

可用作限定外部条件挑战：同记录 11 µM 下的背景比较，以及耐药背景 11/34 µM 两候选排序。不能支持多药、多时长、未见背景、高维表达超越 state-ID 的独立效用或临床效果。没有删除这些有限用途，也没有将它们夸大为完整 G 监督。

### 2.5 TIDYI 和 Bothe：亲读后的边界

TIDYI 原 CSV 为 369×32,858，共 12,124,602 个非负整数计数；按细胞 ID 和明确背景取 368 个 human A549，公共轴逐值核验 11,945,648 个值。原状态标签为 PI-APC- 167、PI+APC+ 108、PI-APC+ 93，均来自 plate 8。没有取得冻结 743 细胞所需的完整板覆盖。269 个细胞的当前计数导出的检测基因数与附带元数据不一致；这可能涉及导出/过滤基因范围，本轮没有据此判定样本无效、删除细胞或重写 QC。分选标签数量也不是自然群体凋亡比例。[官方来源 E-MTAB-16354](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-16354)。

Bothe SDRF 24 行实际为 12 个样本各两个 read mate，四种历史，不是 24 个独立样本。五个 dose 单元格为空白：同一 RNA 的 mate 已知值可核对；其余双 vehicle 条件按原论文 `s13` Hormone re-induction treatments 的 0.1% EtOH 另记协议来源值，原字段不改。百分比基准未明，不转换成摩尔浓度。核查同时看到原源码已经保留该“来源百分比—模型数值未知”的边界，故未制造无必要的 Bothe 生产修改。

三个公开 DE 表各 24,163 行，列为 `baseMean/log2FoldChange/lfcSE/stat/pvalue/padj`，不含 12 个样本表达列。均值/差异统计没有提供个体计数、文库因子或全部样本协方差；不能逆推出并宣称取得真实计数矩阵。本轮读取的 Bothe 原始 count 样本为 **0**；旧 4/12 是历史本机记录，不是本轮重新核验。没有启动大规模 FASTQ 下载。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8403771/)；[官方 E-MTAB-9923](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-9923)。

## 3. 共同轴不是跨来源共同测量

本轮公共轴有序 ENSG 身份 hash 为：

```
6e495561730981f64aca5bafe1b5ab17728c8de0ebde3efd2c9ca838ac727ec5
```

它来源于已下载公共导出交集，**没有与缺失的本机冻结轴逐 ID、逐顺序认证**。本轮重新检查 TIDYI 取值，但未在无 PyArrow 的环境中重新解析 Tahoe 的 Parquet 正文；这些文件只完成字节哈希核验。没有读取 Tahoe 全量细胞，更没有重新认证原 2,762,092 细胞联合训练资产。

FPKM、微阵列 VALUE、估计 counts、原整数 counts、单细胞计数、Bothe intron fragments 不能只因列数相等而直接拼接。推荐先保留来源特异读出/尺度与 mask，比较来源特异建模、条件分布映射和共享潜表示三条路线。共享 E/D 的必要证据是相同任务信息与合法支持下的外部迁移，而非来源内重建通过。混合后 source 分类准确率高本身也不等于失败；必须结合生物信息保留、批次混杂和迁移性能判断，不能机械要求来源不可辨认。

## 4. 全架构问题—数据—验收对应

记 \(W\) 为背景、药物结构、剂量、协议历史、来源和时钟，\(X\) 为合法表达观测，\(Y_e\) 为终点域 \(e\) 中的功能值。独立快照允许在可比实验单元下研究 \(P(X\mid W)\)、均值差与分布差，不唯一确定同一细胞 \(P(X_0,X_1\mid W)\)。本轮没有把所有问题强行要求成同细胞配对；也没有用任意算法 coupling 充当实测轨迹。

| 全架构环节 | 已完成 / 可用层级 | 尚未闭合的关键条件 | 可证伪实验与失败边界 |
|---|---|---|---|
| 状态表示 C/E | plate8 表达—原分选标签；四来源原尺度/公共轴 | 另一板、冻结轴/角色、独立生物初态；共享表示 | 跨板三分类与同信息 source-specific/共享表示比较；仅板内有效不能主张跨来源状态识别 |
| 条件响应 P | GSE96779 四历史×3 源标签，另外三组源值可消费 | 合法配套 E/D/P、批次/维护时长，状态×同动作的充分网格 | observed/zero/mean/state-ID 等基线独立拟合；同记录 W 比较。MSR 与亲本同时改变状态/当前药物，不能单独证明状态因果效应 |
| 未见药/剂量/时间/协议 | 原项目化学留出；本轮增加历史辅助源 | 每个泛化轴独立、全链预处理隔离、支持覆盖 | 新 namespace；每种留出单报；不能沿用看过评价域的旧 AE 或把两个时点当完整动力学 |
| 表达→实测凋亡 G | Sarin 3 条件/10 原倍数及条件级 RNA 参考 | 多药/多时长/状态覆盖；可区分背景 ID 的监督与独立测试 | 分开 fraction、fold、caspase、death/growth 头；外部真实终点比较，无 RNA 的功能来源也可检验排序，但不能证明细胞级中介机制 |
| 药物×剂量×时长选择 | 原候选协议 + 限定 Sarin 两剂量候选 | 全候选合法结构/单位/支持、真正留出 Y、P→G 同链 | 先与直接 G 排序/枚举、drug-only 比较，报告真实 Y regret/top-k；G 自己的奖励不能证明策略收益 |
| 多步模型滚动 | 新历史表达与 Bothe 有序协议元数据 | 已干预状态上的后续响应监督、跨步误差与历史充分性 | 有真实多时点时逐步误差；只有终点快照时明确非辨识性，内部 rollout 稳定不等于真实药效 |
| 真实序贯促凋亡 | 有明确来源的历史/间歇方案线索，原 A375 辅助 | A549 或合法迁移的序贯实测终点及适当顺序/单药/同时对照 | 以真实完整方案为单位保留；没有顺序数据时不宣称序贯收益，但也不删除原研究目标 |

单步有限候选时，直接枚举 \(\arg\max_a\hat G(W,a)\) 是同一评分器的精确最优选择。若奖励仅一次终点，再拟合 Q 没有自动增加信息。RL/规划需在历史依赖、延迟反馈或有限查询等明确条件下，比较同数据、同候选和同终点的实际增益。本轮不改奖励，不加入成本惩罚，也不因没有新 RL 训练而将任务目标改为仅做单步。

## 5. 实验方案与创新主张的边界

本轮最接近的原始工作不是“也用了神经网络”的论文，而是已经提供历史/状态和功能证据的 Kitajima、Sarin、GSE161800 原实验。它们各自证明的实验现象不能自动拼成此项目的完整监督。新基因桥属于数据工程改进，**不声称本身构成论文算法创新**。

可检验差别应是：在预先声明的外推轴和独立实验块上，合法状态信息是否超越药物/背景 ID/平均参考；来源特异或共享表示是否在共同可支持的评价域中改善响应和功能外推；策略是否超越同一预测器的直接枚举。这些仍是待检验主张，不以未检索到某算法作新颖性证明。

正式评估需预先冻结：独立单位（实验/培养块优先、RNA 样本不得冒充跨源功能重复）、新旧角色、参考可用性、训练内归一化/特征选择、基线、条件权重。表达在可比观测域报告均值误差与分布指标；功能按终点原尺度报告误差、排序及选择 regret；统计按来源/实验块/化学组的实际依赖结构比较，不能以细胞、基因或组合对数扩大独立样本量。小样本 Sarin 可报告原数字与描述性不确定性，但不应给出覆盖跨研究迁移误差的虚假置信保证。

失败判据包括：隔离板/来源后状态增益消失；共享模型只改善源内重建而不改善外推；功能排序仅在拟合的 Y 上成立；RL 只改善模拟奖励；新处理协议或低回收细胞域失准。失败不靠修改 final 阈值、删困难样本、将缺失填零或缩小原主问题消除。

检索使用 web 与 Exa 两个互补入口并回到上述原文、GEO/SOFT、官方格式说明；关键代码判断另以源矩阵与注释实际读取支持。本轮是定向补链，不声称再次完成近三年全领域系统综述。PubChem REST 在当前 web 工具返回不可访问，未取得三药新的完整化学响应；没有把失败当作“化学结构未公开”。

## 6. 实现变更、测试与失败记录

| 文件 | 实际行为变化 |
|---|---|
| `src/lung/data/__init__.py` | `TrainingData/prepare_training` 改为按需导入。CSV/NPZ 的数据恢复不再因包初始化提前要求 PyArrow；真正训练/Parquet 路径仍需原依赖 |
| `src/lung/data/history_gene_axis.py` | hg19 symbol+locus/GRCh37+HGNC 身份投影，歧义/重复隔离，mask/尺度保持 |
| `scripts/recover_history_gene_bridge.py` | 使用现有源恢复器；核源/背景/历史、读取公共轴、保存原值与映射、逐值验收后发布，不自动接训练 |
| `tests/test_data_api_optional_dependencies.py` | 实际子进程阻断 PyArrow，检验不相关源读取可导入；保持公开名称 |
| `tests/test_history_gene_axis.py` | 单位、轴顺序、实际零/缺失、重复/歧义、基因组声明/边界等反例 |
| `tests/test_history_gene_bridge_cli.py` | 合成 12 样本完整消费测试、已存在输出保护、错误物种/系列/培养/历史拒绝、样本不全拒绝 |
| `README.md` 与本报告 | 入口、实际成果与未闭合范围，不覆盖旧记录 |

先复现红测，再实现并跑绿测。CSV/NPZ 读取最初由于 `lung.data.__init__` 导入训练模块而在 PyArrow 缺失时失败；没有安装假依赖、删断言或关闭测试。新增 GTF/源协议反例亦分别保存 red/green 日志。

**实际测试结果分开报告：**

- 与本轮改变及可运行相邻消费者有关的八个测试文件：**85 passed，0 skipped，0 errors**，1.17 秒；`portable-tests.txt/xml`。
- 扩大到 `test_bothe_gene_axis_bridge.py` 的九文件运行：**94 passed，6 skipped，15 errors**。15 项 error 来自缺少原 `source_status_ledger.tsv`；6 个 skip 是原测试因未上传真实资产触发，未新增 skip。本轮没有恢复这个冻结元数据文件，也没有宣称扩展测试通过。见 `targeted-tests-final.txt/xml`。
- 全项目测试、GPU、正式模型训练/forward、P/G/Q 最终测试：**未运行**。85 个单元/合成测试不能替代真实科研结果。

第一次历史输出 v1 因重复 NPZ 解压导致读回超时；v2/v3 新目录通过，v1 partial 保留而不登记成功。其他重核脚本的 manifest 路径/NPZ 字段名错误及 Bothe 空白字符串识别问题均保存失败日志，最终重新执行的完整结果独立保存。Bothe 空白值不是矛盾实测剂量，最终侧表保留原值并注明论文协议推导，未改写源 SDRF。

## 7. 可复算命令与交付结构

下面 `S` 为原数据包解压后的 sources，`A` 为包内公共轴 NPZ。输出路径必须不存在。无需本机联网或在线作业来重现本次小规模读回。

```bash
python scripts/recover_history_gene_bridge.py \
  --matrix "$S/GSE96779.fpkm.gz" --soft "$S/GSE96779.soft.gz" \
  --gtf "$S/Homo_sapiens.GRCh37.75.gtf.gz" --hgnc "$S/hgnc_complete_set.txt" \
  --axis-npz "$A" --output "$NEW/history"

python scripts/recover_public_expression.py sarin \
  --soft "$S/GSE108214.soft.gz" --annexin-docx "$S/Sarin_S8.docx" \
  --output "$NEW/native/sarin"
python scripts/recover_public_expression.py hm97662 \
  --soft "$S/HM_family.soft.gz" --matrix "$S/HM_counts.tsv.gz" \
  --output "$NEW/native/hm97662"
python scripts/recover_public_expression.py gse161800 \
  --soft "$S/GSE161800.soft.gz" --matrix "$S/GSE161800_1vs2_counts.csv.gz" \
  --extra-matrix "$S/GSE161800_7vs8_counts.csv.gz" --paper-xml "$S/GSE161800_paper.xml" \
  --output "$NEW/native/pulse"
```

交付中的 `verification/recheck_other_sources.py REPO ARTIFACT_ROOT FRESH_NATIVE_ROOT NEW_OUTPUT` 复算三来源公共轴及 TIDYI；`verification/check_bothe_metadata.py SOURCES NEW_OUTPUT` 复算 Bothe 元数据/公开表。验证脚本位于交付包，不替代项目生产模块。源数据不提交 Git。日志提供原完整运行命令和环境；新数据包保留最终 v3 结果、侧表和原来源下载身份（不依赖过期签名链接）。

补丁应用前：

```bash
git switch -c codex/protocol-source-repair-20261015 ec5be211874c6ccc64ce81b6238a689fc56fad12
git apply --check data-alignment-ec5be21.patch
git apply data-alignment-ec5be21.patch
python -m pytest tests/test_data_api_optional_dependencies.py tests/test_history_gene_axis.py \
  tests/test_history_gene_bridge_cli.py tests/test_public_expression_recovery.py \
  tests/test_external_gene_axis.py tests/test_sarin_gene_mapping.py \
  tests/test_damtc_reference.py tests/test_science_evaluation_comparison.py -q
```

若当前工作树有其他修改，不重置、不覆盖；先保留其独立分支再按补丁基线审阅。补丁包附各变更文件 SHA256，原始 data artifact 链接有有效期，下载副本及来源哈希是后续接续依据。此次没有新建/更新远端 PR。

## 8. 下一阶段优先顺序与决策条件

1. **认证真正冻结轴与公开候选轴。** 在有旧 `genes.parquet` 的环境读取原 `common_gene_index/ensembl_id`，逐 ID/逐顺序比较并重算上面的 UTF-8、LF 哈希，连同 normalization/角色/检查点哈希核对。相同只关闭轴身份；不同必须新建独立投影，不能套旧权重或改写冻结资产。此步骤只读，不恢复训练。
2. **完成 Bothe 原始计数与冻结 bridge 原件。** 核实际 12 个 run 的官方完整性、计数 leaf、`counts_complete` 生产 manifest、`source_status_ledger.tsv/joint_axis.tsv` 及对应原哈希；重新运行 Bothe 专项测试。只有完整发布后才检查现有 `prepare_bothe_observation_inputs.py` 的只读 preflight。数据缺失继续 pending；不得用 DE 表替代计数，也不自行恢复旧后台作业。
3. **补足可区分状态、处理和功能的同条件支持。** 首先将新历史 RNA 与 Sarin 等功能挑战在来源/协议层隔离，补查对应结构、培养密度、采样和独立实验身份。Sarin 限定三条件可作外部反证，不够支持高维状态增量就如实报告；不能从三个条件训练再评价同一 G。
4. **在真实资产齐备后，按同信息基线运行新来源 E/D/P 及独立轴实验。** source-specific、分布映射、共享表示并列，不预设共享一定成功；新剂量/时间留出必须连 AE/归一化一起隔离。任何真实训练仍遵守用户现行停止和授权边界，本文不构成启动记录。
5. **继续真实序贯终点补链。** 保留 A549 顺序/洗脱/间歇目标，核验可以提供独立终点的完整处理方案及对照，再决定 RL 是否具有可检验必要性。找到数据须先处理后验证；找不到足够支持则该环节仍未闭合，不改成单步题目后宣布成功。

最终标志保持：`whole_project_qualified=false`。已做完的源值、身份投影与有限条件连接可继续使用；尚未建立的共享表示、独立功能迁移、联合选择收益和真正序贯验证仍是完整目标的必需部分。
