# 数据链恢复、真实读回与全架构验收（2026-10-15）

基线：`4a78560bdeb2e0d98332d5a9520764a120abf612`。本文件更新早期“仅审计、无矩阵”的记录；恢复发生于 2026-10-14 UTC，日期按本次任务的北京时间。研究主线、完整序贯范围、冻结划分和奖励不变。**本轮已有真实矩阵、功能原值、对应关系和运行证据，但完整科研可行性仍未通过。**

## 1. 本轮实际完成与证据等级

通过只读 GitHub Actions 导出基线 Git bundle，在当前会话云端完整克隆并验证；下载器也是本轮编写的普通确定性 Python 程序，不调用模型、其他 AI 或本机。GitHub 连接器负责分支写入。没有正式训练、GPU 模型推理或原始 FASTQ 批量下载。

两批公开来源分别取得并重新验证 42、46 个文件的大小与 SHA256。失败另存：Inde/Mendeley 返回 403；10x 页面在下载器中返回 429；sci-Plex pData 和休眠论文的大补充文件超过该次下载的 80 MB 单文件限制。这些限制**不是不存在数据的证据**。GEO、BioStudies、Tahoe 固定版本、论文源表均保留实际 URL、源名、版本/快照和校验值，见两个 `configs/public_alignment_*_v1.json`。

### 已实际恢复并逐值读回的表达

| 来源与严格范围 | 样本/细胞 × 原始基因或探针 | 本轮保留的观测域 | 32,461 共同身份轴上的可测 / 缺失基因 | 验收范围 |
|---|---:|---|---:|---|
| GSE306207，A375 四条件 | 8 × 62,710 | 原 TPM，小数保留 | 32,461 / 0 | 全部 8 列、样本标题/别名/背景、处理历史与所有数值；不是 A549 |
| GSE333828，A549 子集 | 6 × 57,905 | Salmon estimated counts，小数保留 | 30,268 / 2,193 | 原 18 样本文件中明确选择 6 个 A549；不是把 H841/SBC5 当 A549 |
| GSE161218，H1792 | 9 × 58,381 | 原整数 featureCounts | 32,461 / 0 | 全部 9 列、DMSO/Lestaurtinib/Trametinib；不是 A549 RNA |
| E-MTAB-16354，官方 TIDYI A549 | 368 × 32,858 | 原单细胞整数 counts | 32,461 / 0 | 官方 369 列中 1 个背景未知者隔离；完整原标签/孔/板/QC 保留 |
| GSE108214，亲本及耐药 A549 | 22 × 29,729 探针 | 原 processed microarray VALUE，log base 不猜测 | 12,070 / 20,391 | 654,038 个原探针数值全保留；仅唯一可信探针映射进共同轴 |

前四包共 13,466,283 个源数值，加 Sarin 全探针为 **14,120,321 个源数值**；它们是表达矩阵元素数，不是细胞数或统计独立样本数。共同轴由固定 Tahoe 基因身份与 TIDYI 来源重新构建，有序 ID 已核；**不是声称复现了未上传冻结 normalization 的全部字节**。所有包同时保留原轴和共同轴、NaN/mask、原始样本身份；没有归一化、补零、插补、训练或角色分配。

Sarin 的 10,831 条多探针对应和 6,828 条未解决/轴外记录留在台账，不平均制造“单基因实测”；HGNC 为本次快照，Ensembl75/GRCh37 是显式注释桥，不冒称作者 2018 年原注释版本。TIDYI 目前仅官方单板来源，不能冒充历史联合索引的 743 个细胞或跨板评测；原 QC 与表达求和不一致记录单列，不静默改原值。

### 已实际恢复的功能与连接

**Macaya 2023：**官方 XLSX 两个 Annexin 面板共 144 个位置、142 个有效百分比、2 个原始缺失；其中 A549 为 **22 个有效值**，不是全部 142 个。百分比与除以 100 后的 fraction 都保留，源工作表/单元格/实验编号/分母/背景和药物组件可追溯。两个缺失不补零；两个面板的同名 Exp 不能自动算成同一或额外独立生物重复。48 个背景—面板—处理条件中，只有 H1792 的 3 个条件连接到本来源 RNA：**16h RNA → 24h Annexin，不同实验的条件级连接**，不是同孔/同细胞/逐重复配对。RNA DMSO 与功能图“untreated”对照等价仍未知。[R1]

**Sarin：**从 2017 原 S8 DOCX 的第 6 表读取 **10 个 Annexin 相对倍数**，对应 2018 年的 22 份 RNA，建立 3 条跨研究条件连接：亲本 11 μM、耐药 11 μM、耐药 34 μM，均为 24h。10 个 Y 始终只存在 10 次，不向每个 RNA 重复复制。倍数不换成绝对凋亡率；绝对对照分母、跨研究培养批次、维持顺铂到挑战的额外洗脱时间未知。[R2–R3]

**药物与协议：**9 个 PubChem 名义全结构、9×2048 位原化学入口 Morgan 指纹已保存、逐位重新计算一致；154 个功能源位置保留完整药物组件列表，不按组合成分扩增 Y。9 个 RNA 条件组保持有序处理事件及样本成员；HM97662 缺可信结构继续 unresolved，不赋虚构指纹。名义结构匹配不认证实验试剂批次、盐型、溶剂化物或金属配位几何。

**sci-Plex：**实际读回 1,920 个 A549 计划孔（24/72h），核 plate+well 唯一；原 dose 列没有单位，依据原论文四剂量网格单独增加 nM→μM 解释侧表，保留“单位不在源列中”与原始值，记录转换公式。观察时刻不自动等于已核连续 active exposure。未读取其表达矩阵，本轮不授予这些孔 RNA-readback 资格。[R4]

### 已排除的错误补齐路线

Bothe 三个官方 DESeq2 文件各 24,163 行，只有 baseMean、log2FoldChange、lfcSE、stat、pvalue、padj，**不是 12 样本表达矩阵**。本轮真实读取了这些对比统计，但不能由此还原十二份计数。历史 4/12 计数是交接记录；未读取本机 count leaf，不改记为已完成。[R5]

SequenTx 原 Supplementary Table 7 的四条待用药名—SMILES 记录与本次名义结构比较不符：Trametinib（Excel154行）、Oxaliplatin（389行）、cisplatin（398行）、midostaurin（400行）。例如 cisplatin 名下字符串没有 Pt。原字段完整保留并排除精确结构连接；**本发现针对该补充表，尚未证明作者可执行训练资产使用了相同记录，不能由此指控整个模型或所有结果错误**。Tahoe 的 Resveratrol/Oxaliplatin 与参照连通图一致但立体信息不完全一致，也没有自动合并为完全相同实体。[R6]

## 2. 哪些数据链闭合、哪些仍未闭合

“闭合”必须明确层级：以下恢复包的数据工程链闭合，不等于完整科研目标闭合。

| 架构环节 | 本轮闭合的实际部分 | 保留的关键缺口与不能升级的结论 |
|---|---|---|
| E/C 状态表示 | 官方 TIDYI 原表达—同细胞分选标签、原/共同身份轴与来源元数据 | 第二板未恢复；FACS 分选类别频率不是自然群体凋亡率；分类效果、跨板迁移未测 |
| P 响应 | 3 个 bulk 来源完整本轮子集；Sarin 22 RNA；相容读回消费者 | Tahoe/sci-Plex 全表达、相容参数、新 protocol 监督、P 真实独立评估未齐；同期 vehicle 非单细胞 t0 |
| 跨来源表示 | 身份、顺序、观测域、缺失语义严格可读 | 共同轴≠共同测量单位；独立 E/D≠共享 latent；无真实跨来源迁移效果 |
| G 独立凋亡 | 142 个 Annexin%＋10 个 Annexin fold 原值；6 条含 RNA 的有限条件连接 | Macaya RNA 在 H1792；Sarin 两状态一药不足高维泛化；HM 仍缺完整原始功能重复与结构；没有已验证 P→G |
| 联合候选选择 | 功能观察身份、名义结构、组件、剂量/时钟及排除记录 | 没有覆盖 A549 状态×多药×剂量×时长的充分独立测试；未训练/验证 G、Q、RL |
| 多步模型滚动 | A375 四条件真实 RNA＋顺序/洗脱未知字段，Bothe 对比统计 | A375 不是 A549；没有普遍第二步转移监督；没有逐步真实预测验收 |
| 真正序贯促凋亡 | 找到 A549 真序贯功能原论文线索；未冒充已读 RNA–Y 轨迹 | A549 顺序、反序、同时、单药、共同终点的可复算原 Y 与相容 RNA 尚未闭合；完整目标保持未通过 |

各来源数据独立存储，只有明确条件/身份边才联接。`role=unassigned` 是本轮恢复对象的实际状态；未通过重命名产生 fit/calibration/final，也未把已读源表称为前瞻盲测。本机正式训练停止安排不变。

## 3. 主问题、变量、可识别性与方法取舍

### 3.1 预测单位不等于实验独立单位

令 S 为测量来源，B 为细胞背景，R 为实验板/培养重复，M 为观测 mask，H 为有序处理历史，a=(化学实体、剂量及单位、暴露/恢复事件)，τ 为实测采样时间，Xref 为允许使用的参考表达分布，X1 为处理后表达，Yᵏ 为 assay k 定义的功能终点。

未配对快照可以在合格随机化/可交换参考下监督条件分布 `P(X1 | a,H,B,R,S)` 与群体差异；可以检验 Xref 是否改善预测。它本身不能唯一识别 `P(X0,X1 | a)` 或个体因果效应：相同两个边缘分布可以有多个联合耦合。最优传输、CFM 选定的是一种模型耦合。**不要求所有任务同细胞药前药后测量，也不制造不存在的配对。**

统计比较应按实验块/独立培养重复或预定化学组件聚类，而非把孔内数千细胞算成数千药效重复。患者个体化只是应用前景，当前模型不具有临床疗效认证。

### 3.2 状态信息的可证伪增量

比较同信息预算的 independently trained context-only、mean-h、observed-h，以及有可交换性依据的分层 state shuffle。预先指定 held-out drug、实验板/重复和来源范围，目标条件与随机抽样一致。零假设为加入参考表达不能降低该外部目标分布的预测风险；用组件/实验块配对 bootstrap 或置换，不伪造独立重复。

如果增益只出现在随机细胞划分，或去掉板/来源捷径后消失，不能宣称生物状态增量。Sarin 共同11μM可否定部分状态不敏感预测，耐药11/34μM可作有限剂量选择；但仅两状态，**不能分离高维表达与 state-ID 的优势**。TIDYI 单板不能替代跨板资格。建议新增10x A549 CRISPR×药物因子数据作为补充候选，但目前下载未闭合，dose/time仍须核对；不把它算入本轮数值成果。

### 3.3 跨来源不默认强制共享

推荐保留原架构中的 P/G 职责，先用**来源适配的观测域与明确的分布目标**，比较来源特异模型、共享表示＋来源 decoder、条件分布映射三条路线。这里只提出比较方案，不替换原主模型。

来源 s 内若 E/D 能重建，任一可逆 T_s 都可构造 `E'_s=T_s∘E_s, D'_s=D_s∘T_s^{-1}` 而保持重建误差，因此重建本身不识别跨来源共同 latent。需要独立可比条件上的 transfer/retrieval、真实响应误差与单位/缺失敏感性。source probe 可诊断，但“能预测来源”不单独否定生物信息，“不能预测来源”也不证明有效共享。缺少重叠化学—剂量—时间锚点时，应明确外推假设与失败域，不能靠 adversarial 去来源抹掉真实生物差异。

### 3.4 功能终点不能由表达自证

Annexin fraction、相对 Annexin fold、caspase 活性/积分、SYTOX 死亡比例、sub-G1、增长抑制各属不同 Yᵏ。原则是原尺度独立头/评价，源特异校准须使用训练或校准集合；不能由 fraction=1、fold=1、growth=0 互换构造目标。

HM97662 原论文正好提供反对证据：A549 的 apoptosis 相关基因集可富集，但作者实测未见对应凋亡诱导，并解释可能为 PRC2 靶基因解除抑制；这不是“表达签名等于凋亡”的正例。原 Fig3A 有3次独立实验的描述，但本轮没有取回逐重复 Y，不从图文编造它们。[R7]

无配套 RNA 的 A549 Annexin 面板仍可作外部候选排序验证：冻结模型后，在精确协议/背景和名义结构可信的交集中，评估排序、校准和错误。它不能证明特定单细胞轨迹导致该 Y，也不能把不相容的16/24/72h条件拼成同一测量。

### 3.5 RL、规划和单次枚举

单次有限候选集 A 下，按同一冻结 G 分数枚举 `argmax_a G(s,a)` 已是该评分器的精确最优；额外 Q 不能以更高同一内部评分证明真实药效增益。若对所有候选有统一真值误差 `|G−Y|≤ε`，所选动作的实测 regret 至多2ε，推导为真最优和预测最优两处误差相加。此上界需要所有候选误差受控，不由平均 MSE 自动推出。

正式 RL 适用于动作改变后续状态、延迟终点、历史作用和可复用转移数据的任务。观测应包括 Xref/状态分布、B/S、已知历史与未知 mask；动作包括药/剂量/暴露/洗脱；转移由真实匹配协议训练的 P 建模；奖励沿既定实测凋亡目标，不新增剂量/成本惩罚；模型 G 的反馈是代理训练信号，独立评价必须使用未用于调参的真实 Y。相比枚举/beam/search 的必要性要按相同候选预算和信息证明，而不是增加模块。

真实序贯需要 `P(Xnext | current state,a,H)` 的支持或经检验的状态充分性/迁移假设。单步药物终点不自动识别第二步；ODE/flow 的生成积分时间不是生物小时。保持单次、模型滚动、真实序贯三个不同验收标签。Gambogic acid→cisplatin 的 A549 原论文包含顺序、反序及同时处理比较，但增长96h与 flow/sub-G1 的24/48h描述不能强行套同一时钟；这是待恢复外部序贯功能证据，不是本轮已完成数据包。[R8]

## 4. 独立实验设计（待审阅，不覆写冻结角色）

| 子问题 | 单位/划分与防泄漏 | 强基线、消融、指标 | 失败判据与决策 |
|---|---|---|---|
| 状态增量 | 以培养块/板/化学组件分组；比较相同a/H/B；fit内归一化/基因筛选 | context-only、state-ID、mean-h、real-h、分层shuffle；mean error、MMD/分布误差 | 去混杂后无增益或CI含无实际意义区间：不扩模型复杂度；仍保留研究目标但结论为未支持 |
| 新药 | 原冻结 chemical holdout不改；近结构连通组件不跨角色 | 简单线性/邻近药、meanresponse、当前P；逐条件及等权组件 | 只在训练药或高回收孔有效则不能称未见药泛化 |
| 独立剂量/时间 | 新namespace，不复用曾见留出轴RNA的AE/normalization/h；原D1/T1提案仍待批准 | dose log-linear插值、24h响应沿用、独立重训P | Tahoe板变化和sciPlex24/72h板变化必须单列；不称纯时间因果或任意连续时间 |
| 表达→功能 | 在来源/背景/完整协议可解释交集中构造条件级连接；group/研究隔离 | drug+dose+protocol-only vs 加预测RNA；各Y域MAE/校准/排序 | RNA特征增益消失或依赖不同域合并则P→G主张未支持 |
| 单次联合选择 | 同一候选全集、同信息、相同真实Y；预先锁定可行候选与方向 | 直接G枚举、药物平均、监督排序、Q；top-k/NDCG/regret | 只赢随机或只提升G自身分数不算额外方法价值；候选不全不能称全域oracle |
| 多步/真正序贯 | 历史/完整方案留出，时钟/洗脱/反序/单药/同时对照；共享对照聚类 | 忽略历史P、greedy、枚举/beam、RL；逐步真实误差、最终真实Y | 只生成稳定轨迹不通过；A375辅助不授A549资格；观察对照不足记不可判定，不强判无效 |

现有源表已被人读过，因此后续是明确隔离的新计算评估，不是回溯声称“从未见过 final”。小样本只有组均值时，不输出虚构的生物学 CI。对当前 Sarin/少量 A549 面板不做夸张显著性承诺。

## 5. 最近工作与可检验差异

两互补检索入口为网页学术/官方数据检索，以及实际下载的 EuropePMC API 检索JSON；关键判断回原论文、GEO/BioStudies、作者补充表。Macaya、Sarin、sci-Plex、Bothe、HM97662、Gambogic、Inde 的相关全文/方法或补充原值已核；SequenTx 主文网页提供摘要与扩展图，完整主文受访问限制，补充 PDF/XLSX 和 GEO 数据已取得，不能写为主文全文无缺阅读。作者 executable asset 对照尚未完成。

| 接近工作 | 已解决与相关边界 | 本课题尚待验证的差别，不是既成创新 |
|---|---|---|
| sci-Plex 2020 [R4] | 化学单细胞、多剂量、24/72h异质性；不是活细胞同细胞t0→t1 | 分离参考信息与板来源、独立协议/功能链接 |
| CPA 2023 [R9] | 扰动组合/剂量、协变量组合预测 | 不将已见drug-ID的剂量外推说成新化学；完整历史和独立凋亡需另外验证 |
| SequenTx 2026 [R6] | RL驱动序贯方案与实验验证；已有相近主线 | 不能主张“首次RL选序贯药”；应验证状态信息、协议可迁移性及实测促凋亡，而非仅生长代理 |
| Ahlmann-Eltze等2025 [R10] | 遗传扰动任务中深度基础模型未胜简单线性基线 | 负面证据要求强基线；不能直接移植为所有化学响应问题的否定定理 |
| PerturbBench [R11] | 跨数据/模型/指标比较揭示评测依赖 | 保持任务、信息、独立单元一致；新模块数不构成创新 |
| Macaya、Sarin、HM、Inde [R1–3,7,12] | 实测功能、少量状态、或死亡动力学证据；终点不同 | 实证检验RNA是否超出drug/protocol-only，而不是将不同终点拼表扩样 |

共享表征＋RL 的堆叠不单独构成创新。可发表的主张需在上述独立比较中通过；出现负结果同样必须保留并改变后续方法选择。

## 6. 实现、调用路径与实际验证

`src/lung/data/external_gene_axis.py`：修复重复样本列允许扩增、空选择、缺失ID和带版本 `_PAR_Y` 身份漏配；新增明确的 TPM、estimated counts、microarray、single-cell raw domain。原 raw counts 整数要求没有放宽。源未测基因不进入真实零/重建目标。

`src/lung/data/public_source_recovery.py`：源hash锁定、GEO literal字段、精确列别名/背景检查、原/共同轴写入和全值回读；消费者拒绝错单位、错基因顺序、错背景或缺hash。独立功能合同拒绝重复Y/源位置、假RNA重复配对、缺失补零和跨终点单位混用。

三个实际入口：`recover_public_expression_bundle.py` → `recover_public_function_links.py` → `audit_public_crosslinks.py`。复用已有 `prepare_gse161218_bridge.extract_annexin`、`align_sarin_state_expression` 和 `build_chemical_catalog`，不重复另造标签抽取器；不调用旧主函数覆盖历史路径。包通过真实源文件运行，而非仅提供空接口。它们仍是**数据准备/读取路径，不是偷偷接入原冻结训练器**。

### 实际命令及产物

```bash
python scripts/recover_public_expression_bundle.py --source-root /mnt/data/public-data --output /mnt/data/recovered-expression-v2
python scripts/recover_public_function_links.py --source-root /mnt/data/public-data --followup-root /mnt/data/followup --expression-root /mnt/data/recovered-expression-v2 --output /mnt/data/recovered-function-v2
python scripts/audit_public_crosslinks.py --source-root /mnt/data/public-data --followup-root /mnt/data/followup --expression-root /mnt/data/recovered-expression-v2 --function-root /mnt/data/recovered-function-v2 --output /mnt/data/crosslinks-v2
python -m pytest tests/test_public_source_recovery.py tests/test_public_function_recovery.py tests/test_external_gene_axis.py tests/test_sarin_gene_mapping.py tests/test_normalization_bridge.py tests/test_endpoint_alignment_v1.py -q
```

三CLI均 exit0；本环境 wall-time 分别19.54s、11.58s、1.69s，最大RSS分别1,199,304、600,720、220,232 KB。只是本次 CPU 数据恢复测量，**不是 RTX3080 完整训练资源估计**。Python3.13.5、pytest9.0.2、torch2.10.0+cpu、numpy2.3.5、pandas2.2.3、scipy1.17.0、RDKit2025.09.4、pyarrow22.0.0、tensorboard2.20.0，CUDA不可用。

针对性 **68 passed / 1.44s**；有真实失败的red-phase与修复后日志。全仓库测试尝试过：先补齐环境依赖；之后完整命令200s未结束，不能称全通过。`-x` 首个失败为 `test_prediction_generation_matches_legacy_and_rng` 的 `AUX_DEPENDENCY_MISMATCH: response`，34passed、2skipped后停止；**在干净基线单独复现同一失败**，未降低依赖绑定断言。扩大到含冻结资产的测试另有4failed、83passed、6skipped、15errors，主要缺本机Bothe台账和Martin资产；这些不记成通过，全部保留日志。

PR 上已有的 vehicle 结构守卫及 Mphahlele 来源变更保留，不覆盖其文件，不将其未在此执行的测试累加到68。审阅本轮应以该次真实输出和文件hash为准。

## 7. 接续：最先做的五项，不自动启动本机正式训练

1. **本轮数据接收验收**：解压交付包，核总 SHA256SUMS；读取每包manifest并调用 `load_expression_packet`，指定域/背景/有序ID；保存 `source_expression.npz/expression.npz/samples.csv/gene_mapping`。本轮包与旧冻结joint只做ID/元数据比较，不能直接替换旧checkpoint依赖。
2. **补 Bothe 十二样本**：在有本机原件的环境核每对官方MD5、完整gzip/配对检查、count执行回执、每个样本计数、17,107可测共同基因及mask，使用现有 `bothe_count_production`/`bothe_observation_inputs`。产物须12/12及全值投影回读；不足则记录具体leaf，不能用本轮DE统计代替。不恢复被停止的历史作业，需按用户新的执行授权启动。
3. **优先补 A549 功能链而非堆细胞数**：取HM原补充逐重复Y/完整结构；恢复 Mendeley/10x 失败来源的原字节或官方替代入口；按背景、剂量、曝光/采样、洗脱及终点域记录匹配/拒绝。结构未解不设零FP。若只有同药单条件，定位为负例/外部校准，不包装为多候选训练集。
4. **先批准新的独立轴与表示实验，再运行**：在原角色不变前提下单列版本；重建该域fit-only统计/AE/P，比较来源特异/共享/分布映射及状态消融。真实数据/参数需求、旧D1/T1板混杂和与当前3080资源测量都必须登记；若仅source shortcut有效，改变表示路线而非改final。
5. **序贯证据与策略必要性并行验收**：先将A375四条件、Bothe历史和A549真实顺序功能分别建表，不跨背景补标签。对真实Y覆盖的同一候选集做直接枚举/排序/规划/RL对比。只有独立真实转移与终点覆盖成立，才能宣告真正序贯优化可行；否则该关键环节继续未闭合，不从主目标删除。

最终总标志为 `whole_project_qualified=false`。数据恢复完成的范围如§1，不因代码可运行、局部对齐成功或某一基线改善而扩大。

## 8. 可定位主要来源

- [R1] Macaya等2023，DOI https://doi.org/10.1038/s41467-023-41828-z ，Methods Apoptosis、Suppl Fig2C/D与3D/E；MOESM7源表Fig Suppl2/3；GSE161218官方SOFT与9counts。
- [R2] Sarin等2017，DOI https://doi.org/10.1371/journal.pone.0181081 ，sec004/sec009、Fig5a、S8第6表；[R3]2018，https://doi.org/10.3390/ijms19030767 ，Methods4.2–4.3、GSE108214/GPL17077。
- [R4] Srivatsan等，https://doi.org/10.1126/science.aax6234 ，Results“sci-Plex scales to thousands of samples and enables HTS”；GSM4150378官方hashTable、gene/cell annotations。
- [R5] Bothe等2021，https://pmc.ncbi.nlm.nih.gov/articles/PMC8403771/ ，E-MTAB-9923官方SDRF/IDF及三DEseq2对比原表。
- [R6] Chen等2026，https://doi.org/10.1038/s42256-026-01192-1 ，SupplementaryTables4–10（MOESM4），Table7指定Excel行、Table10实验清单，GSE306207官方八RNA TPM；不能把预测分数当实测Y。
- [R7] HM97662，https://doi.org/10.1016/j.neo.2026.101349 ，Fig3A、RNA-seq Methods、Discussion；GSE333828官方raw_counts.tsv.gz其实是Salmon估计计数。
- [R8] Gambogic acid/cisplatin，https://doi.org/10.1038/bjc.2013.752 ，Fig1B（48h→48h/洗脱/同时对照），Flow cytometry与Fig2时间/终点另核。
- [R9] CPA，https://doi.org/10.15252/msb.202211517 ，https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/ 。
- [R10] Ahlmann-Eltze等2025，https://doi.org/10.1038/s41592-025-02772-6 ，遗传扰动基准的范围不能泛化成所有化学任务定理。
- [R11] PerturbBench，https://doi.org/10.1038/s41592-025-02980-0 ，比较性评测；本轮用于设计参考，不声称运行其benchmark。
- [R12] Inde等2020，https://pmc.ncbi.nlm.nih.gov/articles/PMC7409774/ ，SYTOX死亡动力学、Mendeley9bccjb75wp；原数值下载本轮403未闭合。

出处与本课题推导/建议分开：§1原数值是实际计算结果；§3数学为明确假设下的推导；§4/7是未实施实验方案，不是已有科学增益。
