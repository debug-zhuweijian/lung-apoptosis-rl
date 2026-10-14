# 数据对齐第二十一轮：真实 RNA 输入与完整协议请求

日期：2026-10-11。继续使用 6.1-sol 专项执行与独立复审。目标仍为完整实现数据对齐及整个课题的数据可行性。**四项真实数据入口已通过实际读回；完整 A/B/C/D 与 E/D/P/C/G/Q 同链尚未验收。模型训练保持停止。**

本报告的A/B/C/D分别指状态信息增量、未见条件可靠性、实测功能一致性、联合候选选择增量；模块E/D/P/C/G/Q则依次为表达编码、来源解码、响应预测、TIDYI原始分选状态评分、实验定义的终点头、动作价值。两组字母的职责不同，详见[已确认研究架构§4.3](G:/研究生科研文件/毕业论文/研究架构.md#43-表示响应与评价层)。RNA数据入口通过不等于Annexin识别或策略选择已完成。

## 1. 本轮进展与证据粒度

| 工作 | 实际结果 | 当前范围 |
|---|---|---|
| Martin RNA 输入包 | 两个19,712×60作者processed表，2,365,440个float64原值全部一致；source/joint双坐标与LOBO/cycle角色可读 | 已观测RNA准备完成，尚无bulk模型/功能资格 |
| Martin RNA协议联动 | 16唯一RNA组、40已报告名义事件、64LOBO+64cycle职责与原RNA/source/FP/mask联动；CPU128视图实际通过 | 名义阶段数据可消费，未补操作耗时/培养日程或model context |
| Martin观测坐标 | 8fold实际准备、每LOBO45/cycle30唯一fit GSM；32461轴/16167观测/16294NaN/mask保留，主代理重算moments与公共API通过 | 新可逆affine数据设计，尚非来源E/D消费或科学迁移通过 |
| 完整协议请求入口 | 32登记身份全部保留，20Viel条件的FP/剂量/暴露/终点/CPU physical实际读回 | 数据实参完成，full producer和匹配decoder仍未接通 |
| 独立复审 | RNA28＋完整请求15＋RNA协议32＋观测坐标13，共88项输入相关测试，各范围独立复审CLOSED；传输定向检查另记 | 不与前轮131项简单相加，不替代科学验证 |
| 主代理实际验证 | 全部源值、joint观测/缺失位置、60样本、32身份及20实参通过 | 没有ML forward、PT加载、训练或核心Y解码 |
| HM97662化学核查 | 供药方/溶剂确认；所查官方入口未取得可认证完整结构 | 结构FP输入资格保持false |
| Bothe数据恢复 | pilot R1整端官方MD5及gzip CRC/EOF已通过；direct-v3续传R2，0296独立direct恢复已执行 | 尚无双端正式先导回执，真实counts验收未完成；旧失败历史保留 |

[主代理逐值读回](../experiments/20261011-data-alignment-round21/root_actual_input_readback.json)是本轮实际数据证据；[第二十轮](DATA_ALIGNMENT_ROUND20_2026-10-11.md)保留前轮来源、计数工程和功能连接背景。

## 2. Martin真实RNA的来源、单位和读取

新[原生RNA输入包](../08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/native-rna-inputs-v1/README.md)、[MartinNativeRNA](../src/lung/data/martin_native_rna.py)及[prepare入口](../scripts/prepare_martin_native_rna.py)均已实际准备和读取。

| 字段 | 实际内容 |
|---|---|
| 原轴 | 19,712个作者gene symbols、60个RNA生物重复样本 |
| log2TPM域 | 作者GSM data_processing注明log2(TPM+0.01)；负值合法 |
| estimated_counts域 | Salmon/tximport lengthScaledTPM估计count；非整数合法，不是UMI |
| joint视图 | 60×32,461；既有固定symbol→joint位置映射，16,167观测、16,294缺失NaN/False |
| 真实样本/组 | 12个parallel parental DMSO reference、48个target、16背景—处理状态组 |
| LOBO64请求 | 每fold三背景fit/一背景final |
| cycle64请求 | 另立每fold两fit/一selection/一final；不覆盖LOBO |

source NPY原字节复制，joint观察值逐值一致。原表是symbol轴，不能写成作者直接提供exact Ensembl ID。128角色请求是16组观测的职责视图，不能扩成128独立实验。没有归一化、四舍五入、缺失补零或把RNA三重复复制成功能Y。

`reference(request_id,system,role,domain,coordinate)`只选参考；`supervision()`才读target。返回只读sample-major float64、同shape mask、GSM/列号、基因身份、单位和真实协议。角色明确开放，final还需allow_final；平行三reference与三target不构成同细胞、同孔或逐重复前后配对。

root实际读取H1975 long washout fit请求：参考3×32,461、source target3×19,712，暴露504h/观察576h，reference-only不选target，Y未接入，模型资格false。完整读回见[执行脚本](../experiments/20261011-data-alignment-round21/verify_round21_inputs.py)和[结果](../experiments/20261011-data-alignment-round21/root_actual_input_readback.json)。

### 2.1 原RNA时钟与四药72h功能实验分开

| RNA状态 | Osi | 暴露 | 无药恢复 | 观察clock |
|---|---:|---:|---:|---:|
| acute，各背景 | 500nM | 24h | 0h | 24h |
| DTP，各背景 | 500nM | 504h | 0h | 504h |
| short washout，各背景 | 500nM | 504h | 24h | 528h |
| long，H1975 | 500nM | 504h | 72h | 576h |
| long，HCC827 | 500nM | 504h | 96h | 600h |
| long，PC9 | 500nM | 504h | 168h | 672h |
| long，HCC2935 | 500nM | 504h | 240h | 744h |

root亲读[PMC10035388 XML](../experiments/20261011-data-alignment-round16/state_function_new_sources/PMC10035388.xml) Methods `sec2-9`：RNA采用21天Osi后两次PBS洗涤、换无药培养基和各背景恢复；parental是平行21天培养后Osi/DMSO24h，不能写给药前t0。

`sec2-4`每周两次洗涤/含药换液属于Incucyte Growth Assay。RNA`sec2-9`没有声明此维护日程，不能移植为504h中的已知media events。`sec2-1`也没有可补齐RNA medium全配方/容积的字段。新接口只编码有出处的名义阶段，维护、精细洗涤时钟、retention/material等unknown保留。

旧gate名称bulk72不是这60个RNA的真实时间。四药72h caspase挑战与本RNA的药物/剂量/时钟不同；本RNA请求保持`function_72h_four_drug_target=false`。[依赖审计](../experiments/20261011-data-alignment-round21/bulk72_input_dependency_audit/README.md)保留GSM与源码定位。Martin为非A549/caspase辅助，不能替代A549 Annexin证据。

### 2.2 bulk消费仍需补的部分

旧single-cell producer使用library10000/log1p/标准化坐标；本source log2(TPM+0.01)/estimated counts、负值、非整数及缺失mask不能直接套进去。现factory、action、standardized、decoder及physical_projection实际调用见[静态依赖审计](../experiments/20261011-data-alignment-round21/bulk72_input_dependency_audit/README.md)。

仍缺source-specific bulk编码/解码、明确预处理与观测损失、真实历史消费路径及匹配参数。这是工程与模型职责缺口，不能概括为只缺权重，不能靠换source字符串关闭。

### 2.3 真实RNA—物理协议—角色联动已经实际补齐

新[observed-protocol-inputs-v1](../08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/observed-protocol-inputs-v1/README.md)由[MartinObservedProtocol](../src/lung/data/martin_observed_protocol.py)消费，[prepare](../scripts/prepare_martin_observed_protocol.py)实际成功。16个RNA组对应40个名义事件：acute/DTP各一个drug interval，short/long各drug→wash→media_change→recovery四阶段。没有添加未报告的21天维护事件。

原有`full_physical_protocol_v1` schema内保留Osi500nM/名义parent FP、真实暴露、恢复、观察clock、PBS2与unknown masks。wash/media的start=end504是已登记transition记号，不是实测操作耗时0。mask中的未知retention不变成生物学零。medium配方/容积、精确洗液时钟、stock/salt和维护日程继续unknown；不能称网络已经消费全部培养条件。

新consumer按登记request/system/fold/role查组和构造physical_batch，reference/supervision委托已关闭的MartinNativeRNA；原source/joint/单位/mask原样保留，不重新制造配对。base_design的source/background/protocol indices均None、numeric_model_kwargs_ready=false，duration24/504与horizon名义endpoint明确为未验证设计；没有给非A549背景填Tahoe/A549模型ID。

owner实际128计算视图/320重复计算事件及256双域读取通过，独立生物计数仍16组/40事件/48targets/12references。root再次核128 CPU视图及16组所有clock/FP/dose/PBS/unknown，两个source/reference与joint/target数值例读精确，见[主代理实际读回](../experiments/20261011-data-alignment-round21/root_observed_protocol_readback.json)。这是显式开放所有RNA角色的数据审计，没有拟合或模型使用final表达。

独立复审发现False→int0重哈希字段仍可通过普通dict等值；修后采用canonical JSON比较m/groups/role_links，类型、列表次序与nonfinite严格核对。32tests及9项独立复验通过，原groups/role_links/RNA字节未改，只刷新新loader绑定。[CLOSED报告](../experiments/20261011-data-alignment-round21/independent_martin_observed_protocol_review/technical_review.md)限定工程/真实协议，未授bulk decoder、参数或功能资格。

### 2.4 bulk观测出口设计与下一步数据准备

新[bulk/source出口设计](../experiments/20261011-data-alignment-round21/bulk_observation_outlet_design/README.md)依据真实reader、角色和旧模型源码。按fit请求的reference与target GSM去重，每LOBO fold为45唯一样本，每cycle fold为30；同一参考被多个请求使用不重复计入拟合统计。

新[observation-coordinates-v1](../08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/observation-coordinates-v1/README.md)及[MartinObservationCoordinates](../src/lung/data/martin_observation_coordinates.py)已实际准备/读回：8个fold共同32461轴，16167观测/16294缺失，每LOBO45/cycle30唯一fit GSM。可逆affine采用float64/ddof0，常数基因scale=1并标记；本次真实8fold的constant均0。作者log2(TPM+0.01)原值/角色不覆盖，留出不参与坐标拟合；这是本课题新设计，不冒称作者算法。

13tests与9项独立检查通过，[独立CLOSED](../experiments/20261011-data-alignment-round21/independent_martin_observation_coordinates_review/technical_review.md)直接按原RNA/roles重算全部8fold NPZ，且修改坐标或scaleNaN后自更新hash仍拒绝，类型/域/轴/角色不能伪造。owner在全部60唯一RNA的回转审计中最大误差3.552713678800501e-15；root另按各fold去重fit重算moments精确一致，并实际用公共API读8fold参考/监督，最大误差1.7763568394002505e-15，见[root实际结果](../experiments/20261011-data-alignment-round21/root_martin_observation_coordinates_readback.json)。两误差来自不同读取范围，不代表模型精度。

有限计算占位可选mask=False时设0，原source values及inverse仍NaN，独立mask同行；缺测不变成实测零。root本次仅fit RNA数据审计，没有功能Y/ML/PT/训练；模型与科学资格恒false。每次read按固定原source及role重建坐标，default只读reference，final仍需原reader显式权限。

现SharedAutoencoder与LatentMember为单decoder，Compact encoder无source/domain/mask职责；NormalizationCoordinates仅支持同轴、同base/单位、全finite的可逆坐标。将Martin变成旧SC的library10000/log1p域不能证明观测域一致。后续E/D需来源输入与mask、匹配连续观测头及真实监督，不能把数据坐标准备称source E/D完成。

## 3. 完整协议入口的32个身份

[FullProtocolRequestInputs](../src/lung/data/full_protocol_request_inputs.py)和[真实新包](../experiments/20261011-data-alignment-round21/full-protocol-request-inputs-v1/README.md)只接受登记candidate/version，实际prepare/load通过。

| 子范围 | 实际状态 |
|---|---|
| Viel20 | 10 continuous24、10 pulse6+recovery18，共同T24，numeric与physical可构造 |
| Nav8 | 原身份、共享Y对应保留，physical缺项pending，未编造events |
| Bothe4 | 四真实历史physical保留，bulk reference/context/decoder pending |
| union32 | 28endpoint候选+4辅助RNA历史；endpoint不同observation IDs为24 |

24是**不同观测身份数**，不是已证明统计独立的24次实验；Nav两种剂量解释仍共享原4个Y，Bothe不增加Annexin Y。当前核心功能Y数值未解码。

Viel base实参为FP[B,2048]、source_nM/1000的μM剂量、active exposure6/24h、horizon24h、elapsed0与vehicle/context；physical保留原nM、真实事件、unknown masks。root逐20候选用batch_size2读回。

本新计划明确采用Tahoe2431fit对照行作统一proxy，原Viel source/state/protocol未改名。按真实词表名称解析model source/background/protocol=0/0/1（Tahoe-100M/CVCL_0023/tahoe24）。旧context_map“Viel=0”是零条mapping记录，不是Viel模型index0。

base duration6/24、horizon24是**本课题未验证设计契约**，不是论文/旧实现/定理强制。有fullphysical时另取base duration24也不必然消除pulse信息。proxy跨来源转移仍未验证。

### 3.1 输入ready还不是full P完成

numeric_kwargs返回NumPy和CPU数据张量；reference_metadata只读登记行、基因轴、mean/scale、词表与bindings。本轮未读Tahoe参考CSR表达本体，也没有encode_h/sample_latent/decoder/forward。

未来producer需真实reference读取与标准化、Torch/device转换、shared.encode_h、FullProtocolLatentAdapter.sample_latent、匹配decoder/观测出口和source-bound feature cache。full adapter只返回latent；旧projection不是来源绝对raw counts。

Viel没有matched RNA限制中间表达与proxy迁移验证，不构成每个Y必须有自有RNA的通用拒绝门槛。条件级预测特征→实测Y路线保留，实际P/G/独立外部评价尚未完成。[依赖审计及root校正](../experiments/20261011-data-alignment-round21/full_protocol_input_dependency_audit/README.md)列出剩余职责。

## 4. 独立复审与主代理验证

Martin初版可通过caller修改role后更新manifest hashes绕过身份。修后固定25项来源anchors并绑定实际loader；独立9项counterchecks通过，角色/clock/数组/axis/generator/loader变化或缺源拒绝，同内容合法clone可读，NPY原字节未动。

完整请求初版接受manifest伪造qualification=True/Poutputs10。修后核完整声明、scope、数目、context及严格type，P/G/Q/PT/forward全0、资格false；六个独立反例拒绝，bool/int不得互冒。

| 验证 | 实际结果 |
|---|---|
| Martin tests | 28 passed；[独立最终复验](../experiments/20261011-data-alignment-round21/independent_martin_native_rna_review/postfix_results.json) |
| 完整请求tests | 15 passed；[独立最终复验](../experiments/20261011-data-alignment-round21/independent_full_protocol_request_inputs_review/postfix_results.json) |
| RNA协议联动tests | 32 passed；[独立最终复验](../experiments/20261011-data-alignment-round21/independent_martin_observed_protocol_review/postfix_results.json)；类型P2关闭 |
| 观测坐标tests | 13 passed；[9项独立检查与全8fold重算](../experiments/20261011-data-alignment-round21/independent_martin_observation_coordinates_review/probe_results.json)；root公共API读回通过 |
| root真实输入 | 2,365,440原值、16167joint观察及16294缺失、60样本、32身份、20实参passed；[真实结果](../experiments/20261011-data-alignment-round21/root_actual_input_readback.json) |
| 固定绑定 | count五个live source hashes未改；研究架构hash仍34e08ddb0f48e035fc6b3ba4f6db4f9a172bb5e159a30bc9ca8c8ffff5fb188e |

[Martin报告CLOSED](../experiments/20261011-data-alignment-round21/independent_martin_native_rna_review/technical_review.md)、[完整请求报告CLOSED](../experiments/20261011-data-alignment-round21/independent_full_protocol_request_inputs_review/technical_review.md)。这些修复防止错误身份/资格被工程入口接受，不能消除生物域偏移或证明研究收益。

## 5. HM97662结构核查的限定结论

root亲读Methods `sec0003`：Hanmi供药、SigmaD2650是DMSO溶剂货号。三种精确PubChem名称查询原JSON均PUGREST.NotFound/NoCID；root又亲看[Hanmi2023AACR整页海报](../experiments/20261011-data-alignment-round21/hm97662_chemical_source/Hanmi2023_AACR_HM97662_page1.png)，其机制图不能充当HM完整分子图。

仅能说所查入口未取得可认证结构，不扩大为全域未公开。CID/CAS/SMILES/InChI/盐型/stock等null，结构P资格false；没有替代药、假FP或商业占位MW0。[原件/响应/状态验证](../experiments/20261011-data-alignment-round21/hm97662_chemical_source/receipt.md)保存完整范围。已有6个A549 RNA与两条近似Annexin未扩增，角色仍unassigned。

### 5.1 新A549状态—Annexin来源核查未补齐关键矩阵

本轮另完成4组web与2组EuropePMC有界查询，重点原文及官方run/BioSample核查两篇，见[完整查证与原件定位](../experiments/20261011-data-alignment-round21/a549_state_annexin_targeted_gap/README.md)。root亲读两原XML的方法与图注，并重核22个源文件hash/size；[全库核对](../experiments/20261011-data-alignment-round21/a549_state_annexin_targeted_gap/root_source_and_library_crosscheck.json)实际穷尽Zotero60个顶层条目，两篇DOI/全标题均未匹配，没有写入Zotero。

| 来源 | 已核实内容 | 当前连接边界 |
|---|---|---|
| FOXN3/p53，DOI10.1002/advs.76449，PMC13339424.2 | PRJNA1166715为6 RNA＋7 ChIP；六RNA来自siNC/siFOXN3，A549归属由原文连接。RNA药物/时钟未明确，尚无表达值读回 | 主要Annexin Fig3H为KO/Dox500nM24h，Fig4J为OE/500nM6h，Fig6I为KO背景WT/突变回补/500nM6h；siKD不能直接当KO/OE/回补相同状态 |
| GATOR1，DOI10.1038/s41419-025-08392-4 | PRJEB98686公开24 RNA明确BEAS-2B WT/三个KO，各untreated/cisplatin；原文Annexin50µM/24h | 同文出现A549不能把BEAS-2B测序转成A549参考；这条设计不能关闭A549高维状态功能增量 |

FOXN3的qPCR/ChIP Dox10µM3h不补作RNA-seq时钟；Co-IP cisplatin40µg/mL不新增Annexin候选。官方补充包请求未完成，没有读取补充或恢复逐重复功能表。此次有新RNA身份与有用排除证据，没有新核心Y/实际表达，也未形成至少两个A549背景×两个相容候选的增量验证矩阵；不据有界搜索宣称全领域没有数据。

## 6. Bothe真实获取与计数进程

**2026-10-10 22:58:49UTC（北京时间次日06:58:49）**root工具观察见[句柄快照](../experiments/20261011-data-alignment-round21/root_process_observation.json)。不靠mtime/锁文件判断存续；后续追加记录优先。

| Root session | 实际状态 |
|---|---|
| 28793旧先导 | 已exit1，完整长度但MD5/CRC错误，失败副本保留 |
| 54677clean重取 | 已exit1，三次ChunkedEncodingError，R1仅1217396736/1507908000bytes，未做完整MD5验收 |
| 5491remaining fleet | 非终止；0296原child SSL exit1，历史保留，其余队列继续 |
| 1277新0296续传 | 非终止；R1重新通过官方MD5，继续R2，原fleet failed记录未覆盖 |
| 10921count waiter | 非终止，等待pilot正式回执+双完整gzip，completed_runs=[]，真实counts0 |

0296不是已证实的第二MD5失败。原程序重新stat实际offset，append前核206/Content-Range/identity/Content-Length；独立审查未发现缺少这些校验或正常提前停读。SSL底层原因及完整R2checksum未知。[限定诊断](../experiments/20261011-data-alignment-round21/bothe_second_transfer_diagnosis/README.md)与[独立续传回执](../experiments/20261011-data-alignment-round21/bothe_network_resume/README.md)分开保存。

clean先导三次均有磁盘进展再截断，尚非第二份完整MD5失败；buffer读数不能替代磁盘长度。该快照时已开始准备另一HTTPS传输的staged Range/header/长度验证方案，审后实际启动见下节；原官方MD5和发布/计数合同保持。不能继续把54677说成运行中，也不无限重启同一个失败方式。

### 6.1 新传输实际启动，原官方验收保持

**root session92729**于23:12:51UTC实际启动[staged Range helper](../experiments/20261011-data-alignment-round21/bothe_staged_range_resume/README.md)，执行目录为`execution-20261010T231251760046Z`。此前两个独立P2已修复：只接受唯一origin206或明确CONNECT200后唯一206，已验证stage的同一buffer在首次append前核大小/hash。7tests及8项独立counterchecks通过，[复审CLOSED](../experiments/20261011-data-alignment-round21/independent_staged_range_review/technical_review.md)。

每8MiB官方HTTPS Range先完整暂存并核206/精确range/identity/length/bytes/SHA，然后在原partial metadata未变条件下追加。23:17轮询92729非终止，先导R1已实际staged追加至1,427,111,936 bytes，尚未整端MD5验收。不是只创建计划。新provenance明确原requests前缀与curl后段，canonical downloader hash只代表未改验收代码，不能冒称所有字节均由旧requests传输。

双端到齐才调用原下载器整文件officialMD5/SHA并发布原receipt；错误段/失败历史保留，不改checksum、不提供partial给count。新运行状态不改变§6的旧时间快照，后续实际结果以新观察和执行receipt为准。

### 6.2 direct-v3与停止样本恢复的最新实际观察

**2026-10-10 23:49:05UTC（北京时间次日07:49:05）**的[最新root句柄与metadata快照](../experiments/20261011-data-alignment-round21/root_process_observation_direct_v3.json)覆盖以下状态。§6与§6.1为较早时间记录，不能再把1277/92729/63888说成仍运行。

| session/样本 | 最新实际状态 |
|---|---|
| 92729 staged proxy-v1 | 已exit1，R1到全长；R2前三段25,165,824bytes通过，后段curl56被拒、失败stage保留 |
| 63888 staged proxy-v2 | 已exit1，R2累计23段192,937,984bytes通过，下一截断段未append |
| 1277 原0296恢复 | 23:27:39.888769UTC exit1，网络截断，未发布双端receipt |
| 61714 pilot direct-v3 | 23:40:16.656726UTC实际启动，23:49工具非终止；R2已确认分段追加至1,140,850,688／1,793,870,336bytes |
| 73351 0296 direct恢复 | 独立复审后实际启动；先整R1复核官方MD5，再R2分段，23:49工具非终止 |
| 0298 原fleet child | 23:25:26.300746UTC exit1，R1正式文件/旧R2prefix保留；新wrapper已准备并独立关闭，尚未execute |
| 5491/10921 | 两root工具非终止；fleet其余队列继续，count仍等待pilot正式双端/receipt，completed_runs=[] |

R1曾在92729退出后由root独立整文件读取：1,507,908,000bytes、官方MD5 `55ae0ef36425d28a286d382923097d5b`匹配，SHA256 `b10ebec10702ff2986d6e9a220daa0ac23212c4ac9650edf8ddbddb769d583c8`，gzip CRC/EOF通过，未压缩7,616,667,743bytes，读取期间metadata稳定。见[实际R1验证](../experiments/20261011-data-alignment-round21/pilot_staged_R1_verification.json)。这是单端完整性认证，尚非pair header/sequence/quality/顺序或RNA count认证；文件仍为.part，不提供给count。

[direct-v3](../experiments/20261011-data-alignment-round21/bothe_direct_route_resume_v3/README.md)只在本下载的官方ENA host使用`--noproxy`，未修改全局proxy配置。root实际8MiB对照probe与先前同范围段SHA一致，[probe证据](../experiments/20261011-data-alignment-round21/bothe_direct_route_probe/result.json)；不据此定位全部SSL/截断根因或证明长流稳定性。版本固定绑定前两次已接受区间，保留R1 requests前缀＋35curl段、R2旧3＋20段，不把旧字节改称直连。唯一206/range/identity/length/实际buffer SHA与metadata/fstat/fsync守卫不变，6项定向复验通过，[独立v3 CLOSED](../experiments/20261011-data-alignment-round21/independent_staged_range_review/v3_technical_review.md)。

新[0296/0298恢复工具](../experiments/20261011-data-alignment-round21/bothe_failed_pairs_direct_resume/README.md)固定两个已停止run/日志/outcome、R1正式文件及R2prefix元数据；旧fleet不重排，root执行前核唯一writer。5项新定向tests、4项独立检查通过，[复审CLOSED](../experiments/20261011-data-alignment-round21/independent_failed_pairs_direct_review/technical_review.md)。执行先核R1整文件官方MD5，不覆盖R1；仅R2走同一冻结guard，最终交原downloader整双端MD5/SHA验收及原canonical receipt。原failed progress/log不改成成功，恢复结果另关联，不授count/model资格。

当前count五个live source hash均未改变，真实counts仍0。双端真正正式验收后10921按固定链执行gzip/pairedFASTQ→Subjunc→pairfilter→featureCounts；不消费.part，不因R1通过就提前发布RNA/功能资格。

## 7. 完整研究尚需的证据

| 方面 | 本轮贡献 | 未完成事项 |
|---|---|---|
| A状态增量 | 已观测RNA与角色可读 | 实际表达增量未跑；统一proxy/非A549caspase不证明A549高维状态对Annexin增量 |
| B未见条件 | 保留实际时钟/合法留出身份 | 合格新P、留出预测及来源转移可靠性未验证 |
| C真实功能一致性 | 预测特征→真实Y路线与终点域分开 | P特征尚无、G未真实校准；caspase/IC50/growth不替代Annexin |
| D完整候选选择 | 完整union、角色与共享Y保留 | 不同fold G不能拼成一个冻结模型对20候选的独立排行；Q/枚举/基线与独立Y评测待完成 |
| E/D/P/C/G/Q同链 | source坐标/mask/role与部分base实参完成 | bulk出口/fullproducer/匹配参数/实际功能出口和整链评测pending |

缺口涉及数据取得、工程连接和独立科学验证，不能统一说只缺训练，也不据此断言所有条件分布预测无法开展。研究主线、奖励、角色、架构不改；完整目标active。

## 8. 接续与修改范围

继续恢复Bothe实际raw及受控counts，生产完成后才执行已有GeneID→joint整数/mask接口。Martin16真实RNA条件的物理阶段和8fold观测坐标本轮已完成，继续补bulk/source模型消费与独立科学验证，原parental/time/unknown/source-domain区别保留。模型执行不随数据准备自动恢复。

[本轮接续说明](../experiments/20261011-data-alignment-round21/CONTINUATION_HANDOFF_2026-10-11.md)列出真实句柄与API。原脏工作树保留，数据/NPY/gzip/运行产物只存本地；未commit/push、作者联系、Zotero/Memory写入或恢复训练。
