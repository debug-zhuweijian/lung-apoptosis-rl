# Bothe Round22 原始数据与计数恢复

本轮恢复的是 `E-MTAB-9923 / ERP125910` 的真实 A549 unique-intron bulk RNA 数据链。完整 12 run、生产矩阵与 joint32461 真实投影仍未完成；本轮没有模型训练、forward 或 PT 加载。当前结果不得解释为表达—Annexin 连接或整个项目可行性通过。

## 实时恢复证据

新证据统一保存在 [bothe_recovery](../experiments/20261011-data-alignment-round22/bothe_recovery/)。[初始真实进程快照](../experiments/20261011-data-alignment-round22/bothe_recovery/initial_exact_bothe_process_snapshot.json)与[初始文件状态](../experiments/20261011-data-alignment-round22/bothe_recovery/initial_actual_state.json)确认，旧 `driver.json` 的 running 状态不能证明进程仍在执行：恢复前没有真正 Bothe 下载或计数进程。

恢复前实际有 8 个完整双端下载回执、3 个已完成真实计数 leaf（ERR5000305、ERR5000296、ERR5000298）。原值与固定命令、source metadata 的既有审核见 [Round21 实际读回](../experiments/20261011-data-alignment-round21/root_first_real_bothe_counts_readback.json)。本轮恢复 preflight 重新检查小回执、当前 metadata 与全部实际 count 值，复用既有 raw/SAM/index 内容验证，未为 preflight 再扫描这些大文件。

| Run | 恢复前真实状态 | 本轮执行 |
|---|---|---|
| ERR5000295 | R1 已发布；R2 partial 982,515,712 / 3,027,710,970 bytes | 严格 direct staged Range 恢复后，固定下载器两端官方 MD5/SHA 已通过并发布 canonical 回执 |
| ERR5000302 | R1 已发布；旧 R2 全长但 MD5 失败，坏副本已隔离保留 | 从零 clean R2；固定下载器两端官方 MD5/SHA 已通过；旧坏副本 size/mtime 与准备记录相同 |
| ERR5000303 | R1 partial 1,729,101,824 / 2,609,542,954 bytes；R2 未开始 | 网络短响应被拒绝并保留；安全恢复后固定整对官方 MD5/SHA 已通过，19:15 canonical 发布 |
| ERR5000306 | 两端未开始 | 同支路正在传输；尚无完整 canonical 双端回执 |
| ERR5000299 | 旧 Subjunc 在约 19% 中断，SAM 3,611,389,952 bytes；没有完成回执或 count | 旧 partial 同卷单叶改名保留；固定 validator/Subjunc/filter/count 从真实原始数据重新执行 |
| 其余完整下载 | 双端 formal 文件和下载回执存在 | 固定真实 FASTQ 全文校验后顺序计数；不把回执存在等同于 FASTQ/QC 全部通过 |

两个下载支路最初 PID 为 32640（0302 clean）与 23592（0295→0303→0306），启动时间 2026-10-11 16:58:24 +08。[下载启动回执](../experiments/20261011-data-alignment-round22/bothe_recovery/download_launch.json)保存实际命令和 PID。18:09 实际核对：0302 的原进程已结束，固定整对校验子调用 exit 0、canonical 回执存在，见[新完成 0302 实际读回](../experiments/20261011-data-alignment-round22/bothe_recovery/ERR5000302_new_complete_pair_readback.json)；0295 同样已有[新完成读回](../experiments/20261011-data-alignment-round22/bothe_recovery/ERR5000295_new_complete_pair_readback.json)。该快照有 10/12 完整双端下载回执及对应当前文件大小，0303 当时正在传输 R2，0306 排在同一支路后续。此次 metadata 读回复用刚完成的固定下载器整文件校验，不重复扫描 raw 正文，不把单端全长 partial 当作完成。

计数恢复经[主代理独立复审](../experiments/20261011-data-alignment-round22/bothe_recovery/root_count_recovery_review.json)关闭后，已实际启动 PID 53960，CreationDate 2026-10-11 17:12:25 +08。[计数启动回执](../experiments/20261011-data-alignment-round22/bothe_recovery/count_launch.json)记录完整源 hash 与实际命令。ERR5000299 的真实 gzip CRC/EOF 和配对同步已通过，实际 49,161,137 对；17:44:37 开始 native Subjunc PID 58392、parent 53960，固定参数 `-T 1`。18:09 [实际数据读回](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_data_readback-20261011T100842246713Z.json)核 4 个完整 FASTQ validation 绑定、3 个已完成 native leaf，0299 native 仍执行。真实 PID、CreationDate、绝对命令及资源见[运行快照](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_runtime_resource_snapshot-20261011T1009218220908Z.json)；driver lifecycle 字段与进程证据分别保留。

## 恢复实现与冻结边界

`resume_counts.py` 使用原 OS `.native-count.lock`，只恢复 `registered-rna-counts-v1` 的可变 driver 生命周期。已完成三个 leaf 不改动、不重跑；旧 driver 已备份到 `count-resume-20261011T091241323249Z/original_driver.json`。仅已停 0299 partial 原子改名为 `ERR5000299.interrupted-20261011T091241323249Z`，源与目标均在原 processed/version 内，metadata 已核对，没有删除或跨卷迁移。

固定 Subjunc、strict paired filter、featureCounts 参数、注释、索引、下载器和生产 parser 保持原样。剩余 9 个 run 串行，native `-T 1`。最终仍调用原 `_publish`、公共 `load_production` 所要求的 source-bound 内容验证；旧 leaf 的恢复前 metadata 检查不代替最终生产内容验收。

0302 复用已审原 helper `49b438cbd75c3cfc2a7d2d14c772211fcd1322ebf89dde958f587347335765a2`，原字节未编辑。Round22 的新 adapter 仅将运行时 `HERE` 证据位置绑定到新目录，固定 ROOT、RAW、MAP、canonical receipt 与原 `__file__` 保持。[下载来源契约](../experiments/20261011-data-alignment-round22/bothe_recovery/download_wrapper_contract.json)分别记录原 helper 审查链、新 adapter hash 与改变的运行时绑定；不把新 adapter 自动冒充为原版本已审资格。

复制的 guard snapshot 字节仍相同，但其 `__file__` 所在层级改变，guard 自身由路径计算的 ROOT 也随之改变。实际调用仅使用其 `metadata`、`sha`、`new_json`、`validate_stage`、`append_stage` 和固定 CURL；未调用其 root-based declarations/preflight/main/lineage。原 clean helper 的权威 MAP/RAW/receipt 绑定保持原位，详见[运行命名空间核对](../experiments/20261011-data-alignment-round22/bothe_recovery/0302_guard_runtime_namespace_note.json)。

0295/0303/0306 新传输程序实际调用固定 `f1fd9e...` 的 `validate_stage`、`append_stage`，每段检查 HTTPS/206、完整 Content-Range、Content-Length、identity encoding、大小、opened file metadata 与 SHA256。每段失败保留正文、headers、日志并终止，不无限重试。只有原 `45ad5f...` 下载器完整核两端官方 MD5/SHA 后才写 canonical 回执；旧 fleet failed/progress 历史不倒改。

## 单机资源与存储

本机约 32 GiB RAM、I 卷 NTFS、4096-byte cluster。恢复时可用内存约 13 GiB，真实卷余量约 809.8 GB（十进制）；后续下载与计数会持续减少余量。

原三个 completed leaf 的逻辑文件量为 53.32、71.36、66.38 GB，其中 SQLite 为 23.28、31.20、29.13 GB，SAM 与 filtered SAM 分别保留。当前[实际 allocation 读回](../experiments/20261011-data-alignment-round22/bothe_recovery/allocation_followup-20261011T092516773817Z.json)分别为 53,321,326,992、71,357,170,064、66,375,905,680 bytes，确认未压缩旧产物的实际占用与逻辑量接近。按这三份真实产物与官方 `read_count` 比例外推，剩余 9 leaf 约 744–750 GB；待下载原始正文约 15 GB，严格 staged 响应正文另约 15 GB。全部按普通存储保留时预计余量约 30–36 GB。该值是外推，未验证剩余 run 的真实峰值，不保证一定足够。

原三个 run 的 Subjunc 实测约 60、83、78 分钟；strict filter 约 39、53、49 分钟；另有 FASTQ 全文扫描和 hash。按固定单线程处理，余 9 run 估计约 20–30 小时，属于实测外推；不设额外墙钟终止门槛。

只在本轮自建 SAM/SQLite fixture 上做了 [NTFS 压缩实测](../experiments/20261011-data-alignment-round22/bothe_recovery/compression-probe-20261011T090852843431Z/result.json)：逻辑约 73.20 MB，压缩前后 SHA256/逻辑字节相同，同文件 mtime 不变；写入约 1.41→1.63 秒，关闭后压缩约 0.33 秒。压缩后立即 API snapshot 的 storage 为 43.73 MB，稍后 FileStandardInfo、CompressionInfo、GetCompressedFileSize 三入口读到合计约 18.41 MB。两个时间点保留；不能把 fixture 比例当作实际 RNA 产物已测比例。

微软 [FSCTL_SET_COMPRESSION 文档](https://learn.microsoft.com/en-us/windows/win32/api/winioctl/ni-winioctl-fsctl_set_compression)规定最大未压缩文件 30 GB。本轮不让 whole leaf/SQLite 继承压缩，不压缩旧三个 leaf 或 live 文件。独立 `compress_completed_sam.py` 只允许本轮已关闭的新 run 下 `aligned.sam` 与 `filtered.sam`、且逻辑字节严格小于 30,000,000,000；执行前后核实际 hash、逻辑 size 与 mtime，并登记 allocated size。每个压缩子进程持独立 `.closed-sam-<run>.lock`，覆盖两文件的真实操作和验收；父监控退出而子进程仍存活时，后续同 run 也会拒绝重复写入。原工程 CI 不构成实际 RNA 压缩证据，真实闭合 run 的执行与结果另行登记。该存储选择不改变原始路径、读字节、科学过滤或计数语义；需要删除或迁移数据时另行处理授权。

存储监控 v1 PID 60940 在确认无真实压缩子进程后仅停止本监控，保留[原 launch](../experiments/20261011-data-alignment-round22/bothe_recovery/storage_monitor_launch.json)、原 source snapshot 和[停止回执](../experiments/20261011-data-alignment-round22/bothe_recovery/storage_monitor_stop_for_lock_delta.json)。独立 delta 复审 CLOSED 后，v2 在 18:04:58 +08 隐藏启动 PID 62236，见[新 launch/pins](../experiments/20261011-data-alignment-round22/bothe_recovery/storage_monitor_launch-v2.json)。监控只守候本轮闭合的新 leaf，始终最多一个压缩 writer；未改或重启计数、下载和 finalizer。

18:09 实际 I 卷余量为 790,952,808,448 bytes，Subjunc working set 约 8.84 GB，系统可用内存 5,761,864 KiB，均为当时快照。主代理[后续模型资源实读](../experiments/20261011-data-alignment-round22/root_model_resource_preflight.json)另估共同轴稀疏缓存数组保守上界约 36.64 GB，不含其余索引/表示/权重；该值不是当前已写缓存。普通未压缩计数外推与此后续缓存合计时余量很窄，尚不能据估算保证整个阶段容量。实际 closed RNA SAM 的前后 allocation 尚待测量；不因此启动模型、改变冻结科学协议、删除已有数据或另设资源门禁。

## 共同基因轴与后续验收

19:32 的[实际数据读回](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_data_readback-20261011T113204626047Z.json)确认 **11/12 canonical 双端回执、4/12 全文 FASTQ 校验回执、3/12 native completed 回执**。ERR5000303 的两端固定下载器完整校验 exit 0，当前 raw 字节与官方声明一致，见[新完成整对读回](../experiments/20261011-data-alignment-round22/bothe_recovery/ERR5000303_new_complete_pair_readback-20261011T113204626047Z.json)；新 metadata 读回未重复 hash raw，全文 MD5/SHA 证据来自刚闭合的固定下载器。0306 尚在传输，没有被计入完整样本。此快照只核当前 metadata、source receipts 与实际进程，不把 receipt-only 审查重新称为 FASTQ 全文验收。

旧下载支路 23592 在 18:52 因 curl 56 短响应终止。0303 R2 的一段声明 8,388,608 bytes，实际只收到 191,680 bytes，guard 在 append 前拒绝，已接受的 prefix 不变，原失败正文、headers、日志和准备记录全部保留。受审新 shim 只切换 prepared 记录，不改原 guard/downloader/官方映射/锁。经[根代理复审](../experiments/20261011-data-alignment-round22/bothe_recovery/root_network_retry_review.json)后，19:01 启动 PID 67148，0303 已完成并转 0306；第二次尝试来源和实际命令见[恢复启动回执](../experiments/20261011-data-alignment-round22/bothe_recovery/remaining-network-retry_launch.json)。不对仍执行的 helper 修改源码或再启动同 run writer。

0299 Subjunc PID 58392 已在 19:14 结束，其新 `aligned.sam` 为 22,745,102,306 logical bytes；计数父进程 53960 仍活，实际 `strict_pairs/templates.sqlite` 正在增加，属于过滤阶段，当前未得到第四个 completed count。19:32 [资源快照](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_runtime_resource_snapshot-20261011T113204626047Z.json)实际剩余 I 卷 741,473,394,688 bytes、可用内存 16,391,284 KiB；不是后续全部阶段容量承诺。新 leaf 尚未闭合，独立存储监控继续等待，不压缩此 live SQLite 或未验收 leaf。

20:15 后已取得第四个实际闭合 leaf：0299 的 strict filter、featureCounts、execution 回执完整，当前 native completed 为 **4/12**。主代理[前三个真实 gene 投影读回](../experiments/20261011-data-alignment-round22/first3_gene_projection_v1/receipt.json)对旧三个实际 counts 重新 parse，保存 51,321 个真实映射观测单元及 46,062 个缺失单元；[第四个真实读回](../experiments/20261011-data-alignment-round22/fourth_gene_projection_v1/receipt.json)核对 49,161,137 pairs、原 207,555 feature 坐标与固定 filter/stage/summary 回执：assigned unique intron fragments 为 4,828,143；共同轴 17,107 个观测基因包含 1,773 个观察零，映射 counts 合计 4,571,023。整数/null/mask 序列化逐值一致，未更改 leaf、未重复 count writer或 raw/SAM 全文 hash。上述是 **四个 partial 实际投影**，没有调用完整 production qualification，也不是 12 样本矩阵或 observation pack通过。

第四个 leaf 闭合后，存储监控启动独占 compression child PID 70368；20:18 [真实 RNA 两文件压缩回执](../experiments/20261011-data-alignment-round22/bothe_recovery/closed-sam-ERR5000299-20261011T121014238803Z/result.json)已闭合。`aligned.sam` allocation 为 22,745,104,384→10,853,658,624 bytes，`filtered.sam` 为 18,901,504,000→8,886,030,336 bytes；两者 logical size、mtime 和 SHA256 均不变，合计实际减少 **21,906,919,424 bytes（约 21.91 GB）**，FSCTL 操作共约 393.57 秒。该比例是 0299 两份真实 SAM 的实测，不能预先推广其他 run 或大于 30GB 的文件。70368 已退出，monitor 62236 保存 completed_storage_runs=[0299]并继续等下一份闭合 leaf；SQLite、旧三个 leaf 和其他 live 文件未被压缩。0306 R1 已全长进入 R2，仍不计作完整双端下载。

新 `finalize_joint_counts.py` 已由隐藏守候进程 v2 PID 57396 接通，等待本轮 `count-resume` 目录的真实 `publication.json` 与完整可解析的正式 source-bound production manifest，核固定 12 run、driver complete、marker 完成字段及 manifest hash 闭合后，再调用既有固定 `project_production_counts(version)`；正式文件刚出现但尚未写完时继续等待。该公共消费路径内部执行完整 `load_production` 验证。导出 32,461×12 的整数/空缺 TSV 和 mask，逐值回读；合法观测 count 0 保持整数 0，不可测或未映射保持空缺/false。实际 17,107 个合格映射对应 205,284 个 gene×sample 观测单元；这个数量是完整 12 样本时的预期，不代表当前已导出。

首次 finalizer launch PID 58520 因 Windows PowerShell 默认编码与 Python UTF-8 捕获不一致而退出，原 failure/launch/log 全保留。只修改已退出的新 helper，显式 PowerShell UTF-8 输出、binary capture 后 UTF-8-sig 解码；中文路径和 PID/executable/command/creation 改变反例均实测。当前 [v2 launch](../experiments/20261011-data-alignment-round22/bothe_recovery/finalizer_launch-v2.json)绑定 count launch 和修复后 source hash，未改变 live count wrapper。需要恢复时以真实进程身份与闭合出版回执为准。

[20:46 bounded readback](../experiments/20261011-data-alignment-round22/bothe_recovery/actual_data_readback-20261011T124617018173Z.json)复用已完成官方完整字节校验及当前 canonical size：**11/12 raw 对、5 份真实 FASTQ 验收回执、4 个真实闭合计数 leaf**，尚余 ERR5000306 raw 与其余 count。该查询没有重 hash 已验 raw/SAM/index，没有重跑 native writer。实际 RAM 可用 6,306,284 KiB，I 卷剩余 727,059,922,944 bytes；这是当前运行资源快照，不是全余量承诺。

[20:51 当前 CIM](../experiments/20261011-data-alignment-round22/bothe_recovery/native_identity-20261011T1251082916051Z.json)证明五个已登记 wrapper 的 PID、creation 与 exact command 仍匹配：53960 count、57396 joint finalizer、62236 storage、67148 downloader、68324 observation waiter。下一真实 ERR5000300 subjunc 子进程 16504 于 20:40:31.1397140+08:00 启动，parent=53960、固定 `-T 1` 参数；下载 curl 的 parent=67148，处理 ERR5000306 R2。旧状态和进程 ID 不能代替这份当前存活证据。完整12计数、共同轴 full production 与 bulk observation pack 仍未验收。

## 21:41 有限只读验收接续

[新快照](../experiments/20261011-data-alignment-round22/bothe_recovery/readonly-continuation-20261011T134154911245Z/snapshot.json)在 21:41:57 +08 核 **12/12 canonical 原始双端下载、5/12 全文 FASTQ 配对验收、4/12 native 闭合计数**。最后 ERR5000306 的固定下载器 `completion.json` 为 validator_exit=0，正式 source-bound 双端回执已发布：R1 为 2,764,214,519 bytes、官方 MD5 `2a49e4a26cff87b693bed764453a1b85`、SHA256 `3fb1c987a432754a42ebdd1f140f34b78faa70db623a2047cfbb69d8d20652fb`；R2 为 3,313,941,354 bytes、官方 MD5 `99cb1a6b28c34baa9a1bed0b17a7c0d7`、SHA256 `a6b99e5865b6f2c27b8786378a71fbdd2a606a59c61d1769e5ddd50fe458b571`。来源 mapping、固定 downloader 和 genome declaration 当前 SHA 与原件一致，12份回执逐字段及当前文件 byte size 均吻合；本快照复用刚闭合/此前完整官方 MD5/SHA 校验，不重复读取 raw 正文，也没有替代仍缺的7份 gzip CRC/EOF 和配对验收。

五个 wrapper 的 launch、源 pin、creation、实际 executable 和 exact command 已逐项核对，全部 source pin 无漂移。count 53960、finalizer 57396、storage 62236、observation 68324 当前真实匹配存活、各只有一个相同命令实例；download 67148 已不存在，306 completion=0 与 canonical publication 是其任务闭合依据，未仅凭进程消失认定成功，也没有虚构另存的 wrapper exit code。没有重启或停止任何作业。

ERR5000300 的 native PID 16504、parent 53960、creation 20:40:31.1397140+08、executable 和原 `subjunc.command.json` 的完整命令均匹配，仍执行冻结 `-T 1`。新 live `aligned.sam` 当时 logical 为 **14,533,955,584 bytes**、FileStandardInfo allocation 为 **14,533,984,256 bytes**，compressed=false；各 allocation API 非同时取值，live 文件持续写入，不当作闭合验收。本快照没有 hash/压缩该文件；native 日志末项为映射阶段 57%、55分钟，此数不表示整个计数/全项目进度。该 run 尚无 execution 回执，因此仍只计4个闭合 leaf。

当前 I 卷 NTFS剩余 **709,049,917,440 bytes（约709.05GB）**；系统可用 **3,491,016 KiB（约3.33GiB）**，native working set约8.87GB。这仅是运行资源快照，模型授权仍停止，未新增资源门禁或改变参数。当前只有0299有本轮新闭合 SAM 压缩 result；其两文件 logical size、mtime、allocation与先前真实 before/after回执一致，实际已减少 **21,906,919,424 bytes**；复用原关闭后 hash校验，不重 hash SAM，不对 SQLite、旧leaf或0300 live文件操作。正式 production manifest 尚未出现，joint与bulk waiter准确保持 pending。

新只读脚本第一次误用 `reference/genome_declaration.json` 路径，在元数据核验阶段退出，保留[失败](../experiments/20261011-data-alignment-round22/bothe_recovery/readonly-continuation-20261011T134052211522Z/failure.json)；亲读冻结 downloader 的 REF 后，改为其真实 `reference/current-ucsc-hg19-20261011`，第二次快照成功。只修改本轮新只读快照脚本，原 writer、live源码和样本均未改动。

后续依次完成：其余7份实际 gzip CRC/EOF与配对同步验收；其余8个 run 的固定 native 和完整 feature count；12列 source-bound matrix；公共 production readback 与 joint 导出；按来源组织的 RNA/QC 与模型消费。当前真实生产矩阵与 joint 投影未发布，E/D、表达—功能连接及完整项目效果仍待各自验收。数据作业连续运行，快照完成不停止它们，原T1的20–30小时总时长估算仍为实测外推。

已实际通过恢复边界 7 tests 与真实只读 preflight；[独立存储 delta 复审](../experiments/20261011-data-alignment-round22/independent_bothe_storage_delta_review/results.json)实际执行 20 个 owner tests 与 11 个独立反例，共 31 passed，覆盖真实多进程 fixture 同 run 锁拒绝、父监控锁释放后的子锁、Win32 PID/CreationDate/绝对命令绑定、中文/BOM、partial publication、压缩失败留存及整数/空缺序列化。这些工程测试不作为真实 RNA 压缩或科学效果通过证据。

该独立 31 项命令遗漏 `--basetemp`，实际 TEMP/TMP 为 `C:/Users/Windows11/AppData/Local/Temp`、conftest 未重定向，因此其合成 fixture 写到 C 临时目录。这是本轮执行路径错误；源码工程断言仍通过，不能据此声称这次所有测试产物均在 I 项目内。复审者追加实际路径与限制，现存文件保留不删除；后续 fixture 显式指定 I 项目内新的、尚不存在的 basetemp。本 owner 的此前定向 tests 已使用 I 树 basetemp，真实下载/计数/压缩守候/投影守候与回执均在 I 树执行。

## 真实 bulk 样本消费与闭合守候

新增 [BotheObservationInputs](../src/lung/data/bothe_observation_inputs.py) 与 [数据准备入口](../scripts/prepare_bothe_observation_inputs.py)，把已闭合的真实 production→joint 投影接成样本×共同基因轴的数据包。完整源出现后读取 exact 12×32,461 整数/JSON null 与 observed bool mask；观测零保持 `0/true`，未映射缺失保持 `null/false`，不经浮点转换，也不转换成 UMI、成熟 mRNA 或拟合归一化。五份数据文件分别保留 counts、mask、gene axis、sample contexts、原四种 physical protocol contexts；样本—run 按官方身份连接，不按计数执行顺序连接。

每种历史保留 3 个来源重复、A549 bulk total RNA、百万细胞提取单位、nominal first-treatment-relative T32 观测时钟和来源角色 `unassigned_history_RNA_auxiliary`；没有擅自分配 fit/selection/final。参考为三份平行 T32 double-vehicle 样本，**不是同细胞 t0**，重复编号不建立生物配对。完整处理事件、剂量 nM、两次洗涤、24h 恢复、培养基范围及缺失 mask 沿用冻结 physical schema；0.1% EtOH 的具体 basis、前药残留和历史 stock form 未知仍未知。旧 physical record 的 pending reference 等字段作为历史原件保留并明确标记，不因新 count 输入存在就赋予 latent/模型资格。

[独立复审](../experiments/20261011-data-alignment-round22/independent_bothe_observation_review/review_receipt.json)已关闭两项工程 P2：不能把 `python -c` 的位置参数误认成实际 wrapper；count 完成退出而 finalizer 正投影时，需重新核闭合 count publication，不能用旧 running lifecycle 误报失败。受审 consumer SHA `14088b23776a1517a58fd4fe39b3c2511801a1bae179fd6ed52d28d4b86f3640`，CLI SHA `9dbf6c6271a834ed30e918790512765215bca2613d60a12f8b0ff5198ece3c48`；29 owner + 14 独立反例实际 **43 passed / 0.90s**，全部在新 I 项目 `experiments/bovr3` basetemp。16 项原件/协议/桥/名义结构 pins 实际匹配。默认真实 [preflight](../experiments/20261011-data-alignment-round22/bothe_observation/real_default_preflight.json)仍为 pending、未生成 pack；以上是工程闭合，不代替真实 12 样本数据验收。

在该受审版本上已隐藏启动唯一 observation 守候 **PID 68324，CreationDate 2026-10-11 19:35:07.528380 +08:00**。[launch](../experiments/20261011-data-alignment-round22/bothe_observation/waiter_launch-20261011T1135075124239Z.json)记录实际 CIM 身份、精确命令、source/review hashes；[首次 state](../experiments/20261011-data-alignment-round22/bothe_observation/waiter-20261011T1135075124239Z/state.json)确认 count/finalizer 当前身份均匹配、闭合输入尚缺。它持独立 observation lock，等待 12 run 正式 publication、固定 joint manifest 与匹配 actual result 闭合后，再读取新增 TSV、写新版本 pack、逐文件及逐值读回，并保存 result 或 failure；复用已审实际 `project_production_counts/load_production` 内容核验，不重复 hash raw/SAM/index。此 job 不触发 native、模型或归一化；已有 count/finalizer/storage/download job 未改动。

启动前两个只读 PowerShell preflight 误判已分别保留 [PID 类型失败](../experiments/20261011-data-alignment-round22/bothe_observation/launch_initial_preflight_failure.json)和[创建时间失败](../experiments/20261011-data-alignment-round22/bothe_observation/launch_creation_preflight_failure.json)：JSON 解析返回 Int64 与自动 DateTime，后者隐式转字符串丢失小数。修为正数 Win32 范围的整数类型，以及直接保留 typed DateTime ticks；两次均在 Start-Process 前停止，没有额外进程或数据写入。受审 Python consumer/CLI 字节保持不变。

完整 count/joint 未闭合前此包仍未验收。E/D/P、原始 C、G、Q、来源解码/归一化等价、共享 latent、Annexin 连接、模型实验与完整项目可行性全部继续登记为 false/待验；训练停止安排不因后台数据 job 改变。

## 用户限定的已启动数据阶段收尾

用户先要求暂告一段落，随后明确把当前已启动的Bothe数据阶段做完。当前继续范围仅为原12 raw、完整FASTQ、真实count、全量joint投影与观察包验收；其他科研和模型保持暂停，整体研究目标不据此标记完成。

[22:39实际接续](../experiments/20261011-data-alignment-round22/bothe_recovery/stage-scope-readback-20261011T1439150923570Z/snapshot.json)重新核原四PID53960/57396/62236/68324实际exe、creation和完整命令，source分别仍为dd85/76c/5eb/9db，consumer1408亦未变。0300原Subjunc16504已不存在；原 `subjunc.command.json` 完整status=complete、exit_code=0、finished=22:21:59.025696+08，log计101.5分钟。源码在该step完成后同步进入parent53960的strict filter；当前parent实例真实活跃、无featureCounts子进程/command、无filter receipt和第五execution，strict_pairs/templates.sqlite已22,080,008,192bytes。该数据路径与filter阶段一致，不把SQLite mtime/driver running字段单独当作进程证据，不授第五leaf闭合。当前I余675,925,000,192bytes、RAM14,141,076KiB；只读metadata，无live raw/SAM hash或压缩。

结束链路核查另在合成fixture复现两项原结束窗口，见[两个tiny反例](../experiments/20261011-data-alignment-round22/bothe_recovery/completion-window-fixtures-20261011T142712014039Z/receipt.json)：finalizer140–148行旧ready_pub为空后CIM期间count完成退出，未刷新publication而提前failure；storage93–94行旧completed列表之后count结束，未再次访问第9new run而failure。实际native/compression/model调用0，模拟8条stdout不能当真实压缩；76c/5eb原源前后SHA完全一致，当前没有该真实失败。活跃脚本未被修改、停止或换版。

[收尾恢复准备](../experiments/20261011-data-alignment-round22/bothe_recovery/CLOSURE_RECOVERY_PREPARATION.md)列出Root审核后的最小原源恢复、全部必须实际验收的文件/入口，以及明确未执行的[preflight脚本](../experiments/20261011-data-alignment-round22/bothe_recovery/prepare_closure_recovery.py)。仅在旧实例实际退出、没有对应/orphan writer、12实际leaf/publication已闭合、目标不存在且锁可取时才允许独占恢复；目标partial存在必须保留，不能覆盖或删除。finalizer复用原76c；闭合joint之后observer可复用原9db无watch，不再依赖旧finalizer PID；storage凭真实child exit0 outcome与匹配closed compression result跳过已验run，只处理缺项。全部是准备，preflight仅AST检查，尚未运行或恢复。

建议下次有意义检查在23:15–23:30 +08，优先看0300新execution、storage实际outcome及异常；等待期间不频繁CIM。若新failure/receipt或用户消息先出现则按事件核查。原唯一count与三个守候继续，训练和额外科研均未恢复。

## 与当前研究架构的对应（边界保持）

与[研究架构 §4.3、§5.1、§6](G:/研究生科研文件/毕业论文/研究架构.md)一致：统一基因身份保留来源观测域，bulk unique-intron counts 不变成单细胞表达或群体凋亡概率。已亲读 [PMC8403771 原文](../experiments/20261011-data-alignment-round17/bothe_protocol_source/PMC8403771.xml) Figure 3、Methods s10/s15/s27，并在[本轮原文读回](../experiments/20261011-data-alignment-round22/bothe_recovery/primary_bothe_protocol_readback.json)登记版本和定位：RNA 从百万 A549 细胞提取，Figure 3 为 4h 暴露 + 24h 撤药恢复 + 4h 再处理；四历史×三重复为平行 T32 终点，没有 t0/t28 实测初态或 Annexin 标签，也不是同一细胞的连续轨迹。Methods 的“hormone-free”含 5% FBS，指无额外加入激素，作者明确其含低水平 cortisol；不能据此填为绝对零浓度。原作者 STAR2.7.0a/RefSeq unique-intron 方案与本课题冻结的 Subjunc/当前注释方案分别登记，不宣称 exact 作者复现。此数据补充历史条件下的表达辅助评价，不补出未测的 A549 凋亡标签，也不证明任意真实序贯收益。

后续模型训练与正式实验仍遵循当前停止安排；数据工程不自动恢复模型运行。本轮未提交、推送、写 Zotero、Memory 或外部系统。
