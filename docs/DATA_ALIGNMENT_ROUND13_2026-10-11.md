# 第十三轮：方法审查与真实完整协议输入接入

日期：2026-10-11；本轮于10月10日晚开始。接续[第十二轮](DATA_ALIGNMENT_ROUND12_2026-10-10.md)。上一轮属于实质进展：验收契约校正、真实参考索引和来源读取均已完成。本轮继续“完整实现数据对齐，确保整个课题在数据上的可行性”，未修改[研究架构](G:/研究生科研文件/毕业论文/研究架构.md)，未恢复训练。

## 1. 本轮结论

**实际完成了真实候选的完整协议输入消费者、终点隔离的输入桥、四条方法证据报告；完整P→G/Q的数据生产和功能评价仍未完成。** 数据可以按有边界的组合设计检验原课题，不能再以缺少每条Y同测RNA为普遍否定理由；数据消费者通过也不等于模型有效或完整课题已验收。

方法与可行性逻辑详见[数据对齐的方法依据与实际接入方案](DATA_ALIGNMENT_METHOD_SOLUTION_2026-10-11.md)。主代理亲读CSG2A原文/源码、CellOT原始XML、LLP/MIL关键PDF页、Neural CDE与决策方法原文相关部分；四条方法任务按用户要求指定`gpt-6.1-sol`。论文中的IC50、活力、群体增殖/死亡及临床结局没有被替换成Annexin。

## 2. 实际数据读回

[输入包manifest](../08-datasets/LungCancer_A549_Integration/endpoint-protocol-inputs-v1/manifest.json)绑定[消费者](../src/lung/data/endpoint_protocol_inputs.py)、[CLI](../scripts/build_endpoint_protocol_inputs.py)和110项来源文件。

| 实际单位 | 数量 |
|---|---:|
| Vielhauer完整药物/剂量/暴露候选 | 20 |
| Nav-Gal两状态/两药/两剂量解释输入 | 8 |
| 独立活性终点身份 | 24：20＋4，不是24个独立研究 |
| 标准连续24h P请求 | 14：10＋4 |
| 读取的实际指纹输入 | 28×2048 |
| 真实P描述符 | **0** |

消费包manifest SHA256：`2284d771170ee67fc81793e35e1465e805abef5ca08ea221f35818533f307075`。冻结消费者SHA256：`898569f8232b1167e0944fb7acc1ac77e337f62a4bae365e86d4fab07645b28e`。

持续24h和6h暴露＋18h恢复保留不同候选、不同数值和换液事件，共同T*=24h；它们可以共享同药/剂量的标准P请求，但不能共享整个G/Q输入。Nav两种剂量解释共享同一个源Y身份，分别作为敏感性场景；不把8输入说成8条真实药效观测。

输入保留状态类别、密度、前药撤除未知、洗脱浮游细胞回收未知、物理时间和检测分母。预测输入不含实测Y、像素位置或误差条。fraction和percent分别保留，读图范围不是生物置信区间。原核心r5/v6的42个Annexin汇总未被本轮覆盖，也未与挑战数机械相加。

## 3. 独立审查与真实特征入口缺口

[独立协议审查](../experiments/20261010-data-alignment-round13/protocol_consumer_independent_review.md)重核真实包和读取路径，确认上述局部输入资格；同时发现旧手工descriptor入口只验证格式及自报散列，允许任意命名的向量，不能证明数值来自真实P。

主代理随后核验[标签隔离输入桥](../src/lung/engine/endpoint_protocol_features.py)与[回执](../experiments/20261010-data-alignment-round13/protocol_feature_bridge_receipt.md)：新路径固定已审核包身份，只读取候选/请求/指纹，不反序列化终点文件，不调用会读取Y的旧prepare。未经审查的cache仍被拒绝；缺实际producer时保持`pending_unsupported_producer`。

这隔离了旧原型入口的P2风险，**尚未实现真实P cache接入**。现有`endpoint_features.generate_feature_rows`按实验条件输出population/bio/context/history；它没有本轮标准P请求的固定定义与生产回执。不能因存在旧生成器就声称14个新请求已支持。旧消费者与v1包字节保持不变，新桥与后续生产版本分开。

## 4. 与研究架构逐项核对

| 原架构要求 | 本轮实际对应 | 验收边界 |
|---|---|---|
| §2/§4.2：状态、药物、剂量、真实时长与协议 | 28条输入保留完整候选和状态背景；6/24暴露、0/18恢复、共同24h分开 | 输入消费通过，未证明预测收益 |
| §4.1：无同细胞配对时学习分布 | 标准参考P请求明确预测身份，不造expression_sample_id或命运标签 | 请求已定义，实际P特征为0 |
| §4.3：实测凋亡才监督G | 24活性Y独立保存，观测定义/单位不混合，多个解释不扩样 | G尚未接入完整协议；没有ready训练manifest |
| §4.4：完整候选价值与排序 | 候选ID区分撤药协议；相同剂量/药物不能合并整动作 | 新API尚未进入实际Q/排名调用 |
| §6：状态增量与公平基线 | Round12完整集合/均值设计保留；本轮规定同块直接监督、P→G枚举、P→G→Q比较 | 未执行模型效果比较 |
| §6：独立真实功能评价 | 来源观测可构成限定回顾性挑战，剂量场景及读图敏感性明确 | 训练/选择/评价版本仍需冻结，不能宣称盲测 |

因此本轮输出与架构的局部输入要求一致；没有将主线改为一般毒性、只做表达、只做类别或只做固定条件头，也没有以这些局部通过替代整体验收。

## 5. 主代理实际验证

`F:/anaconda/envs/study/python.exe -m pytest tests/test_endpoint_protocol_inputs.py tests/test_vielhauer_duration_challenge.py tests/test_nav_gal_challenge_evaluation.py -q`：**56 passed**。初次新测试把标称dose对象中的`value`误识别为Y，已仅允许该明确dose位置并重跑通过。

`F:/anaconda/envs/study/python.exe -m pytest tests/test_endpoint_protocol_feature_binding.py -q`：**10 passed**。覆盖终点文件不解析、旧prepare禁止、任意Annexin descriptor拒绝、缺真实工件、跨请求/协议错配以及自洽合成cache仍保持unsupported。

`scripts/build_endpoint_protocol_inputs.py --verify`：实际读回28输入/24Y/14P请求/110绑定/0真实descriptor。新包未重建；上述命令没有模型执行。

第十二轮文档收尾实际通过13文档/257本地链接、证据哈希与协作报告绑定；本轮最终状态由[整轮验证](../experiments/20261010-data-alignment-round13/validation.json)及[文档验证](../experiments/20261010-data-alignment-round13/document_validation.json)登记。测试证明相应输入/拒绝行为，不证明模型精度、跨域迁移或生物获益。

## 6. 活动目标的完成审查

当前还必须完成：真实P标准请求生产与工件身份；完整协议/状态G消费；完整候选Q及排名消费；符合实际单位的独立划分；同一冻结程序的真实终点评价。这些缺口属于仍需实现或运行的工作，不是已经证明没有公开数据。

更强主张如目标实验细粒度初态RNA的真实凋亡增益、同一细胞未来命运、状态×暴露时长因果交互和任意序贯收益仍无对应验收证据，不得仅靠参考转移假设补齐。

**`all_project_data_alignment_complete=false`，`full_scientific_feasibility_passed=false`；活动目标保持active。** 本轮属于实质进展，未满足完成审查，也不存在连续三轮无法推进的阻塞条件。没有正式训练、模型推理、作者联系、Zotero/Memory写入、提交或推送。

Cloud/Dot本轮复用已经归档的报告，没有新派单；本地新代码与结论不能说成它们已重新审核。[协作记录](CLOUD_DOT_COLLABORATION_2026-10-10.md)。
