# 真实完整协议输入 v1

2026-10-11治理说明；数据包于10月10日晚生成。该README不属于冻结manifest的输出文件，未改动原始或派生数据字节。

状态：**完整协议输入可读取，真实P描述符缺失，G/Q生产链尚未接入**。本包不是ready训练集或完整科研可行性证明。

| 文件 | 内容 |
|---|---|
| `candidate_records.json` | 28输入：Vielhauer20完整暴露方案；Nav-Gal4真实状态/药物条件的两种剂量解释，共8输入 |
| `observations.json` | 24独立活性观测身份：20＋4；保留原尺度、来源及读图边界，不代表24个独立研究 |
| `observation_mapping.json` | 候选→真实观测；Nav10/11µM两解释映射同一Y |
| `p_descriptor_requests.json` | 14标准连续24h请求；没有实际预测向量 |
| `reference_observations.json` | 分列图vehicle参考，非模型输入或配对初态RNA |
| `contract.json` | 局部用途与未完成状态 |
| `manifest.json` | 消费者、110项来源、6个输出文件的冻结身份 |

manifest SHA256：`2284d771170ee67fc81793e35e1465e805abef5ca08ea221f35818533f307075`。

只读回验：在执行项目根运行 `F:/anaconda/envs/study/python.exe -X utf8 scripts/build_endpoint_protocol_inputs.py --verify`。重新生成必须另取输出目录，不覆盖本包。

[数据消费者](../../../src/lung/data/endpoint_protocol_inputs.py)可以审计原始观测和输入的对应；其手工descriptor参数只验证原型格式，不认证特征来源。模型接线使用[终点隔离读取桥](../../../src/lung/engine/endpoint_protocol_features.py)，只解析已冻结候选/请求与真实指纹，不解析终点文件；当前缺已审核producer，P cache保持unsupported。

Vielhauer持续24h与6h给药＋18h恢复均在T*=24h，但保持不同候选和换液事件。未知浮游细胞回收不填0。Nav前药撤除未知保留类别，群体状态与密度差异不解释为单独状态因果效应。标准P请求不是洗脱终点RNA，不是Nav同孔实测初态。

fraction与percent保留，不直接混合不同分母/检测方式。像素区间不是SEM/CI；生成细胞不增加Y样本量。详细依据、验证和剩余工作见[第十三轮报告](../../../docs/DATA_ALIGNMENT_ROUND13_2026-10-11.md)和[方法接入方案](../../../docs/DATA_ALIGNMENT_METHOD_SOLUTION_2026-10-11.md)。
