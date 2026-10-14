# 真实完整协议终点监督视图 v1

本目录于 2026-10-11 实际构建并逐项读回，清单见 [manifest.json](manifest.json)。治理说明不属于 manifest 中冻结的三个标签文件。

- Vielhauer source_nominal：20 个活性条件均值，真实持续 24h 或 6h 给药后洗脱＋18h 恢复，共同 T*=24h。
- Nav figure_caption 10μM：4 个来源观测；methods 11μM 是相同 4 个 Y 的完整敏感性视图。
- 合计 24 个 unique observation_id；禁止将两个 Nav 视图合并为 8 个独立观测。

单位可换算为 fraction，原单位、检测定义、分母、source_record 与读图上下界保留。读图范围不是生物置信区间，一条发表均值也没有因此变成原始重复。当前全部角色 unassigned；真实 P 描述符为 0，formal_training_ready=false。

来源是[冻结输入包](../endpoint-protocol-inputs-v1/manifest.json)；显式读标签代码为 [endpoint_protocol_supervision.py](../../../src/lung/data/endpoint_protocol_supervision.py)。[CLI](../../../scripts/build_endpoint_protocol_supervision.py) 的 --verify 从实际冻结来源重建后逐项比较；不要修改旧标签文件或 reader 后继续沿用 v1。

当前生产与评价边界见[第十四轮报告](../../../docs/DATA_ALIGNMENT_ROUND14_2026-10-11.md)。公开结局已经读取，后续划分是回顾性设计；条件读取完成不等于模型效果或全课题数据可行性通过。
