# Mphahlele 2023：A549 24 h Annexin 数值案例

## 2026-10-15：归档转录与名义结构的云端对齐

已从固定 Git blob 的旧采集脚本恢复全部 12 条已归档观测，并依据原论文的命名、取代位置和 E 构型登记 11 个名义结构；真实 RDKit Morgan 指纹与剂量、24 h 终点、untreated 对照、来源 ID 已连接并读回。30 项针对性测试通过，限于部分源码快照，不代表全仓库或模型效果验收。

新入口：[align_mphahlele_review.py](../../../scripts/align_mphahlele_review.py)；[结构登记](../../../configs/mphahlele2023_nominal_structures_v1.json)；[完整数据链核查与限制](../../../docs/DATA_CHAIN_CLOSURE_RESEARCH_2026-10-15.md)。在项目根目录执行 `python scripts/align_mphahlele_review.py --output experiments/cloud-data-chain-review/mphahlele-v1`，输出目录必须尚不存在；不下载、不训练、不改冻结资产。

3k 的论文 HRMS 文字与名义含四个 F 的结构冲突，显式登记 `source_identity_conflict=true`。名义结构并非实验材料鉴定。RNA、批次与逐生物重复仍缺失，训练角色未分配，`use_for_training=false`、`training_ready=false`。本轮没有重新取得原 DOCX/FCS，因此只是旧转录的实际恢复/计算与名义结构对齐，不是全部原始流式数据的独立再核验，更不是完整研究数据链已闭合。

## 2026-10-09 历史来源核查

以下保留原记录；其中原件读取和哈希属于当时核查，不升级为本轮原件重验。

日期：2026-10-09。[原论文](https://link.springer.com/article/10.1007/s00044-023-03136-5)，DOI 10.1007/s00044-023-03136-5。

已核实 Methods 的A549、1×各自IC50、24 h、Annexin-V/PI流程；实际读取Table2的A549浓度列，亲看补充TableS3。

- `44_2023_3136_MOESM1_ESM.docx`：3,231,707字节，SHA256 `6ac58a4129072e6a7956438a9bb8c2d091a85c76472c74bbc2d0db20c0a7e7fb`。
- `TableS3_original_image26.png`：原DOCX中`word/media/image26.png`原样提取，无图像编辑。
- `table2_verified.html`：216,173字节，含真实HTML表，SHA256 `c06f7de9bcff4579df3270190862083840a90535714178069e739e0d276304b2`。
- `TableS3_event_transcription_review.csv`：11种处理＋untreated，原始Q2/Q4整数事件数、10,000事件分母、由事件数重算的百分比。
- `source_verification_20261009.json`：实际源文件URL和哈希。

旧`table2.html`为一次HTTP 200访问挑战响应（3038字节），不是论文表；仅作失败取证，禁止引用为剂量来源。旧download manifest保留其请求记录，内容核验以`table2_verified.html`及新的source verification为准。

没有逐重复Annexin原表或FCS，没有配套RNA；MTT IC50的SD不当作Annexin重复误差。药物结构登记待完成；全部`use_for_training=false`。该同骨架系列的剂量7.75–31.35 μM，不是当前Tahoe 0.05/0.5/5 μM条件的直接配对。

提取脚本：[collect_a549_endpoint_review_sources_20261009.py](../../../scripts/collect_a549_endpoint_review_sources_20261009.py)；[研究说明](G:/研究生科研文件/paperstudy/02-notes/LungCancer_Data_Gap_Resolution_Study_2026-10-09.md)。源文件和派生产物保持本地。
