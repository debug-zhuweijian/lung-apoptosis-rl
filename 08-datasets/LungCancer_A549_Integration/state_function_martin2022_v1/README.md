# Martin2022 同来源 RNA 参考与相对 caspase 候选

2026-10-11。非 A549 的跨 LUAD 补充来源；完整研究主线保持 A549 优先。原 RNA GSE193258 曾作为历史辅助来源，伴随论文 DOI 10.1158/2767-9764.CRC-22-0066 的功能读出是本轮新增信息，不重复计算 GEO 来源。

- [RNA 完整读取说明](README_RNA.md)：19,712 symbol 行 × 60 bulk 样本，四背景五状态各三生物 RNA 重复；12 亲本平行 reference，16,167 / 32,461 唯一 symbol 共同轴覆盖。缺失保持 mask。
- [功能 manifest](function/manifest.json)与[数值 CSV](function/published_means.csv)：S2F/M 共 40 个发表柱身份，20 单药、8 vehicle、12 组合；相对 caspase 72h AUC，不是 Annexin fraction 或独立生物重复。
- [化学说明](chemistry/README.md)：四个官方名义 parent 实体、真实 4×2048 chiral Morgan 指纹和 20 单药绑定；实验 stock form 未认证，组合未生成伪单药输入。
- 当前读取版本 [bridge_v2/manifest.json](bridge_v2/manifest.json)：四 RNA 参考、20 单药、两测量域、八决策块、四回顾性 LOBO fold。每 fold 15 fit / 5 final，S2F 9/3、S2M 6/2；无 selection，不获得 Q 奖励资格。bridge_v1 因早于化学 manifest 最终更新而失效，保留原字节供追溯。

脚本 `scripts/build_martin_state_function.py --verify` 只核验 metadata 和真实资产身份；`reference_expression` 返回原轴 bulk，`model_input` 返回合法候选和真实指纹，serving 不解析功能 Y。显式标签 API 与最终预测冻结仍须按正式评价入口另行连接。原生 bulk/state/72h P、测量域 G、Q 及效果未执行。

同来源平行背景桥是 explicit_assumption_unvalidated。没有同孔/同批次 RNA—功能配对，没有把 Y 复制到 RNA 三重复。acute RNA 500nM/24h 不是功能 160nM/72h 的匹配药后 RNA。AUC 算法/完整单位未知，不能校准成绝对凋亡比例。RNA 与背景共变，LOBO 不能直接证明 RNA 状态增量。

完整解释及实际验收见[第十六轮报告](../../../docs/DATA_ALIGNMENT_ROUND16_2026-10-11.md)、[独立审查](../../../experiments/20261011-data-alignment-round16/martin_independent_review/technical_review.md)。原件、数据本体与运行产物均只保存在本地，不进入 Git/Git LFS。
