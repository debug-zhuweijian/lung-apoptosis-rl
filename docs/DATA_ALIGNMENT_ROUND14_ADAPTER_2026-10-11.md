# Round14 完整协议 G/Q adapter 实际接线

实现范围为新 `endpoint_protocol_features_v2.py` 与 `endpoint_protocol_model.py`，未替换 Round13 bridge、冻结消费包、旧 endpoints/world 或 DA-AC checkpoint。V2 只在显式 cache 路径时调用 producer 的 `load_standard_cache(path, input_bridge)`，无缓存仍 pending。

`ProtocolConditionAdapter.encode` 把候选实际剂量、2048结构指纹、真实6/24h暴露、洗脱时钟及适用mask、18/0h恢复、共同24h终点、换液事件计数、前药保留/浮游回收类别及unknown、source-defined state/state_kind、名义接种数与单位、容器及诱导历史编码成具名NumPy数组。名义接种数不是测量时密度。标准P只有population/bio[27]/context进入预测数组；standard history/elapsed/horizon只作溯源，不能冒充pulse实际历史。

`enumerate_with_g` 和 `rank_with_candidate_q` 接受独立注册scorer，实际将完整数组传到各scorer；记录模型/算法/证据身份与同一个 terminal Annexin定义和尺度。G枚举不是已训练RL。模型和cache缺失时不输出预测；不会构造随机网络或执行fit。排名保留并列，不加入剂量、时长、风险或其他成本。

scope固定完整endpoint_definition、source/cell/state/state_kind/dose_scenario；不固定protocol_arm或dose。因此Vielhauer同一状态下20个合法候选可比较两臂；Nav必须固定状态与10/11剂量敏感性场景后比较两药。没有读取observations、observation_mapping或reference_observations，未扩张Nav的4个真实Y为8个独立样本。未新增训练view。

架构对应：§2/§4.2完整候选条件与真实物理时间；§4.3群体P摘要加实验协议；§4.4独立候选Q、同terminal尺度与ties；§6直接G枚举baseline与RL候选价值需分别验证。代码接线通过mock数组测试不证明模型精度、真实Annexin预测、RL训练或跨源状态迁移；真实pipeline尚未供给，实际descriptor为0，正式功能评价仍pending。

验证：`F:/anaconda/envs/study/python.exe -m pytest tests/test_endpoint_protocol_model.py -q`，5 passed（纯mock cache/scorer、真实输入身份与数组路由）。旧bridge测试需与producer测试由主代理做组合回归。
