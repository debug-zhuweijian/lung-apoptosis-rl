# sci-Plex 源结构输入 v2

日期：2026-10-10。本版替代 `chemical_inputs_v1` 作为新联合数据包的消费入口，旧版与旧生成脚本保持原样。**188 条源目录记录全部保留；187 条可编码为当前单结构 Morgan 输入，1 条相对立体实体保留但暂不进入该输入。** 这不是对 187 种历史实验材料的独立鉴定，也不是 RNA 与凋亡已经配对。

`chemical_identity.parquet/csv` 保留源名、SMILES、CAS、盐片段和原有立体信息。`fingerprints.npy` 为 `187 × 2048` 二值数组，radius=2、includeChirality=true。只允许对 `eligible_for_structural_features=true` 的行按 `fingerprint_row` 取值；合格行恰覆盖 0–186，不合格行为 -1，禁止负索引或零向量冒充有效结构。与旧 v3 的 18 个完整记录结构匹配仍只是结构关系，不继承物料或实验配对证明。

S4246 的源字符串含孤立 `*`。主代理亲读原厂 SDF，确认其原子 alias 为 `relative stereochemistry`。删除注释后直接使用所画绝对构型会丢失语义；按 CAS 查询取得的 CID6419961 又是其镜像，不能直接替换。该记录使用来源限定的相对实体 ID，保留原字符串及歧义状态，`eligible_for_structural_features=false`。依据见[独立原件审查](../chemical_review_S4246/independent_review.md)和[官方 SDF 原件](../chemical_review_S4246/independent_Selleck_S4246.sdf)。

`relative_stereo_candidates.csv` 与 `relative_stereo_candidate_fingerprints.npy` 保存所画关系允许的两个绝对解释，保留氯化物片段。候选没有混合权重、没有观测物料身份声明，不是两种实验，也不进入主指纹矩阵。未来若实现不确定结构输入，需另行验证；本版没有用忽略手性或平均指纹绕过这一限制。

[manifest.json](manifest.json) 绑定 12 个输入及 6 个输出。`inputs_sha256` 从项目根解析，`outputs_sha256` 从本目录解析。生成入口为 `scripts/build_sciplex_structure_sidecar_v2.py`，拒绝覆盖已有目录。针对原结构保留、未知虚拟原子拒绝、相对关系证据对应、镜像与指纹资格的 8 项定向测试全部通过，0.72 秒；主代理另核对全部 18 个登记哈希与真实表行号。

当前没有删除 S4246 的源表达或计划孔记录，没有生成真实模型预测或启动正式训练。完整数据可行性继续按课题的状态、表达、功能终点和候选选择分别验收。
