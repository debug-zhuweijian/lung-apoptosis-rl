# sci-Plex来源结构输入：独立于物料身份结论

日期：2026-10-10。已从官方hashTable的A549记录读取188个catalogue条目，保留原SMILES、CAS、名称、分子量，实际生成188×2048手性Morgan指纹。**本包允许编码来源所报告的结构，不声称188种实验物料均已独立鉴定。**

- 188条均可由RDKit解析，188个完整结构ID不同。
- 41条源SMILES有未指定立体信息；保持原样，不按药名补造。
- 29条有多个片段；保留全形，不用母体替换。
- 1条不能生成InChIKey；保留来源SMILES和其内容身份，并标明限制。
- 18个条目与当前training-identity-v3的完整结构相同；仅为结构记录相同，不是试剂、批次或实验样本对应证明。

[chemical_identity.csv](chemical_identity.csv)和[Parquet](chemical_identity.parquet)保存实际输入字段；[fingerprints.npy](fingerprints.npy)按fingerprint_row读取。完整结构ID继续使用原项目的full_isomeric_smiles_v2定义。`eligible_for_structural_features`只表示来源结构可编码，资格范围另以`eligibility_scope`明确；`experimental_material_verified`和`cross_source_pair_verified`均为false。

[v3_full_matches.csv](v3_full_matches.csv)列出全部相同完整结构对应及旧角色，供新划分审计。非立体连接图或最大有机片段可以另外用于保守防泄漏分组，但不能借此替换指纹、合并动作或称材料等价。新的全局联合数据包可以重新划分；旧v3的角色不是新实验不可改变的科学条件。

[manifest.json](manifest.json)的`input_path_base=project_root`明确所有inputs_sha256相对于项目根，`input_base=.`也在该基准下解释；outputs_sha256相对于本目录。六项输入包括官方metadata、旧v3化学/条件/manifest、实际结构解析代码和生成脚本。[生成脚本](../../../../../../scripts/build_sciplex_structure_sidecar.py)不覆盖已有版本。

源内离散处理条件可按来源限定catalogue ID定义。涉及跨药结构泛化时，必须同时报告这些结构不确定性与敏感性范围；单纯可解析不能升级为完全准确的药物身份。当前包没有响应表达、训练划分或终点标签，`formal_training_ready=false`。
