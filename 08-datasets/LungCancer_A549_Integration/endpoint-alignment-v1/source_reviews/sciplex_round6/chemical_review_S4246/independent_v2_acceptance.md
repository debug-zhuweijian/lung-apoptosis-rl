# chemical_inputs_v2 独立验收

日期：2026-10-10。结论：**通过本次化学输入包验收**。范围为只读数据、来源和指纹核对；没有检索、训练、改主表或改源码，不代表真实读取器集成或模型效果已经验收。

已核对 [v2 manifest](../chemical_inputs_v2/manifest.json) SHA256 为 `c13e5a9be54692a7a708dbf8a82dea3000fafa0edbcdd92eeec12b48e87f5a05`，与指定版本一致。

| 检查 | 实际结果 |
|---|---|
| 来源与输出完整性 | 12 个输入、6 个输出的 SHA256 全部与清单一致 |
| 原目录保留 | 从原 sci-Plex metadata 重新提取 A549 非 vehicle 目录；188 个唯一目录全部保留，名称、原 SMILES、CAS、分子量逐项一致 |
| 主指纹 | `(187, 2048)` uint8；187 个 eligible 记录的行号恰为 0–186，行序与 manifest 的 drug ID 完全一致；采用 RDKit Morgan radius=2、includeChirality=true 独立重算，187 行逐位一致 |
| S4246 | 主表仍含原相对立体字符串及专门 `source_relative_stereo` 身份，`eligible_for_structural_features=false`、`fingerprint_row=-1`、InChIKey 缺失；未用任何单一绝对对映体替换主身份 |
| 两个候选 | 两条均链接同一 S4246 相对实体；两个指定中心整体翻转后，候选 1 与候选 2 完整匹配，且与官方 SDF 的相对立体注释兼容 |
| 候选指纹 | 独立重算两行均一致；两行相差 10 位，只表示指纹区分，不推断生物学效应 |
| 组成与实验单位 | `mixture_weight` 均空，`observed_material_identity=false`，明确为 alternative interpretations；没有写入主目录成为额外实验，源目录仍为 188 |
| v1 保留 | [chemical_inputs_v1](../chemical_inputs_v1/manifest.json)仍存在；188 条原目录保留，4 个已登记输出 hash 仍与 v1 清单一致 |

另外，CSV/Parquet 内容按明确的空字段口径比较一致。S4246 的 `parent_*` 字段仍属于对绘图的派生解析信息；主身份、eligible 标记和专用候选表没有把它升级为绝对材料身份。

本验收未改变立体结论：两候选没有 50:50 或其他比例依据，不能解释为已确认单一对映体或确定混合物。消费端是否在索引指纹前遵守 eligible / `-1` 标志，由其真实读取器集成另行验证，本报告不代替该执行检查。

本次只新增本文；v2 manifest 绑定的原独立审查、下载记录、结构比较、官方 SDF 和 PubChem 文件均未修改。
