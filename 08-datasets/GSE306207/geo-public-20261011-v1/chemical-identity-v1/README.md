# 两个来源名称的独立名义结构

2026-10-11实际读取PubChem PUG REST的 `name_type=complete` 名称→CID、CID properties和synonyms记录。范围只含原GSE306207的 I-BET726、Oxaliplatin。两个查询各返回唯一CID，并在返回CID的官方synonyms核同名称；Oxaliplatin在synonyms为同词小写，不作其它名称替换。

| 源名称 | 唯一CID | InChIKey | 完整图组分 | FP置位 |
|---|---|---|---|---|
| I-BET726 | 52912222 | FAWSUKOIROHXAP-NPMXOYFQSA-N | 1 | 50 |
| Oxaliplatin | 9887053 | DRMCATBEKSVAPL-BNTLRKBRSA-N | 3 | 17 |

完整立体SMILES、InChI和原始字段见 [nominal_chemical_identity.json](nominal_chemical_identity.json)与 `raw_pubchem/CID*_properties.json`；每个原响应保存URL、UTC读取日期和SHA。两药均有2个已定义立体中心，未定义原子/键立体数为0；RDKit从完整SMILES得到的InChIKey与PubChem一致。

FP采用现有冻结 `GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)`，RDKit2026.03.6，uint8的2×2048。保留所有组分、金属与电荷，不取largest organic fragment。Oxaliplatin复用原identity-v3的正式CID属性响应（`properties_0200.json` SHA8d777944…）并与原FP第25行逐位相同；I-BET726是本次独立新增名义图，没有写入全局drug table。

[actual_chemical_readback_receipt.json](actual_chemical_readback_receipt.json)独立用旧Morgan API复算全部FP位、核每个官方字段与源名称；[source_protocol_event_links.json](source_protocol_event_links.json)连接8个源样本中的6次I-BET726与2次Oxaliplatin阶段。DMSO/撤药没有伪造药物图或假零指纹。所有连接仍为unassigned，原协议未知项保留。

这是 `nominal_name_graph_only` 的结构资格，**没有认证本RNA实验的stock、供应商/批号、盐型或实际浓度basis**；也没有授A549配对、Annexin、fullP执行、源decoder或latent兼容资格。原RNA包manifest保持原哈希，此化学附属版本单独登记，不能把旧RNA manifest的历史化学pending字段直接改成全部通过。

7项边界检查实际0.74s通过、新I basetemp `experiments/gseqchem1`先确认不存在；真实模型/PT/forward/fit为0。[manifest](manifest.json)绑定冻结资产及来源原响应。专属报告的本轮化学追加见 [DATA_ALIGNMENT_SEQUENCE_SOURCE_ROUND22_2026-10-11.md](../../../../docs/DATA_ALIGNMENT_SEQUENCE_SOURCE_ROUND22_2026-10-11.md)。
