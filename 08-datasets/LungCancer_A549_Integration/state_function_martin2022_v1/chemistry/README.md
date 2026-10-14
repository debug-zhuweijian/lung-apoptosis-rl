# Martin2022 名义parent结构输入

2026-10-11。四个真实名义实体的结构与指纹已核验。不是实验stock盐型/批次认证。论文Methods sec2-2说明小分子按已发表方法合成，没有明确供应商/stock form。Supplement Fig S2F/M（Martin2022_supplement.md第182行）亲读剂量：osimertinib160nM、venetoclax300nM、AZD4320 300nM、K-975100nM，72h；500nM DTP/RNA诱导不是此parental challenge剂量。

| 名称 | CID | InChIKey | 指纹行 |
|---|---:|---|---:|
| osimertinib | 71496458 | DUYJMQONPNNFPI-UHFFFAOYSA-N | 0 |
| venetoclax | 49846579 | LQBVNQSMGBZMKD-UHFFFAOYSA-N | 1 |
| AZD4320 | 86661883 | UNEJSHNDABUZNY-UJNHCCGESA-N | 2 |
| K-975 | 155353714 | KPXAHQSIROUBPH-UHFFFAOYSA-N | 3 |

Venetoclax复用identity-v3/drug_identity.csv已登记实体，实际验证官方响应哈希、CID与SMILES→InChIKey；另三药PubChem官方PUG REST exact name查询各返回一个CID，并核对官方CID synonyms中的源名。所有参考均单fragment，AZD4320立体信息保留；不改实验stock form未知。不存在待猜结构，四药均resolved nominal parent。

items.json/chemical_items.csv保存CID、SMILES、InChIKey、parent身份、行号和逐行fingerprint SHA；raw/保存原响应与URL/query/status。morgan2048_chiral.npy为4×2048 uint8 binary，radius=2/includeChirality=true，沿用scripts/build_reviewed_identity_v3.py既有约定。没有组合伪指纹。

single_candidate_bindings.csv保存20个来源single观测的observation_id/domain/block/background/源剂量/精确uM/chemical_id/fp row：四背景×(S2F三单药+S2M两单药)。同一osimertinib跨两个panel仍是两项assay观测，绝不合并测量域。nM/1000采用Decimal，160→0.16、300→0.3、100→0.1。没有Y列；function来源表中的组合保留，未修改。

化学输入structure_type固定parent_normalized_not_experiment_stock_form_certified，experimental_stock_form=unknown，experimental_stock_form_certified=false；parent归一只代表PubChem名义parent参考，不推断实验counterion或剂量的盐/活性母核口径。

验证：14项pytest通过，四结构SMILES/InChIKey一致，4行独立FP重算及hash、源响应hash、20 observation bindings、dose精确转换、错误单位/结构/组件/panel剂量均检查。该脚本读取本次已缓存官方响应并建立资产，不重新在线检索或推理。manifest保存源论文/功能表/补充caption哈希及输出清单。

与毕业论文架构一致：仅source-specific跨LUAD背景候选输入，不改变A549优先或state-conditioned apoptosis主线；不表明same-well RNA+function配对、患者获益或功能转移通过，未模型/训练/forward/预测。
