# S4246 独立名义化学身份复核

日期：2026-10-10。范围限于已有 PubChem CAS 查询结果及 Selleck S4246 官方产品页、说明书、产品 SDF；没有广域检索、联系厂商、修改原数据或主代理 sidecar。

**结论：不能用 CID 6419961 替换原结构并称为确定的单一对映体。官方 Selleck SDF 直接解释了那个孤立 `*`：它是带有 `relative stereochemistry` 标签的绘图伪原子，不能丢掉其相对立体语义后宣称绝对构型已确定。** 可以采用 S4246 的名义相对立体实体及其明确的连接关系；绝对立体组成、两 CAS 的注册关系和对映体比例仍应保留歧义。

## 实算：删除孤立伪原子并不等于 CID 6419961

只删除原分子中 degree=0 / atomic number=0 的一个伪原子，未删除 `.Cl`，未作去盐、重离子化或手性改写。用 RDKit 完整 isomeric SMILES 和 InChIKey 比较：

| 对象 | canonical isomeric SMILES | 完整 InChIKey |
|---|---|---|
| 原字符串删除孤立伪原子后 | `Cl.N[C@@H]1C[C@H]1c1ccccc1` | `ZPEFMSTTZXJOTM-OULXEKPRSA-N` |
| 原始 PubChem CAS4548-34-9 查询返回的 CID6419961 | `Cl.N[C@H]1C[C@@H]1c1ccccc1` | `ZPEFMSTTZXJOTM-RJUBDTSPSA-N` |
| 当前 Selleck 网页的未指定立体 SMILES | `Cl.NC1CC1c1ccccc1` | `ZPEFMSTTZXJOTM-UHFFFAOYSA-N` |

前两个分子的连接关系和 C9H12ClN 分子式相同，但两个中心整体反转：原绘图按绝对构型读取时，氨基所连环碳 R、苯基所连环碳 S；CID6419961 对应位置为 S、R。程序将原绘图两个中心全部翻转后才与 CID6419961 完整匹配。因此一个 CAS 字符串命中该 CID，并不能证明它与原 SMILES 完全相同。

全部计算、RDKit 版本和来源 hash 保存于 [independent_structure_comparison.json](independent_structure_comparison.json)。这只是对两个结构编码的独立计算，不是对历史实验材料绝对构型的测定。

## 原厂 SDF 的决定性信息

从[产品页所链接的官方 SDF](https://file.selleckchem.com/downloads/product-sdf/tranylcypromine-2-pcpa-hcl-s4246.sdf)实际取得 1597 字节文件，保存为 [independent_Selleck_S4246.sdf](independent_Selleck_S4246.sdf)。SHA256 为 `e0b52281e2f19be69dea55f0e249d81fe2f485b062cc010f5f0c223884e13b41`。

其字段为 Catalog Number `S4246`、CAS Number `4548-34-9`，完整解析结果恰好等于原 sci-Plex SMILES：`*.Cl.N[C@@H]1C[C@H]1c1ccccc1`。第 12 个原子为孤立 `R` 伪原子；`A 12` alias 的原文是 **relative stereochemistry**。V2000 chiral flag 为 0；本审查的解释主要依赖明确的文字标签，而非单凭 flag。

该文件支持所画两个中心之间的相对关系；将整套中心同时反转仍保持该相对关系。它没有规定某一个绝对对映体就是产品，也没有提供可用于断言 50:50 外消旋组成的比例。原始 SDF 没有被修改。不能因 RDKit 将未识别为 enhanced-stereo group 的楔线直接转为 `@/@@`，就忽略原始文本的限制。

## 当前网页与说明书能支持什么

[Selleck 产品页](https://www.selleckchem.com/products/tranylcypromine-2-pcpa-hcl.html)列 S4246、CAS1986-47-6，并给出无手性标记的 SMILES；[S424608 说明书](https://www.selleckchem.com/datasheet/tranylcypromine-2-pcpa-hcl-S424608-DataSheet.html)同列该 CAS。两者在这里均未明确指定产品的单一绝对对映体或比例。它们介绍 R、S、± 的生物活性文献，不能把该综述句当作本产品立体组成的质检声明。

网页与 SDF 的 CAS 字段不同，本轮保留这个事实，不推断注册关系或宣布哪个号码“正确”。两份原始网页成功以 HTTP200 保存；SDF 首次普通请求403，第二次普通浏览器 UA/产品 Referer 请求200并成功解析，见 [independent_acquisition.json](independent_acquisition.json)。没有把失败响应冒充 SDF。

## 建议供主代理采用的表示

建议状态：`nominal_relative_stereochemistry_verified_absolute_composition_unspecified`。

- 可保留 S4246 的源条件、原厂来源、名义盐形式、连接关系、相对立体约束及 CAS 冲突说明，不必因为没有历史 lot 而删除整条实验资料。
- 若下游模型要求确定的完整绝对立体指纹，这一条不能直接冒充已经唯一核定的单结构。原绘图的绝对解释与其镜像可作为**输入解释的两个候选**，分别保留完整身份；这不是两份实验，也不能假设两者等比例混合。
- 若仅使用不含手性的连接特征，明确记录模型忽略了立体信息；当前网页的无手性 SMILES本身还会丢掉原 SDF 的相对约束，不能称为精确完整结构的等价替换。
- 严禁把 `*.Cl...` 简单去星后标为绝对构型已证实，或用 CAS 查询直接替换为 CID6419961。立体组成未唯一确定与“连接结构不可解析”是不同状态。

本审查仅新增 `independent_*` 文件；未覆盖 `PubChem_CAS4548-34-9.json`、`acquisition.json`、任何已冻结包或主代理结构表。最终 v2 输入表示由主代理综合本证据决定。
