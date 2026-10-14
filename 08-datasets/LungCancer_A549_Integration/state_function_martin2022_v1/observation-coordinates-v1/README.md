# Martin observation coordinates v1 实际准备

已用原始只读RNA与已关闭MartinNativeRNA准备8fold坐标，未修改作者原值或旧来源。数据位置：08-datasets/LungCancer_A549_Integration/state_function_martin2022_v1/observation-coordinates-v1/（8个fold NPZ和manifest）。共同轴32461，实测mask16167，缺失16294维持续NaN；source原值仍由原reader提供。

作者log2TPM域，附加fit-only affine为本课题新可逆数据设计：float64/ddof0，LOBO每fold45唯一GSM、cycle每fold30唯一GSM，重复reference按sample ID去重。constant标准差0时scale1并显式constant_mask；本次真实8fold constant数均0，固定合成测试覆盖constant分支。各fold坐标只用fit表达；selection/final只在明确授权的回转验证读取，未参与moments。

API：MartinObservationCoordinates(allowed_roles=('fit',),allow_final=False)；read_request(request_id,system,fold_id,role,domain='log2TPM',coordinate='joint',supervision=False,finite_placeholder=False)。访问角色完全委托原reader；final须allowed_roles及allow_final=True且bool严格。返回source values/metadata/mask、model_values、inverse_values、fit_sample_ids、constant_mask与False模型/科学资格。可选finite_placeholder只在mask=False设计算零，mask独立返回，不称旧E已消费。没有bulk→SC library10000/log1p，也没有来源decoder/损失。

固定原reader代码/manifest pins后由原reader核其source/code/roles/arrays；新包加载及每次read重建fit并核字节hash+逐数组source重建等值/dtype，不能自更新包hash伪造坐标。manifest固定schema、axis/domain、角色fold/sample身份、0输出与False资格类型，拒False-as-int0/int0-asFalse；不接受caller替代axis/domain/source/ref/hash。输出exclusive已存在拒绝，失败不删除旧包。

验证：实际prepare成功；13 tests passed，包含fit去重、holdout ID不进入fit、mask/实测0/constant、observed逆变换、domain/axis/system/fold/role/类型/源hash拒绝、旧source重建拒绝tampered坐标、final显式权限。verify_readback.py实际核8fold全部60唯一源样本参考/处理回转，float64最大误差3.552713678800501e-15；缺失全保NaN。

源码hash与原source pins见validation.json，新consumer SHA256 7ae4cba93a55ff7c6da2c15fcb32e06b7b5b416288526604c6e15ab9ac658645。P/G/Q0，PT/forward/训练0，coreY未解码；E/D/norm/context模型消费和共享表示的生物迁移仍pending。
