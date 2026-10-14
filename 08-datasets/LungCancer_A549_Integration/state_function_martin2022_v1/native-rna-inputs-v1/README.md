# Martin observed native RNA inputs v1

2026-10-11。这是**真实已观测RNA输入包**，不含潜在h、模型生成表达、功能Y或decoder资格。

来源：GSE193258已登记的两个official processed矩阵；原基因symbol顺序19712、60个RNA biological replicate样本。原source matrix/样本/基因轴/共同轴与builder源码的SHA256由manifest绑定，reader每次构造从登记源重建并逐值核对，不能仅靠caller自填矩阵与重算hash冒充来源。

独立复审修复后的loader revision为`fixed_registered_snapshot_v2`：25项manifest/请求/角色/RNA/轴/原gz及生成器代码先对固定expected hash校验；loader副本另须与实际执行模块字节一致，合计26个source binding。允许相同固定内容身份的复制根；caller改角色/时钟/数值再重算manifest不能成为新登记源。复审前manifest已归档，原数组未重写。

| 文件 | 内容与单位 |
|---|---|
| source_log2TPM.npy | 原19712×60 float64，log2(TPM+0.01)，负值合法；原NPY字节复制 |
| source_estimated_counts.npy | 原19712×60 float64，Salmon/tximport lengthScaledTPM估计count，非整数合法；原NPY字节复制 |
| joint_log2TPM.npy / joint_estimated_counts.npy | 60×32461共同Ensembl位置上的原观测值；16167可测位置精确保留，16294缺失为NaN |
| joint_observed_mask.npy | 32461个bool，所有60样本使用同一固定source覆盖；不是过滤/补零矩阵 |
| samples.json | 60列原GSM/column_index/matrix_column/state/生物重复/真实协议metadata，functional_y仍not_attached |
| coordinates.json | ordered source symbols、joint Ensembl IDs、common_to_source_rows（缺失为null）、两个域原单位 |
| lobo_requests.json | 原64个LOBO职责请求保持原样，各fold3背景fit/1背景final |
| cycle_requests.json | 按已有function metadata的背景cycle派生的64个RNA职责请求，各fold2背景fit/1背景selection/1背景final；是独立角色体系 |
| manifest.json | 文件/source/axis/code bindings、真实计数与能力边界 |

两体系的64职责请求均是16个真实背景—处理状态组的角色视图；真实仍48处理RNA样本+12parental DMSO reference。不能计成128独立实验，也不能把RNA三生物重复用于复制功能Y。reference为parallel parental群体，不是同细胞t0，不声明逐行配对。

RNA真实协议：osi500nM acute24h；DTP处理504h；short washout为504+24h；long washout为504h后按背景恢复72/96/168/240h。H1975恢复72h对应observation576h，不能改成连续72h处理；所有request保持function_72h_four_drug_target=false。

使用：

```python
from lung.data.martin_native_rna import MartinNativeRNA

with MartinNativeRNA(ROOT, PACKAGE, allowed_roles=('fit',)) as data:
    rows = data.requests(system='cycle', fold_id='cycle_final:HCC2935', role='fit')
    cid = rows[0]['request_id']
    reference = data.reference(cid, system='cycle', role='fit', domain='log2TPM', coordinate='source')
    observed = data.supervision(cid, system='cycle', role='fit', domain='estimated_counts', coordinate='joint')
```

values为readonly、sample-major float64，observed_mask与值同shape，返回GSM/source columns/ordered gene identity、真实protocol、role system和原单位。reference不选target，只有supervision打开对应target表达。需要selection时显式授该role，final_test还需allow_final=True。LOBO/cycle request ID不能混用。public axes/manifest/request metadata均返回副本。

没有归一化、四舍五入、缺失补零或fraction clipping；两域不能送入现single-cell library10000/log1p路径。数据准备完成不授予bulk decoder/模型权重、function72h RNA或G/Q资格，仍pending。

实际准备命令：`F:/anaconda/envs/study/python.exe scripts/prepare_martin_native_rna.py`。同目录已存在时拒绝覆盖。固定anchor修复后轻量边界测试28项通过；两official processed TSV全部2365440值和512次全职责/双域/双坐标监督+reference-only再次读回精确一致；最新结果见Round21 bulk72_input_dependency_audit/prepared_full_readback_fixed_anchors.json，初次准备结果继续保留。未改原RNA/角色/功能标签，未运行训练/forward/PT加载。
