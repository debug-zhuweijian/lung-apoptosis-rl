# Tahoe 全部已发布 A549 表达：下载与完整性验证

本文件对应主工作区 `I:\科研\肺癌细胞凋亡-强化学习`。此任务只准备数据，不执行响应预测器、DA-AC 或 DLPA 训练；不改变两套架构的奖励、实验终点或数据划分。

## 官方来源与范围

- 官方仓库：[tahoebio/Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M)。
- 锁定 revision：`2dc57900b7981cfcf5e211527169a0b006546a95`。
- 以 Cellosaurus ID **CVCL_0023** 筛选 A549，不按名称模糊匹配。
- 官方表达发布有 **3,388 个混合细胞 Parquet 分片**。本次完整官方文件清单实算源文件共 **337,644,770,670 字节（337.64 GB／314.46 GiB）**；这是取得所有 A549 行所需遍历的发布分片，不表示项目研究范围变成全部细胞系。
- 已保存的完整 obs 包含 **2,664,482** 条 A549 观测：`full` 2,567,838、`minimal` 96,644。观测元数据不是表达落盘证据；最终表达行数以全分片提取及逐条码核对为准，不预先保证 minimal 条码在表达发布中存在。
- 3,388 个分片的完整路径、大小、官方 LFS SHA256 记录在版本目录下 `full_expression/source_manifest.json`。数据卡登记的已发布表达总行数为 95,624,334，全部分片读完后核对该数。

## 为什么遍历混合分片

官方 Parquet 每个行组混合不同细胞；A549 的基因与计数不能通过只读元数据列取得。本次复核 HF 过滤接口得到 HTTP 500（服务繁忙），不能用它保证完整子集。当前采用官方固定版本分片、逐源哈希和本地精确筛选；不需要账户、付费数据入口或第三方镜像。

源文件初次下载/接受时核验字节数及官方 LFS SHA256。原始分片保留在版本目录 `data/`，既有文件校验不通过时直接报错，不覆盖历史数据。网络采用每次最多 8 MiB 的有界 HTTP Range，已写入的 `.partial` 可从实际长度恢复；仅完整且哈希匹配后改名为正式文件。

## 当前运行与恢复

2026-09-18 已运行全量任务；**运行中不是全量完成**。状态以磁盘文件为准，本文不固定不断变化的进度数字。

入口：`scripts/download_tahoe_a549_full.py`。

```powershell
python scripts/download_tahoe_a549_full.py --workers 16 --retries 3 --verify-hashes
```

- 网络最多 16 路，Parquet 解码与 CSR 构建通过信号量限制为 **2 路**，不把下载并发直接扩成 16 份解码内存。
- 主状态：`08-datasets/Tahoe-100M/2dc57900b7981cfcf5e211527169a0b006546a95/full_expression/download_status.json`。
- 当前后台执行日志：`logs/tahoe_a549_full_20260918_final_v2.stdout.log`、对应 `.stderr.log`。较早运行日志保留作为调整下载机制的记录。下载错误仅记异常类型与 HTTP 状态码，不打印重定向后的签名 URL。
- `download.lock` 记录运行 PID，防止两个下载器覆盖相同分片。异常退出后，先确认该 PID 已结束，再移除这一锁文件并运行相同命令；不能在旧进程仍运行时清锁。
- 断点恢复默认复用已经完成且大小/修改时间不变的 checkpoint；这一步**不会重新计算全部哈希**。当前实际启动命令带 `--verify-hashes`，在所有分片完成后的最终审计会重新计算每个源文件和派生文件的 SHA256。
- 单片最多尝试 3 次；连续失败会记入 `failed_shards` 并停止该片；其他分片继续处理。最后如仍有失败，状态为 `incomplete_retry_required`，查明失败原因后可重复同一命令继续。
- `--limit 2` 仅为读取验证，不代表全量；其终态专门标为 `smoke_complete`。

全部下载完成后也可独立重验：

```powershell
python scripts/download_tahoe_a549_full.py --audit-only --verify-hashes
```

## 新增输出与读取

均位于锁定版本的 `full_expression/`，原冻结脚本和既有派生结果不修改。

| 路径 | 含义 |
|---|---|
| `raw_a549/train-xxxxx-of-03388.parquet` | 分片内全部 A549 原始行，保留完整 genes／expressions 列及 CLS 首项，增加 source_shard |
| `csr/train-xxxxx-of-03388.npz` | 相同细胞顺序的稀疏原始计数矩阵；已去除每行唯一 CLS 首项；行对应同名 obs，列对应 var |
| `obs/train-xxxxx-of-03388.parquet` | 相同细胞顺序的条码、sample、plate、药物等源元数据 |
| `var.parquet` | 62,710 个基因按 token_id 排序；含 Ensembl ID、gene_symbol、token_id；所有 CSR 分片使用相同列顺序 |
| `checkpoints/*.json` | 单片源与派生文件大小、SHA256、完成时间、源行数、A549 行数与非零数 |
| `a549_expression_coverage.parquet` | 审计后生成的逐 obs 条码覆盖标记；未完成运行时可能仍为较早读取验证结果，必须结合 scope/状态使用 |
| `a549_expression_sample_coverage.csv` | 按 sample、plate、pass_filter 统计观测数与实际表达数 |
| `coverage_audit.json` | 全部条码、sample、plate 核对结果及缺失过滤级别计数；scope 区分 partial_expression / complete_published_expression |

读取示例（数据接口，不是训练或模型精度证明）：

```python
from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz

root = Path(r"I:\科研\肺癌细胞凋亡-强化学习")
base = root / "08-datasets/Tahoe-100M/2dc57900b7981cfcf5e211527169a0b006546a95/full_expression"
stem = "train-00000-of-03388"
X = load_npz(base / "csr" / f"{stem}.npz")
obs = pd.read_parquet(base / "obs" / f"{stem}.parquet")
var = pd.read_parquet(base / "var.parquet")
assert X.shape == (len(obs), len(var))
```

本轮交付分块 CSR＋obs／var，避免强制将数百万细胞一次拼入内存；尚未创建合并 H5AD。没有对 counts 归一化、筛选高变基因或划分训练集。后续模型准备应依实验协议处理，不能把下载器的固定 gene 轴误当作已选择的训练面板。

## 完成判据与已运行检查

完整成功要求：3,388 个分片均有有效 checkpoint；源行数之和符合官方发布；每个提取条码在完整 A549 obs 中唯一存在，sample 与 plate 精确一致；参考 obs 中所有 `full` A549 条码均已覆盖；报告明确各过滤级别的表达覆盖和未覆盖；`download_status.json` 最终为 `complete`。如遍历完但源总行数或 full A549 覆盖不符，则标记 `coverage_mismatch` 并以非零退出码结束，不登记为成功。当前执行使用 `--verify-hashes`，还要求最终再次核对所有源/派生 SHA256；意外异常标记 `failed`。

已完成的针对性验证：

1. 对前两个已核源文件重跑，重现 **1,813** 个 A549 表达行，并与完整 obs 逐条码、sample、plate 对齐；该结果只证明读取验证子集。
2. 8 项 Python 标准库 unittest 检查通过：基因轴与计数保持；错误 CLS、负计数、非整数计数、长度不匹配、重复/未知 token 拒绝；同大小且同 mtime 的损坏能被严格哈希验证检出。执行 `python -m unittest discover -s tests -p test_download_tahoe_a549_full.py -v`，未引入新测试框架。
3. 独立复审对首片源文件、原始 A549 子集、CSR、obs 四个文件重算哈希均一致，并从原始 token 重建抽查行，与 CSR 逐值一致。

整批数据完成不意味着凋亡评价器、序贯预测或强化学习路线已验证。跨来源 TIDYI 状态标签、真实药效结局、真实序贯终点继续按现有架构与验证协议分别审查。
