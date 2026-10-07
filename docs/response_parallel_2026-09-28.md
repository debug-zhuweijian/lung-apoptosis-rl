# 响应成员并行训练

日期：2026-09-28。状态：代码与配置已修改；按用户要求，未运行测试、性能基准或正式训练。

## 当前设置

根 `train.py` 的响应阶段使用以下 `configs/response.yaml` 设置：

```yaml
members: 5
member_workers: 2
member_cpu_threads: 4
conditions_per_update: 16
condition_microbatch: 4
prefetch_group: true
pin_memory: true
```

- 共训练五个成员，同时运行两个独立的 Windows spawn 进程；一个完成后立即调度下一成员。
- 每个进程有自己的模型、优化器、随机状态和读取器。成员编号、初始化种子及训练抽样种子继续按全局成员编号计算。
- 每个成员将四个条件合批计算，累计十六个条件才更新参数。`member_cpu_threads` 控制成员内 PyTorch CPU 运算线程，不是数据预取线程数。
- `member_workers` 为正整数，实际并发数不超过 `members`；设为 5 即请求五成员并发。设为 1 使用原串行入口。
- `condition_microbatch` 允许任意正整数，实际计算块还受每次更新的条件数约束。不同骨干和参考模式继续使用原有适配规则。

五成员并发的显存和主机内存需求未实测，当前两个成员的设置也未经过运行验证；不保证 GPU 达到 100% 或固定提速倍数。

## 数据准备

启用预取时，先依次生成每个条件的 CFM 与 MMD 样本行选择，再将本组读取请求合并。同一次读取中，每个源分片只打开和审计一次，随后恢复各请求的行顺序，并分别调用原归一化函数。

该改动保留来源审计、跨调用文件变化检查和原随机抽样次序，不增加常驻分片堆缓存。整组稀疏矩阵会暂时增加主机内存占用。代码目标是减少重复读取；实际收益与数值一致性本轮未运行验证。

## 产物与进度

- 权重继续保存在同一运行的 `checkpoints/<run_id>/response/member_N/`。
- 每个成员的进度、控制台和独立指标保存在 `logs/<run_id>/response_members/<segment>/member_N/`。
- 父进程展示一条成员总进度，并收集各成员的训练指标；共享 CSV、曲线及运行状态由父进程写入。
- 每个成员独立执行验证、早停和检查点保存。采样数、全部 32,461 个基因、CFM/MMD 目标、ODE 步数和数值精度沿用现有配置。
- 某个成员失败时，父进程结束其余成员进程并报告失败，不将未完成成员记为成功。

## 启动与旧进度

用户在 PyCharm 选择现有 `F:/anaconda/envs/study/python.exe`，运行项目根 `train.py`，参数留空。

本轮源码及配置已变化，套件身份也会变化；默认自动恢复只匹配同版本套件，因此会新建训练套件。旧检查点没有被删除、覆盖或迁移，不应将新运行表述为原训练的无缝续训。

本轮没有启动正式训练，也没有执行测试、合成验收或显存压力检查。修改前源码备份位于 `.omo/backups/aggressive-config-20260928/`、`.omo/backups/aggressive-parallel-20260928/` 和 `.omo/backups/aggressive-reader-20260928/`。
