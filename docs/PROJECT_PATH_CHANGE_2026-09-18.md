# 项目路径规则变更核验

日期：2026-09-18。用户明确指定新的主工作目录；paperstudy 项目规则已先备份再修改，新项目规则为首次创建。

| 文件 | SHA256 | 全部标题数 | H2 节数 |
|---|---|---:|---:|
| `G:\研究生科研文件\paperstudy\AGENTS.md` | `1e3577f2e090fea04f38f6b94fe77f40d4986836b9c83a4d27a571d627ad0187` | 24 | 14 |
| `I:\科研\肺癌细胞凋亡-强化学习\AGENTS.md` | `30320decdcb371a3b0e8208edd1559f1d00e0dc952133beca00d3b0c20423369` | 6 | 5 |
| `G:\研究生科研文件\paperstudy\99-archive\path-migration-2026-09-18\AGENTS.before-project-root-change.md` | `12559ba74cb3f330f0b9177b19501f364433da1036b280991b360ebf7fb537c3` | 24 | 14 |

路径一致性：通过。两份生效 AGENTS 均指向 `I:\科研\肺癌细胞凋亡-强化学习`，数据目录均为该路径下 `08-datasets`。Windows 路径大小写不构成另一份配置；未创建重复的 agent.md／agents.md。

调整内容：本课题执行主目录迁至 I 盘；G 盘保留论文、笔记与历史副本。其他数据集的默认治理规则保持原范围。PROJECT_MEMORY.md 与全局规则未修改。

生效方式：当前任务已按新目录执行；未来在该目录启动任务读取本项目 AGENTS。无需软件重启。当前聊天所属 paperstudy 的规则也显式指向新执行目录。

Git：未初始化新仓库、未提交、未推送；已有脏工作树和历史文件不回滚。

复制后治理文档调整：新主目录中的 `08-datasets/LungCancer_A549_Integration/README.md` 将执行报告链接改为指向 G 盘 paperstudy 的绝对路径；`08-datasets/Tahoe-100M/README.md` 增加完整下载任务入口，区分历史小样本核验与当前进度。复制清单记录的是复制时的逐字节核验；这两处有意的文档导航调整不改变数据、代码或冻结实验。
