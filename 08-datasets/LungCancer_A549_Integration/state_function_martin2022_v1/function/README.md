# Martin2022 S2F/S2M published function means v1

原始来源：[Martin et al., PMC10035388](https://pmc.ncbi.nlm.nih.gov/articles/PMC10035388/)，Methods “Apoptosis Assay” 与 Supplementary Figure S2F/S2M。

`published_means.csv` 保存四背景、两个原图 panel 的 40 个发表柱均值与有限像素读图界。`manifest.json` 保存原件哈希、log 轴校准、8 个完整候选 decision blocks 及限制。20 行单药活性、8 行 vehicle、12 行 combination；组合保留完整组分，不拆为单药监督。

使用 `scripts/digitize_martin2022_function.py` 重建，测试 `tests/test_martin2022_function.py`。端点为 72 h relative caspase-3/7 AUC，展示尺度未完整说明；读图界不是 CI，不含 SD 或生物不确定性。原文 triplicate wells 不是已确认三次生物重复。

同一背景在 S2F/M 的同药柱是不同发表观测身份，不因此证明独立实验批次。RNA 与功能桥仅为同源、同背景的平行桥假设，不是样本配对；不得向 RNA 重复复制 Y，不得换算 Annexin、单细胞凋亡率或作为 A549 实测标签。S2N DTP 仅 quarantine 元数据，未提取数值。详细回执见 `experiments/20261011-data-alignment-round16/martin_function_receipt.md`。
