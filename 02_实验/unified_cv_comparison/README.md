# 统一三折折外对照：计算平台运行包

本包把当前中文稿固定划分表中的模型改用同一份三折 ID 清单评价。每折有 2380 条训练记录、420 条验证记录和 1400 条折外测试记录；每个模型使用种子 9、17、29，各训练 100 轮，batch size 64。每个种子最终为全部 4200 条记录各生成一次折外预测。主指标为完全匹配率，样本 F1 为次指标。

`reference_results/` 已包含并核验四组 M03 三折结果：小型独立 sigmoid、近容量独立 sigmoid、同编码器分类器链、主 GRU。它们不重复训练。本包新增下列训练方法：

| 组别 | 方法代码名 | 结构或对照内容 | 参数量 |
|---|---|---|---:|
| 直接多标签 | `mlp` | 旧版二元症状 MLP | 13,578 |
| 直接多标签 | `cnn` | 症状 token CNN | 157,066 |
| 直接多标签 | `symptom_transformer` | 症状 Transformer 编码器与独立输出头 | 271,882 |
| 直接多标签 | `asl_mlp` | 非对称损失 MLP | 13,578 |
| 标签依赖 | `classifier_chain_legacy` | 旧版分类器链结构 | 13,678 |
| 直接多标签 | `label_graph` | 训练折标签共现图 MLP | 13,578 |
| 直接多标签 | `set_attention` | 症状集合注意力模型 | 338,058 |
| 逐步集合构造 | `transformer_sequence_matched` | 均值症状编码器与近容量 Transformer 标签解码器 | 520,715 |
| 逐步集合构造 | `tcn_sequence` | TCN 症状编码器与 GRU 标签解码器 | 1,274,699 |
| 逐步集合构造 | `patchtst_style_sequence` | PatchTST-style 症状编码器与 GRU 标签解码器 | 1,695,563 |
| 逐步集合构造 | `mamba_style_v3` | 四种症状排列取均值的 Mamba-style 编码器与 GRU 解码器 | 682,827 |
| 集合条件策略 | `set_nll` | 精确集合似然训练；输出贪心、beam-5、精确 MAP 三种解码 | 48,395 |
| 集合条件策略 | `set_rl` | 50 轮监督初始化，接续 50 轮自批评策略梯度 | 48,395 |

共新增 13 种训练方法 × 3 折 × 3 种子，即 117 次训练。`set_nll` 的三种解码共用同一个入选检查点，因此新增预测输出为 135 份，而非 135 次训练。连同 36 份既有参考结果，分析脚本共核验 171 份折外输出。

容量匹配的 Transformer 改的是**标签解码器**；症状 Transformer 是直接多标签模型；Mamba-style v3 改的是**症状编码器**。三者不可合并成一个“Transformer/Mamba 编码”对照。Mamba-style 是本任务的纯 PyTorch 适配实现，不是官方 Mamba/Mamba-2。TCN 和 PatchTST-style 的参数量明显高于 GRU，表格须列出参数量，不能称为近容量比较。

## 上传与运行

解压压缩包后，进入 `unified_cv_comparison` 文件夹，在 CPU 节点运行：

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 run_all.py --smoke-test --cpu-threads 2
python3 run_all.py --cpu-threads 6
python3 analyze_results.py
```

正式运行完成时会显示 `COMPLETE 135/135 outputs (117 trainings)`。随后 `analyze_results.py` 核对每份结果的 ID、参考标签、预测、指标、训练轮次和文件哈希，生成 `results/analysis_report.md` 与 `results/unified_summary.json`。把整个 `results/` 文件夹复制回本机同名包内即可继续修改论文。

一次作业跑不完时，原命令再次执行会跳过已完成的模型/折/种子。训练中每轮保存到 `results/_checkpoints/`；未完成的一次训练会从最后完整轮次续跑。Set-NLL、Set-RL 每轮打印进度，其他方法每十轮打印一次。集群允许按方法拆任务时，先用一个作业运行：

```bash
python3 run_all.py --init-only --cpu-threads 2
```

然后给不同作业分配互不重叠的组合，例如：

```bash
python3 run_all.py --method set_nll --fold 2 --seed 17 --cpu-threads 6
```

不要让两个作业同时写同一个模型/折/种子的目录。`set_nll` 和 `set_rl` 的单次训练成本高于其他模型，集群作业时限应为它们留出余量。数据、划分或训练源码改变后，runner 会拒绝继续写入已有 `results/`。

## 训练与选模口径

所有训练使用 AdamW（初始学习率 0.0003、权重衰减 0.0001、AMSGrad）、梯度范数裁剪 1.0 和参数 EMA（衰减上限 0.995）。验证集完全匹配率优先、样本 F1 次之、仍并列时取较早轮次。直接预测方法与旧版分类器链在验证集搜索 0.05 至 0.95、步长 0.01 的统一阈值；分类器链按每个候选阈值重新逐标签解码。阈值和检查点均不使用折外测试记录。

序贯方法训练时随机排列参考标签，使用独立的随机数流。TCN 和 PatchTST-style 训练时还随机排列输入症状，验证和测试使用固定词表顺序。TCN 输入统一补齐到 40 个症状位，以免其卷积编码受同批其他记录的补齐长度影响。原固定划分的 CNN 曾使用文件中的症状顺序；本次按统一协议使用词表顺序。这些输入规则均写入 `results/protocol.json`。

`set_nll` 按验证集贪心解码选一个检查点，再用该权重计算贪心、beam-5 和精确 MAP 三种折外预测。`set_rl` 从前 50 轮中验证集表现最好的监督 EMA 权重启动，后 50 轮使用学习率 0.00001；最终检查点只能从第 51–100 轮选取。包内同时保存 RL 启动时的监督权重和折外预测，用来检验 RL 更新相对于自身起点的变化。

这些结果仍属于模型开发后、同一批记录上的内部评价。文件没有患者 ID，不能把三折结果写成患者级划分或独立外部验证。
