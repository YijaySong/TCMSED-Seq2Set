# M04 GRU 机制对照：三折运行包

此文件夹可单独上传计算平台。数据、三折 ID 清单和完整 GRU 源码来自 M03 运行包；启动时会核对数据及划分 SHA-256。四种方法都在每折独立训练，不能用旧版 2940/630/630 单次划分的结果替代。每种方法运行 3 折 × 3 种子，共 36 次训练；每次 100 轮、batch size 64，使用 CPU。

| 代码名 | 相对完整 GRU 的改动 |
|---|---|
| `gru_stop` | M03 的完整 GRU；随机参考标签顺序、联合标签/STOP 头及合法动作掩码 |
| `gru_fixed_order` | 训练目标按词表顺序排列；解码仍可按任意顺序添加标签 |
| `gru_no_label_history` | 前一动作嵌入换为固定 BOS；保留 GRU 隐状态、动作掩码与 STOP |
| `gru_no_stop_cardinality` | 从症状上下文先预测标签数，再用 GRU 添加相应数量的不同标签；标签交叉熵与基数交叉熵相加 |

最后一项同时改变输出头、训练损失和终止规则，属于替代方案比较，不能将差值单独归于 STOP。四种方法使用相同的症状均值编码器结构和随机初始化规则，各自端到端训练。完整 GRU 与 M03 使用同一网络、折、种子、批次顺序、独立目标排列随机数流、AdamW、梯度裁剪 1.0 和 EMA 上限 0.995。每轮按验证集完全匹配率、样本 F1、较早轮次选检查点。

在计算平台进入该文件夹，执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
python3 run_all.py --smoke-test --cpu-threads 2
python3 run_all.py --cpu-threads 6
python3 analyze_results.py --verify-checkpoints --cpu-threads 6
```

冒烟测试只做单次更新和验证集解码，不写正式结果。完整运行结束应显示 `COMPLETE 36/36 runs`。`results/analysis_report.md` 汇总 4,200 条折外记录的三种子指标、折别成绩和配对病例重采样区间。把整个 `results` 文件夹复制回此目录即可复核；它包含每次运行的 `summary.json`、`history.json`、`predictions.json` 与 `best.pt`。核验脚本检查 36 次运行的协议、文件哈希、100 轮历史、病例 ID、参考标签、动作轨迹和保存指标。加上 `--verify-checkpoints` 后，会重载每个检查点并重新预测测试折，核对最终集合；因并列动作分数造成的轨迹顺序差异另行计数。

一次运行中断后，原命令会跳过协议与文件哈希均一致的完整运行，并重做残缺运行。可按方法、折和种子分配计算作业，例如：

```bash
python3 run_all.py --init-only
python3 run_all.py --method gru_fixed_order --fold 2 --seed 17 --cpu-threads 6
```

并行作业先由一个作业执行 `--init-only`；不同作业不得写同一方法、折和种子的目录。若 M03 结果也在本机，可额外核对完整 GRU 的每条预测集合：

```bash
python3 analyze_results.py --verify-checkpoints --m03-results ../matched_encoder_comparison/results
```

三折按记录划分，每折训练/验证/测试为 2380/420/1400。原始记录此前已经参与方法探索；结果属于开发后的记录级内部评价，不能称为独立队列或患者级验证。病例级重采样区间条件于现有划分和训练实例。
