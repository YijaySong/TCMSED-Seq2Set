# TCMDeS-Seq2Set 主模型

本文件夹包含 GRU 序列集合模型的独立源码、数据、固定三折划分、训练和预测脚本。它不调用旁边的论文实验包，也不需要其他对照模型代码。

## 文件

- `gru_sequence.py`：症状集合编码、GRU 标签解码、随机目标排列、联合 ADD/STOP 决策、动作掩码和训练损失。
- `train.py`：按冻结协议训练 `gru_stop` 主模型，默认三折、种子 9/17/29、每次 100 轮、batch size 64。
- `predict.py`：读取训练检查点并输出无序证候要素集合及动作轨迹。
- `splits/`：固定三折 ID 划分和数据/协议哈希。
- `广安门心绞痛4200例.txt`：与划分协议对应的数据。

## 训练

```bash
python3 -m pip install -r requirements.txt
python3 train.py --cpu-threads 6
```

程序依次训练三折和三个种子，结果保存在 `outputs/gru_stop/fold_<fold>/seed_<seed>/`。每次运行会保存检查点、每轮历史、测试预测和指标。中断后再次执行会从最后保存的完整轮次继续；已完成的运行会跳过。也可以用 `--folds 1 --seeds 9 --output outputs_fold1_seed9` 单独训练一折一个种子；若随后要跑完整协议，请使用新的输出目录。

先做短检查：

```bash
python3 train.py --smoke-test
```

## 预测

新病例文件使用 `id symptom` 两列即可，也可以保留第三列 `Se`，程序不会读取它。症状以英文逗号分隔，症状词需要与训练词表一致。示例：

```bash
python3 predict.py \
  --checkpoint outputs/gru_stop/fold_1/seed_9/best.pt \
  --input new_cases.txt \
  --output predictions.json
```

该检查点对应折 1 的模型。部署时应根据训练用途确定采用哪一折或另行用全部训练数据重训；不能把折外模型直接称为全量数据最终模型。输出包含每条记录的预测集合、多热向量和 ADD/STOP 动作轨迹。词表外症状会报错，应先使用一致的临床词项规范化规则处理。
