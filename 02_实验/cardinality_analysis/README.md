# M05 数量与病例分析

本目录读取既有 M03、M04 的三折折外预测，不训练模型，也不修改上游结果。输入包默认位于同级的 `matched_encoder_comparison/` 与 `gru_mechanism/`。先调用各自的 `collect()` 校验全部 72 次运行，再对 4,200 条记录汇总逐证候要素识别、预测数量、配对差值及固定规则病例。需要 Python、NumPy 和 PyTorch（用于上游校验器）。

```bash
cd cardinality_analysis
python3 -m unittest discover -s tests
python3 analyze.py
```

输出保存在本目录 `results/`：

- `analysis_report.md`：分层结果、两项诊断基线、配对差值、逐要素指标、数量交叉表及病例。
- `stratified_metrics.csv`：各真实标签数与 1–2、3–4、5–6 合组的三种子指标。
- `per_label_metrics.csv`、`per_label_seed_metrics.csv`：完整 GRU 与近容量 sigmoid 的逐要素阳性记录数、TP/FP/FN、精确率、召回率和 F1；前者为三种子均值与样本标准差，后者保留逐种子值。
- `cardinality_summary.csv`：完整 GRU、近容量 sigmoid 与无 STOP 独立基数头的数量正确、低估、高估比例及数量 MAE，包含总体、逐个真实数量、1–3 与 4–5 标签组。
- `paired_differences.csv`：同一记录 GRU 对照其他方法的三种子平均差；仅总体给出记录级重采样区间。
- `cardinality_matrix.csv`：三个种子的真实数量与预测数量交叉计数。
- `diagnostic_baselines.csv`、`baseline_definitions.json`：每折由训练记录确定的边际前六标签及最常见完整集合。
- `case_examples.json`：按脚本内固定规则抽取的成功、仅漏选、仅多选记录和三个种子的动作轨迹。
- `analysis_manifest.json`：输入协议和数据的哈希。

诊断基线只看每折 2,380 条训练记录，验证及测试标签不参与确定预测集合。三个种子产生的 4,200 条折外预测不是 12,600 个独立病例：配对分析先在每条记录内平均三个种子的差值，再按记录有放回抽样。区间仅描述现有训练实例条件下的记录级差异，不覆盖重新训练、重新选模或患者内相关。1、2 标签组的记录数很少，分组结果供描述，不单独作显著性推断。病例只导出原数据已有的顺序 ID、症状和证候要素，不含患者身份字段。

展示病例须在三个种子中得到相同的预测集合与错误类型。仅为让三个例子更易读，选例时排除同时含有以下成对描述的记录：食欲不振/食欲旺盛、口渴喜冷饮/口渴喜热饮、胸痛遇寒加重/胸痛遇热加重、苔厚腻/苔薄白。报告同时列出筛选前、后的候选数；之后按症状数与全数据中位数的距离选例，同距时选最小记录 ID。此展示筛选不改数据、指标或基线，也不评价这些共现的临床效度。
