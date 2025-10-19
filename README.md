# WiFi CSI Transformer Training

这是一个最小可复现项目，用于对已处理的 CSI 幅度时间序列（`.npy` 文件）使用 Transformer 进行二分类（是否有人体活动）。

结构说明
- `data/` - 包含 `nexmon_csi_data_*.npy` 文件（每个文件为 (T, F) 的幅度矩阵）
- `dataset.py` - 数据加载和窗口化
- `models.py` - TransformerClassifier 实现
- `train.py` - 训练/评估/快速 smoke test
- `requirements.txt` - 需要安装的依赖

默认标签假设
- 默认 label_map 会将前 60% 的文件标为 0（无活动），其余标为 1（有活动）。
- 你可以在创建 `CSIDataset(..., label_map=your_map)` 时传入具体文件到标签的映射字典，例如：

```
label_map = {
  'data/nexmon_csi_data_0.npy': 0,
  'data/nexmon_csi_data_1.npy': 0,
  'data/nexmon_csi_data_2.npy': 0,
  'data/nexmon_csi_data_3.npy': 1,
  'data/nexmon_csi_data_4.npy': 1,
}
```

快速开始（Windows PowerShell）

# 在虚拟环境中安装依赖
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

# 仅运行 smoke test（前向推理）
python train.py

# 运行完整训练（示例）
python train.py --epochs 20 --batch 16 --window 128 --stride 64

## 标注工具（keyboard-only）

项目中包含了一个纯终端标注工具 `label_tool.py`，不需要鼠标，适合你键盘输入精确区间。

基本用法：

```powershell
python label_tool.py
```

常用命令举例：
- `add 100:200 walking` 或 `add 100-200 walking` — 添加区间 [100,200) 并标注为 walking
- `list` — 列出当前文件的标注
- `save` — 保存当前文件标注到 `labels/<basename>.json`
- `next` / `prev` — 切换文件（若有未保存更改会提示保存）
- `export` — 导出所有 labels 到 `labels/all_labels.json`

标注文件格式（JSON）为：

```json
[
  {"start": 0, "end": 830, "label": "walking"},
  {"start": 831, "end": 1000, "label": "nowalking"}
]
```

把所有文件标注完成后，可以使用这些注释来为滑动窗口生成训练标签（见下一节）。

## 在训练中使用注释（annotations）

`train.py` 支持把 `labels/*.json` 中的区间映射到每个窗口的标签（默认 `label_from_annotations=False`）。使用方法：

```powershell
# 使用 annotations（任意重叠即判为活动）
python train.py --use-annotations --overlap-threshold 0.0 --epochs 10 --window 128 --stride 64

# 只有当窗口与注释重叠比例 >= 0.3 时才判为活动
python train.py --use-annotations --overlap-threshold 0.3 --epochs 10 --window 128 --stride 64
```

训练开始时脚本会打印每个标签的样本数量（便于确认标签分布）。如果某个文件没有对应的 `labels/<basename>.json`，会使用按文件统一的 `label_map` 回退标签。

后续建议
- 准备更清晰的标签（如手动标注或利用视频同步）以提高监督学习效果
- 尝试数据增强（噪声注入、频域扰动）和更深的模型或预训练策略
- 使用滑动窗口的同时保留时间戳以做序列标注任务（而不是窗口分类）

