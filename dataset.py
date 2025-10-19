import os
import glob
import numpy as np
from torch.utils.data import Dataset
import torch


class CSIDataset(Dataset):
    """
    加载目录下的多个 .npy 文件，拼接为时间序列样本。

    假设：每个 .npy 文件是 shape (n_frames, n_subcarriers) 的幅度时间序列。

    标签策略（默认）：
      - 默认将文件按文件名顺序分为无活动(0)和有活动(1)：
        例如 data/nexmon_csi_data_0.npy ... _4.npy
        默认映射: files 0-2 -> label 0, files 3-4 -> label 1
      - 你可以传入 label_map 参数以覆盖。

    主要参数：
      - dirpath: 存放 .npy 文件的目录
      - window_size: 每个样本的时间步长
      - stride: 滑动步长
      - normalize: 是否对每个窗口进行 z-score 归一化
    """

    def __init__(self, dirpath="data", window_size=100, stride=50, normalize=True, label_map=None,
                 label_from_annotations=True, overlap_threshold=0.0, labels_dir="labels", ann_label_map=None):
        self.dirpath = dirpath
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.label_from_annotations = label_from_annotations
        self.overlap_threshold = overlap_threshold
        self.labels_dir = labels_dir
        # ann_label_map: dict mapping annotation label string -> int class (e.g. {'walking':1, 'nowalking':0})
        self.ann_label_map = ann_label_map

        files = sorted(glob.glob(os.path.join(dirpath, "*.npy")))
        if len(files) == 0:
            raise FileNotFoundError(f"No .npy files found in {dirpath}")
        self.files = files

        # 默认 label_map
        if label_map is None:
            # simple default: first 60% files -> 0, rest ->1
            n = len(files)
            cutoff = max(1, int(n * 0.6))
            label_map = {f: 0 if i < cutoff else 1 for i, f in enumerate(files)}
        self.label_map = label_map

        # 预加载所有数据到内存（小数据集）
        self.raw = []  # list of arrays
        self.labels = []
        self.annotations = []  # list of annotations per file (list of dicts)
        for f in files:
            arr = np.load(f)
            if arr.ndim == 1:
                arr = arr[:, None]
            self.raw.append(arr.astype(np.float32))
            self.labels.append(self.label_map.get(f, 0))
            # 尝试加载 annotations
            base = os.path.splitext(os.path.basename(f))[0]
            ann_path = os.path.join(self.labels_dir, base + ".json")
            if os.path.exists(ann_path):
                try:
                    import json
                    with open(ann_path, 'r', encoding='utf-8') as fo:
                        ann = json.load(fo)
                        # ann 应为 [{"start":int,"end":int,"label":...}, ...]
                        self.annotations.append(ann)
                except Exception:
                    self.annotations.append([])
            else:
                self.annotations.append([])

        # 构建窗口索引: (file_idx, start_frame)
        self.index = []
        for i, arr in enumerate(self.raw):
            n = arr.shape[0]
            if n < self.window_size:
                continue
            for s in range(0, n - self.window_size + 1, self.stride):
                self.index.append((i, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        arr = self.raw[file_idx]
        win = arr[start: start + self.window_size]
        # win: (T, F)
        if self.normalize:
            mean = win.mean(axis=0, keepdims=True)
            std = win.std(axis=0, keepdims=True) + 1e-6
            win = (win - mean) / std
        # 转为 torch tensor: (T, F) -> (T, F)
        x = torch.from_numpy(win)
        # 如果未启用注释标注，直接使用默认标签
        if not self.label_from_annotations:
            y = torch.tensor(self.labels[file_idx], dtype=torch.long)
            return x, y

        # 根据注释 JSON 中的 'label' 判断活动窗口，超过重叠阈值即停止扫描
        w_s, w_e = start, start + self.window_size
        is_active = False
        for ann in self.annotations[file_idx]:
            try:
                a_s, a_e = int(ann.get('start', 0)), int(ann.get('end', 0))
            except Exception:
                continue
            # 计算重叠比例
            ov = max(0, min(w_e, a_e) - max(w_s, a_s))
            if ov / float(self.window_size) > self.overlap_threshold:
                # 直接读取注释中的 label 字段 (0 或 1)
                try:
                    is_active = (int(ann.get('label', 0)) == 1)
                except Exception:
                    is_active = False
                break
        y = torch.tensor(1 if is_active else 0, dtype=torch.long)

        return x, y


if __name__ == "__main__":
    import argparse
    # 导出分割后窗口数据集
    parser = argparse.ArgumentParser(description="导出 CSI 窗口数据集")
    parser.add_argument('--data', default='data', help='原始 .npy 文件目录')
    parser.add_argument('--labels-dir', default='labels', help='注释 JSON 文件目录')
    parser.add_argument('--window', type=int, default=20, help='窗口长度')
    parser.add_argument('--stride', type=int, default=10, help='滑动步长')
    parser.add_argument('--normalize', action='store_true', help='是否执行 z-score 归一化')
    parser.add_argument('--use-annotations', action='store_true', help='使用注释计算标签')
    parser.add_argument('--overlap-threshold', type=float, default=0.0, help='注释重叠阈值')
    parser.add_argument('--output', default='exported', help='导出目录')
    args = parser.parse_args()
    # 构建数据集
    ds = CSIDataset(dirpath=args.data,
                    window_size=args.window,
                    stride=args.stride,
                    normalize=args.normalize,
                    label_from_annotations=args.use_annotations,
                    overlap_threshold=args.overlap_threshold,
                    labels_dir=args.labels_dir)
    # 收集所有窗口和标签
    xs, ys = [], []
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.item())
    xs = np.stack(xs)
    ys = np.array(ys)
    # 保存到磁盘
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, 'X.npy'), xs)
    np.save(os.path.join(args.output, 'Y.npy'), ys)
    print(f"已导出 {len(ds)} 个窗口到目录: {args.output}")
