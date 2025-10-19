import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import numpy as np
from models import TransformerClassifier


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total




def main():
    parser = argparse.ArgumentParser(description='基于 X.npy/Y.npy 训练模型')
    parser.add_argument('--x', default='exported/X.npy', help='X.npy 文件路径')
    parser.add_argument('--y', default='exported/Y.npy', help='Y.npy 文件路径')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=1, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--val-split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--save', default='checkpoints', help='保存模型目录')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    # 加载数据
    X = np.load(args.x)
    Y = np.load(args.y)
    X_t = torch.from_numpy(X).float()
    Y_t = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_t, Y_t)
    # 划分训练/验证集
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False)
    # 将验证集拆分为单独的窗口文件，方便逐个推断
    val_indices = val_set.indices
    val_dir = os.path.join(args.save, 'val')
    os.makedirs(val_dir, exist_ok=True)
    for i, idx in enumerate(val_indices):
        # 获取真实标签并添加到文件名
        lbl = int(Y[idx])
        np.save(os.path.join(val_dir, f'X_{i}_label{lbl}.npy'), X[idx])
        np.save(os.path.join(val_dir, f'Y_{i}_label{lbl}.npy'), Y[idx])
    print(f"验证集窗口已保存到目录: {val_dir}/X_*.npy 和 Y_*.npy 共 {len(val_indices)} 个文件")

    # 将训练集拆分为单独的窗口文件，方便逐个推断
    train_indices = train_set.indices
    train_dir = os.path.join(args.save, 'train')
    os.makedirs(train_dir, exist_ok=True)
    for i, idx in enumerate(train_indices):
        # 获取真实标签并添加到文件名
        lbl = int(Y[idx])
        np.save(os.path.join(train_dir, f'X_{i}_label{lbl}.npy'), X[idx])
        np.save(os.path.join(train_dir, f'Y_{i}_label{lbl}.npy'), Y[idx])
    print(f"训练集窗口已保存到目录: {train_dir}/X_*.npy 和 Y_*.npy 共 {len(train_indices)} 个文件")

    device = torch.device(args.device)

    # 打印训练/验证集分布
    from collections import Counter
    train_labels = [y.item() for _, y in train_set]
    val_labels = [y.item() for _, y in val_set]
    print('Train distribution:', Counter(train_labels))
    print('Val distribution:', Counter(val_labels))

    # 构建模型
    input_dim = X_t.size(2)
    model = TransformerClassifier(input_dim=input_dim).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.4f} acc: {train_acc:.3f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save, 'best_model.pt'))


if __name__ == '__main__':
    main()
