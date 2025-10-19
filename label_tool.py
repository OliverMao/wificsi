#!/usr/bin/env python3
"""
终端标注工具（基于键盘输入区间）

用法:
  python label_tool.py

命令集（在提示符处输入命令）:
  help                显示帮助
  show                打印当前文件的基本信息（帧数、子载波）
  add START:END L     添加区间标签（包含 START，排除 END），标签为整数或字符串
  list                列出当前文件的所有标注区间
  del IDX             删除指定索引的区间（从 0 开始计数）
  save                保存当前文件的标注到 labels/<filename>.json
  next                切换到下一个文件
  prev                切换到上一个文件
  goto N              跳转到第 N 个文件（从 0 开始）
  export              导出所有 labels 到 labels/all_labels.json
  quit                退出（会提示保存未保存的更改）

标注数据格式:
  labels/<basename>.json -> [{"start": int, "end": int, "label": str}, ...]

说明: 所有输入均通过键盘完成，不需要鼠标。
"""

import os
import glob
import json
import numpy as np

DATA_DIR = "data"
LABEL_DIR = "labels"


def ensure_dirs():
    os.makedirs(LABEL_DIR, exist_ok=True)


def list_files():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
    return files


def load_labels_for_file(filepath):
    base = os.path.splitext(os.path.basename(filepath))[0]
    label_path = os.path.join(LABEL_DIR, base + ".json")
    if os.path.exists(label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f), label_path
    else:
        return [], label_path


def save_labels(label_path, labels):
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def print_help():
    with open(__file__, 'r', encoding='utf-8') as f:
        head = f.read().split('\n')[:40]
    print('\n'.join(head))


def show_file_info(arr):
    print(f"shape: {arr.shape}  (frames, subcarriers)")
    print(f"frames: {arr.shape[0]}")


def parse_range(s):
    try:
        parts = s.split(':')
        if len(parts) != 2:
            return None
        a = int(parts[0])
        b = int(parts[1])
        return a, b
    except Exception:
        return None


def normalize_labels(labels):
    # 合并或排序，确保无重叠（简单按开始排序）
    labels = sorted(labels, key=lambda x: x['start'])
    return labels


def repl():
    ensure_dirs()
    files = list_files()
    if len(files) == 0:
        print("No .npy files in data/ - 请把你的文件放到 data/ 文件夹下")
        return
    idx = 0
    modified = False
    current_idx = None
    arr = None
    labels = []
    label_path = None

    while True:
        # 仅在文件索引变化时加载数据 / labels，避免覆盖未保存的更改
        if current_idx != idx:
            filepath = files[idx]
            arr = np.load(filepath)
            labels, label_path = load_labels_for_file(filepath)
            labels = normalize_labels(labels)
            current_idx = idx
            modified = False
            print(f"\nFile [{idx}/{len(files)-1}]: {os.path.basename(filepath)}")
            show_file_info(arr)
            print(f"Loaded {len(labels)} labels. (saved: {os.path.exists(label_path)})")

        cmd = input('label> ').strip()
        if cmd == '':
            continue
        parts = cmd.split()
        c = parts[0].lower()
        if c == 'help':
            print_help()
        elif c == 'show':
            show_file_info(arr)
        elif c == 'add':
            if len(parts) < 3:
                print('用法: add START:END LABEL  或 add START-END LABEL')
                continue
            rng = parse_range(parts[1])
            if rng is None:
                print('区间格式错误，使用 START:END 或 START-END，整数')
                continue
            start, end = rng
            lab = ' '.join(parts[2:])
            if start < 0 or end > arr.shape[0] or start >= end:
                print('区间越界或无效')
                continue
            labels.append({'start': int(start), 'end': int(end), 'label': lab})
            labels = normalize_labels(labels)
            modified = True
            print(f'Added: {start}:{end} -> {lab}')
        elif c == 'list':
            if len(labels) == 0:
                print('No labels')
            else:
                for i, L in enumerate(labels):
                    print(i, f"{L['start']}:{L['end']} -> {L['label']}")
        elif c == 'del':
            if len(parts) < 2:
                print('用法: del IDX')
                continue
            try:
                irem = int(parts[1])
                if irem < 0 or irem >= len(labels):
                    print('索引越界')
                    continue
                removed = labels.pop(irem)
                modified = True
                print('Removed', removed)
            except Exception as e:
                print('索引解析错误', e)
        elif c == 'save':
            save_labels(label_path, labels)
            modified = False
            print('Saved to', label_path)
        elif c == 'next':
            if modified:
                yn = input('有未保存的更改，保存? (y/n) ')
                if yn.lower().startswith('y'):
                    save_labels(label_path, labels)
                    modified = False
            if idx < len(files) - 1:
                idx += 1
            else:
                print('已是最后一个文件')
        elif c == 'prev':
            if modified:
                yn = input('有未保存的更改，保存? (y/n) ')
                if yn.lower().startswith('y'):
                    save_labels(label_path, labels)
                    modified = False
            if idx > 0:
                idx -= 1
            else:
                print('已是第一个文件')
        elif c == 'goto':
            if len(parts) < 2:
                print('用法: goto N')
                continue
            try:
                n = int(parts[1])
                if n < 0 or n >= len(files):
                    print('索引越界')
                    continue
                if modified:
                    yn = input('有未保存的更改，保存? (y/n) ')
                    if yn.lower().startswith('y'):
                        save_labels(label_path, labels)
                        modified = False
                idx = n
            except Exception as e:
                print('解析错误', e)
        elif c == 'export':
            # 导出所有 label 文件到 single json
            all_labels = {}
            for f in files:
                labs, _ = load_labels_for_file(f)
                all_labels[os.path.basename(f)] = labs
            out = os.path.join(LABEL_DIR, 'all_labels.json')
            with open(out, 'w', encoding='utf-8') as fo:
                json.dump(all_labels, fo, ensure_ascii=False, indent=2)
            print('Exported to', out)
        elif c == 'quit' or c == 'exit':
            if modified:
                yn = input('有未保存的更改，保存? (y/n) ')
                if yn.lower().startswith('y'):
                    save_labels(label_path, labels)
                    modified = False
            print('Bye')
            break
        else:
            print('Unknown command, 输入 help 查看帮助')


if __name__ == '__main__':
    repl()
