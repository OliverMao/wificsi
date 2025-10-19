import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import TransformerClassifier


def make_windows(arr, window_size, stride):
    # arr: (T, F)
    T, F = arr.shape
    if T < window_size:
        # pad at end with zeros
        pad = np.zeros((window_size - T, F), dtype=arr.dtype)
        arr = np.vstack([arr, pad])
        T = window_size
    windows = []
    for s in range(0, T - window_size + 1, stride):
        windows.append(arr[s: s + window_size])
    return np.stack(windows, axis=0)  # (N, window, F)


def zscore_windows(windows):
    # windows: (N, T, F)
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-6
    return (windows - mean) / std


def infer(npy_path, model_path, window, stride, batch, device, num_classes, threshold):
    arr = np.load(npy_path)
    if arr.ndim == 1:
        arr = arr[:, None]
    windows = make_windows(arr, window, stride)
    windows = windows.astype(np.float32)
    windows = zscore_windows(windows)

    N, T, F = windows.shape
    ds = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(ds, batch_size=batch, shuffle=False)

    model = TransformerClassifier(input_dim=F, num_classes=num_classes)
    state = torch.load(model_path, map_location='cpu')
    # allow state dict or whole model saved
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        # possibly trained with DataParallel
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        state = new_state
    try:
        model.load_state_dict(state)
    except Exception:
        # maybe state is nested
        if 'state_dict' in state:
            sd = state['state_dict']
            new_state = {}
            for k, v in sd.items():
                nk = k.replace('module.', '')
                new_state[nk] = v
            model.load_state_dict(new_state)
        else:
            raise

    model.to(device)
    model.eval()

    all_probs = []
    all_preds = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = softmax(logits)
            preds = probs.argmax(dim=1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, C)
    all_preds = np.concatenate(all_preds, axis=0)  # (N,)

    # Aggregate
    from collections import Counter
    cnt = Counter(map(int, all_preds.tolist()))
    avg_prob = all_probs.mean(axis=0)

    print(f'Input: {npy_path}  windows: {N}  classes: {num_classes}')
    print('Per-window prediction counts:', dict(cnt))
    print('Average probs:', avg_prob.tolist())

    if num_classes == 2:
        # for binary, report positive prob
        pos_prob = float(avg_prob[1])
        final = 1 if pos_prob >= threshold else 0
        print(f'Final (avg prob >= {threshold} -> class): {final}  (pos_prob={pos_prob:.4f})')
    else:
        final = int(avg_prob.argmax())
        print(f'Final (argmax avg prob) class: {final}')

    # also print per-window predictions (optional)
    print('\nPer-window preds:')
    print(all_preds.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', default='checkpoints/train/X_0_label1.npy', help='Path to input .npy file')
    parser.add_argument('--model', default='checkpoints/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5, help='binary positive threshold on avg prob')
    args = parser.parse_args()

    device = torch.device(args.device)
    infer(args.npy, args.model, args.window, args.stride, args.batch, device, args.num_classes, args.threshold)


if __name__ == '__main__':
    main()
