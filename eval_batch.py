"""
Batch evaluation over a date range.

Usage:
    python batch_eval.py --start 20240101 --end 20241231 --model_path logs/xxx/best_model.pth
    python batch_eval.py --start 20240101 --end 20241231 --model_path logs/xxx/best_model.pth --out results.csv
"""

import argparse
import csv
import math
import os
from datetime import date, timedelta

import numpy as np
import torch
from tqdm import tqdm

from Data_Config import (
    SURFACE_VARS, _SURFACE_INDEX, data_index,
    RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END, CROP_COL_START, CROP_COL_END,
    NAN_FILL_VALUE,
)
from load_datasets import OceanDatasetLoader
from models.simple_convnext_net import ConvNeXtUNet as mymodel

IN_CHANNELS = len(SURFACE_VARS) + 40


def load_model(model_path: str, device: torch.device):
    model = mymodel(in_channels=IN_CHANNELS, out_channels=40).to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    norm_stats = ckpt.get('norm_stats', None)
    return model, norm_stats


def normalize(arr, var_name, norm_stats):
    if norm_stats is None:
        return arr
    mean = norm_stats[var_name]['mean']
    std  = norm_stats[var_name]['std']
    return (arr - mean) / std


def prepare_input(raw_data, norm_stats) -> torch.Tensor:
    channels = []
    for v in SURFACE_VARS:
        name = _SURFACE_INDEX[v][2]
        arr = raw_data[name]
        if norm_stats is not None:
            arr = normalize(arr, name, norm_stats)
        channels.append(np.expand_dims(arr, axis=0))
    bg_t = raw_data['bg_t_3d']
    bg_s = raw_data['bg_s_3d']
    if norm_stats is not None:
        bg_t = normalize(bg_t, 'bg_t_3d', norm_stats)
        bg_s = normalize(bg_s, 'bg_s_3d', norm_stats)
    channels.extend([bg_t, bg_s])
    inp = np.concatenate(channels, axis=0).astype(np.float32)
    inp = inp[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
    inp = np.nan_to_num(inp, nan=NAN_FILL_VALUE)
    return torch.from_numpy(inp)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    mask = ~np.isnan(target) & ~np.isnan(pred)
    if mask.sum() == 0:
        return float('nan')
    diff = pred[mask] - target[mask]
    return float(math.sqrt(np.mean(diff ** 2)))


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model, norm_stats = load_model(args.model_path, device)
    loader = OceanDatasetLoader(RAW_DATASET_PATH)

    start = date(int(args.start[:4]), int(args.start[4:6]), int(args.start[6:]))
    end   = date(int(args.end[:4]),   int(args.end[4:6]),   int(args.end[6:]))
    dates = [(start + timedelta(days=i)).strftime('%Y%m%d')
             for i in range((end - start).days + 1)]

    out_path = args.out or f"eval_{args.start}_{args.end}.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    rows = []
    skipped = []

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['date',
                         'rmse_temp', 'rmse_salt', 'rmse_mean',
                         'bg_rmse_temp', 'bg_rmse_salt', 'bg_rmse_mean'])

        for d in tqdm(dates, desc='Evaluating'):
            raw = loader.load_single_date(d, data_index, isLog=False)

            # 检查数据是否完整
            if not all(k in raw for k in ('bg_t_3d', 'bg_s_3d', 'label_t_3d', 'label_s_3d')):
                skipped.append(d)
                continue

            inp = prepare_input(raw, norm_stats).unsqueeze(0).to(device)

            bg = np.concatenate([raw['bg_t_3d'], raw['bg_s_3d']], axis=0).astype(np.float32)
            bg = bg[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

            target = np.concatenate([raw['label_t_3d'], raw['label_s_3d']], axis=0).astype(np.float32)
            target = target[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

            with torch.no_grad():
                residual = model(inp).squeeze(0).cpu().numpy()

            pred = bg + residual

            rt  = rmse(pred[:20], target[:20])
            rs  = rmse(pred[20:], target[20:])
            rm  = rmse(pred,      target)
            bgt = rmse(bg[:20],   target[:20])
            bgs = rmse(bg[20:],   target[20:])
            bgm = rmse(bg,        target)

            writer.writerow([d,
                             f'{rt:.6f}',  f'{rs:.6f}',  f'{rm:.6f}',
                             f'{bgt:.6f}', f'{bgs:.6f}', f'{bgm:.6f}'])
            f.flush()
            rows.append((rt, rs, rm, bgt, bgs, bgm))

    # 汇总
    valid = [r for r in rows if not any(math.isnan(x) for x in r)]
    n = len(valid)

    print(f"\n{'='*50}")
    print(f"Date range : {args.start} – {args.end}")
    print(f"Total days : {len(dates)}  |  evaluated: {n}  |  skipped: {len(skipped)}")
    if n > 0:
        def rms_of(idx): return math.sqrt(sum(r[idx]**2 for r in valid) / n)
        print(f"{'':20s} {'Pred':>12} {'BG':>12}")
        print(f"{'Overall RMSE temp':20s} {rms_of(0):>12.6f} {rms_of(3):>12.6f}")
        print(f"{'Overall RMSE salt':20s} {rms_of(1):>12.6f} {rms_of(4):>12.6f}")
        print(f"{'Overall RMSE mean':20s} {rms_of(2):>12.6f} {rms_of(5):>12.6f}")
    if skipped:
        print(f"Skipped dates: {skipped}")
    print(f"CSV saved  : {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # python batch_eval.py --start 20240101 --end 20241231 --model_path logs/xxx/best_model.pth --out results.csv
    parser.add_argument('--start',      default='20250101',                          help='Start date YYYYMMDD')
    parser.add_argument('--end',        default='20251230',                          help='End date YYYYMMDD')
    parser.add_argument('--model_path', default='./logs/20260520_190103_finetune/best_model.pth', help='Path to best_model.pth')
    parser.add_argument('--out',        default=None,  help='Output CSV path (default: eval_START_END.csv)')
    run(parser.parse_args())
