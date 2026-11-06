"""
预处理脚本：把所有日期的 NetCDF 数据裁剪好存成 .npz
输入：F:\datasets（原始 NetCDF）
输出：F:\datasets_full_npz\{YYYYMMDD}.npz

用法：python preprocess_to_npz.py
已存在的日期自动跳过，可中断后续跑。
"""

import numpy as np
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from utils.load_datasets import OceanDatasetLoader
from Data_Config import (
    INPUT_VARS, RAW_DATASET_PATH,
    CROP_ROW_START, CROP_ROW_END,
    CROP_COL_START, CROP_COL_END,
)

# ── 配置 ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(r"F:\datasets_20layer_npz")
N_WORKERS  = 8        # 并行线程数，按 CPU/IO 情况调整
START_DATE = date(2015, 1, 1)
END_DATE   = date(2025, 12, 31)
# ─────────────────────────────────────────────────────────────────────────────


def date_range(start, end):
    n = (end - start).days + 1
    return [(start + timedelta(days=i)).strftime('%Y%m%d') for i in range(n)]


_loader = None

def _init_worker(base_path):
    global _loader
    _loader = OceanDatasetLoader(base_path)


def process_one(date_str):
    out_path = OUTPUT_DIR / f"{date_str}.npz"
    if out_path.exists():
        return date_str, "skip"

    try:
        raw = _loader.load_single_date(date_str, INPUT_VARS, isLog=False)
    except Exception as e:
        return date_str, f"error: {e}"

    if not raw:
        return date_str, "error: empty result"

    # 检查所有必需的 key 都存在
    required_keys = [e[2] for e in INPUT_VARS]
    missing = [k for k in required_keys if k not in raw]
    if missing:
        return date_str, f"error: missing keys {missing}"

    arrays = {}
    for key, arr in raw.items():
        arr = arr.astype(np.float32)
        if arr.ndim == 2:
            arrays[key] = arr[CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]
        else:
            arrays[key] = arr[:, CROP_ROW_START:CROP_ROW_END, CROP_COL_START:CROP_COL_END]

    # np.savez(out_path, **arrays)
    np.savez_compressed(out_path, **arrays)
    return date_str, "ok"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = date_range(START_DATE, END_DATE)

    existing = sum(1 for d in dates if (OUTPUT_DIR / f"{d}.npz").exists())
    print(f"Total dates : {len(dates)}")
    print(f"Already done: {existing}")
    print(f"To process  : {len(dates) - existing}")
    print(f"Output dir  : {OUTPUT_DIR}")

    errors = []
    with ProcessPoolExecutor(max_workers=N_WORKERS,
                             initializer=_init_worker,
                             initargs=(RAW_DATASET_PATH,)) as pool:
        futures = {pool.submit(process_one, d): d for d in dates}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            d, status = fut.result()
            if status.startswith("error"):
                errors.append((d, status))

    print(f"\nDone. Errors: {len(errors)}")
    for d, e in errors:
        print(f"  {d}: {e}")


if __name__ == "__main__":
    main()
