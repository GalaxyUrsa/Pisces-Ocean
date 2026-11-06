import numpy as np

PATH       = r"F:\datasets_full_npz_test\20210101.npz"
KEY_FILTER = None  # 只显示包含该字符串的 key，如 "bg_t"；None 显示全部

npz  = np.load(PATH)
keys = sorted(npz.files)

if KEY_FILTER:
    keys = [k for k in keys if KEY_FILTER in k]

print(f"\nFile : {PATH}")
print(f"Keys : {len(keys)}\n")
print(f"{'variable':<20} {'shape':<24} {'dtype':<8} {'min':>10} {'max':>10} {'mean':>10} {'nan%':>8}")
print("-" * 96)

for key in keys:
    arr     = npz[key].astype(np.float32)
    valid   = arr[~np.isnan(arr)]
    nan_pct = 100.0 * np.isnan(arr).sum() / arr.size
    if valid.size > 0:
        print(f"{key:<20} {str(arr.shape):<24} {str(arr.dtype):<8} "
              f"{valid.min():>10.4f} {valid.max():>10.4f} {valid.mean():>10.4f} {nan_pct:>7.2f}%")
    else:
        print(f"{key:<20} {str(arr.shape):<24} {str(arr.dtype):<8} {'all NaN':>10}")

