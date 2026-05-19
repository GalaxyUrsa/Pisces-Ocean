"""快速测量数据加载耗时，定位瓶颈。"""
import time
from datetime import date, timedelta

from load_datasets import OceanDatasetLoader
from Data_Config import data_index


def measure(n_samples=10, n_runs=10):
    loader = OceanDatasetLoader()
    dates = [(date(2024, 1, 8) + timedelta(days=i)).strftime('%Y%m%d')
             for i in range(n_samples + 2)]

    # warmup（OS 文件 cache + folder index 缓存）
    loader.load_single_date(dates[0], data_index, isLog=False)
    loader.load_single_date(dates[1], data_index, isLog=False)

    per_run_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for d in dates[2:2 + n_samples]:
            loader.load_single_date(d, data_index, isLog=False)
        elapsed = time.perf_counter() - t0
        per_run_ms.append(elapsed / n_samples * 1000)

    avg = sum(per_run_ms) / len(per_run_ms)
    mn, mx = min(per_run_ms), max(per_run_ms)
    print(f"\n{n_runs} runs × {n_samples} samples:")
    for i, ms in enumerate(per_run_ms):
        print(f"  run {i + 1}: {ms:.0f} ms/sample")
    print(f"\navg: {avg:.0f} ms/sample (min {mn:.0f}, max {mx:.0f})")


if __name__ == "__main__":
    measure(n_samples=10, n_runs=10)

