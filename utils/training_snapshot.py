"""保存训练使用的配置和模型源码。"""

import inspect
import shutil
from datetime import datetime
from pathlib import Path

import Data_Config


def save_training_sources(log_dir, model_class, resume=False):
    """新训练保存到日志根目录；续训另存快照，避免覆盖原始源码。"""
    sources = [Path(inspect.getsourcefile(Data_Config)),
               Path(inspect.getsourcefile(model_class))]
    destination = Path(log_dir)
    if resume or any((destination / source.name).exists() for source in sources):
        destination = destination / 'source_snapshots' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, destination / source.name)
    print(f"Training source snapshot saved to: {destination}")
    return destination
