"""逐变量的 epoch 指标、终端输出及训练曲线。"""

import csv
from pathlib import Path

import numpy as np
import torch


def loss_ratio(loss, baseline):
    return loss / baseline if baseline > 0 else float('nan')


class VariableLossMeter:
    """聚合未乘变量权重的损失，不保留计算图。"""

    def __init__(self):
        self.names = []
        self.totals = None
        self.count = 0

    def update(self, losses, baselines, weight=1):
        self.names = list(losses)
        values = torch.stack([
            torch.stack([losses[name].detach(), baselines[name].detach()])
            for name in self.names
        ]).to(torch.float64) * weight
        self.totals = values if self.totals is None else self.totals + values
        self.count += weight

    def compute(self, scale=1):
        values = (self.totals / self.count * scale).cpu().tolist()
        return {
            name: {'loss': loss, 'bg_loss': baseline,
                   'ratio': loss_ratio(loss, baseline)}
            for name, (loss, baseline) in zip(self.names, values)
        }


def record_variable_metrics(history, train_metrics, val_metrics):
    """记录一轮指标；训练和验证分别与同一数据划分的背景基线比较。"""
    record = {}
    print(f"{'Variable':<20} {'Train loss':>12} {'Val loss':>12} "
          f"{'Train BG':>12} {'Val BG':>12} {'Train ratio':>12} {'Val ratio':>12}")
    for name, train in train_metrics.items():
        val = val_metrics[name]
        record[name] = {
            'train_loss': train['loss'], 'val_loss': val['loss'],
            'train_bg_loss': train['bg_loss'], 'bg_loss': val['bg_loss'],
            'train_ratio': train['ratio'], 'val_ratio': val['ratio'],
        }
        print(f"{name:<20} {train['loss']:12.6f} {val['loss']:12.6f} "
              f"{train['bg_loss']:12.6f} {val['bg_loss']:12.6f} "
              f"{train['ratio']:12.4f} {val['ratio']:12.4f}")
    history.append(record)


def plot_loss_curves(train_losses, val_losses, save_path='loss_curve.png',
                     bg_losses=None, variable_history=None):
    """总损失与各变量各占一行，左侧 loss，右侧 model/background ratio。

    AR 指标先按预测日权重求和；旧断点缺失的逐变量历史显示为空白。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)
    bg_losses = bg_losses if bg_losses is not None else []
    history = variable_history if variable_history is not None else []
    names = list(dict.fromkeys(name for record in history if record for name in record))
    fields = ('train_loss', 'val_loss', 'train_bg_loss', 'bg_loss', 'train_ratio', 'val_ratio')

    def at(values, index):
        return values[index] if index < len(values) else float('nan')

    def variable_at(index, name, field):
        record = at(history, index)
        return record.get(name, {}).get(field, float('nan')) if isinstance(record, dict) else float('nan')

    fig, axes = plt.subplots(len(names) + 1, 2, figsize=(14, 3.2 * (len(names) + 1)),
                             squeeze=False, sharex=True)
    total_axis, ratio_axis = axes[0]
    total_axis.plot(epochs, train_losses, label='Train')
    total_axis.plot(epochs, val_losses, label='Validation')
    if len(bg_losses):
        total_axis.plot(epochs, [at(bg_losses, i) for i in range(len(train_losses))],
                        'g--', label='Validation background')
        ratio_axis.plot(epochs, [loss_ratio(val, at(bg_losses, i))
                                for i, val in enumerate(val_losses)], label='Validation ratio')
    if len(val_losses):
        best = int(np.nanargmin(val_losses))
        total_axis.plot(best + 1, val_losses[best], 'r*', markersize=10,
                        label=f'Best: {val_losses[best]:.6f} (epoch {best + 1})')
    total_axis.set(title='Total (weighted)', ylabel='Loss')
    ratio_axis.set(title='Total validation / background', ylabel='Ratio')

    for row, name in enumerate(names, 1):
        loss_axis, ratio_axis = axes[row]
        for field, label, style in [('train_loss', 'Train', '-'), ('val_loss', 'Validation', '-'),
                                     ('train_bg_loss', 'Train background', ':'),
                                     ('bg_loss', 'Validation background', '--')]:
            loss_axis.plot(epochs, [variable_at(i, name, field) for i in range(len(train_losses))],
                           style, label=label)
        for field, label in [('train_ratio', 'Train ratio'), ('val_ratio', 'Validation ratio')]:
            ratio_axis.plot(epochs, [variable_at(i, name, field) for i in range(len(train_losses))],
                            label=label)
        loss_axis.set(title=f'{name} (unweighted)', ylabel='Normalized loss')
        ratio_axis.set(title=f'{name}: model / background', ylabel='Ratio')
    for loss_axis, ratio_axis in axes:
        ratio_axis.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Background = 1')
        for axis in (loss_axis, ratio_axis):
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel('Epoch')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    csv_path = Path(save_path).with_suffix('.csv')
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        header = ['epoch', 'train_loss', 'val_loss']
        if len(bg_losses):
            header += ['bg_loss', 'val_ratio']
        header += [f'{name}_{field}' for name in names for field in fields]
        writer.writerow(header)
        for i, (train, val) in enumerate(zip(train_losses, val_losses)):
            values = [train, val]
            if len(bg_losses):
                baseline = at(bg_losses, i)
                values += [baseline, loss_ratio(val, baseline)]
            values += [variable_at(i, name, field) for name in names for field in fields]
            writer.writerow([i + 1] + [f'{value:.8f}' if np.isfinite(value) else '' for value in values])
    print(f"Loss curve saved to: {save_path}")
    print(f"Loss data saved to: {csv_path}")
