#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pisces Model RMSE Evaluation & Seasonal Analysis
================================================
一键运行脚本：只需修改 csv_path 变量指向你的评估 CSV 文件，
即可自动生成全套统计报告 + 6 张高清图表。

CSV 格式要求：
    date, rmse_temp, rmse_salt, rmse_mean, bg_rmse_temp, bg_rmse_salt, bg_rmse_mean
    20250101, 0.1496, 0.0406, 0.1096, 0.2071, 0.0458, 0.1500
    ...

输出：
    - 控制台统计报告
    - rmse_timeseries_3panel.png   (时间序列三面板)
    - rmse_boxplot_3panel.png      (箱线图三面板)
    - eval_analysis_2025.png       (全年总览六宫格)
    - eval_analysis_detail.png     (深度诊断四宫格)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# ========================== 用户配置区 ==========================
csv_path = "finetune_eval_20250101_20251230.csv"   # <-- 只需改这里
output_dir = "./eval"                          # 图表输出目录
# ==============================================================

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date').reset_index(drop=True)

# 计算改进量
df['improve_temp'] = df['bg_rmse_temp'] - df['rmse_temp']
df['improve_salt'] = df['bg_rmse_salt'] - df['rmse_salt']
df['improve_mean'] = df['bg_rmse_mean'] - df['rmse_mean']
df['improve_temp_pct'] = df['improve_temp'] / df['bg_rmse_temp'] * 100
df['improve_salt_pct'] = df['improve_salt'] / df['bg_rmse_salt'] * 100
df['improve_mean_pct'] = df['improve_mean'] / df['bg_rmse_mean'] * 100

# 月度 / 季节
df['month'] = df['date'].dt.month
df['season'] = df['month'].map({12:1, 1:1, 2:1, 3:2, 4:2, 5:2,
                                 6:3, 7:3, 8:3, 9:4, 10:4, 11:4})

# ==============================================================
# 1. 控制台统计报告
# ==============================================================
print("=" * 70)
print(" Pisces Model Annual Evaluation Report")
print("=" * 70)
print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
print(f"Total samples: {len(df)} days\n")

print("[Model Prediction RMSE]")
print(df[['rmse_temp', 'rmse_salt', 'rmse_mean']].describe().round(4).to_string())
print("\n[Background RMSE]")
print(df[['bg_rmse_temp', 'bg_rmse_salt', 'bg_rmse_mean']].describe().round(4).to_string())
print("\n[Improvement (%)]")
print(df[['improve_temp_pct', 'improve_salt_pct', 'improve_mean_pct']].describe().round(2).to_string())

# 月度
monthly = df.groupby('month').agg({
    'rmse_temp': 'mean', 'rmse_salt': 'mean', 'rmse_mean': 'mean',
    'bg_rmse_temp': 'mean', 'bg_rmse_salt': 'mean', 'bg_rmse_mean': 'mean',
    'improve_temp_pct': 'mean', 'improve_salt_pct': 'mean', 'improve_mean_pct': 'mean'
}).round(4)
print("\n[Monthly Statistics]")
print(monthly.to_string())

# 季节
seasonal = df.groupby('season').agg({
    'rmse_temp': 'mean', 'rmse_salt': 'mean', 'rmse_mean': 'mean',
    'improve_temp_pct': 'mean', 'improve_salt_pct': 'mean', 'improve_mean_pct': 'mean'
}).round(4)
season_names = {1:'Winter(Dec-Feb)', 2:'Spring(Mar-May)', 3:'Summer(Jun-Aug)', 4:'Autumn(Sep-Nov)'}
seasonal.index = [season_names[i] for i in seasonal.index]
print("\n[Seasonal Statistics]")
print(seasonal.to_string())

# 极值
best_mean = df.loc[df['rmse_mean'].idxmin()]
worst_mean = df.loc[df['rmse_mean'].idxmax()]
print(f"\nBest RMSE day:  {best_mean['date'].strftime('%Y-%m-%d')} → mean={best_mean['rmse_mean']:.4f}")
print(f"Worst RMSE day: {worst_mean['date'].strftime('%Y-%m-%d')} → mean={worst_mean['rmse_mean']:.4f}")

best_imp = df.loc[df['improve_mean_pct'].idxmax()]
worst_imp = df.loc[df['improve_mean_pct'].idxmin()]
print(f"Best improvement:  {best_imp['date'].strftime('%Y-%m-%d')} → {best_imp['improve_mean_pct']:.2f}%")
print(f"Worst improvement: {worst_imp['date'].strftime('%Y-%m-%d')} → {worst_imp['improve_mean_pct']:.2f}%")

worse_days = df[df['rmse_mean'] >= df['bg_rmse_mean']]
print(f"\nDays worse than BG: {len(worse_days)} / {len(df)} ({len(worse_days)/len(df)*100:.1f}%)")

# ==============================================================
# 2. 图1: 时间序列三面板 (参考 XiHe 风格)
# ==============================================================
mean_temp_model = df['rmse_temp'].mean()
mean_temp_bg = df['bg_rmse_temp'].mean()
mean_salt_model = df['rmse_salt'].mean()
mean_salt_bg = df['bg_rmse_salt'].mean()
mean_ave_model = df['rmse_mean'].mean()
mean_ave_bg = df['bg_rmse_mean'].mean()

color_model = '#1f77b4'
color_bg = '#ff7f0e'

fig, axes = plt.subplots(3, 1, figsize=(14, 14))
fig.suptitle('Pisces Model Evaluation: RMSE Time Series', fontsize=16, fontweight='bold', y=0.98)

ax = axes[0]
ax.plot(df['date'], df['rmse_temp'], color=color_model, linewidth=0.8, alpha=0.9,
        label=f'Model={mean_temp_model:.4f}')
ax.plot(df['date'], df['bg_rmse_temp'], color=color_bg, linewidth=0.8, alpha=0.9,
        label=f'BG={mean_temp_bg:.4f}')
ax.set_title('RMSE of Temperature', fontsize=14)
ax.set_ylabel('RMSE (°C)', fontsize=12)
ax.legend(loc='upper center', ncol=2, fontsize=10, frameon=True)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(df['date'], df['rmse_salt'], color=color_model, linewidth=0.8, alpha=0.9,
        label=f'Model={mean_salt_model:.4f}')
ax.plot(df['date'], df['bg_rmse_salt'], color=color_bg, linewidth=0.8, alpha=0.9,
        label=f'BG={mean_salt_bg:.4f}')
ax.set_title('RMSE of Salinity', fontsize=14)
ax.set_ylabel('RMSE (psu)', fontsize=12)
ax.legend(loc='upper center', ncol=2, fontsize=10, frameon=True)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(df['date'], df['rmse_mean'], color=color_model, linewidth=0.8, alpha=0.9,
        label=f'Model={mean_ave_model:.4f}')
ax.plot(df['date'], df['bg_rmse_mean'], color=color_bg, linewidth=0.8, alpha=0.9,
        label=f'BG={mean_ave_bg:.4f}')
ax.set_title('RMSE of Average (Mean)', fontsize=14)
ax.set_ylabel('RMSE', fontsize=12)
ax.set_xlabel('Time (day)', fontsize=12)
ax.legend(loc='upper center', ncol=2, fontsize=10, frameon=True)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(output_dir, 'rmse_timeseries_3panel.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\n[保存] rmse_timeseries_3panel.png")

# ==============================================================
# 3. 图2: 箱线图三面板
# ==============================================================
data_temp = [df['rmse_temp'].values, df['bg_rmse_temp'].values]
data_salt = [df['rmse_salt'].values, df['bg_rmse_salt'].values]
data_ave = [df['rmse_mean'].values, df['bg_rmse_mean'].values]
labels = ['Model', 'BG']
colors = ['#1f77b4', '#ff7f0e']

fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.suptitle('Pisces Model Evaluation: RMSE Boxplot', fontsize=15, fontweight='bold', y=1.02)

for ax, data, title, unit in zip(axes, [data_temp, data_salt, data_ave],
                                 ['Temperature', 'Salinity', 'Average (Mean)'],
                                 ['RMSE (°C)', 'RMSE (psu)', 'RMSE']):
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(linewidth=2, color='#2c3e50'),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel(unit, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    for i, d in enumerate(data):
        offset = (np.max(d) - np.min(d)) * 0.03
        ax.text(i+1, np.mean(d)+offset, f'μ={np.mean(d):.4f}',
                ha='center', fontsize=10, fontweight='bold', color='#2c3e50')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rmse_boxplot_3panel.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[保存] rmse_boxplot_3panel.png")

# ==============================================================
# 4. 图3: 全年总览六宫格
# ==============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Pisces Model Annual Evaluation', fontsize=16, fontweight='bold', y=0.995)

# 温度时间序列
ax = axes[0, 0]
ax.plot(df['date'], df['bg_rmse_temp'], label='BG', color='#e74c3c', alpha=0.7, linewidth=1.2)
ax.plot(df['date'], df['rmse_temp'], label='Model', color='#2ecc71', alpha=0.9, linewidth=1.2)
ax.fill_between(df['date'], df['rmse_temp'], df['bg_rmse_temp'], alpha=0.2, color='#2ecc71', label='Improvement')
ax.set_title('Temperature RMSE Time Series', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (°C)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 盐度时间序列
ax = axes[0, 1]
ax.plot(df['date'], df['bg_rmse_salt'], label='BG', color='#e74c3c', alpha=0.7, linewidth=1.2)
ax.plot(df['date'], df['rmse_salt'], label='Model', color='#3498db', alpha=0.9, linewidth=1.2)
ax.fill_between(df['date'], df['rmse_salt'], df['bg_rmse_salt'], alpha=0.2, color='#3498db', label='Improvement')
ax.set_title('Salinity RMSE Time Series', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE (psu)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 平均时间序列
ax = axes[1, 0]
ax.plot(df['date'], df['bg_rmse_mean'], label='BG', color='#e74c3c', alpha=0.7, linewidth=1.2)
ax.plot(df['date'], df['rmse_mean'], label='Model', color='#9b59b6', alpha=0.9, linewidth=1.2)
ax.fill_between(df['date'], df['rmse_mean'], df['bg_rmse_mean'], alpha=0.2, color='#9b59b6', label='Improvement')
ax.set_title('Mean RMSE Time Series', fontsize=13, fontweight='bold')
ax.set_ylabel('RMSE')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# 改进百分比
ax = axes[1, 1]
ax.plot(df['date'], df['improve_temp_pct'], label='Temp', color='#2ecc71', alpha=0.8, linewidth=1)
ax.plot(df['date'], df['improve_salt_pct'], label='Salt', color='#3498db', alpha=0.8, linewidth=1)
ax.plot(df['date'], df['improve_mean_pct'], label='Mean', color='#9b59b6', alpha=0.9, linewidth=1.2)
ax.axhline(y=df['improve_mean_pct'].mean(), color='#9b59b6', linestyle='--', alpha=0.5,
           label=f'Avg: {df["improve_mean_pct"].mean():.1f}%')
ax.set_title('Improvement over BG (%)', fontsize=13, fontweight='bold')
ax.set_ylabel('Improvement (%)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 月度柱状图
ax = axes[2, 0]
x = np.arange(1, 13)
width = 0.35
bars1 = ax.bar(x - width/2, monthly['bg_rmse_mean'], width, label='BG', color='#e74c3c', alpha=0.7)
bars2 = ax.bar(x + width/2, monthly['rmse_mean'], width, label='Model', color='#9b59b6', alpha=0.8)
ax.set_title('Monthly Mean RMSE Comparison', fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Mean RMSE')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i in x:
    imp = monthly.loc[i, 'improve_mean_pct']
    ax.text(i, monthly.loc[i, 'rmse_mean'] + 0.003, f'{imp:.1f}%',
            ha='center', va='bottom', fontsize=8, color='#333')

# 散点图
ax = axes[2, 1]
ax.scatter(df['bg_rmse_mean'], df['rmse_mean'], alpha=0.5, s=30, color='#9b59b6', edgecolors='none')
lim_min = min(df['bg_rmse_mean'].min(), df['rmse_mean'].min()) - 0.01
lim_max = max(df['bg_rmse_mean'].max(), df['rmse_mean'].max()) + 0.01
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='1:1 line')
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_title('BG vs Model RMSE Scatter', fontsize=13, fontweight='bold')
ax.set_xlabel('BG RMSE')
ax.set_ylabel('Model RMSE')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(os.path.join(output_dir, 'eval_analysis_overview.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[保存] eval_analysis_overview.png")

# ==============================================================
# 5. 图4: 深度诊断四宫格
# ==============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Pisces Diagnostic Analysis', fontsize=15, fontweight='bold')

# 11月放大
nov = df[(df['date'] >= df['date'].min().replace(month=11, day=1)) &
         (df['date'] <= df['date'].min().replace(month=11, day=30))]
if len(nov) == 0:
    nov = df[df['month'] == 11]
ax = axes[0, 0]
ax.plot(nov['date'], nov['bg_rmse_mean'], 'o-', color='#e74c3c', alpha=0.7, label='BG', markersize=4)
ax.plot(nov['date'], nov['rmse_mean'], 'o-', color='#9b59b6', alpha=0.9, label='Model', markersize=4)
ax.fill_between(nov['date'], nov['rmse_mean'], nov['bg_rmse_mean'], alpha=0.2, color='#9b59b6')
ax.set_title('November Detail (mean RMSE)', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE')
ax.legend()
ax.grid(True, alpha=0.3)

# 温盐联合分布
ax = axes[0, 1]
scatter = ax.scatter(df['rmse_temp'], df['rmse_salt'], c=df['rmse_mean'], cmap='viridis', alpha=0.6, s=25)
plt.colorbar(scatter, ax=ax, label='mean RMSE')
ax.set_xlabel('Temperature RMSE')
ax.set_ylabel('Salinity RMSE')
ax.set_title('Temperature-Salinity RMSE Joint Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 背景场难度 vs 改进率
ax = axes[1, 0]
ax.scatter(df['bg_rmse_mean'], df['improve_mean_pct'], alpha=0.5, s=25, color='#2ecc71', edgecolors='none')
z = np.polyfit(df['bg_rmse_mean'], df['improve_mean_pct'], 1)
p = np.poly1d(z)
ax.plot(sorted(df['bg_rmse_mean']), p(sorted(df['bg_rmse_mean'])), "r--", alpha=0.6)
ax.set_xlabel('BG mean RMSE (difficulty)')
ax.set_ylabel('Improvement (%)')
ax.set_title('BG Difficulty vs Model Improvement', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# 月度箱线图
ax = axes[1, 1]
monthly_data = [df[df['month']==m]['improve_mean_pct'].values for m in range(1, 13)]
bp = ax.boxplot(monthly_data, tick_labels=range(1, 13), patch_artist=True,
                boxprops=dict(facecolor='#9b59b6', alpha=0.6),
                medianprops=dict(color='#e74c3c', linewidth=2))
ax.set_xlabel('Month')
ax.set_ylabel('Improvement (%)')
ax.set_title('Monthly Improvement Distribution', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(os.path.join(output_dir, 'eval_analysis_detail.png'), dpi=300, bbox_inches='tight')
plt.close()
print("[保存] eval_analysis_detail.png")

print("\n" + "=" * 70)
print(" Done! Charts saved to:", os.path.abspath(output_dir))
print("=" * 70)