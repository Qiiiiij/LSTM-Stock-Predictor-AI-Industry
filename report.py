# -*- coding: utf-8 -*-
"""report.py - AI 产业链舆情可视化报告

生成：
- 环节情绪走势图（上/中/下游日均情绪得分曲线）
- 情绪热力图（日期 × 环节）
- 环节等权指数走势对比图
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from industry_config import (DATA_FEATURE_DIR, DATA_MARKET_DIR,
                             OUTPUT_DIR, SEGMENTS)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SENTIMENT_FILE = os.path.join(DATA_FEATURE_DIR, '环节情绪每日宽表.csv')
INDEX_FILE = os.path.join(DATA_MARKET_DIR, '环节等权指数.csv')
TREND_FIG = os.path.join(OUTPUT_DIR, '环节情绪走势.png')
HEATMAP_FIG = os.path.join(OUTPUT_DIR, '环节情绪热力图.png')
INDEX_FIG = os.path.join(OUTPUT_DIR, '环节指数走势.png')

SEG_COLORS = {'upstream': '#d62728', 'midstream': '#1f77b4', 'downstream': '#2ca02c'}


def plot_sentiment_trend(wide):
    fig, ax = plt.subplots(figsize=(14, 6))
    for seg_key, seg in SEGMENTS.items():
        col = f'{seg_key}_情绪'
        if col in wide.columns:
            ax.plot(wide['日期'], wide[col], marker='o', markersize=4,
                    linewidth=2, color=SEG_COLORS[seg_key], label=seg['name'])
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title('AI 产业链各环节每日新闻情绪走势', fontsize=15, pad=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('日均情绪得分', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(TREND_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {TREND_FIG}")
    plt.close(fig)


def plot_heatmap(wide):
    seg_cols = [f'{k}_情绪' for k in SEGMENTS if f'{k}_情绪' in wide.columns]
    data = wide.set_index('日期')[seg_cols].T
    data.index = [SEGMENTS[c.replace('_情绪', '')]['name'] for c in seg_cols]

    fig, ax = plt.subplots(figsize=(14, 4))
    values = data.values.astype(float)
    vmax = max(float(np.nanmax(np.abs(values))), 1.0)
    cmap = plt.get_cmap('RdYlGn_r').copy()  # A 股惯例：红=利好，绿=利空
    cmap.set_bad('white')                   # 无新闻的日期显示为白色
    im = ax.imshow(np.ma.masked_invalid(values), aspect='auto',
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(data.index)), data.index, fontsize=11)
    dates = [d.strftime('%m-%d') for d in data.columns]
    step = max(1, len(dates) // 15)
    ax.set_xticks(range(0, len(dates), step), dates[::step],
                  rotation=45, ha='right', fontsize=9)
    ax.set_title('AI 产业链新闻情绪热力图（红=利好，绿=利空，白=无新闻）', fontsize=15, pad=12)
    fig.colorbar(im, ax=ax, label='情绪得分')
    fig.tight_layout()
    plt.savefig(HEATMAP_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {HEATMAP_FIG}")
    plt.close(fig)


def plot_segment_index():
    if not os.path.exists(INDEX_FILE):
        print(f"未找到 {INDEX_FILE}，跳过环节指数图")
        return
    idx = pd.read_csv(INDEX_FILE, parse_dates=['日期'])
    fig, ax = plt.subplots(figsize=(14, 6))
    for seg_key, seg in SEGMENTS.items():
        if seg_key in idx.columns:
            ax.plot(idx['日期'], idx[seg_key], linewidth=2,
                    color=SEG_COLORS[seg_key], label=seg['name'])
    ax.set_title('AI 产业链环节等权指数走势（基点=100）', fontsize=15, pad=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('指数点位', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(INDEX_FIG, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"已保存: {INDEX_FIG}")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(SENTIMENT_FILE):
        print(f"未找到 {SENTIMENT_FILE}，请先运行: python news_analyze.py")
        return
    wide = pd.read_csv(SENTIMENT_FILE, parse_dates=['日期'])
    plot_sentiment_trend(wide)
    plot_heatmap(wide)
    plot_segment_index()
    print("\n报告生成完成，图表位于 output/ 目录。")


if __name__ == '__main__':
    main()
