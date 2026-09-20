# -*- coding: utf-8 -*-
"""feature_builder.py - 特征工程与产业链传导分析

将环节情绪宽表与 ETF 行情按交易日对齐，生成 LSTM 输入特征宽表；
并计算各环节情绪得分与 ETF 收益的滞后相关（传导分析）。

说明：新闻情绪仅覆盖近 30 天，更早的交易日情绪特征以 0（中性）填充。
"""

import os
import glob

import pandas as pd

from industry_config import DATA_FEATURE_DIR, DATA_MARKET_DIR, ETFS, SEGMENTS

SENTIMENT_FILE = os.path.join(DATA_FEATURE_DIR, '环节情绪每日宽表.csv')
FEATURE_FILE = os.path.join(DATA_FEATURE_DIR, 'LSTM特征宽表.csv')
TRANSMISSION_FILE = os.path.join(DATA_FEATURE_DIR, '传导相关性分析.csv')
MAX_LAG = 5


def load_etf(etf):
    path = os.path.join(DATA_MARKET_DIR, f"ETF_{etf['code']}_{etf['name']}.csv")
    df = pd.read_csv(path, parse_dates=['日期'])
    df = df[['日期', '开盘价', '收盘价']]
    df.columns = ['日期', f"{etf['code']}_开盘", f"{etf['code']}_收盘"]
    return df


def build_feature_table():
    """以主 ETF（配置中第一只）的交易日为基准，左连接情绪特征"""
    sentiment = pd.read_csv(SENTIMENT_FILE, parse_dates=['日期'])

    base_etf = ETFS[0]
    base = load_etf(base_etf)
    feature = base.merge(sentiment, on='日期', how='left')

    # 无新闻的交易日：情绪=0（中性）、新闻数=0
    fill_cols = [c for c in feature.columns if c.endswith('_情绪') or c.endswith('_新闻数')]
    feature[fill_cols] = feature[fill_cols].fillna(0)

    # 附加其他 ETF 收盘价作为参考特征
    for etf in ETFS[1:]:
        extra = load_etf(etf)[['日期', f"{etf['code']}_收盘"]]
        feature = feature.merge(extra, on='日期', how='left').ffill()

    feature = feature.sort_values('日期').reset_index(drop=True)
    feature.to_csv(FEATURE_FILE, index=False, encoding='utf_8_sig')
    print(f"特征宽表已保存到: {FEATURE_FILE}（{len(feature)} 个交易日 × {len(feature.columns)} 列）")
    return feature


def transmission_analysis(feature):
    """传导分析：环节情绪得分 与 ETF 未来 N 日收益率 的相关系数"""
    base_etf = ETFS[0]
    close_col = f"{base_etf['code']}_收盘"

    # 只在有新闻覆盖的窗口内分析（情绪非全 0 的时段）
    sentiment_cols = [f'{k}_情绪' for k in SEGMENTS]
    window = feature[feature[sentiment_cols].abs().sum(axis=1) > 0].copy()
    if len(window) < MAX_LAG + 3:
        print("新闻覆盖窗口过短，传导分析结果仅供参考。")

    rows = []
    for seg_key, seg in SEGMENTS.items():
        sent = window[f'{seg_key}_情绪']
        returns = window[close_col].pct_change()
        for lag in range(0, MAX_LAG + 1):
            future_ret = returns.shift(-lag)
            corr = sent.corr(future_ret)
            rows.append({'环节': seg['name'], '滞后交易日': lag,
                         '相关系数': round(corr, 3) if pd.notna(corr) else None})

    result = pd.DataFrame(rows)
    result.to_csv(TRANSMISSION_FILE, index=False, encoding='utf_8_sig')
    print(f"传导相关性已保存到: {TRANSMISSION_FILE}")

    print(f"\n===== 产业链传导分析（{base_etf['name']}，n={len(window)}） =====")
    pivot = result.pivot(index='环节', columns='滞后交易日', values='相关系数')
    print(pivot.to_string())
    print("（正相关=情绪领先于价格；滞后 0 为当日同步关系，样本量小仅供参考）")
    return result


def main():
    if not os.path.exists(SENTIMENT_FILE):
        print(f"未找到 {SENTIMENT_FILE}，请先运行: python news_analyze.py")
        return
    etf_glob = glob.glob(os.path.join(DATA_MARKET_DIR, 'ETF_*.csv'))
    if not etf_glob:
        print("未找到 ETF 行情数据，请先运行: python market_collect.py")
        return

    feature = build_feature_table()
    transmission_analysis(feature)


if __name__ == '__main__':
    main()
