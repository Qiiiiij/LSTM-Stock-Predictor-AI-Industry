# -*- coding: utf-8 -*-
"""stock_picker.py - AI 产业链选股打分

对环节股票池逐只抓取公司新闻（双源）并做情绪打分，
结合近 20 日行情动量，输出环节内与全产业综合排名。

综合分 = 0.6 × 情绪标准分 + 0.4 × 动量标准分（z-score 标准化）
"""

import os

import pandas as pd

from industry_config import (DATA_FEATURE_DIR, DATA_MARKET_DIR,
                             DATA_NEWS_DIR, DEFAULT_DAYS, SEGMENTS)
from news_collect import fetch_keywords
from news_analyze import DictSentimentAnalyzer

MOMENTUM_DAYS = 20
WEIGHT_SENTIMENT = 0.6
WEIGHT_MOMENTUM = 0.4
STOCK_NEWS_DIR = os.path.join(DATA_NEWS_DIR, 'stocks')
PICKER_FILE = os.path.join(DATA_FEATURE_DIR, '选股榜单.csv')


def stock_sentiment(stock, analyzer, days=DEFAULT_DAYS):
    """抓取单只股票新闻并返回 (平均情绪得分, 新闻数量)"""
    df = fetch_keywords([stock['name']], days)
    if df.empty:
        print(f"  {stock['name']}: 无新闻")
        return 0.0, 0
    os.makedirs(STOCK_NEWS_DIR, exist_ok=True)
    df.to_csv(os.path.join(STOCK_NEWS_DIR, f"{stock['code']}_{stock['name']}.csv"),
              index=False, encoding='utf_8_sig')
    scores = [analyzer.analyze(f"{r['标题']}。{r['摘要']}")[0]
              for _, r in df.assign(摘要=df['摘要'].fillna('')).iterrows()]
    return round(sum(scores) / len(scores), 3), len(scores)


def stock_momentum(stock):
    """近 MOMENTUM_DAYS 个交易日的涨跌幅（%）"""
    path = os.path.join(DATA_MARKET_DIR,
                        f"股票_{stock['code']}_{stock['name']}.csv")
    try:
        df = pd.read_csv(path)
        if len(df) < MOMENTUM_DAYS + 1:
            return 0.0
        close = df['收盘价']
        return round((close.iloc[-1] / close.iloc[-MOMENTUM_DAYS - 1] - 1) * 100, 2)
    except FileNotFoundError:
        print(f"  [警告] 缺少 {stock['name']} 行情，动量计 0")
        return 0.0


def main():
    analyzer = DictSentimentAnalyzer()
    rows = []
    for seg_key, seg in SEGMENTS.items():
        print(f"\n===== 打分环节：{seg['name']} =====")
        for stock in seg['stocks']:
            sent, news_n = stock_sentiment(stock, analyzer)
            mom = stock_momentum(stock)
            rows.append({'环节': seg['name'], '代码': stock['code'],
                         '名称': stock['name'], '情绪均分': sent,
                         '新闻数量': news_n, f'{MOMENTUM_DAYS}日动量(%)': mom})
            print(f"  {stock['name']}: 情绪 {sent:+.2f}（{news_n} 条）, 动量 {mom:+.2f}%")

    df = pd.DataFrame(rows)

    # z-score 标准化后加权
    df['情绪标准分'] = (df['情绪均分'] - df['情绪均分'].mean()) / df['情绪均分'].std()
    df['动量标准分'] = (df[f'{MOMENTUM_DAYS}日动量(%)'] - df[f'{MOMENTUM_DAYS}日动量(%)'].mean()) \
                       / df[f'{MOMENTUM_DAYS}日动量(%)'].std()
    df['综合分'] = (WEIGHT_SENTIMENT * df['情绪标准分']
                    + WEIGHT_MOMENTUM * df['动量标准分']).round(3)
    df = df.sort_values('综合分', ascending=False).reset_index(drop=True)
    df.insert(0, '排名', df.index + 1)

    os.makedirs(DATA_FEATURE_DIR, exist_ok=True)
    df.to_csv(PICKER_FILE, index=False, encoding='utf_8_sig')
    print(f"\n选股榜单已保存到: {PICKER_FILE}")

    print("\n===== 各环节 TOP 2 =====")
    for seg_name in df['环节'].unique():
        top = df[df['环节'] == seg_name].head(2)
        for _, r in top.iterrows():
            print(f"  [{seg_name}] {r['名称']}({r['代码']}) 综合分 {r['综合分']:+.2f} "
                  f"(情绪 {r['情绪均分']:+.2f} / 动量 {r[f'{MOMENTUM_DAYS}日动量(%)']:+.2f}%)")


if __name__ == '__main__':
    main()
