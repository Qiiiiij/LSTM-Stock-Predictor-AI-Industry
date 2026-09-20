# -*- coding: utf-8 -*-
"""market_collect.py - 行情数据采集（东方财富 K 线接口）

采集 AI 主题 ETF 与各环节代表股票池的日线行情，
并按环节合成等权指数（基点=100），供特征工程与选股使用。
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from industry_config import (DATA_MARKET_DIR, ETFS, HISTORY_DAYS, SEGMENTS)

KLINE_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
INDEX_FILE = os.path.join(DATA_MARKET_DIR, '环节等权指数.csv')


def fetch_kline(code, market, days=HISTORY_DAYS):
    """抓取单只证券的日线 K 线（股票/ETF 通用）"""
    end = datetime.now()
    beg = end - timedelta(days=int(days * 1.6))  # 预留非交易日
    params = {
        'secid': f"{market}.{code}",
        'fields1': 'f1,f2,f3,f4,f5,f6',
        'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
        'klt': '101',  # 日K线
        'fqt': '1',    # 前复权
        'beg': beg.strftime('%Y%m%d'),
        'end': end.strftime('%Y%m%d'),
    }
    try:
        resp = requests.get(KLINE_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        klines = (data.get('data') or {}).get('klines') or []
        records = []
        for line in klines:
            it = line.split(',')
            records.append({
                '日期': pd.to_datetime(it[0]),
                '开盘价': float(it[1]),
                '收盘价': float(it[2]),
                '最高价': float(it[3]),
                '最低价': float(it[4]),
                '成交量': float(it[5]),
                '成交金额': float(it[6]),
            })
        df = pd.DataFrame(records).tail(days).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  {code} 行情抓取失败: {e}")
        return pd.DataFrame()


def save_kline(df, category, code, name):
    path = os.path.join(DATA_MARKET_DIR, f"{category}_{code}_{name}.csv")
    df.to_csv(path, index=False, encoding='utf_8_sig')
    return path


def build_segment_index(days=HISTORY_DAYS):
    """按环节合成等权指数：环节内个股收盘价归一化（首日=100）后取均值"""
    index_df = None
    for seg_key, seg in SEGMENTS.items():
        series = []
        for stock in seg['stocks']:
            df = fetch_kline(stock['code'], stock['market'], days)
            if df.empty:
                print(f"  [警告] {stock['name']} 无行情，跳过")
                continue
            path = save_kline(df, '股票', stock['code'], stock['name'])
            norm = df['收盘价'] / df['收盘价'].iloc[0] * 100
            series.append(pd.Series(norm.values, index=df['日期'], name=stock['name']))
            print(f"  {seg['name']} {stock['name']}({stock['code']}): {len(df)} 条 -> {path}")
        if not series:
            continue
        panel = pd.concat(series, axis=1).sort_index().ffill()
        seg_index = panel.mean(axis=1).round(2).rename(seg_key)
        index_df = seg_index.to_frame() if index_df is None else index_df.join(seg_index)

    if index_df is not None:
        index_df.index.name = '日期'
        index_df = index_df.reset_index()
        index_df.to_csv(INDEX_FILE, index=False, encoding='utf_8_sig')
        print(f"\n环节等权指数已保存到: {INDEX_FILE}")
    return index_df


def collect_etfs(days=HISTORY_DAYS):
    for etf in ETFS:
        df = fetch_kline(etf['code'], etf['market'], days)
        if df.empty:
            print(f"  [警告] {etf['name']} 无行情")
            continue
        path = save_kline(df, 'ETF', etf['code'], etf['name'])
        print(f"  {etf['name']}({etf['code']}): {len(df)} 条 -> {path}")


def main():
    os.makedirs(DATA_MARKET_DIR, exist_ok=True)
    print("===== 采集 AI 主题 ETF 行情 =====")
    collect_etfs()
    print("\n===== 采集环节股票池行情并合成环节指数 =====")
    build_segment_index()


if __name__ == '__main__':
    main()
