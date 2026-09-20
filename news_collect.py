# -*- coding: utf-8 -*-
"""news_collect.py - AI 产业链多源新闻采集（东方财富 + 新浪财经）

按 industry_config.SEGMENTS 的环节配置逐个关键词检索双源新闻，
合并去重后过滤时间窗，按环节保存语料 CSV + 一份全量汇总 CSV。
"""

import json
import os
import re

import pandas as pd
import requests

from industry_config import DATA_NEWS_DIR, DEFAULT_DAYS, SEGMENTS

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
}
PAGE_SIZE = 50
COMBINED_FILE = os.path.join(DATA_NEWS_DIR, '新闻数据_AI产业链.csv')


def _clean_html(text):
    """去除 <em> 等 HTML 标签与多余空白"""
    if not text:
        return ''
    return re.sub(r'<[^>]+>', '', str(text)).strip()


def fetch_eastmoney(keyword, days=DEFAULT_DAYS):
    """东方财富资讯搜索接口（JSONP）"""
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner = {
        "uid": "",
        "keyword": keyword,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "time",
                "pageIndex": 1,
                "pageSize": PAGE_SIZE,
                "preTag": "",
                "postTag": "",
            }
        },
    }
    params = {"cb": "jsonpCallback", "param": json.dumps(inner, ensure_ascii=False)}
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        m = re.search(r'\((.*)\)', resp.text.strip(), re.S)
        if not m:
            print(f"  [东方财富] '{keyword}': 返回格式异常")
            return pd.DataFrame()
        data = json.loads(m.group(1))
        items = (data.get('result') or {}).get('cmsArticleWebOld') or []
        records = []
        for it in items:
            media = _clean_html(it.get('mediaName'))
            records.append({
                '日期': pd.to_datetime(it.get('date'), errors='coerce'),
                '来源': f"东方财富·{media}" if media else '东方财富',
                '标题': _clean_html(it.get('title')),
                '摘要': _clean_html(it.get('content'))[:200],
                '链接': it.get('url', ''),
            })
        df = pd.DataFrame(records)
        print(f"  [东方财富] '{keyword}': {len(df)} 条")
        return df
    except Exception as e:
        print(f"  [东方财富] '{keyword}' 抓取失败: {e}")
        return pd.DataFrame()


def fetch_sina(keyword, days=DEFAULT_DAYS):
    """新浪搜索新闻接口（JSON）"""
    url = "https://search.sina.com.cn/api/news"
    headers = {**HEADERS, 'Referer': 'https://search.sina.com.cn/'}
    params = {'q': keyword, 'page': 1, 'size': PAGE_SIZE}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get('code') != 0:
            print(f"  [新浪财经] '{keyword}': 接口返回异常 {data.get('message')}")
            return pd.DataFrame()
        items = (data.get('data') or {}).get('list') or []
        records = []
        for it in items:
            ts = it.get('ctime')
            dt = pd.to_datetime(ts, unit='s') if ts else \
                pd.to_datetime(it.get('dataTime'), errors='coerce')
            media = _clean_html(it.get('media_show') or it.get('media'))
            records.append({
                '日期': dt,
                '来源': f"新浪财经·{media}" if media else '新浪财经',
                '标题': _clean_html(it.get('title')),
                '摘要': _clean_html(it.get('intro'))[:200],
                '链接': it.get('url', ''),
            })
        df = pd.DataFrame(records)
        print(f"  [新浪财经] '{keyword}': {len(df)} 条")
        return df
    except Exception as e:
        print(f"  [新浪财经] '{keyword}' 抓取失败: {e}")
        return pd.DataFrame()


def fetch_keywords(keywords, days=DEFAULT_DAYS):
    """对一组关键词抓取双源新闻并合并去重（供环节采集与选股复用）"""
    parts = []
    for keyword in keywords:
        for fetcher in (fetch_eastmoney, fetch_sina):
            part = fetcher(keyword, days)
            if not part.empty:
                part['关键词'] = keyword
                parts.append(part)
    if not parts:
        return pd.DataFrame()
    return deduplicate(pd.concat(parts, ignore_index=True), days)


def deduplicate(df, days=DEFAULT_DAYS):
    """清洗：日期解析、时间窗过滤、标题+日期去重"""
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    df = df.dropna(subset=['日期', '标题'])
    df = df[df['标题'].str.len() > 0]
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    df = df[df['日期'] >= cutoff]
    df = df.drop_duplicates(subset=['标题', '日期'])
    return df.sort_values('日期', ascending=False).reset_index(drop=True)


def collect_all_segments(days=DEFAULT_DAYS):
    """逐环节采集新闻，按环节保存语料 CSV + 全量汇总 CSV"""
    os.makedirs(DATA_NEWS_DIR, exist_ok=True)
    all_parts = []
    for seg_key, seg in SEGMENTS.items():
        print(f"\n===== 采集环节：{seg['name']} =====")
        df = fetch_keywords(seg['keywords'], days)
        if df.empty:
            print(f"  {seg['name']} 未采集到新闻")
            continue
        df['环节'] = seg_key
        df['环节名称'] = seg['name']
        seg_file = os.path.join(DATA_NEWS_DIR, f"新闻语料_{seg_key}_{seg['name']}.csv")
        df.to_csv(seg_file, index=False, encoding='utf_8_sig')
        print(f"  -> {len(df)} 条，已保存 {seg_file}")
        all_parts.append(df)

    if not all_parts:
        print("所有环节均未采集到新闻。")
        return pd.DataFrame()

    combined = pd.concat(all_parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=['标题', '日期', '环节'])
    combined = combined.sort_values('日期', ascending=False).reset_index(drop=True)
    combined.to_csv(COMBINED_FILE, index=False, encoding='utf_8_sig')

    print(f"\n采集完成：共 {len(combined)} 条，汇总已保存到 {COMBINED_FILE}")
    print(combined.groupby(['环节名称', '来源']).size().groupby(level=0).sum().to_string())
    return combined


if __name__ == '__main__':
    collect_all_segments(days=DEFAULT_DAYS)
