# -*- coding: utf-8 -*-
"""news_analyze.py - AI 产业链新闻情绪分析

两档设计：
- DictSentimentAnalyzer：本地金融情感词典规则打分（默认，免费离线）
- LLMSentimentAnalyzer：预留 LLM API 接口（DeepSeek/通义等，填入 Key 后启用）

输出：
- 逐条情绪明细（带环节标签，即标注语料）
- 环节 × 日期 的情绪宽表（供特征工程与 LSTM 融合）
- 终端报告：分环节正负占比与 TOP 利好/利空
"""

import os
from abc import ABC, abstractmethod

import pandas as pd

from industry_config import DATA_FEATURE_DIR, DATA_NEWS_DIR, SEGMENTS
from sentiment_dict import DEGREE_WORDS, NEGATION_WORDS, NEGATIVE_WORDS, POSITIVE_WORDS

COMBINED_FILE = os.path.join(DATA_NEWS_DIR, '新闻数据_AI产业链.csv')
DETAIL_FILE = os.path.join(DATA_NEWS_DIR, '新闻情绪明细_标注语料.csv')
DAILY_WIDE_FILE = os.path.join(DATA_FEATURE_DIR, '环节情绪每日宽表.csv')

POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


class SentimentAnalyzer(ABC):
    """情绪分析器抽象接口：analyze(text) -> (score: float, label: str)"""

    @abstractmethod
    def analyze(self, text):
        """对一段文本打分。score > 0 偏利好，< 0 偏利空；label ∈ {利好, 利空, 中性}"""
        raise NotImplementedError


class DictSentimentAnalyzer(SentimentAnalyzer):
    """基于金融情感词典的规则分析器：正词加分、负词减分，
    程度副词加权、否定词反转。词典见 sentiment_dict.py，可自行扩充。"""

    def analyze(self, text):
        score = self._score(text)
        if score >= POSITIVE_THRESHOLD:
            return score, '利好'
        if score <= NEGATIVE_THRESHOLD:
            return score, '利空'
        return score, '中性'

    def _score(self, text):
        if not text:
            return 0.0
        total = 0.0
        for word in POSITIVE_WORDS:
            total += self._weighted_count(text, word, sign=1)
        for word in NEGATIVE_WORDS:
            total += self._weighted_count(text, word, sign=-1)
        return round(total, 3)

    @staticmethod
    def _weighted_count(text, word, sign):
        count = text.count(word)
        if count == 0:
            return 0.0
        weight = 1.0
        prefix = text[max(0, text.find(word) - 3):text.find(word)]
        for deg, mult in DEGREE_WORDS.items():
            if deg in prefix:
                weight *= mult
                break
        if any(neg in prefix for neg in NEGATION_WORDS):
            weight = -weight
        return sign * weight * count


class LLMSentimentAnalyzer(SentimentAnalyzer):
    """预留实现：接入 DeepSeek / 通义千问等 LLM API 做新闻情绪分析。

    使用步骤：
        1. 在 __init__ 传入 api_key 与 endpoint；
        2. 实现 _call_api(text) 调用 LLM 并解析情绪得分；
        3. 在 main() 中把 analyzer 替换为本类实例。
    """

    def __init__(self, api_key='', endpoint=''):
        self.api_key = api_key
        self.endpoint = endpoint

    def analyze(self, text):
        raise NotImplementedError(
            'LLMSentimentAnalyzer 为预留接口：请填入 API Key 并实现 API 调用后使用。'
        )


def analyze_news(df, analyzer):
    """逐条新闻打分：以 标题+摘要 作为分析文本"""
    df = df.copy()
    df['摘要'] = df['摘要'].fillna('')
    results = [analyzer.analyze(f"{row['标题']}。{row['摘要']}")
               for _, row in df.iterrows()]
    df['情绪得分'] = [r[0] for r in results]
    df['情绪标签'] = [r[1] for r in results]
    return df


def aggregate_daily_wide(df):
    """按 环节 × 日期 聚合：日均情绪得分与新闻数量，透视成宽表"""
    d = df.copy()
    d['日期'] = pd.to_datetime(d['日期']).dt.normalize()
    grouped = (d.groupby(['日期', '环节'])
                 .agg(score=('情绪得分', 'mean'), count=('情绪得分', 'count'))
                 .reset_index())
    score_wide = grouped.pivot(index='日期', columns='环节', values='score')
    count_wide = grouped.pivot(index='日期', columns='环节', values='count')
    score_wide.columns = [f'{c}_情绪' for c in score_wide.columns]
    count_wide.columns = [f'{c}_新闻数' for c in count_wide.columns]
    wide = pd.concat([score_wide, count_wide], axis=1).reset_index()
    return wide.sort_values('日期')


def print_summary(df):
    """终端报告：分环节正负占比 + TOP2 利好/利空"""
    total = len(df)
    counts = df['情绪标签'].value_counts()
    print(f"\n===== 全产业情绪汇总（共 {total} 条） =====")
    for label in ('利好', '中性', '利空'):
        n = counts.get(label, 0)
        print(f"  {label}: {n} 条 ({n / total * 100:.1f}%)")

    for seg_key, seg in SEGMENTS.items():
        seg_df = df[df['环节'] == seg_key]
        if seg_df.empty:
            continue
        mean_score = seg_df['情绪得分'].mean()
        print(f"\n--- {seg['name']}（{len(seg_df)} 条，均分 {mean_score:+.2f}） ---")
        for label, ascending in (('利好', False), ('利空', True)):
            top = seg_df[seg_df['情绪标签'] == label].sort_values(
                '情绪得分', ascending=ascending).head(2)
            for _, row in top.iterrows():
                print(f"  [{label} {row['情绪得分']:+.2f}] {row['日期']:%m-%d} {row['标题'][:40]}")


def main():
    try:
        df = pd.read_csv(COMBINED_FILE, parse_dates=['日期'])
    except FileNotFoundError:
        print(f"未找到 {COMBINED_FILE}，请先运行: python news_collect.py")
        return

    if df.empty:
        print("新闻数据为空，无法进行情绪分析。")
        return

    analyzer = DictSentimentAnalyzer()
    df = analyze_news(df, analyzer)

    os.makedirs(DATA_NEWS_DIR, exist_ok=True)
    os.makedirs(DATA_FEATURE_DIR, exist_ok=True)

    df.to_csv(DETAIL_FILE, index=False, encoding='utf_8_sig')
    print(f"逐条情绪明细（标注语料）已保存到: {DETAIL_FILE}")

    print_summary(df)

    wide = aggregate_daily_wide(df)
    wide.to_csv(DAILY_WIDE_FILE, index=False, encoding='utf_8_sig')
    print(f"\n环节情绪每日宽表已保存到: {DAILY_WIDE_FILE}（供特征工程与 LSTM 融合）")


if __name__ == '__main__':
    main()
