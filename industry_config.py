# -*- coding: utf-8 -*-
"""industry_config.py - AI 产业链全局配置（环节划分 / 关键词库 / 代表股票池 / ETF）"""

# ====== 全局参数 ======
DEFAULT_DAYS = 30      # 新闻采集时间窗（天）
HISTORY_DAYS = 250     # 行情回看天数（约一年交易日）

# ====== AI 主题 ETF（market：0=深市，1=沪市） ======
ETFS = [
    {'code': '159819', 'name': '人工智能ETF', 'market': 0},
    {'code': '588200', 'name': '科创芯片ETF', 'market': 1},
]

# ====== 产业链环节配置 ======
# 每个环节：新闻检索关键词 + 代表股票池（用于环节指数与选股打分）
SEGMENTS = {
    'upstream': {
        'name': '上游·算力硬件',
        'keywords': ['半导体', '芯片', 'GPU', '光模块', '先进封装', 'HBM', '存储芯片', '晶圆'],
        'stocks': [
            {'code': '002371', 'name': '北方华创', 'market': 0},
            {'code': '688981', 'name': '中芯国际', 'market': 1},
            {'code': '300308', 'name': '中际旭创', 'market': 0},
            {'code': '300502', 'name': '新易盛', 'market': 0},
            {'code': '603986', 'name': '兆易创新', 'market': 1},
            {'code': '000977', 'name': '浪潮信息', 'market': 0},
        ],
    },
    'midstream': {
        'name': '中游·模型平台',
        'keywords': ['大模型', 'AIGC', '多模态', '云计算', 'DeepSeek', '智能体', 'Agent'],
        'stocks': [
            {'code': '002230', 'name': '科大讯飞', 'market': 0},
            {'code': '601360', 'name': '三六零', 'market': 1},
            {'code': '688111', 'name': '金山办公', 'market': 1},
            {'code': '300229', 'name': '拓尔思', 'market': 0},
            {'code': '688327', 'name': '云从科技', 'market': 1},
        ],
    },
    'downstream': {
        'name': '下游·行业应用',
        'keywords': ['智能驾驶', '人形机器人', '具身智能', 'AI医疗', 'AI手机', 'AIPC', '机器人'],
        'stocks': [
            {'code': '002920', 'name': '德赛西威', 'market': 0},
            {'code': '300496', 'name': '中科创达', 'market': 0},
            {'code': '002415', 'name': '海康威视', 'market': 0},
            {'code': '300024', 'name': '机器人', 'market': 0},
            {'code': '300253', 'name': '卫宁健康', 'market': 0},
        ],
    },
}

# ====== 路径 ======
DATA_NEWS_DIR = 'data/news'        # 语料数据集
DATA_MARKET_DIR = 'data/market'    # 行情数据
DATA_FEATURE_DIR = 'data/features' # 特征与结果
OUTPUT_DIR = 'output'              # 图表
