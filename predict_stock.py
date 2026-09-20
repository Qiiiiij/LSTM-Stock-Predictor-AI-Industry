# -*- coding: utf-8 -*-
"""predict_stock.py - 多特征 LSTM 预测股票池中任意个股

用法: python predict_stock.py <股票代码>   例: python predict_stock.py 300308
默认: 300308 中际旭创（上游·光模块）

输入特征：个股开盘/收盘价 + 上中下游环节情绪得分
预测目标：未来 30 个交易日收盘价
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from industry_config import (DATA_FEATURE_DIR, DATA_MARKET_DIR,
                             OUTPUT_DIR, SEGMENTS)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(42)
np.random.seed(42)

SENTIMENT_FILE = os.path.join(DATA_FEATURE_DIR, '环节情绪每日宽表.csv')
SEQ_LENGTH = 15
NUM_EPOCHS = 300
FUTURE_DAYS = 30


def find_stock(code):
    for seg_key, seg in SEGMENTS.items():
        for stock in seg['stocks']:
            if stock['code'] == code:
                return stock, seg['name']
    return None, None


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i + seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else '300308'
    stock, seg_name = find_stock(code)
    if stock is None:
        pool = [s['code'] for seg in SEGMENTS.values() for s in seg['stocks']]
        print(f"代码 {code} 不在股票池中，可选: {pool}")
        return

    path = os.path.join(DATA_MARKET_DIR, f"股票_{code}_{stock['name']}.csv")
    if not os.path.exists(path):
        print(f"未找到 {path}，请先运行: python market_collect.py")
        return

    df = pd.read_csv(path, parse_dates=['日期'])[['日期', '开盘价', '收盘价']]
    sentiment = pd.read_csv(SENTIMENT_FILE, parse_dates=['日期'])
    sentiment_cols = [f'{k}_情绪' for k in SEGMENTS]
    df = df.merge(sentiment[['日期'] + sentiment_cols], on='日期', how='left')
    df[sentiment_cols] = df[sentiment_cols].fillna(0)

    feature_cols = ['开盘价', '收盘价'] + sentiment_cols
    print(f"预测标的: {stock['name']}（{code}，{seg_name}）")
    print(f"输入特征: {feature_cols}")

    X_raw = df[feature_cols].values
    y_raw = df['收盘价'].values.reshape(-1, 1)
    dates = df['日期']

    # 先切分再归一化，防止数据泄漏
    train_size = int(len(X_raw) * 0.8)
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_train = scaler_x.fit_transform(X_raw[:train_size])
    X_test = scaler_x.transform(X_raw[train_size:])
    y_train = scaler_y.fit_transform(y_raw[:train_size])
    y_test = scaler_y.transform(y_raw[train_size:])

    Xtr, ytr = create_sequences(X_train, y_train, SEQ_LENGTH)
    Xte, yte = create_sequences(X_test, y_test, SEQ_LENGTH)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.FloatTensor(ytr)),
                              batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(Xte), torch.FloatTensor(yte)),
                             batch_size=32, shuffle=False)

    model = LSTMModel(input_size=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("开始训练...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss / len(train_loader):.6f}')

    torch.save(model.state_dict(), f'stock_{code}_model.pt')

    # 测试集评估
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(model(bx.to(device)).cpu().numpy())
            actuals.extend(by.numpy())
    preds = scaler_y.inverse_transform(np.array(preds))
    actuals = scaler_y.inverse_transform(np.array(actuals))
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    print(f"测试集 RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    # 未来 30 个交易日滚动预测（情绪特征以最近一期值填充）
    X_full = np.vstack([X_train, X_test])
    input_seq = X_full[-SEQ_LENGTH:].copy()
    last_row = X_full[-1].copy()

    future_scaled = []
    with torch.no_grad():
        for _ in range(FUTURE_DAYS):
            inp = torch.FloatTensor(input_seq.reshape(1, SEQ_LENGTH, -1)).to(device)
            pred_close = model(inp).cpu().numpy()[0, 0]
            future_scaled.append(pred_close)
            next_row = last_row.copy()
            next_row[0] = pred_close   # 开盘价
            next_row[1] = pred_close   # 收盘价
            input_seq = np.vstack([input_seq, next_row])[1:]

    future_close = scaler_y.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1),
                                 periods=FUTURE_DAYS, freq='B')

    future_df = pd.DataFrame({'日期': future_dates,
                              '预测收盘价': np.round(future_close, 2)})
    pred_file = os.path.join(DATA_FEATURE_DIR, f"个股预测_{code}_{stock['name']}.csv")
    future_df.to_csv(pred_file, index=False, encoding='utf_8_sig')
    print(f"\n未来 {FUTURE_DAYS} 个交易日预测已保存到: {pred_file}")
    print(future_df.head(10).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig_file = os.path.join(OUTPUT_DIR, f"个股预测_{code}_{stock['name']}.png")
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_raw, label='历史收盘价', color='#1f77b4', linewidth=1.5)
    test_dates = dates.iloc[train_size + SEQ_LENGTH:]
    plt.plot(test_dates, preds, label='测试集拟合', color='#ff7f0e',
             linewidth=1, alpha=0.8)
    plt.plot(future_dates, future_close, label='预测收盘价',
             linestyle='--', color='#d62728', linewidth=2)
    plt.title(f"{stock['name']}（{code}，{seg_name}）多特征 LSTM 价格预测",
              fontsize=15, pad=15)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格 (元)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"预测图已保存到: {fig_file}")
    plt.show()


if __name__ == '__main__':
    main()
