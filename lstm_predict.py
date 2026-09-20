# -*- coding: utf-8 -*-
"""lstm_predict.py - 多特征 LSTM 预测 AI 主题 ETF 价格

输入特征：ETF 开盘/收盘价 + 上中下游环节情绪得分（+ 关联 ETF 收盘价）
预测目标：主 ETF 未来 30 个交易日的收盘价

注意：未来日的环节情绪不可知，滚动预测时以最近一期情绪值填充（恒定假设）。
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from industry_config import DATA_FEATURE_DIR, ETFS, OUTPUT_DIR, SEGMENTS

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 固定随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

FEATURE_FILE = os.path.join(DATA_FEATURE_DIR, 'LSTM特征宽表.csv')
PREDICT_FILE = os.path.join(DATA_FEATURE_DIR, 'ETF未来30日预测.csv')
FIGURE_FILE = os.path.join(OUTPUT_DIR, 'ETF价格预测.png')
MODEL_FILE = 'ai_lstm_model.pt'

SEQ_LENGTH = 15
NUM_EPOCHS = 300
FUTURE_DAYS = 30


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
    if not os.path.exists(FEATURE_FILE):
        print(f"未找到 {FEATURE_FILE}，请先运行: python feature_builder.py")
        return

    base_etf = ETFS[0]
    open_col = f"{base_etf['code']}_开盘"
    close_col = f"{base_etf['code']}_收盘"

    df = pd.read_csv(FEATURE_FILE, parse_dates=['日期'])
    sentiment_cols = [f'{k}_情绪' for k in SEGMENTS]
    extra_cols = [c for c in df.columns if c.endswith('_收盘') and c != close_col]
    feature_cols = [open_col, close_col] + sentiment_cols + extra_cols
    print(f"输入特征: {feature_cols}")

    X_raw = df[feature_cols].values
    y_raw = df[close_col].values.reshape(-1, 1)
    dates = df['日期']

    # 先切分再归一化：Scaler 只拟合训练集，避免测试集信息泄漏
    train_size = int(len(X_raw) * 0.8)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
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
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss / len(train_loader):.6f}')
    print("训练完成。")

    torch.save(model.state_dict(), MODEL_FILE)
    print(f"模型权重已保存到 '{MODEL_FILE}'。")

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
    close_idx = feature_cols.index(close_col)
    open_idx = feature_cols.index(open_col)

    future_scaled = []
    with torch.no_grad():
        for _ in range(FUTURE_DAYS):
            inp = torch.FloatTensor(input_seq.reshape(1, SEQ_LENGTH, -1)).to(device)
            pred_close = model(inp).cpu().numpy()[0, 0]
            future_scaled.append(pred_close)
            # 构造下一日特征：开盘/收盘=预测值，情绪与其他特征沿用最近值
            next_row = last_row.copy()
            next_row[open_idx] = pred_close
            next_row[close_idx] = pred_close
            input_seq = np.vstack([input_seq, next_row])[1:]

    future_close = scaler_y.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1),
                                 periods=FUTURE_DAYS, freq='B')

    future_df = pd.DataFrame({'日期': future_dates,
                              f"预测收盘价": np.round(future_close, 3)})
    os.makedirs(DATA_FEATURE_DIR, exist_ok=True)
    future_df.to_csv(PREDICT_FILE, index=False, encoding='utf_8_sig')
    print(f"\n未来 {FUTURE_DAYS} 个交易日预测已保存到: {PREDICT_FILE}")
    print(future_df.head(10).to_string(index=False))

    # 可视化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_raw, label='历史收盘价', color='#1f77b4', linewidth=1.5)
    plt.plot(future_dates, future_close, label='预测收盘价',
             linestyle='--', color='#d62728', linewidth=2)
    test_dates = dates.iloc[train_size + SEQ_LENGTH:]
    plt.plot(test_dates, preds, label='测试集拟合', color='#ff7f0e',
             linewidth=1, alpha=0.8)
    plt.title(f"{base_etf['name']}（{base_etf['code']}）多特征 LSTM 价格预测",
              fontsize=15, pad=15)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('价格 (元)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"预测图已保存到: {FIGURE_FILE}")
    plt.show()


if __name__ == '__main__':
    main()
