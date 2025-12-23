"""
基于机器学习的智能网格间距调整

使用LSTM预测未来波动率，动态调整网格间距
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习框架
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    print("⚠️ TensorFlow未安装。请运行: pip install tensorflow")


def prepare_lstm_data(data_series, lookback=20, forecast_horizon=5):
    """
    准备LSTM训练数据
    
    参数:
        data_series: 时间序列数据（如收盘价或波动率）
        lookback: 回看窗口（使用过去多少天的数据）
        forecast_horizon: 预测未来多少天
    
    返回:
        X, y: 特征和目标
    """
    X, y = [], []
    
    for i in range(lookback, len(data_series) - forecast_horizon + 1):
        # 特征：过去lookback天的数据
        X.append(data_series[i-lookback:i])
        # 目标：未来forecast_horizon天的平均值
        y.append(np.mean(data_series[i:i+forecast_horizon]))
    
    return np.array(X), np.array(y)


def build_lstm_model(lookback=20, units=50, dropout_rate=0.2):
    """
    构建LSTM模型
    
    参数:
        lookback: 输入序列长度
        units: LSTM单元数量
        dropout_rate: Dropout比率
    
    返回:
        编译好的Keras模型
    """
    if not HAS_TENSORFLOW:
        raise ImportError("需要安装TensorFlow: pip install tensorflow")
    
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(dropout_rate),
        LSTM(units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def train_volatility_predictor(ohlc_df, lookback=20, forecast_horizon=5, 
                               epochs=50, validation_split=0.2):
    """
    训练波动率预测模型
    
    参数:
        ohlc_df: 包含OHLC数据的DataFrame
        lookback: 回看窗口
        forecast_horizon: 预测窗口
        epochs: 训练轮数
        validation_split: 验证集比例
    
    返回:
        model: 训练好的模型
        scaler: 数据缩放器
        history: 训练历史
    """
    if not HAS_TENSORFLOW:
        raise ImportError("需要安装TensorFlow: pip install tensorflow")
    
    # 计算历史波动率（使用高低价范围）
    ohlc_df = ohlc_df.copy()
    ohlc_df['volatility'] = (ohlc_df['HighPx'] - ohlc_df['LowPx']) / ohlc_df['LastPx']
    volatility = ohlc_df['volatility'].values
    
    # 数据标准化
    scaler = MinMaxScaler()
    volatility_scaled = scaler.fit_transform(volatility.reshape(-1, 1)).flatten()
    
    # 准备训练数据
    X, y = prepare_lstm_data(volatility_scaled, lookback, forecast_horizon)
    
    # 重塑为LSTM输入格式 [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    print(f"训练数据形状: X={X.shape}, y={y.shape}")
    
    # 构建模型
    model = build_lstm_model(lookback=lookback)
    
    # 早停回调
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 训练模型
    print("开始训练LSTM模型...")
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✓ 模型训练完成")
    
    return model, scaler, history


def predict_future_volatility(model, scaler, recent_volatility, lookback=20):
    """
    预测未来波动率
    
    参数:
        model: 训练好的LSTM模型
        scaler: 数据缩放器
        recent_volatility: 最近的波动率数据（至少lookback个点）
        lookback: 回看窗口
    
    返回:
        预测的未来波动率
    """
    if not HAS_TENSORFLOW:
        raise ImportError("需要安装TensorFlow: pip install tensorflow")
    
    # 标准化
    recent_scaled = scaler.transform(recent_volatility[-lookback:].reshape(-1, 1)).flatten()
    
    # 重塑为LSTM输入格式
    X_pred = recent_scaled.reshape(1, lookback, 1)
    
    # 预测
    y_pred_scaled = model.predict(X_pred, verbose=0)
    
    # 反标准化
    y_pred = scaler.inverse_transform(y_pred_scaled)[0, 0]
    
    return float(y_pred)


def run_ml_adaptive_grid_backtest(ohlc_df,
                                  total_capital=100000.0,
                                  trade_fraction=0.02,
                                  lookback=20,
                                  forecast_horizon=5,
                                  train_ratio=0.6,
                                  volatility_multiplier=2.0,
                                  retrain_interval=20):
    """
    基于机器学习的自适应网格交易回测
    
    工作流程:
    1. 使用前train_ratio的数据训练LSTM模型
    2. 在剩余数据上进行回测
    3. 每retrain_interval天预测未来波动率并调整网格间距
    
    参数:
        ohlc_df: 包含OHLC数据的DataFrame
        total_capital: 初始资金
        trade_fraction: 每次交易比例
        lookback: LSTM回看窗口
        forecast_horizon: 预测未来多少天
        train_ratio: 训练数据比例（如0.6表示前60%用于训练）
        volatility_multiplier: 波动率乘数（grid_step = predicted_vol * price * multiplier）
        retrain_interval: 每隔多少天更新预测
    
    返回:
        equity_df: 资产曲线
        trades_df: 交易记录
        model: 训练好的模型
        predictions_df: 预测记录
    """
    if not HAS_TENSORFLOW:
        raise ImportError("需要安装TensorFlow: pip install tensorflow")
    
    df = ohlc_df.copy().sort_values("Date").reset_index(drop=True)
    
    # 计算历史波动率
    df['volatility'] = (df['HighPx'] - df['LowPx']) / df['LastPx']
    
    # 分割训练集和测试集
    train_size = int(len(df) * train_ratio)
    df_train = df.iloc[:train_size].copy()
    df_test = df.iloc[train_size:].copy()
    
    print(f"训练集大小: {len(df_train)} 天")
    print(f"测试集大小: {len(df_test)} 天")
    
    # 训练模型
    model, scaler, history = train_volatility_predictor(
        df_train,
        lookback=lookback,
        forecast_horizon=forecast_horizon,
        epochs=50,
        validation_split=0.2
    )
    
    # 在测试集上进行回测
    prices = df_test['LastPx'].values
    dates = df_test['Date'].values
    volatilities = df_test['volatility'].values
    
    # 初始化交易状态
    cash = total_capital
    position = 0
    
    # 初始网格间距（使用最后lookback天的平均波动率）
    initial_vol = np.mean(volatilities[:min(lookback, len(volatilities))])
    grid_step = float(prices[0] * initial_vol * volatility_multiplier)
    
    mid_price = float(prices[0])
    qty_per_trade = int(total_capital * trade_fraction / mid_price)
    if qty_per_trade <= 0:
        qty_per_trade = 1
    
    last_trade_price = float(prices[0])
    
    equity_records = []
    trade_records = []
    prediction_records = []
    
    for i in range(len(df_test)):
        date = dates[i]
        p = float(prices[i])
        current_vol = float(volatilities[i])
        
        # 定期更新网格间距（基于LSTM预测）
        if i > 0 and i % retrain_interval == 0 and i >= lookback:
            # 获取最近的波动率数据
            recent_vol = volatilities[max(0, i-lookback):i]
            
            # 预测未来波动率
            predicted_vol = predict_future_volatility(
                model, scaler, recent_vol, lookback
            )
            
            # 更新网格间距
            grid_step = float(p * predicted_vol * volatility_multiplier)
            
            # 记录预测
            prediction_records.append({
                'Date': date,
                'ActualVol': current_vol,
                'PredictedVol': predicted_vol,
                'GridStep': grid_step,
                'Price': p
            })
            
            print(f"Day {i}: 预测波动率={predicted_vol:.4f}, 新网格间距={grid_step:.3f}")
        
        # 网格交易逻辑（与之前相同）
        price_min = float(prices[:i+1].min())
        price_max = float(prices[:i+1].max())
        min_level = (price_min // grid_step) * grid_step if grid_step > 0 else price_min
        max_level = ((price_max // grid_step) + 1) * grid_step if grid_step > 0 else price_max
        
        diff = p - last_trade_price
        steps = int(diff // grid_step) if grid_step > 0 else 0
        
        while steps != 0:
            if steps > 0:  # 卖出
                next_price = last_trade_price + grid_step
                if next_price > max_level:
                    break
                
                if position >= qty_per_trade:
                    position -= qty_per_trade
                    cash += qty_per_trade * next_price
                    last_trade_price = next_price
                    
                    trade_records.append({
                        'Date': date,
                        'Side': 'SELL',
                        'Price': next_price,
                        'Qty': qty_per_trade,
                        'GridStep': grid_step,
                        'PredictedVol': prediction_records[-1]['PredictedVol'] if prediction_records else np.nan,
                        'CashAfter': cash,
                        'PositionAfter': position
                    })
                    steps -= 1
                else:
                    break
            
            elif steps < 0:  # 买入
                next_price = last_trade_price - grid_step
                if next_price < min_level:
                    break
                
                cost = qty_per_trade * next_price
                if cost <= cash:
                    position += qty_per_trade
                    cash -= cost
                    last_trade_price = next_price
                    
                    trade_records.append({
                        'Date': date,
                        'Side': 'BUY',
                        'Price': next_price,
                        'Qty': qty_per_trade,
                        'GridStep': grid_step,
                        'PredictedVol': prediction_records[-1]['PredictedVol'] if prediction_records else np.nan,
                        'CashAfter': cash,
                        'PositionAfter': position
                    })
                    steps += 1
                else:
                    break
        
        # 记录每日资产
        equity = cash + position * p
        equity_records.append({
            'Date': date,
            'ClosePrice': p,
            'Cash': cash,
            'Position': position,
            'Equity': equity,
            'GridStep': grid_step,
            'ActualVol': current_vol
        })
    
    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trade_records)
    predictions_df = pd.DataFrame(prediction_records)
    
    return equity_df, trades_df, model, predictions_df


# 如果没有TensorFlow，提供简化版本（使用移动平均）
def run_simple_ml_grid_backtest(ohlc_df,
                                total_capital=100000.0,
                                trade_fraction=0.02,
                                volatility_window=20,
                                volatility_multiplier=2.0):
    """
    简化版ML网格策略（不需要TensorFlow）
    使用移动平均预测波动率
    """
    print("使用简化版本（移动平均预测）")
    
    from grid import run_adaptive_atr_grid_backtest
    
    # 使用ATR策略作为替代
    return run_adaptive_atr_grid_backtest(
        ohlc_df,
        atr_period=volatility_window,
        atr_multiplier=volatility_multiplier,
        total_capital=total_capital,
        trade_fraction=trade_fraction,
        recalc_interval=5
    )
