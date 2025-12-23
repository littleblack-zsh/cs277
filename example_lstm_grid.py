"""
基于LSTM的智能网格交易策略示例

使用深度学习预测未来波动率，动态调整网格间距
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 配置中文字体
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

from data_clean import load_all_dayk, clean_dayk
from grid import (
    get_price_series, run_grid_backtest, 
    calculate_performance_metrics, print_performance_report
)

try:
    from grid_ml import run_ml_adaptive_grid_backtest, HAS_TENSORFLOW
    if not HAS_TENSORFLOW:
        print("⚠️  TensorFlow未安装，将使用简化版本")
        print("    安装TensorFlow: pip install tensorflow")
except ImportError:
    print("⚠️  grid_ml模块导入失败")
    HAS_TENSORFLOW = False


def main():
    # 1. 数据加载
    print("="*60)
    print("基于LSTM的智能网格交易策略")
    print("="*60)
    print("\n正在加载数据...")
    
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)
    print("✓ 数据加载完成")

    # 2. 参数设置
    security_id = "000008"
    start_date = "2020-01-01"
    end_date = "2024-06-20"
    total_capital = 100000.0

    print(f"\n测试股票: {security_id}")
    print(f"数据范围: {start_date} ~ {end_date}")

    # 3. 获取OHLC数据
    ohlc_df = get_price_series(df_clean, security_id, start_date, end_date, include_ohlc=True)
    price_df = ohlc_df[['Date', 'LastPx']]
    
    print(f"总交易日数: {len(ohlc_df)} 天")

    if not HAS_TENSORFLOW:
        print("\n⚠️  由于TensorFlow未安装，无法运行LSTM策略")
        print("请安装: pip install tensorflow")
        return

    # 4. 运行LSTM智能网格策略
    print("\n" + "="*60)
    print("策略：LSTM预测波动率 + 动态网格间距")
    print("="*60)
    
    print("\n训练LSTM模型并执行回测...")
    print("(前60%数据用于训练，后40%用于回测)")
    
    equity_ml, trades_ml, model, predictions_df = run_ml_adaptive_grid_backtest(
        ohlc_df,
        total_capital=total_capital,
        trade_fraction=0.1,
        lookback=20,              # LSTM回看20天
        forecast_horizon=5,       # 预测未来5天
        train_ratio=0.6,          # 前60%数据训练
        volatility_multiplier=2.0,  # 波动率乘数
        retrain_interval=10       # 每10天更新一次预测
    )
    
    # 获取回测期数据（后40%）
    test_start_idx = int(len(ohlc_df) * 0.6)
    test_dates = ohlc_df['Date'].iloc[test_start_idx:].values
    backtest_start = test_dates[0]
    backtest_end = test_dates[-1]
    
    print(f"\n回测期: {backtest_start} ~ {backtest_end}")
    print(f"回测天数: {len(test_dates)} 天")
    
    # 5. 对比：固定网格策略
    print("\n" + "="*60)
    print("对比策略：固定网格间距")
    print("="*60)
    
    from grid import suggest_grid_step
    
    # 获取回测期数据
    price_backtest = price_df[price_df['Date'] >= backtest_start].copy()
    
    # 使用训练期确定固定网格间距（避免前视偏差）
    price_train = price_df[price_df['Date'] < backtest_start].copy()
    fixed_step = suggest_grid_step(price_train, factor=1.5)
    print(f"固定网格间距: {fixed_step:.3f}")
    
    equity_fixed, trades_fixed = run_grid_backtest(
        price_backtest,
        grid_step=fixed_step,
        total_capital=total_capital,
        trade_fraction=0.1
    )
    
    # 6. 计算评估指标
    equity_ml_simple = equity_ml[['Date', 'ClosePrice', 'Cash', 'Position', 'Equity']]
    metrics_ml = calculate_performance_metrics(equity_ml_simple, trades_ml, price_backtest)
    metrics_fixed = calculate_performance_metrics(equity_fixed, trades_fixed, price_backtest)
    
    print_performance_report(metrics_ml, f"{security_id} - LSTM智能网格")
    print_performance_report(metrics_fixed, f"{security_id} - 固定网格")
    
    # 7. 策略对比
    print("\n" + "="*60)
    print("策略对比总结")
    print("="*60)
    
    comparison = pd.DataFrame({
        '指标': ['年化收益率', '夏普比率', '最大回撤', '交易次数', '超额收益'],
        'LSTM智能': [
            f"{metrics_ml['annual_return']:.2%}",
            f"{metrics_ml['sharpe_ratio']:.2f}",
            f"{metrics_ml['max_drawdown']:.2%}",
            f"{metrics_ml['num_trades']:.0f}",
            f"{metrics_ml['excess_vs_bh']:.2%}"
        ],
        '固定网格': [
            f"{metrics_fixed['annual_return']:.2%}",
            f"{metrics_fixed['sharpe_ratio']:.2f}",
            f"{metrics_fixed['max_drawdown']:.2%}",
            f"{metrics_fixed['num_trades']:.0f}",
            f"{metrics_fixed['excess_vs_bh']:.2%}"
        ]
    })
    print(comparison.to_string(index=False))
    
    # 8. 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 8.1 资产曲线对比
    axes[0, 0].plot(equity_fixed['Date'], equity_fixed['Equity'], 
                    label='固定网格', linewidth=2, alpha=0.8)
    axes[0, 0].plot(equity_ml['Date'], equity_ml['Equity'], 
                    label='LSTM智能', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('日期')
    axes[0, 0].set_ylabel('资产 (¥)')
    axes[0, 0].set_title('资产曲线对比')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 8.2 网格间距动态变化
    axes[0, 1].plot(equity_ml['Date'], equity_ml['GridStep'], 
                    color='green', linewidth=2, label='LSTM动态间距')
    axes[0, 1].axhline(y=fixed_step, color='blue', linestyle='--', 
                      label=f'固定间距 = {fixed_step:.3f}')
    axes[0, 1].set_xlabel('日期')
    axes[0, 1].set_ylabel('网格间距 (¥)')
    axes[0, 1].set_title('LSTM动态调整网格间距')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 8.3 波动率预测 vs 实际
    if len(predictions_df) > 0:
        axes[1, 0].plot(predictions_df['Date'], predictions_df['ActualVol'], 
                       label='实际波动率', linewidth=2, alpha=0.8)
        axes[1, 0].plot(predictions_df['Date'], predictions_df['PredictedVol'], 
                       label='LSTM预测', linewidth=2, alpha=0.8, linestyle='--')
        axes[1, 0].set_xlabel('日期')
        axes[1, 0].set_ylabel('波动率')
        axes[1, 0].set_title('LSTM波动率预测 vs 实际')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
    
    # 8.4 累计交易次数对比
    if len(trades_fixed) > 0:
        trades_fixed_cumsum = range(1, len(trades_fixed) + 1)
        axes[1, 1].plot(trades_fixed['Date'], trades_fixed_cumsum, 
                       label='固定网格', linewidth=2)
    if len(trades_ml) > 0:
        trades_ml_cumsum = range(1, len(trades_ml) + 1)
        axes[1, 1].plot(trades_ml['Date'], trades_ml_cumsum, 
                       label='LSTM智能', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('累计交易次数')
    axes[1, 1].set_title('累计交易次数对比')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{security_id}_lstm_comparison.png', dpi=150)
    print(f"\n✓ 对比图表已保存到 {security_id}_lstm_comparison.png")
    plt.show()
    
    # 9. 保存结果
    trades_ml.to_csv(f'{security_id}_lstm_trades.csv', index=False)
    trades_fixed.to_csv(f'{security_id}_fixed_trades.csv', index=False)
    if len(predictions_df) > 0:
        predictions_df.to_csv(f'{security_id}_lstm_predictions.csv', index=False)
    
    print(f"✓ 交易记录已保存")
    print(f"✓ 预测记录已保存到 {security_id}_lstm_predictions.csv")
    
    print("\n" + "="*60)
    print("说明：")
    print("1. LSTM模型在前60%数据上训练")
    print("2. 在后40%数据上进行回测验证")
    print("3. 每10天更新一次波动率预测")
    print("4. 根据预测的波动率动态调整网格间距")
    print("="*60)


if __name__ == "__main__":
    main()
