"""
Complete Strategy Comparison: Buy & Hold vs MA Crossover vs Grid Strategies

This script compares five trading strategies:
1. Buy & Hold: Passive investment baseline
2. MA Crossover: Moving average crossover strategy (5-day vs 20-day)
3. Fixed Grid: Constant grid spacing
4. ATR Adaptive: Dynamic spacing based on Average True Range
5. LSTM Adaptive: ML-based volatility prediction

Time Separation:
- Training Period: 2020-01-01 ~ 2021-12-31 (parameter optimization)
- Backtest Period: 2022-01-01 ~ 2024-06-20 (strategy validation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure Chinese font and larger font sizes
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['font.size'] = 12           # Base font size
mpl.rcParams['axes.titlesize'] = 14      # Title font size
mpl.rcParams['axes.labelsize'] = 12      # Axis label font size
mpl.rcParams['xtick.labelsize'] = 11     # X-tick label size
mpl.rcParams['ytick.labelsize'] = 11     # Y-tick label size
mpl.rcParams['legend.fontsize'] = 11     # Legend font size

from data_clean import load_all_dayk, clean_dayk
from grid import (
    get_price_series, suggest_grid_step, suggest_grid_step_atr,
    run_grid_backtest, run_adaptive_atr_grid_backtest,
    run_ma_crossover_backtest,
    calculate_performance_metrics, print_performance_report
)

try:
    from grid_ml import run_ml_adaptive_grid_backtest, HAS_TENSORFLOW
    if not HAS_TENSORFLOW:
        print("⚠️  TensorFlow not installed, LSTM strategy will be skipped")
        print("    Install: pip install tensorflow")
except ImportError:
    print("⚠️  grid_ml module not found, LSTM strategy will be skipped")
    HAS_TENSORFLOW = False


def main():
    # ========== 1. Data Loading ==========
    print("="*70)
    print("Grid Trading Strategy Comparison")
    print("="*70)
    print("\nLoading data...")
    
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)
    print("✓ Data loaded successfully")

    # ========== 2. Time Separation (Avoid Look-Ahead Bias) ==========
    security_id = "600900"
    
    # Training period (for parameter optimization)
    train_start = "2020-01-01"
    train_end = "2021-12-31"
    
    # Backtest period (for strategy validation)
    backtest_start = "2022-01-01"
    backtest_end = "2024-06-20"
    
    total_capital = 100000.0
    
    print(f"\n{'='*70}")
    print(f"Time Separation (Avoid Look-Ahead Bias)")
    print(f"{'='*70}")
    print(f"Stock: {security_id}")
    print(f"Training Period: {train_start} ~ {train_end}  (parameter optimization)")
    print(f"Backtest Period:  {backtest_start} ~ {backtest_end}  (strategy validation)")

    # ========== 3. Get Training Data ==========
    print(f"\nGetting training period data for parameter optimization...")
    ohlc_train = get_price_series(df_clean, security_id, 
                                  train_start, train_end, 
                                  include_ohlc=True)
    price_train = ohlc_train[["Date", "LastPx"]]
    print(f"Training days: {len(ohlc_train)}")

    # ========== 4. Determine Strategy Parameters (on Training Data) ==========
    print(f"\n{'='*70}")
    print(f"Determining Parameters on Training Period")
    print(f"{'='*70}")
    
    # Fixed Grid parameters
    fixed_step = suggest_grid_step(price_train, factor=1.5)
    print(f"Fixed Grid Spacing: {fixed_step:.3f}")
    
    # ATR parameters
    atr_period = 14
    atr_multiplier = 1.0
    atr_step = suggest_grid_step_atr(ohlc_train, atr_period=atr_period, atr_multiplier=atr_multiplier)
    print(f"\nATR Parameters:")
    print(f"  - ATR Period: {atr_period} days")
    print(f"  - ATR Multiplier: {atr_multiplier}")
    print(f"  - Initial ATR Spacing: {atr_step:.3f}")
    
    # MA Crossover parameters
    short_window = 5
    long_window = 20
    print(f"\nMA Crossover Parameters:")
    print(f"  - Short MA: {short_window} days")
    print(f"  - Long MA: {long_window} days")

    # ========== 5. Get Backtest Data ==========
    print(f"\nGetting backtest period data...")
    ohlc_backtest = get_price_series(df_clean, security_id, 
                                     backtest_start, backtest_end, 
                                     include_ohlc=True)
    price_backtest = ohlc_backtest[["Date", "LastPx"]]
    print(f"Backtest days: {len(ohlc_backtest)}")

    # ========== 6. Baseline: Buy and Hold ==========
    print(f"\n{'='*70}")
    print(f"Baseline: Buy and Hold Strategy")
    print(f"{'='*70}")
    
    # Calculate buy-and-hold performance
    initial_price = price_backtest.iloc[0]['LastPx']
    final_price = price_backtest.iloc[-1]['LastPx']
    bh_shares = total_capital / initial_price
    bh_final_value = bh_shares * final_price
    bh_return = (bh_final_value - total_capital) / total_capital
    
    # Create buy-and-hold equity curve
    equity_bh = pd.DataFrame({
        'Date': price_backtest['Date'],
        'ClosePrice': price_backtest['LastPx'],
        'Cash': 0.0,
        'Position': bh_shares,
        'Equity': price_backtest['LastPx'] * bh_shares
    })
    
    trades_bh = pd.DataFrame()  # No trades for buy-and-hold
    metrics_bh = calculate_performance_metrics(equity_bh, trades_bh, price_backtest)
    print_performance_report(metrics_bh, f"{security_id} - Buy and Hold")
    
    # ========== 7. Strategy 1: MA Crossover ==========
    print(f"\n{'='*70}")
    print(f"Strategy 1: MA Crossover (Backtest Performance)")
    print(f"{'='*70}")
    
    equity_ma, trades_ma = run_ma_crossover_backtest(
        price_backtest,
        short_window=short_window,
        long_window=long_window,
        total_capital=total_capital
    )
    
    metrics_ma = calculate_performance_metrics(equity_ma, trades_ma, price_backtest)
    print_performance_report(metrics_ma, f"{security_id} - MA Crossover")
    
    # ========== 8. Strategy 2: Fixed Grid ==========
    print(f"\n{'='*70}")
    print(f"Strategy 2: Fixed Grid (Backtest Performance)")
    print(f"{'='*70}")
    
    equity_fixed, trades_fixed = run_grid_backtest(
        price_backtest,
        grid_step=fixed_step,
        total_capital=total_capital,
        trade_fraction=0.1
    )
    
    metrics_fixed = calculate_performance_metrics(equity_fixed, trades_fixed, price_backtest)
    print_performance_report(metrics_fixed, f"{security_id} - Fixed Grid")

    # ========== 9. Strategy 3: ATR Adaptive ==========
    print(f"\n{'='*70}")
    print(f"Strategy 3: ATR Adaptive (Backtest Performance)")
    print(f"{'='*70}")
    
    equity_atr, trades_atr = run_adaptive_atr_grid_backtest(
        ohlc_backtest,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        total_capital=total_capital,
        trade_fraction=0.1,
        recalc_interval=5
    )
    
    equity_atr_simple = equity_atr[["Date", "ClosePrice", "Cash", "Position", "Equity"]]
    metrics_atr = calculate_performance_metrics(equity_atr_simple, trades_atr, price_backtest)
    print_performance_report(metrics_atr, f"{security_id} - ATR Adaptive")

    # ========== 10. Strategy 4: LSTM Adaptive ==========
    equity_lstm = None
    trades_lstm = None
    metrics_lstm = None
    predictions_df = None
    
    if HAS_TENSORFLOW:
        print(f"\n{'='*70}")
        print(f"Strategy 4: LSTM Adaptive (Backtest Performance)")
        print(f"{'='*70}")
        
        # For LSTM, we need continuous data from train to backtest
        ohlc_full = get_price_series(df_clean, security_id, 
                                     train_start, backtest_end, 
                                     include_ohlc=True)
        
        # Calculate train_ratio based on date split
        train_days = len(ohlc_train)
        total_days = len(ohlc_full)
        train_ratio = train_days / total_days
        
        vol_multiplier = 2.0
        
        print(f"LSTM training on first {train_ratio*100:.1f}% data ({train_days} days)")
        print(f"LSTM backtesting on last {(1-train_ratio)*100:.1f}% data ({len(ohlc_backtest)} days)")
        print(f"Volatility Multiplier: {vol_multiplier}")
        
        equity_lstm, trades_lstm, model, predictions_df = run_ml_adaptive_grid_backtest(
            ohlc_full,
            total_capital=total_capital,
            trade_fraction=0.1,
            lookback=20,
            forecast_horizon=5,
            train_ratio=train_ratio,
            volatility_multiplier=vol_multiplier,
            retrain_interval=10
        )
        
        equity_lstm_simple = equity_lstm[["Date", "ClosePrice", "Cash", "Position", "Equity"]]
        metrics_lstm = calculate_performance_metrics(equity_lstm_simple, trades_lstm, price_backtest)
        print_performance_report(metrics_lstm, f"{security_id} - LSTM Adaptive")
    else:
        print(f"\n{'='*70}")
        print(f"Strategy 4: LSTM Adaptive - SKIPPED (TensorFlow not available)")
        print(f"{'='*70}")

    # ========== 11. Strategy Comparison Summary ==========
    print(f"\n{'='*70}")
    print(f"Strategy Comparison Summary")
    print(f"{'='*70}")
    
    comparison_data = {
        "Metric": ["Annual Return", "Sharpe Ratio", "Max Drawdown", "Num Trades", "Excess Return"],
        "Buy & Hold": [
            f"{metrics_bh['annual_return']:.2%}",
            f"{metrics_bh['sharpe_ratio']:.2f}",
            f"{metrics_bh['max_drawdown']:.2%}",
            f"{metrics_bh['num_trades']:.0f}",
            f"{metrics_bh['excess_vs_bh']:.2%}"
        ],
        "MA Crossover": [
            f"{metrics_ma['annual_return']:.2%}",
            f"{metrics_ma['sharpe_ratio']:.2f}",
            f"{metrics_ma['max_drawdown']:.2%}",
            f"{metrics_ma['num_trades']:.0f}",
            f"{metrics_ma['excess_vs_bh']:.2%}"
        ],
        "Fixed Grid": [
            f"{metrics_fixed['annual_return']:.2%}",
            f"{metrics_fixed['sharpe_ratio']:.2f}",
            f"{metrics_fixed['max_drawdown']:.2%}",
            f"{metrics_fixed['num_trades']:.0f}",
            f"{metrics_fixed['excess_vs_bh']:.2%}"
        ],
        "ATR Adaptive": [
            f"{metrics_atr['annual_return']:.2%}",
            f"{metrics_atr['sharpe_ratio']:.2f}",
            f"{metrics_atr['max_drawdown']:.2%}",
            f"{metrics_atr['num_trades']:.0f}",
            f"{metrics_atr['excess_vs_bh']:.2%}"
        ]
    }
    
    if metrics_lstm:
        comparison_data["LSTM Adaptive"] = [
            f"{metrics_lstm['annual_return']:.2%}",
            f"{metrics_lstm['sharpe_ratio']:.2f}",
            f"{metrics_lstm['max_drawdown']:.2%}",
            f"{metrics_lstm['num_trades']:.0f}",
            f"{metrics_lstm['excess_vs_bh']:.2%}"
        ]
    
    comparison = pd.DataFrame(comparison_data)
    print(comparison.to_string(index=False))

    # ========== 12. Visualization ==========
    print(f"\nGenerating comparison charts...")
    
    # Define consistent colors for each strategy
    colors = {
        'buy_hold': 'gray',
        'ma_crossover': 'purple',
        'fixed_grid': 'blue',
        'atr_adaptive': 'orange',
        'lstm': 'green'
    }
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    
    # ========== 12.1 Equity Curves Comparison ==========
    axes[0, 0].plot(equity_bh["Date"], equity_bh["Equity"], 
                    label="Buy & Hold", linewidth=2.5, alpha=0.9, linestyle="--", color=colors['buy_hold'])
    axes[0, 0].plot(equity_ma["Date"], equity_ma["Equity"], 
                    label="MA Crossover", linewidth=2.5, alpha=0.9, color=colors['ma_crossover'])
    axes[0, 0].plot(equity_fixed["Date"], equity_fixed["Equity"], 
                    label="Fixed Grid", linewidth=2.5, alpha=0.9, color=colors['fixed_grid'])
    axes[0, 0].plot(equity_atr["Date"], equity_atr["Equity"], 
                    label="ATR Adaptive", linewidth=2.5, alpha=0.9, color=colors['atr_adaptive'])
    if equity_lstm is not None:
        axes[0, 0].plot(equity_lstm["Date"], equity_lstm["Equity"], 
                        label="LSTM Adaptive", linewidth=2.5, alpha=0.9, color=colors['lstm'])
    axes[0, 0].set_xlabel("Date")
    axes[0, 0].set_ylabel("Equity")
    axes[0, 0].set_title("Equity Curve Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # ========== 12.2 Dynamic Grid Spacing ==========
    axes[0, 1].plot(equity_atr["Date"], equity_atr["GridStep"], 
                    color=colors['atr_adaptive'], linewidth=2.5, label="ATR Adaptive", alpha=0.9)
    if equity_lstm is not None:
        axes[0, 1].plot(equity_lstm["Date"], equity_lstm["GridStep"], 
                        color=colors['lstm'], linewidth=2.5, label="LSTM Adaptive", alpha=0.9)
    axes[0, 1].axhline(y=fixed_step, color=colors['fixed_grid'], linestyle="--", 
                      linewidth=2, label=f"Fixed Grid = {fixed_step:.3f}")
    axes[0, 1].set_xlabel("Date")
    axes[0, 1].set_ylabel("Grid Spacing")
    axes[0, 1].set_title("Dynamic Grid Spacing Adjustment")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # ========== 12.3 ATR Indicator ==========
    axes[1, 0].plot(equity_atr["Date"], equity_atr["ATR"], 
                    color="orange", linewidth=2.5, alpha=0.9)
    axes[1, 0].set_xlabel("Date")
    axes[1, 0].set_ylabel("ATR Value")
    axes[1, 0].set_title("ATR (Average True Range) Indicator")
    axes[1, 0].grid(alpha=0.3)
    
    # ========== 12.4 LSTM Volatility Prediction ==========
    if predictions_df is not None and len(predictions_df) > 0:
        axes[1, 1].plot(predictions_df["Date"], predictions_df["ActualVol"], 
                       label="Actual Volatility", linewidth=2.5, alpha=0.9)
        axes[1, 1].plot(predictions_df["Date"], predictions_df["PredictedVol"], 
                       label="LSTM Prediction", linewidth=2.5, alpha=0.9, linestyle="--")
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Volatility")
        axes[1, 1].set_title("LSTM Volatility Prediction vs Actual")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "LSTM Not Available\nor No Predictions", 
                       ha='center', va='center', fontsize=14)
        axes[1, 1].set_title("LSTM Volatility Prediction")
        axes[1, 1].axis('off')
    
    # ========== 12.5 Cumulative Trade Count ==========
    if len(trades_ma) > 0:
        axes[2, 0].plot(trades_ma["Date"], range(1, len(trades_ma) + 1), 
                       label="MA Crossover", linewidth=2.5, alpha=0.9, color=colors['ma_crossover'])
    if len(trades_fixed) > 0:
        axes[2, 0].plot(trades_fixed["Date"], range(1, len(trades_fixed) + 1), 
                       label="Fixed Grid", linewidth=2.5, alpha=0.9, color=colors['fixed_grid'])
    if len(trades_atr) > 0:
        axes[2, 0].plot(trades_atr["Date"], range(1, len(trades_atr) + 1), 
                       label="ATR Adaptive", linewidth=2.5, alpha=0.9, color=colors['atr_adaptive'])
    if trades_lstm is not None and len(trades_lstm) > 0:
        axes[2, 0].plot(trades_lstm["Date"], range(1, len(trades_lstm) + 1), 
                       label="LSTM Adaptive", linewidth=2.5, alpha=0.9, color=colors['lstm'])
    axes[2, 0].set_xlabel("Date")
    axes[2, 0].set_ylabel("Cumulative Trade Count")
    axes[2, 0].set_title("Cumulative Trade Count Comparison")
    axes[2, 0].legend()
    axes[2, 0].grid(alpha=0.3)
    
    # ========== 12.6 Performance Metrics Bar Chart ==========
    metrics_names = ["Annual\nReturn", "Sharpe\nRatio", "Max\nDrawdown"]
    bh_values = [metrics_bh["annual_return"]*100, 
                 metrics_bh["sharpe_ratio"], 
                 abs(metrics_bh["max_drawdown"])*100]
    ma_values = [metrics_ma["annual_return"]*100, 
                 metrics_ma["sharpe_ratio"], 
                 abs(metrics_ma["max_drawdown"])*100]
    fixed_values = [metrics_fixed["annual_return"]*100, 
                   metrics_fixed["sharpe_ratio"], 
                   abs(metrics_fixed["max_drawdown"])*100]
    atr_values = [metrics_atr["annual_return"]*100, 
                 metrics_atr["sharpe_ratio"], 
                 abs(metrics_atr["max_drawdown"])*100]
    
    x = np.arange(len(metrics_names))
    width = 0.16
    
    axes[2, 1].bar(x - width*2, bh_values, width, label="Buy & Hold", alpha=0.8, color=colors['buy_hold'])
    axes[2, 1].bar(x - width, ma_values, width, label="MA Crossover", alpha=0.8, color=colors['ma_crossover'])
    axes[2, 1].bar(x, fixed_values, width, label="Fixed Grid", alpha=0.8, color=colors['fixed_grid'])
    axes[2, 1].bar(x + width, atr_values, width, label="ATR Adaptive", alpha=0.8, color=colors['atr_adaptive'])
    
    if metrics_lstm:
        lstm_values = [metrics_lstm["annual_return"]*100, 
                      metrics_lstm["sharpe_ratio"], 
                      abs(metrics_lstm["max_drawdown"])*100]
        axes[2, 1].bar(x + width*2, lstm_values, width, label="LSTM Adaptive", alpha=0.8, color=colors['lstm'])
    
    axes[2, 1].set_ylabel("Value")
    axes[2, 1].set_title("Performance Metrics Comparison")
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(metrics_names)
    axes[2, 1].legend()
    axes[2, 1].grid(alpha=0.3, axis='y')
    
    # Note: Return and Drawdown in %, Sharpe Ratio is absolute
    # axes[2, 1].text(0.02, 0.98, "Note: Return & Drawdown in %, Sharpe is absolute", 
    #                transform=axes[2, 1].transAxes, fontsize=9,
    #                verticalalignment='top')
    
    plt.tight_layout()
    output_file = f"{security_id}_strategy_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison chart saved to {output_file}")
    plt.show()

    # ========== 13. Save Results ==========
    trades_ma.to_csv(f"{security_id}_ma_crossover_trades.csv", index=False)
    trades_fixed.to_csv(f"{security_id}_fixed_trades.csv", index=False)
    trades_atr.to_csv(f"{security_id}_atr_trades.csv", index=False)
    if trades_lstm is not None:
        trades_lstm.to_csv(f"{security_id}_lstm_trades.csv", index=False)
    if predictions_df is not None and len(predictions_df) > 0:
        predictions_df.to_csv(f"{security_id}_lstm_predictions.csv", index=False)
    
    print(f"✓ Trade records saved")
    
    # ========== 14. Summary ==========
    print(f"\n{'='*70}")
    print(f"Backtest Completed Successfully!")
    print(f"{'='*70}")
    print(f"Key Points:")
    print(f"1. Baseline: Buy & Hold (passive investment)")
    print(f"2. MA Crossover: {short_window}-day vs {long_window}-day moving average")
    print(f"3. Grid Strategies with independent initialization:")
    print(f"   - Fixed Grid: Constant spacing = {fixed_step:.3f}")
    print(f"   - ATR Adaptive: Initial ATR-based spacing = {atr_step:.3f}, then adjusts dynamically")
    print(f"   - LSTM Adaptive: ML-predicted spacing based on volatility forecasts")
    print(f"4. Parameters optimized on training period ({train_start} ~ {train_end})")
    print(f"5. Strategies validated on backtest period ({backtest_start} ~ {backtest_end})")
    print(f"6. Five strategies compared: Buy & Hold, MA Crossover, Fixed Grid, ATR, LSTM")
    print(f"7. No look-ahead bias - all parameters determined before backtest")
    print(f"{'='*70}")
    


if __name__ == "__main__":
    main()
