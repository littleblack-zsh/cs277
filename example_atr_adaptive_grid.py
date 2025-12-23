"""
基于ATR的自适应网格间距策略示例

ATR (Average True Range) 是衡量市场波动性的技术指标：
- 当市场波动大时，ATR值高，网格间距自动扩大
- 当市场波动小时，ATR值低，网格间距自动缩小

优势：
1. 动态适应市场波动性
2. 避免在高波动期过度交易
3. 在低波动期捕捉更多小幅波动
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from data_clean import load_all_dayk, clean_dayk
from grid import (
    get_price_series, suggest_grid_step, suggest_grid_step_atr,
    run_grid_backtest, run_adaptive_atr_grid_backtest,
    calculate_performance_metrics, print_performance_report
)

# 配置matplotlib中文字体，避免乱码
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
mpl.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def main():
    # 1. 数据加载与清洗
    print("正在加载数据...")
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)
    print("✓ 数据加载完成")

    # 2. ===== 关键：时间分割，避免前视偏差 =====
    security_id = "000008"  # 测试股票
    
    # 选股期（用于确定策略参数）
    selection_start = "2019-11-13"
    selection_end = "2021-12-31"
    
    # 回测期（用于验证策略表现）
    backtest_start = "2022-01-01"
    backtest_end = "2024-06-20"
    
    total_capital = 100000.0
    
    print(f"\n{'='*60}")
    print(f"时间分割（避免前视偏差）")
    print(f"{'='*60}")
    print(f"选股期: {selection_start} ~ {selection_end}  (用于确定参数)")
    print(f"回测期: {backtest_start} ~ {backtest_end}  (验证策略表现)")

    # 3. 在选股期获取OHLC数据，用于确定策略参数
    print(f"\n获取选股期数据用于参数确定...")
    ohlc_selection = get_price_series(df_clean, security_id, 
                                     selection_start, selection_end, 
                                     include_ohlc=True)
    price_selection = ohlc_selection[["Date", "LastPx"]]

    # 4. 在选股期确定策略参数
    print(f"\n在选股期 ({selection_start} ~ {selection_end}) 确定策略参数:")
    
    # 固定网格参数（基于选股期）
    fixed_step = suggest_grid_step(price_selection, factor=1.5)
    print(f"  - 固定网格间距: {fixed_step:.3f}")
    
    # ATR参数（基于选股期）
    atr_multiplier = 1.0
    atr_period = 14
    atr_step_initial = suggest_grid_step_atr(ohlc_selection, 
                                             atr_period=atr_period, 
                                             atr_multiplier=atr_multiplier)
    print(f"  - ATR初始网格间距: {atr_step_initial:.3f}")
    print(f"  - ATR周期: {atr_period} 天")
    print(f"  - ATR乘数: {atr_multiplier}")

    # 5. 在回测期获取数据
    print(f"\n获取回测期数据...")
    ohlc_backtest = get_price_series(df_clean, security_id, 
                                    backtest_start, backtest_end, 
                                    include_ohlc=True)
    price_backtest = ohlc_backtest[["Date", "LastPx"]]
    print(f"回测期交易日数: {len(ohlc_backtest)} 天")

    # 6. 策略对比：在回测期执行
    
    # ========== 策略1: 固定网格间距 ==========
    print("\n" + "="*60)
    print("策略1: 固定网格间距（回测期表现）")
    print("="*60)
    
    equity_fixed, trades_fixed = run_grid_backtest(
        price_backtest,
        grid_step=fixed_step,  # 使用选股期确定的参数
        total_capital=total_capital,
        trade_fraction=0.1
    )
    
    metrics_fixed = calculate_performance_metrics(equity_fixed, trades_fixed, price_backtest)
    print_performance_report(metrics_fixed, f"{security_id} - 固定网格")
    
    # ========== 策略2: ATR自适应网格 ==========
    print("\n" + "="*60)
    print("策略2: ATR自适应网格（回测期表现）")
    print("="*60)
    
    # ========== 策略2: ATR自适应网格 ==========
    print("\n" + "="*60)
    print("策略2: ATR自适应网格（回测期表现）")
    print("="*60)
    
    equity_atr, trades_atr = run_adaptive_atr_grid_backtest(
        ohlc_backtest,          # 使用回测期数据
        atr_period=atr_period,  # 使用选股期确定的参数
        atr_multiplier=atr_multiplier,
        total_capital=total_capital,
        trade_fraction=0.1,
        recalc_interval=5
    )
    
    # 将ATR版本的equity_df转换为标准格式用于metrics计算
    equity_atr_simple = equity_atr[["Date", "ClosePrice", "Cash", "Position", "Equity"]]
    metrics_atr = calculate_performance_metrics(equity_atr_simple, trades_atr, price_backtest)
    print_performance_report(metrics_atr, f"{security_id} - ATR自适应")
    
    # 5. 策略对比总结
    print("\n" + "="*60)
    print("策略对比总结")
    print("="*60)
    comparison = pd.DataFrame({
        "指标": ["年化收益率", "夏普比率", "最大回撤", "交易次数", "超额收益"],
        "固定网格": [
            f"{metrics_fixed['annual_return']:.2%}",
            f"{metrics_fixed['sharpe_ratio']:.2f}",
            f"{metrics_fixed['max_drawdown']:.2%}",
            f"{metrics_fixed['num_trades']:.0f}",
            f"{metrics_fixed['excess_vs_bh']:.2%}"
        ],
        "ATR自适应": [
            f"{metrics_atr['annual_return']:.2%}",
            f"{metrics_atr['sharpe_ratio']:.2f}",
            f"{metrics_atr['max_drawdown']:.2%}",
            f"{metrics_atr['num_trades']:.0f}",
            f"{metrics_atr['excess_vs_bh']:.2%}"
        ]
    })
    print(comparison.to_string(index=False))
    
    # 6. 可视化对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 6.1 资产曲线对比
    axes[0, 0].plot(equity_fixed["Date"], equity_fixed["Equity"], 
                    label="固定网格", linewidth=2)
    axes[0, 0].plot(equity_atr["Date"], equity_atr["Equity"], 
                    label="ATR自适应", linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel("日期")
    axes[0, 0].set_ylabel("资产 (¥)")
    axes[0, 0].set_title("资产曲线对比")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 6.2 网格间距动态变化（仅ATR策略）
    axes[0, 1].plot(equity_atr["Date"], equity_atr["GridStep"], 
                    color="orange", linewidth=2)
    axes[0, 1].axhline(y=fixed_step, color="blue", linestyle="--", 
                      label=f"固定间距 = {fixed_step:.3f}")
    axes[0, 1].set_xlabel("日期")
    axes[0, 1].set_ylabel("网格间距 (¥)")
    axes[0, 1].set_title("ATR自适应网格间距变化")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # 6.3 ATR指标变化
    axes[1, 0].plot(equity_atr["Date"], equity_atr["ATR"], 
                    color="green", linewidth=2)
    axes[1, 0].set_xlabel("日期")
    axes[1, 0].set_ylabel("ATR")
    axes[1, 0].set_title("ATR (平均真实波动范围) 变化")
    axes[1, 0].grid(alpha=0.3)
    
    # 6.4 交易次数累计对比
    trades_fixed_cumsum = range(1, len(trades_fixed) + 1)
    trades_atr_cumsum = range(1, len(trades_atr) + 1)
    
    if len(trades_fixed) > 0:
        axes[1, 1].plot(trades_fixed["Date"], trades_fixed_cumsum, 
                       label="固定网格", linewidth=2)
    if len(trades_atr) > 0:
        axes[1, 1].plot(trades_atr["Date"], trades_atr_cumsum, 
                       label="ATR自适应", linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel("日期")
    axes[1, 1].set_ylabel("累计交易次数")
    axes[1, 1].set_title("累计交易次数对比")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{security_id}_atr_comparison.png", dpi=150)
    print(f"\n✓ 对比图表已保存到 {security_id}_atr_comparison.png")
    plt.show()
    
    # 7. 保存交易记录
    trades_fixed.to_csv(f"{security_id}_fixed_grid_trades.csv", index=False)
    trades_atr.to_csv(f"{security_id}_atr_adaptive_trades.csv", index=False)
    print(f"✓ 交易记录已保存")
    
    # 8. 参数敏感性分析：不同ATR乘数的表现（在回测期验证）
    print("\n" + "="*60)
    print("ATR乘数参数扫描（回测期验证）")
    print("="*60)
    print("注意：参数范围基于选股期分析，这里在回测期验证不同参数的表现")
    
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    scan_results = []
    
    for mult in multipliers:
        equity_temp, trades_temp = run_adaptive_atr_grid_backtest(
            ohlc_backtest,      # 使用回测期数据
            atr_period=atr_period,
            atr_multiplier=mult,
            total_capital=total_capital,
            trade_fraction=0.1,
            recalc_interval=5
        )
        
        equity_temp_simple = equity_temp[["Date", "ClosePrice", "Cash", "Position", "Equity"]]
        metrics_temp = calculate_performance_metrics(equity_temp_simple, trades_temp, price_backtest)
        
        scan_results.append({
            "ATR乘数": mult,
            "年化收益率": metrics_temp["annual_return"],
            "夏普比率": metrics_temp["sharpe_ratio"],
            "最大回撤": metrics_temp["max_drawdown"],
            "交易次数": metrics_temp["num_trades"]
        })
    
    scan_df = pd.DataFrame(scan_results)
    print(scan_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ 回测完成！")
    print("="*60)
    print(f"说明：")
    print(f"1. 策略参数在选股期 ({selection_start}~{selection_end}) 确定")
    print(f"2. 策略表现在回测期 ({backtest_start}~{backtest_end}) 验证")
    print(f"3. 这样避免了使用未来信息（前视偏差）")
    print("="*60)


if __name__ == "__main__":
    main()
