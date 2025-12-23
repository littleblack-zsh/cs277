"""
正确的网格交易回测工作流 - 避免前视偏差

这个示例展示了如何正确分离选股期和回测期，避免使用未来信息进行选股。
"""

import pandas as pd
from data_clean import load_all_dayk, clean_dayk
from grid import (
    get_price_series, suggest_grid_step, sweep_grid_steps,
    run_grid_backtest, calculate_performance_metrics, print_performance_report, plot_step_performance
)

def main():
    # 1. 数据加载与清洗
    print("正在加载数据...")
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)
    print("✓ 数据加载完成")

    # 2. ===== 关键：时间分割，避免前视偏差 =====
    security_id = "000008"  # 你手动指定的股票
    
    # 选股期（用于确定策略参数）
    selection_start = "2020-01-01"
    selection_end = "2022-12-31"
    
    # 回测期（用于验证策略表现）
    backtest_start = "2023-01-01"
    backtest_end = "2024-06-20"
    
    total_capital = 100000.0
    
    print(f"\n{'='*60}")
    print(f"时间分割（避免前视偏差）")
    print(f"{'='*60}")
    print(f"选股期: {selection_start} ~ {selection_end}  (用于确定参数)")
    print(f"回测期: {backtest_start} ~ {backtest_end}  (验证策略表现)")

    # 3. 在选股期获取价格序列，用于确定策略参数
    print(f"\n在选股期确定策略参数...")
    price_selection = get_price_series(df_clean, security_id, selection_start, selection_end)

    # 4. 基于选股期数据推荐基准网格间距
    base_step = suggest_grid_step(price_selection, factor=1.5)
    print(f"  - 基于选股期推荐网格间距: {base_step:.3f}")

    # 5. 在选股期进行参数扫描（寻找最优参数）
    print(f"\n在选股期进行参数扫描...")
    result_df_selection = sweep_grid_steps(
        price_selection,
        base_step=base_step,
        total_capital=total_capital,
        trade_fraction=0.1,
        min_factor=0.5,
        max_factor=2.0,
        num=10
    )
    print("\n选股期不同网格间距表现：")
    print(result_df_selection[["factor", "step", "final_equity", "pnl_pct", "num_trades"]])

    # 可视化参数敏感性（选股期）
    plot_step_performance(result_df_selection)

    # 6. 选取最优参数（基于选股期表现）
    best_row = result_df_selection.loc[result_df_selection["pnl_pct"].idxmax()]
    best_step = best_row["step"]
    print(f"\n✓ 选股期最优网格间距: {best_step:.3f} (factor={best_row['factor']:.2f})")

    # 7. ===== 在回测期使用选股期确定的最优参数 =====
    print(f"\n{'='*60}")
    print(f"在回测期验证策略（使用选股期确定的参数）")
    print(f"{'='*60}")
    
    # 获取回测期数据
    price_backtest = get_price_series(df_clean, security_id, backtest_start, backtest_end)
    print(f"回测期交易日数: {len(price_backtest)} 天")

    # 使用最优参数在回测期执行
    equity_df, trades_df = run_grid_backtest(
        price_backtest, 
        grid_step=best_step,  # 使用选股期确定的最优参数
        total_capital=total_capital, 
        trade_fraction=0.1
    )
    
    # 8. 计算回测期评估指标
    metrics = calculate_performance_metrics(equity_df, trades_df, price_backtest)
    print_performance_report(metrics, security_id)
    
    # 保存交易记录
    trades_df.to_csv(f"{security_id}_optimized_grid_trades.csv", index=False)
    print(f"\n✓ 交易记录已保存到 {security_id}_optimized_grid_trades.csv")
    
    print(f"\n{'='*60}")
    print("说明：")
    print(f"1. 在选股期 ({selection_start}~{selection_end}) 通过参数扫描确定最优网格间距")
    print(f"2. 在回测期 ({backtest_start}~{backtest_end}) 使用该参数验证策略表现")
    print(f"3. 这样避免了使用未来信息优化参数（前视偏差）")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()