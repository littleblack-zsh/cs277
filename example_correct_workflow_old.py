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
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)

    # 2. 设定标的与回测区间
    security_id = "000008"  # 你手动指定的股票
    start_date = "2023-01-01"
    end_date = "2024-06-20"
    total_capital = 100000.0

    # 3. 获取价格序列
    price_df = get_price_series(df_clean, security_id, start_date, end_date)

    # 4. 自动推荐基准网格间距
    base_step = suggest_grid_step(price_df, factor=1.5)
    print(f"{security_id} 推荐基准网格间距: {base_step:.3f}")

    # 5. 参数扫描（不同网格间距的回测表现）
    result_df = sweep_grid_steps(
        price_df,
        base_step=base_step,
        total_capital=total_capital,
        trade_fraction=0.1,
        min_factor=0.5,
        max_factor=2.0,
        num=10
    )
    print(result_df[["factor", "step", "final_equity", "pnl_pct", "num_trades"]])

    # 可视化参数敏感性
    plot_step_performance(result_df)

    # 6. 选取最优参数回测并输出详细指标
    best_row = result_df.loc[result_df["pnl_pct"].idxmax()]
    best_step = best_row["step"]
    print(f"\n最优网格间距: {best_step:.3f}")

    equity_df, trades_df = run_grid_backtest(
        price_df, grid_step=best_step, total_capital=total_capital, trade_fraction=0.1
    )
    metrics = calculate_performance_metrics(equity_df, trades_df, price_df)
    print_performance_report(metrics, security_id)

if __name__ == "__main__":
    main()