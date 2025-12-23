import data_clean
import select_stock
df_raw = load_all_dayk(root_dir)
df_clean = clean_dayk(df_raw)

# 2. 计算每个股票的指标
metrics = compute_stock_metrics(df_clean, lookback_days=252, min_trading_days=200)

# 3. 筛选网格候选标的
candidates = select_grid_candidates(
    metrics,
    min_avg_amount=2e7,
    vol_range=(0.15, 0.5),
    max_abs_ret=0.10
)

print("候选标的列表：")
print(candidates[["SecurityID", "trading_days", "avg_amount", "vol_annual", "ret_annual"]].head())

# 4. 画图（放到报告里）
plot_liquidity_hist(metrics)
plot_vol_return_scatter(metrics, candidates)

# 5. 选一两只具体股票做时间序列图
if not candidates.empty:
    example_id = candidates.iloc[0]["SecurityID"]
    plot_price_series(df_clean, example_id)
    plot_return_hist(df_clean, example_id)