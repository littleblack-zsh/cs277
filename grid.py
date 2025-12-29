import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_clean import clean_dayk, load_all_dayk

#获取某个股票在某时间段的价格序列
def get_price_series(df_clean: pd.DataFrame,
                     security_id: str,
                     start_date: str,
                     end_date: str,
                     include_ohlc: bool = False) -> pd.DataFrame:
    """
    返回某个 SecurityID 在指定时间区间内的价格序列。
    start_date, end_date: 'YYYYMMDD' 或 'YYYY-MM-DD' 格式都可以。
    include_ohlc: 是否返回完整的OHLC数据（用于ATR计算）
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    df_one = df_clean[df_clean["SecurityID"] == security_id].copy()
    df_one = df_one[(df_one["Date"] >= start) & (df_one["Date"] <= end)]
    df_one = df_one.sort_values("Date").reset_index(drop=True)

    if df_one.empty:
        raise ValueError(f"{security_id} 在 {start_date}~{end_date} 没有数据")

    if include_ohlc:
        # 返回完整的OHLC数据用于ATR计算
        return df_one[["Date", "OpenPx", "HighPx", "LowPx", "LastPx"]]
    else:
        return df_one[["Date", "LastPx"]]

# ==========================
# 自动推荐网格间距
# ==========================
def suggest_grid_step(price_df: pd.DataFrame,
                      factor: float = 1.5) -> float:
    """
    根据历史收盘价序列，推荐一个网格间距（单位：价格，比如人民币）。
    factor: 放大倍数，越大网格越宽，交易频率越低。
    """
    p = price_df["LastPx"].values
    if len(p) < 2:
        raise ValueError("价格序列太短，无法建议网格间距")

    # 每日绝对价格变化
    abs_change = np.abs(np.diff(p))
    median_abs_change = np.median(abs_change)

    mid_price = np.median(p)

    # 初步建议
    raw_step = factor * median_abs_change

    # 控制在 [0.5%, 5%] * 中位价 之间
    min_step = 0.005 * mid_price
    max_step = 0.05 * mid_price
    grid_step = np.clip(raw_step, min_step, max_step)

    return float(grid_step)

# ==========================
# ATR (Average True Range) 计算
# ==========================
def calculate_atr(ohlc_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    计算ATR（平均真实波动范围）
    
    参数:
        ohlc_df: 包含 OpenPx, HighPx, LowPx, LastPx 的DataFrame
        period: ATR计算周期（默认14天）
    
    返回:
        ATR序列
    """
    high = ohlc_df["HighPx"].values
    low = ohlc_df["LowPx"].values
    close = ohlc_df["LastPx"].values
    
    # 计算True Range
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]  # 第一天没有前一天收盘价
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # 计算ATR（使用EMA平滑）
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    
    return atr


def suggest_grid_step_atr(ohlc_df: pd.DataFrame, 
                          atr_period: int = 14,
                          atr_multiplier: float = 1.0) -> float:
    """
    基于ATR推荐网格间距
    
    参数:
        ohlc_df: 包含OHLC数据的DataFrame
        atr_period: ATR计算周期
        atr_multiplier: ATR乘数（网格间距 = ATR * multiplier）
    
    返回:
        推荐的网格间距
    """
    atr = calculate_atr(ohlc_df, period=atr_period)
    # 使用最近的ATR值
    recent_atr = atr.iloc[-min(20, len(atr)):].mean()  # 最近20天ATR的平均
    grid_step = float(recent_atr * atr_multiplier)
    
    return grid_step

# ==========================
# 5. 网格交易回测
# ==========================
def run_grid_backtest(price_df: pd.DataFrame,
                      grid_step: float,
                      total_capital: float = 100000.0,
                      trade_fraction: float = 0.02):
    """
    对单只股票的日线收盘价做网格交易模拟。
    price_df: 包含 ['Date','LastPx'] 的 DataFrame，按日期排序
    grid_step: 网格间距（价格单位）
    total_capital: 初始资金
    trade_fraction: 每次交易使用的资金占总资金的比例（决定每次买/卖的股数）
    返回:
        equity_df: 每日资产曲线
        trades_df: 每笔成交记录
    """
    df = price_df.copy().sort_values("Date").reset_index(drop=True)
    prices = df["LastPx"].values
    dates = df["Date"].values

    # 根据历史波动确定网格上下界
    price_min = float(prices.min())
    price_max = float(prices.max())
    min_level = (price_min // grid_step) * grid_step
    max_level = ((price_max // grid_step) + 1) * grid_step

    # 初始资金与持仓
    cash = total_capital
    position = 0  # 持有股数

    # 每次交易使用的股数（固定）
    mid_price = float(prices[0])
    qty_per_trade = int(total_capital * trade_fraction / mid_price)
    if qty_per_trade <= 0:
        qty_per_trade = 1  # 至少 1 股/份

    # 初始 last_trade_price 设为第一天收盘价，并对齐到网格上
    last_trade_price = round(mid_price / grid_step) * grid_step
    last_trade_price = float(np.clip(last_trade_price, min_level, max_level))

    equity_records = []
    trade_records = []

    for i in range(len(df)):
        date = dates[i]
        p = float(prices[i])

        # 当天开盘前不做任何事，使用收盘价来判断是否触发网格
        # 计算与上次成交价的价差
        diff = p - last_trade_price
        steps = int(diff // grid_step) if grid_step != 0 else 0

        # 逐步执行每一个“网格步长”的交易
        # 为了简单，用每个网格的价格近似为 last_trade_price ± grid_step
        while steps != 0:
            # 向上突破 → 卖出
            if steps > 0:
                next_price = last_trade_price + grid_step
                if next_price > max_level:
                    break  # 超出上边界则不再交易

                # 有仓才卖
                if position >= qty_per_trade:
                    trade_price = next_price
                    position -= qty_per_trade
                    cash += qty_per_trade * trade_price
                    last_trade_price = next_price

                    trade_records.append({
                        "Date": date,
                        "Side": "SELL",
                        "Price": trade_price,
                        "Qty": qty_per_trade,
                        "CashAfter": cash,
                        "PositionAfter": position
                    })

                    steps -= 1
                else:
                    # 没仓可卖
                    break

            # 向下突破 → 买入
            elif steps < 0:
                next_price = last_trade_price - grid_step
                if next_price < min_level:
                    break  # 超出下边界不再买

                trade_price = next_price
                cost = qty_per_trade * trade_price
                if cost <= cash:
                    position += qty_per_trade
                    cash -= cost
                    last_trade_price = next_price

                    trade_records.append({
                        "Date": date,
                        "Side": "BUY",
                        "Price": trade_price,
                        "Qty": qty_per_trade,
                        "CashAfter": cash,
                        "PositionAfter": position
                    })

                    steps += 1
                else:
                    # 现金不足
                    break

        # 记录当日资产情况（按收盘价估值）
        equity = cash + position * p
        equity_records.append({
            "Date": date,
            "ClosePrice": p,
            "Cash": cash,
            "Position": position,
            "Equity": equity
        })

    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trade_records)

    return equity_df, trades_df


def buy_and_hold_backtest(price_df: pd.DataFrame,
                          total_capital: float = 100000.0) -> pd.DataFrame:
    """
    简单买入持有回测：第一天按收盘价全仓买入，之后一直持有。
    返回 equity_df：每天的市值曲线
    """
    df = price_df.copy().sort_values("Date").reset_index(drop=True)
    prices = df["LastPx"].values
    dates = df["Date"].values

    first_price = float(prices[0])
    qty = int(total_capital // first_price)
    cash = total_capital - qty * first_price

    equity_records = []
    for date, p in zip(dates, prices):
        equity = cash + qty * float(p)
        equity_records.append({
            "Date": date,
            "ClosePrice": float(p),
            "Cash": cash,
            "Position": qty,
            "Equity": equity
        })

    equity_df = pd.DataFrame(equity_records)
    return equity_df


def run_adaptive_atr_grid_backtest(ohlc_df: pd.DataFrame,
                                   atr_period: int = 14,
                                   atr_multiplier: float = 1.0,
                                   total_capital: float = 100000.0,
                                   trade_fraction: float = 0.02,
                                   recalc_interval: int = 5):
    """
    基于ATR自适应调整网格间距的网格交易回测
    
    参数:
        ohlc_df: 包含 Date, OpenPx, HighPx, LowPx, LastPx 的DataFrame
        atr_period: ATR计算周期（默认14天）
        atr_multiplier: ATR乘数，用于确定网格间距（grid_step = ATR * multiplier）
        total_capital: 初始资金
        trade_fraction: 每次交易使用的资金比例
        recalc_interval: 每隔多少天重新计算ATR和调整网格间距（默认5天）
    
    返回:
        equity_df: 每日资产曲线
        trades_df: 每笔成交记录
    """
    df = ohlc_df.copy().sort_values("Date").reset_index(drop=True)
    
    # 计算整个周期的ATR
    atr_series = calculate_atr(df, period=atr_period)
    
    prices = df["LastPx"].values
    dates = df["Date"].values
    
    # 初始资金与持仓
    cash = total_capital
    position = 0
    
    # 初始网格间距
    grid_step = float(atr_series.iloc[max(atr_period, 0)] * atr_multiplier)
    
    # 每次交易股数（会随grid_step动态调整）
    mid_price = float(prices[0])
    qty_per_trade = int(total_capital * trade_fraction / mid_price)
    if qty_per_trade <= 0:
        qty_per_trade = 1
    
    # 初始交易价格
    last_trade_price = float(prices[0])
    
    equity_records = []
    trade_records = []
    
    for i in range(len(df)):
        date = dates[i]
        p = float(prices[i])
        
        # 定期重新计算网格间距（基于当前ATR）
        if i > 0 and i % recalc_interval == 0 and i >= atr_period:
            # 使用最近的ATR值
            current_atr = float(atr_series.iloc[i])
            grid_step = current_atr * atr_multiplier
            # 可选：重新调整每次交易股数
            # qty_per_trade = int(total_capital * trade_fraction / p)
            # if qty_per_trade <= 0:
            #     qty_per_trade = 1
        
        # 网格边界（动态调整）
        price_min = float(prices[:i+1].min())
        price_max = float(prices[:i+1].max())
        min_level = (price_min // grid_step) * grid_step
        max_level = ((price_max // grid_step) + 1) * grid_step
        
        # 计算与上次成交价的价差
        diff = p - last_trade_price
        steps = int(diff // grid_step) if grid_step != 0 else 0
        
        # 逐步执行每一个网格的交易
        while steps != 0:
            # 向上突破 → 卖出
            if steps > 0:
                next_price = last_trade_price + grid_step
                if next_price > max_level:
                    break
                
                if position >= qty_per_trade:
                    trade_price = next_price
                    position -= qty_per_trade
                    cash += qty_per_trade * trade_price
                    last_trade_price = next_price
                    
                    trade_records.append({
                        "Date": date,
                        "Side": "SELL",
                        "Price": trade_price,
                        "Qty": qty_per_trade,
                        "GridStep": grid_step,
                        "ATR": float(atr_series.iloc[i]),
                        "CashAfter": cash,
                        "PositionAfter": position
                    })
                    
                    steps -= 1
                else:
                    break
            
            # 向下突破 → 买入
            elif steps < 0:
                next_price = last_trade_price - grid_step
                if next_price < min_level:
                    break
                
                trade_price = next_price
                cost = qty_per_trade * trade_price
                if cost <= cash:
                    position += qty_per_trade
                    cash -= cost
                    last_trade_price = next_price
                    
                    trade_records.append({
                        "Date": date,
                        "Side": "BUY",
                        "Price": trade_price,
                        "Qty": qty_per_trade,
                        "GridStep": grid_step,
                        "ATR": float(atr_series.iloc[i]),
                        "CashAfter": cash,
                        "PositionAfter": position
                    })
                    
                    steps += 1
                else:
                    break
        
        # 记录当日资产情况
        equity = cash + position * p
        equity_records.append({
            "Date": date,
            "ClosePrice": p,
            "Cash": cash,
            "Position": position,
            "Equity": equity,
            "GridStep": grid_step,
            "ATR": float(atr_series.iloc[i])
        })
    
    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trade_records)
    
    return equity_df, trades_df


def sweep_grid_steps(price_df: pd.DataFrame,
                     base_step: float,
                     total_capital: float = 100000.0,
                     trade_fraction: float = 0.02,
                     min_factor: float = 0.5,
                     max_factor: float = 2.0,
                     num: int = 10) -> pd.DataFrame:
    """
    在 base_step 周围扫描多种网格间距，统计每种间距的最终收益和交易次数。
    返回 result_df，包含：
        factor, step, final_equity, pnl, pnl_pct, num_trades
    """
    factors = np.linspace(min_factor, max_factor, num)
    results = []

    for f in factors:
        step = base_step * f
        equity_df, trades_df = run_grid_backtest(
            price_df,
            grid_step=step,
            total_capital=total_capital,
            trade_fraction=trade_fraction
        )
        start_eq = equity_df["Equity"].iloc[0]
        end_eq = equity_df["Equity"].iloc[-1]
        pnl = end_eq - start_eq
        pnl_pct = pnl / start_eq * 100.0

        results.append({
            "factor": f,
            "step": step,
            "final_equity": end_eq,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "num_trades": len(trades_df)
        })

    result_df = pd.DataFrame(results)
    return result_df


def calculate_performance_metrics(equity_df: pd.DataFrame, 
                                 trades_df: pd.DataFrame,
                                 price_df: pd.DataFrame,
                                 risk_free_rate: float = 0.03) -> dict:
    """
    计算回测的各项评估指标
    
    参数:
        equity_df: 资产曲线 DataFrame (包含 Date, Equity 列)
        trades_df: 交易记录 DataFrame
        price_df: 价格序列 DataFrame (用于计算买入持有基准)
        risk_free_rate: 无风险利率 (默认3%)
    
    返回:
        包含各项指标的字典
    """
    # 基本收益指标
    initial_equity = equity_df["Equity"].iloc[0]
    final_equity = equity_df["Equity"].iloc[-1]
    total_return = (final_equity - initial_equity) / initial_equity
    
    # 计算年化收益率
    days = (equity_df["Date"].iloc[-1] - equity_df["Date"].iloc[0]).days
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    
    # 计算日收益率序列
    equity_df = equity_df.copy()
    equity_df["daily_return"] = equity_df["Equity"].pct_change()
    daily_returns = equity_df["daily_return"].dropna()
    
    # 夏普比率 (假设252个交易日)
    if len(daily_returns) > 0:
        excess_return = daily_returns.mean() * 252 - risk_free_rate
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
    else:
        sharpe_ratio = 0
    
    # 最大回撤 (Maximum Drawdown)
    equity_series = equity_df["Equity"]
    cummax = equity_series.expanding().max()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # 卡玛比率 (Calmar Ratio) = 年化收益率 / 最大回撤绝对值
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 交易统计
    num_trades = len(trades_df)
    if num_trades > 0:
        # 计算每笔交易的盈亏 (简化版：相邻买卖配对)
        buys = trades_df[trades_df["Side"] == "BUY"]
        sells = trades_df[trades_df["Side"] == "SELL"]
        
        # 胜率和盈亏比 (需要配对交易，这里简化处理)
        if len(sells) > 0 and len(buys) > 0:
            avg_buy_price = buys["Price"].mean()
            avg_sell_price = sells["Price"].mean()
            win_rate = (sells["Price"] > avg_buy_price).mean() if len(sells) > 0 else 0
        else:
            win_rate = 0
        
        # 交易频率 (年化)
        trade_frequency = num_trades / years if years > 0 else 0
    else:
        win_rate = 0
        trade_frequency = 0
    
    # 买入持有基准对比
    bh_equity = buy_and_hold_backtest(price_df, initial_equity)
    bh_return = (bh_equity["Equity"].iloc[-1] - bh_equity["Equity"].iloc[0]) / bh_equity["Equity"].iloc[0]
    bh_annual_return = (1 + bh_return) ** (1 / years) - 1 if years > 0 else 0
    excess_vs_bh = annual_return - bh_annual_return
    
    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "volatility": daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0,
        "num_trades": num_trades,
        "trade_frequency": trade_frequency,
        "win_rate": win_rate,
        "bh_annual_return": bh_annual_return,
        "excess_vs_bh": excess_vs_bh,
        "backtest_days": days,
        "backtest_years": years
    }


def print_performance_report(metrics: dict, security_id: str = ""):
    """
    打印格式化的回测报告
    """
    print(f"\n{'='*60}")
    print(f"  回测绩效报告 - {security_id}")
    print(f"{'='*60}")
    print(f"回测周期: {metrics['backtest_days']:.0f} 天 ({metrics['backtest_years']:.2f} 年)")
    print(f"\n【收益指标】")
    print(f"  总收益率:        {metrics['total_return']:>8.2%}")
    print(f"  年化收益率:      {metrics['annual_return']:>8.2%}")
    print(f"  买入持有年化:    {metrics['bh_annual_return']:>8.2%}")
    print(f"  超额收益:        {metrics['excess_vs_bh']:>8.2%}")
    print(f"\n【风险指标】")
    print(f"  年化波动率:      {metrics['volatility']:>8.2%}")
    print(f"  最大回撤:        {metrics['max_drawdown']:>8.2%}")
    print(f"  夏普比率:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"  卡玛比率:        {metrics['calmar_ratio']:>8.2f}")
    print(f"\n【交易统计】")
    print(f"  交易次数:        {metrics['num_trades']:>8.0f}")
    print(f"  年化交易频率:    {metrics['trade_frequency']:>8.1f} 次/年")
    print(f"  胜率:            {metrics['win_rate']:>8.2%}")
    print(f"{'='*60}\n")


def plot_step_performance(result_df: pd.DataFrame):
    """
    画 网格间距(step) vs 收益率(pnl_pct) & 交易次数 的曲线。
    """
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(result_df["step"], result_df["pnl_pct"], marker="o")
    ax1.set_xlabel("Grid Step (Price)")
    ax1.set_ylabel("PnL (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    # 第二个 y 轴画交易次数
    ax2 = ax1.twinx()
    ax2.plot(result_df["step"], result_df["num_trades"], marker="x", linestyle="--", color="tab:orange")
    ax2.set_ylabel("Number of Trades", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Grid Step vs Performance")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ===== 数据加载 =====
    root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
    df_raw = load_all_dayk(root_dir)
    df_clean = clean_dayk(df_raw)

    security_id = "000008" 
    total_capital = 100000.0

    # ===== 重要：避免前视偏差的正确做法 =====
    # 选股期（用于筛选标的和确定参数）: 2019-07-01 ~ 2022-12-31
    # 回测期（真实模拟交易）: 2023-01-01 ~ 2024-06-20
    # 
    # 错误做法：使用全部数据既做选股又做回测
    # 正确做法：在选股期筛选标的，在回测期评估表现
    
    selection_start = "2019-07-01"  # 选股期开始
    selection_end = "2022-12-31"    # 选股期结束
    backtest_start = "2023-01-01"   # 回测期开始（不与选股期重叠！）
    backtest_end = "2024-06-20"     # 回测期结束

    # Step 1: 在选股期获取价格数据（用于确定网格参数）
    price_df_selection = get_price_series(df_clean, security_id, selection_start, selection_end)
    
    # Step 2: 基于选股期数据推荐网格间距
    auto_step = suggest_grid_step(price_df_selection, factor=1.5)
    print(f"\n基于选股期 ({selection_start} ~ {selection_end}) 数据:")
    print(f"{security_id} 建议网格间距: {auto_step:.3f}")

    # Step 3: 在回测期（out-of-sample）执行回测
    price_df_backtest = get_price_series(df_clean, security_id, backtest_start, backtest_end)
    
    print(f"\n在回测期 ({backtest_start} ~ {backtest_end}) 执行模拟交易...")
    equity_grid, trades_grid = run_grid_backtest(
        price_df_backtest,
        grid_step=auto_step,
        total_capital=total_capital,
        trade_fraction=0.1
    )

    # Step 4: 买入持有基准
    equity_bh = buy_and_hold_backtest(price_df_backtest, total_capital=total_capital)

    # Step 5: 计算详细的评估指标
    metrics = calculate_performance_metrics(equity_grid, trades_grid, price_df_backtest)
    print_performance_report(metrics, security_id)

    # 保存交易记录
    trades_grid.to_csv(f"{security_id}_grid_trades.csv", index=False)
    print(f"✓ 交易记录已保存到 {security_id}_grid_trades.csv")

    # Step 6: 可选 - 在选股期进行参数扫描（用于优化参数）
    print("\n【可选】在选股期进行参数优化扫描:")
    result_df = sweep_grid_steps(
        price_df_selection,
        base_step=auto_step,
        total_capital=total_capital,
        trade_fraction=0.1,
        min_factor=0.5,
        max_factor=2.0,
        num=10
    )
    print("\n不同网格间距在选股期的表现：")
    print(result_df[["step", "pnl_pct", "num_trades"]].head())

    # Step 7: 可视化（回测期数据）
    plt.figure(figsize=(12, 5))
    plt.plot(equity_grid["Date"], equity_grid["Equity"], label="Grid Trading", linewidth=2)
    plt.plot(equity_bh["Date"], equity_bh["Equity"], label="Buy & Hold", linewidth=2, linestyle="--")
    plt.legend(fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Equity (¥)", fontsize=12)
    plt.title(f"Backtest Period Equity Curve - {security_id}", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 参数敏感性分析（选股期数据）
    plot_step_performance(result_df)


# ==========================
# Moving Average Crossover Strategy
# ==========================
def run_ma_crossover_backtest(
    price_df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    total_capital: float = 100000.0
) -> tuple:
    """
    移动平均交叉策略回测
    
    策略逻辑：
    - 当短期均线上穿长期均线（金叉）时，全仓买入
    - 当短期均线下穿长期均线（死叉）时，全仓卖出
    
    参数：
        price_df: 价格数据，必须包含 'Date' 和 'LastPx' 列
        short_window: 短期移动平均窗口（天数）
        long_window: 长期移动平均窗口（天数）
        total_capital: 初始资金
    
    返回：
        equity_df: 资产曲线DataFrame
        trades_df: 交易记录DataFrame
    """
    df = price_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    
    # 计算移动平均线
    df['MA_Short'] = df['LastPx'].rolling(window=short_window).mean()
    df['MA_Long'] = df['LastPx'].rolling(window=long_window).mean()
    
    # 初始化
    cash = total_capital
    position = 0.0  # 持有股数
    
    equity_records = []
    trade_records = []
    
    prev_signal = None  # 上一个信号：None, 'buy', 'sell'
    
    for i in range(len(df)):
        date = df.loc[i, 'Date']
        price = df.loc[i, 'LastPx']
        ma_short = df.loc[i, 'MA_Short']
        ma_long = df.loc[i, 'MA_Long']
        
        # 需要足够的数据才能产生信号
        if pd.isna(ma_short) or pd.isna(ma_long):
            equity = cash + position * price
            equity_records.append({
                'Date': date,
                'ClosePrice': price,
                'Cash': cash,
                'Position': position,
                'Equity': equity
            })
            continue
        
        # 检测交叉信号
        signal = None
        if ma_short > ma_long:
            signal = 'buy'
        elif ma_short < ma_long:
            signal = 'sell'
        
        # 执行交易
        if signal == 'buy' and prev_signal != 'buy' and position == 0:
            # 金叉且当前空仓：全仓买入
            shares_to_buy = cash // price  # 只买整数股
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                cash -= cost
                position += shares_to_buy
                
                trade_records.append({
                    'Date': date,
                    'Side': 'BUY',
                    'Price': price,
                    'Shares': shares_to_buy,
                    'Amount': cost,
                    'Cash': cash,
                    'Position': position
                })
        
        elif signal == 'sell' and prev_signal != 'sell' and position > 0:
            # 死叉且当前持仓：全仓卖出
            shares_to_sell = position
            revenue = shares_to_sell * price
            cash += revenue
            position = 0
            
            trade_records.append({
                'Date': date,
                'Side': 'SELL',
                'Price': price,
                'Shares': shares_to_sell,
                'Amount': revenue,
                'Cash': cash,
                'Position': position
            })
        
        prev_signal = signal
        
        # 记录当日资产
        equity = cash + position * price
        equity_records.append({
            'Date': date,
            'ClosePrice': price,
            'Cash': cash,
            'Position': position,
            'Equity': equity
        })
    
    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trade_records)
    
    return equity_df, trades_df
    