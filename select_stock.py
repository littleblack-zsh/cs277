import pandas as pd
import numpy as np
from data_clean import clean_dayk, load_all_dayk

def compute_stock_metrics(
    df_clean: pd.DataFrame,
    lookback_days: int = 252,
    min_trading_days: int = 200
) -> pd.DataFrame:
    """
    根据清洗后的 df_clean 计算每个 SecurityID 的流动性、波动率、收益率指标。
    """
    df = df_clean.copy()
    # 确保按日期排序
    df = df.sort_values(["SecurityID", "Date"])

    # 先按股票分组
    grouped = df.groupby("SecurityID", group_keys=False)

    # 只保留最近 lookback_days 的数据来计算波动率和收益率（防止历史太久）
    def last_n_days(g):
        return g.tail(lookback_days)

    df_recent = grouped.apply(last_n_days)

    # 计算每个股票的基本指标
    metrics = df_recent.groupby("SecurityID").agg(
        trading_days = ("Date", "nunique"),
        avg_amount   = ("Amount", "mean"),
        # 日收益率波动
        ret_std      = ("return", "std"),
        # 收盘价头尾
        first_price  = ("LastPx", "first"),
        last_price   = ("LastPx", "last")
    ).reset_index()

    # 年化波动率（假设一年 252 个交易日）
    metrics["vol_annual"] = metrics["ret_std"] * np.sqrt(252)

    # 简单年化收益：近 lookback_days 的整体收益 * (252 / trading_days)
    # 避免除零
    metrics["gross_return"] = metrics["last_price"] / metrics["first_price"] - 1
    metrics["ret_annual"] = metrics["gross_return"] * (252 / metrics["trading_days"].clip(lower=1))

    # 过滤掉交易天数太少的股票
    metrics = metrics[metrics["trading_days"] >= min_trading_days].copy()

    # 丢掉明显异常：价格为 0 或 NaN 的
    metrics = metrics.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["avg_amount", "vol_annual", "ret_annual"]
    )

    return metrics


def select_grid_candidates(
    metrics: pd.DataFrame,
    min_avg_amount: float = 2e7,   # 日均成交额下限
    vol_range: tuple = (0.15, 0.5),
    max_abs_ret: float = 0.10      # 年化收益绝对值上限
) -> pd.DataFrame:
    """
    在指标表 metrics 上筛选适合做网格交易的标的。
    """
    cond_liquidity = metrics["avg_amount"] >= min_avg_amount
    cond_vol = (metrics["vol_annual"] >= vol_range[0]) & (metrics["vol_annual"] <= vol_range[1])
    cond_ret = metrics["ret_annual"].abs() <= max_abs_ret

    candidates = metrics[cond_liquidity & cond_vol & cond_ret].copy()
    return candidates


import matplotlib.pyplot as plt

def plot_liquidity_hist(metrics: pd.DataFrame, bins: int = 50):
    """
    画日均成交额的分布直方图（对数刻度更清楚）
    """
    plt.figure(figsize=(8, 5))
    x = metrics["avg_amount"]

    plt.hist(x, bins=bins)
    plt.xlabel("Average Daily Amount")
    plt.ylabel("Number of Stocks")
    plt.ylim(0, 6000)
    plt.title("Distribution of Average Daily Trading Amount")

    plt.xscale("log")  # 成交额跨度通常很大，用 log 更直观
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_vol_return_scatter(metrics: pd.DataFrame, candidates: pd.DataFrame = None):
    """
    画所有股票的 年化收益率 vs 年化波动率 散点图，
    并高亮标注筛选出的 candidates（如果提供的话）。
    """
    plt.figure(figsize=(8, 6))

    # 全部股票
    plt.scatter(
        metrics["ret_annual"],
        metrics["vol_annual"],
        s=10,
        alpha=0.4,
        label="All Stocks"
    )

    if candidates is not None and not candidates.empty:
        plt.scatter(
            candidates["ret_annual"],
            candidates["vol_annual"],
            s=30,
            alpha=0.9,
            marker="x",
            label="Grid Candidates"
        )

        # 也可以顺手标几个代码（只标前几个，避免太乱）
        for _, row in candidates.head(10).iterrows():
            plt.text(row["ret_annual"], row["vol_annual"],
                     row["SecurityID"], fontsize=8,
                     ha="left", va="bottom")

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Annualized Return")
    plt.ylabel("Annualized Volatility")
    plt.xlim(-1.5,1.5)
    plt.ylim(0,1.5)
    plt.title("Volatility vs Annualized Return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_price_series(df_clean: pd.DataFrame, security_id: str):
    """
    画某一只股票的收盘价时间序列
    """
    df_one = df_clean[df_clean["SecurityID"] == security_id].sort_values("Date")
    plt.figure(figsize=(10, 4))
    plt.plot(df_one["Date"], df_one["LastPx"])
    plt.title(f"Price Series of {security_id}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_return_hist(df_clean: pd.DataFrame, security_id: str, bins: int = 50):
    """
    画某一只股票的日收益率分布直方图
    """
    df_one = df_clean[df_clean["SecurityID"] == security_id].sort_values("Date")
    r = df_one["return"].dropna()

    plt.figure(figsize=(8, 4))
    plt.hist(r, bins=bins)
    plt.title(f"Daily Return Distribution of {security_id}")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def merge_plots(df_clean: pd.DataFrame, security_id: str):
    """
    画某一只股票的收盘价时间序列和日收益率分布直方图，合并为一个1*2的图。
    """
    df_one = df_clean[df_clean["SecurityID"] == security_id].sort_values("Date")
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # 收盘价时间序列
    axs[0].plot(df_one["Date"], df_one["LastPx"])
    axs[0].set_title(f"Price Series of {security_id}")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Close Price")
    axs[0].grid(alpha=0.3)

    # 日收益率分布直方图
    r = df_one["return"].dropna()
    axs[1].hist(r, bins=50)
    axs[1].set_title(f"Daily Return Distribution of {security_id}")
    axs[1].set_xlabel("Daily Return")
    axs[1].set_ylabel("Frequency")
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root_dir = r'D:\研究生课\cs277\project\上证信息数据2024\his_sh1_201907-202406'
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
        # example_id = candidates.iloc[0]["SecurityID"]
        example_id= '000008'
        # plot_price_series(df_clean, example_id
        
        # plot_return_hist(df_clean, example_id)
        merge_plots(df_clean,example_id)