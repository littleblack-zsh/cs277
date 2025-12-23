# Grid Trading System - AI Agent Instructions

## 项目概述

这是一个课程项目：基于历史数据的股票网格交易回测系统，用于研究和分析网格交易策略的表现。核心流程包括数据清洗、波动率筛选、网格回测和性能评估。

**重要提示**: `grid-trading-system-template/` 是参考的外部项目，可以忽略。`grid2.py` 是简化的教学版本，不在主工作流中使用。

## 核心架构

### 数据流架构
1. **数据加载** ([data_clean.py](my-grid/data_clean.py)): `load_all_dayk()` → 遍历日期文件夹(YYYYMMDD/)下的 `Day.csv`
2. **数据清洗**: `clean_dayk()` → 处理价格/成交量异常、停牌过滤、日收益率计算
3. **标的筛选** ([select_stock.py](my-grid/select_stock.py)): 基于流动性、波动率、收益率筛选适合网格交易的股票
4. **策略回测** ([grid.py](my-grid/grid.py)): 网格交易模拟 vs 买入持有基准对比
5. **结果分析**: 交易明细CSV导出、资产曲线可视化、参数敏感性扫描

### 关键组件

- **数据源**: 上证数据存储在 `上证信息数据2024/his_sh1_201907-202406/YYYYMMDD/Day.csv`
  - 目录结构: 每个日期一个文件夹，包含所有股票的当日K线数据
  - 字段: `SecurityID`, `DateTime`, `OpenPx`, `HighPx`, `LowPx`, `LastPx`, `Volume`, `Amount`
  - 数据时间范围: 2019-07 至 2024-06

- **核心回测模块**: [grid.py](my-grid/grid.py) - 功能完整的网格回测（推荐间距、参数扫描、评估指标、可视化）

## 关键约定

### 数据处理约定
- **日期格式**: 统一转换为 `pd.to_datetime()` 对象，原始格式为 `YYYYMMDD` (无分隔符)
- **价格字段**: `LastPx` 为收盘价（核心字段），清洗时检查 `HighPx >= max(OpenPx, LastPx)` 和 `LowPx <= min(OpenPx, LastPx)`
- **停牌过滤**: 当日 `Volume == 0 AND Amount == 0` 则标记为停牌并删除
- **异常值处理**: 价格 <= 0 或 NaN 的行直接删除，不尝试修复

### 网格交易逻辑 (grid.py)

**固定网格策略**:
```python
# 核心参数
grid_step: float          # 网格间距（价格单位，如 0.5 元）
total_capital: float      # 初始资金（如 100000）
trade_fraction: float     # 每次交易占总资金比例（如 0.02 = 2%）
qty_per_trade: int        # 每次交易固定股数（基于 total_capital * trade_fraction / mid_price）

# 交易触发条件
当前价格 >= last_trade_price + grid_step → 卖出 qty_per_trade 股（如果持仓充足）
当前价格 <= last_trade_price - grid_step → 买入 qty_per_trade 股（如果现金充足）
```

**ATR自适应网格策略** (推荐):
```python
# ATR (Average True Range) 自适应调整网格间距
run_adaptive_atr_grid_backtest(
    ohlc_df,              # 需要包含 OpenPx, HighPx, LowPx, LastPx
    atr_period=14,        # ATR计算周期（默认14天）
    atr_multiplier=1.0,   # 网格间距 = ATR × multiplier
    recalc_interval=5     # 每5天重新计算ATR并调整间距
)

# 优势：
# - 波动大时自动扩大网格间距，减少交易频率
# - 波动小时自动缩小网格间距，捕捉更多小幅波动
# - 动态适应市场状态
```

### 标的筛选标准 (select_stock.py)
```python
compute_stock_metrics()  # 计算流动性、波动率、收益率
select_grid_candidates() # 筛选条件:
    - 日均成交额 >= 2e7 (2千万)
    - 年化波动率在 [0.15, 0.5] 区间（15%-50%）
    - 年化收益率绝对值 <= 0.10（避免强趋势股票）
```

## 典型工作流

### 正确的时间分割方法 ⚠️ 避免前视偏差

```python
# ===== 关键：分离选股期和回测期，避免用未来信息选股 =====
# 数据时间线: 2019-07 至 2024-06
# 选股期 (in-sample): 2019-07-01 ~ 2022-12-31 (用于筛选标的和确定参数)
# 回测期 (out-of-sample): 2023-01-01 ~ 2024-06-20 (真实模拟交易)

from data_clean import load_all_dayk, clean_dayk
from select_stock import compute_stock_metrics, select_grid_candidates
from grid import get_price_series, suggest_grid_step, run_grid_backtest, calculate_performance_metrics

root_dir = r'D:\研究生课\cs277\my-grid\上证信息数据2024\his_sh1_201907-202406'
df_raw = load_all_dayk(root_dir)
df_clean = clean_dayk(df_raw)

# Step 1: 在选股期筛选候选标的
df_selection = df_clean[(df_clean["Date"] >= "2019-07-01") & (df_clean["Date"] <= "2022-12-31")]
metrics = compute_stock_metrics(df_selection, lookback_days=252)
candidates = select_grid_candidates(metrics, min_avg_amount=2e7, vol_range=(0.15, 0.5))

# Step 2: 对筛选出的股票在回测期进行模拟
for security_id in candidates["SecurityID"].head(5):
    price_df = get_price_series(df_clean, security_id, 
                                start_date="2023-01-01",  # 回测期开始
                                end_date="2024-06-20")
    
    grid_step = suggest_grid_step(price_df, factor=1.5)
    equity_df, trades_df = run_grid_backtest(price_df, grid_step, total_capital=100000)
    
    # Step 3: 计算评估指标
    metrics_result = calculate_performance_metrics(equity_df, trades_df, price_df)
    print(f"{security_id} 回测结果: 年化收益={metrics_result['annual_return']:.2%}, "
          f"夏普比率={metrics_result['sharpe_ratio']:.2f}, "
          f"最大回撤={metrics_result['max_drawdown']:.2%}")
    
    trades_df.to_csv(f"{security_id}_grid_trades.csv", index=False)
```

### 参数优化（在选股期数据上进行）
```python
# 在选股期扫描不同网格间距，找到最优参数
result_df = sweep_grid_steps(price_df_selection, base_step=auto_step, 
                             min_factor=0.5, max_factor=2.0, num=10)
# 选出最优参数后，在回测期验证
```

## 回测评估指标

项目实现了以下专业评估指标（见 `calculate_performance_metrics()`）：

- **年化收益率**: 考虑复利的年化回报
- **夏普比率**: 风险调整后收益 (假设无风险利率 3%)
- **最大回撤 (MDD)**: 从峰值到谷值的最大跌幅
- **卡玛比率**: 年化收益 / 最大回撤（风险收益比）
- **胜率**: 盈利交易占比
- **交易频率**: 年化交易次数
- **超额收益**: 相对买入持有的超额年化收益

使用示例：
```python
metrics = calculate_performance_metrics(equity_df, trades_df, price_df)
print_performance_report(metrics, security_id="000008")
```

## 开发注意事项

### ⚠️ 避免前视偏差（Look-Ahead Bias）
- **问题**: 用包含未来信息的数据选股，会导致回测结果虚高
- **解决方案**: 严格分离选股期（in-sample）和回测期（out-of-sample）
- **时间分割建议**: 选股期 2019-2022，回测期 2023-2024
- **检查清单**: 
  - ✅ 选股指标计算时只用选股期数据
  - ✅ 回测时不能用回测期数据重新筛选标的
  - ✅ 网格参数优化在选股期完成

### 路径处理
- **使用绝对路径**: 所有数据目录使用 `r'D:\研究生课\cs277\...'` 形式的原始字符串
- **Path 对象**: 文件遍历使用 `from pathlib import Path`, `Path.rglob("Day.csv")`

### 性能优化
- `load_all_dayk()` 会加载**所有日期的所有股票**数据（数据量大），清洗后约数百MB
- 建议缓存清洗结果: `df_clean.to_csv("dayk_cleaned.csv")` 后续直接读取
- 筛选单个股票后再做计算，避免在大表上执行复杂操作

### 可视化设置
- 使用 `matplotlib.pyplot`, 中文字体已配置（grid-trading-system-template 中设置了 `KaiTi`）
- 常用图表: 资产曲线对比、交易点标记、参数敏感性分析、流动性分布直方图

### 输出文件命名规范
- 交易记录CSV: `{SecurityID}_grid_trades.csv` (如 `000008_grid_trades.csv`)
- 清洗后数据: `dayk_cleaned.csv`

## 关键文件引用
- 数据处理核心: [data_clean.py](my-grid/data_clean.py) - `load_all_dayk()`, `clean_dayk()`
- 回测引擎: [grid.py](my-grid/grid.py) - `run_grid_backtest()`, `run_adaptive_atr_grid_backtest()`, `suggest_grid_step()`, `calculate_performance_metrics()`
- 标的筛选: [select_stock.py](my-grid/select_stock.py) - `compute_stock_metrics()`, `select_grid_candidates()`
- 可视化分析: [data_vis.py](my-grid/data_vis.py) - 流动性分布、波动率散点图等
- **ATR自适应示例**: [example_atr_adaptive_grid.py](my-grid/example_atr_adaptive_grid.py) - 完整的ATR策略演示和对比

## 策略选择建议

- **固定网格**: 适合震荡市、波动性相对稳定的股票
- **ATR自适应网格**: 适合波动性变化较大的股票，能自动适应市场状态
- 建议先用固定网格建立基准，再用ATR策略对比改进效果

## 不要做的事
- ❌ **不要用回测期数据选股**（前视偏差是最严重的错误）
- ❌ 不要修改原始数据文件 (Day.csv)
- ❌ 不要在未清洗数据上直接执行回测（必须先调用 `clean_dayk()`）
- ❌ 不要假设所有股票数据完整（检查 `df.empty` 和处理日期缺失）
- ❌ 不要在回测期重新调整参数（参数优化应在选股期完成）
