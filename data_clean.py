from pathlib import Path
import pandas as pd
import numpy as np

def load_all_dayk(root_dir: str) -> pd.DataFrame:
    """
    遍历 root_dir 下所有子目录中的 Day.csv，并合并成一个大表
    """
    root = Path(root_dir)
    files = sorted(root.rglob("Day.csv"))
    dfs = []
    for f in files:
        df_day = pd.read_csv(f,dtype={0: str})
        dfs.append(df_day)
    if not dfs:
        raise ValueError("没有找到任何 Day.csv 文件，请检查路径。")
    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def clean_dayk(df: pd.DataFrame) -> pd.DataFrame:
    """
    对合并后的日 K 线数据做统一清洗
    （逻辑与单文件版相同，只是去掉了与“单日”相关的限制）
    """
    df = df.copy()

    # 1) 日期格式统一
    df["Date"] = pd.to_datetime(df["DateTime"].astype(str),
                                format="%Y%m%d",
                                errors="coerce")
    df = df.dropna(subset=["Date"])

    # 2) 数值列统一
    numeric_cols = [
        "PreClosePx", "OpenPx", "HighPx", "LowPx", "LastPx",
        "Volume", "Amount"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 3) 过滤明显错误
    for c in ["OpenPx", "HighPx", "LowPx", "LastPx"]:
        if c in df.columns:
            df.loc[df[c] <= 0, c] = np.nan

    if "Volume" in df.columns:
        df.loc[df["Volume"] < 0, "Volume"] = np.nan
    if "Amount" in df.columns:
        df.loc[df["Amount"] < 0, "Amount"] = np.nan

    # 4) 高低价一致性检查
    max_oc = df[["OpenPx", "LastPx"]].max(axis=1)
    min_oc = df[["OpenPx", "LastPx"]].min(axis=1)

    invalid_high = df["HighPx"] < max_oc
    invalid_low = df["LowPx"] > min_oc

    df.loc[invalid_high, "HighPx"] = np.nan
    df.loc[invalid_low, "LowPx"] = np.nan

    df = df.dropna(subset=["OpenPx", "HighPx", "LowPx", "LastPx"])

    # 5) 标记并删除停牌
    df["is_suspended"] = False
    if "Volume" in df.columns and "Amount" in df.columns:
        df["is_suspended"] = (df["Volume"] == 0) & (df["Amount"] == 0)
        df = df[~df["is_suspended"]].copy()

    # 6) 排序 & 去重
    df = df.sort_values(["SecurityID", "Date"])
    df = df.drop_duplicates(subset=["SecurityID", "Date"], keep="last")

    # 7) 计算日收益率
    df["return"] = df.groupby("SecurityID")["LastPx"].pct_change()

    return df


# ===== 实际调用示例 =====
# root_dir = r'D:\研究生课\cs277\project\上证信息数据2024\his_sh1_201907-202406'
# df_raw = load_all_dayk(root_dir)
# df_clean = clean_dayk(df_raw)
# df_clean.to_csv("dayk_cleaned.csv", index=False)
