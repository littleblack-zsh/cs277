import os
import pandas as pd

# 假设主文件夹路径
main_dir = r'D:\研究生课\cs277\project\上证信息数据2024\his_sh1_201907-202406'  # 修改为你的实际路径
# 你自定义的10个header
new_header_12 =  ['SecurityID', 'DateTime', 'PreClosePx','OpenPx','HighPx','LowPx','LastPx','Volume','Amount','IOPV','fp_Volume','fp_Amount']
new_header_13 =  ['SecurityID', 'DateTime', 'PreClosePx','OpenPx','HighPx','LowPx','LastPx','Volume','Amount','IOPV','fp_Volume','fp_Amount','AvgPx']
for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)
    day_csv = os.path.join(subfolder_path, 'Day.csv')

    # 检查Day.csv是否存在
    if os.path.isfile(day_csv):
        # 读取时不指定header，把所有行都读进来
        df = pd.read_csv(day_csv, header=None,skiprows=1,dtype={0: str})
        # 如果数据列数为10
        # if df.shape[1] == 12:
            
        # # 替换第一行为自定义header
        #     df.columns = new_header_12
        # # 去掉第一行（原来的header行，多余了）
        # elif df.shape[1] ==13:
        #     df.columns = new_header_13
        # elif df.shape[1] >13:
        #     df = df.iloc[:, :13]
        #     df.columns = new_header_13
        if df.shape[1] ==13:
           df.columns = new_header_13
        elif df.shape[1] >13:
            df = df.iloc[:, :13]
            df.columns = new_header_13
            
        else:
            raise ValueError(f'Unexpected number of columns ({df.shape[1]}) in file: {day_csv}')
            
        
        # 保存，index不保存，header写入新的
        df.to_csv(day_csv, index=False)
        print(f'处理成功: {day_csv}')
        
    else:
        print(f'文件不存在: {day_csv}')