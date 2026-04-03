import pandas as pd
import numpy as np

def load_and_prepare_data(file_path='汇总.xlsx'):
    """
    加载汇总.xlsx中的各个Sheet数据并进行对齐
    """
    print(f"正在从 {file_path} 加载数据...")
    price_df = pd.read_excel(file_path, sheet_name='日前电价', index_col=0)
    wind_df = pd.read_excel(file_path, sheet_name='风电24', index_col=0)
    pv_df = pd.read_excel(file_path, sheet_name='光伏24', index_col=0)
    load_df = pd.read_excel(file_path, sheet_name='负荷24', index_col=0)
    thermal_df = pd.read_excel(file_path, sheet_name='负载24', index_col=0)

    # 尝试转换索引为 datetime 类型，以方便提取时间特征
    try:
        price_df.index = pd.to_datetime(price_df.index)
        wind_df.index = pd.to_datetime(wind_df.index)
        pv_df.index = pd.to_datetime(pv_df.index)
        load_df.index = pd.to_datetime(load_df.index)
        thermal_df.index = pd.to_datetime(thermal_df.index)
    except Exception as e:
        print("时间索引转换失败，部分时间周期特征可能受影响:", e)

    return price_df, wind_df, pv_df, load_df, thermal_df

def create_features_for_hour(price_df, wind_df, pv_df, load_df, thermal_df, hour_idx):
    """
    为一天中的特定小时(0-23)构建深度特征工程
    """
    df = pd.DataFrame(index=price_df.index)
    
    # ==========================
    # 1. 目标变量与当天已知协变量
    # ==========================
    target = price_df.iloc[:, hour_idx]
    
    df['wind'] = wind_df.iloc[:, hour_idx]
    df['pv'] = pv_df.iloc[:, hour_idx]
    df['load'] = load_df.iloc[:, hour_idx]
    df['thermal'] = thermal_df.iloc[:, hour_idx]

    # ==========================
    # 2. 历史电价滞后特征 (Lag)
    # ==========================
    # 前1天、前2天、前3天、前7天的相同时段电价
    df['price_lag_1d'] = price_df.iloc[:, hour_idx].shift(1)
    df['price_lag_2d'] = price_df.iloc[:, hour_idx].shift(2)
    df['price_lag_3d'] = price_df.iloc[:, hour_idx].shift(3)
    df['price_lag_7d'] = price_df.iloc[:, hour_idx].shift(7)

    # 前一天的24点电价作为一个强烈的基准锚点
    df['price_yesterday_last'] = price_df.iloc[:, 23].shift(1)
    
    # 距离当前预测点最近的一个已知电价（如果预测1点，则是前一天的24点，若预测8点，同样只用前一天24点来防止数据穿越）
    # 为保证严格的“日前”预测，只能使用前一天的信息

    # ==========================
    # 3. 滚动统计特征 (Rolling Windows)
    # ==========================
    # 过去3天均值
    df['price_roll_mean_3d'] = price_df.iloc[:, hour_idx].shift(1).rolling(window=3).mean()
    
    # 过去7天各项统计值
    roll_7d = price_df.iloc[:, hour_idx].shift(1).rolling(window=7)
    df['price_roll_mean_7d'] = roll_7d.mean()
    df['price_roll_std_7d'] = roll_7d.std()
    df['price_roll_max_7d'] = roll_7d.max()
    df['price_roll_min_7d'] = roll_7d.min()

    # ==========================
    # 4. 差分特征 (Diff)
    # ==========================
    # 日环比变化
    df['price_diff_1d'] = df['price_lag_1d'] - df['price_lag_2d']
    # 周环比变化
    df['price_diff_7d'] = df['price_lag_1d'] - df['price_lag_7d']

    # 协变量差分特征 (当天预报与前一天实际的差值)
    df['wind_diff'] = df['wind'] - wind_df.iloc[:, hour_idx].shift(1)
    df['pv_diff'] = df['pv'] - pv_df.iloc[:, hour_idx].shift(1)
    df['load_diff'] = df['load'] - load_df.iloc[:, hour_idx].shift(1)

    # ==========================
    # 5. 时间周期编码特征 (Time Encodings)
    # ==========================
    if isinstance(df.index, pd.DatetimeIndex):
        day_of_week = df.index.dayofweek
        month = df.index.month
        
        # 捕捉星期和月份的周期性
        df['sin_dow'] = np.sin(2 * np.pi * day_of_week / 7)
        df['cos_dow'] = np.cos(2 * np.pi * day_of_week / 7)
        df['sin_month'] = np.sin(2 * np.pi * month / 12)
        df['cos_month'] = np.cos(2 * np.pi * month / 12)
        
        # 是否周末
        df['is_weekend'] = day_of_week.isin([5, 6]).astype(int)

    # 丢弃因 Lag 和 Rolling 产生的 NaN 数据
    valid_idx = df.dropna().index
    
    return df.loc[valid_idx], target.loc[valid_idx]
