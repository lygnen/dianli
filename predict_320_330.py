import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
from advanced_features import load_and_prepare_data, create_features_for_hour

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def train_and_predict():
    # 1. 加载基础数据
    price_df, wind_df, pv_df, load_df, thermal_df = load_and_prepare_data('汇总.xlsx')
    
    # 找到真实电价非空的行作为训练集
    price_not_null = price_df.notna().all(axis=1)
    train_indices = price_df[price_not_null].index
    
    print(f"训练数据结束于: {train_indices[-1].strftime('%Y-%m-%d')}")
    
    # 预测目标日期: 2026-03-20 至 2026-03-30
    target_dates = pd.date_range(start='2026-03-20', end='2026-03-30')
    print(f"需要预测的日期: {target_dates[0].strftime('%Y-%m-%d')} 至 {target_dates[-1].strftime('%Y-%m-%d')}")
    
    # 我们将训练24个模型
    models = {}
    print("\n正在训练24小时的独立模型 (基于历史所有有效数据)...")
    for hour in range(24):
        X, y = create_features_for_hour(price_df.loc[train_indices], 
                                        wind_df.loc[train_indices], 
                                        pv_df.loc[train_indices], 
                                        load_df.loc[train_indices], 
                                        thermal_df.loc[train_indices], 
                                        hour)
        # 使用一组合理的默认参数，保证速度和精度
        model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)
        models[hour] = model
        if (hour + 1) % 6 == 0:
            print(f"已完成 {hour + 1} 个小时模型的训练")

    print("\n模型训练完成，开始自回归预测...")
    
    # 复制一个用于迭代更新的电价表
    # 截取到我们要预测的最后一天
    end_date = target_dates[-1]
    all_dates = price_df.index[price_df.index <= end_date]
    
    iter_price_df = price_df.loc[all_dates].copy()
    iter_wind_df = wind_df.loc[all_dates]
    iter_pv_df = pv_df.loc[all_dates]
    iter_load_df = load_df.loc[all_dates]
    iter_thermal_df = thermal_df.loc[all_dates]
    
    PRICE_MIN = 200
    
    # 逐天预测
    for current_date in target_dates:
        print(f"正在预测: {current_date.strftime('%Y-%m-%d')}")
        
        # 提取到 current_date 的子集
        sub_dates = iter_price_df.index[iter_price_df.index <= current_date]
        
        day_preds = []
        for hour in range(24):
            X_all, _ = create_features_for_hour(
                iter_price_df.loc[sub_dates],
                iter_wind_df.loc[sub_dates],
                iter_pv_df.loc[sub_dates],
                iter_load_df.loc[sub_dates],
                iter_thermal_df.loc[sub_dates],
                hour
            )
            # 提取当天的特征
            X_current = X_all.loc[[current_date]]
            
            # 预测
            pred = models[hour].predict(X_current)[0]
            pred = max(pred, PRICE_MIN)
            day_preds.append(pred)
            
        # 将预测结果填入 iter_price_df，供下一天的特征工程使用 (滞后特征等)
        iter_price_df.loc[current_date] = day_preds

    # 提取预测结果
    result_df = iter_price_df.loc[target_dates]
    result_df.columns = [f'{h+1}时' for h in range(24)]
    
    output_file = '预测结果_0320_0330.xlsx'
    result_df.to_excel(output_file, sheet_name='预测电价')
    print(f"\n预测完成！结果已保存至 '{output_file}'")
    
    # 绘制预测结果的热力图或折线图
    plt.figure(figsize=(12, 6))
    for i, row in result_df.iterrows():
        plt.plot(range(1, 25), row.values, marker='o', label=i.strftime('%m-%d'))
    
    plt.title('2026年3月20日至3月30日 电价预测曲线', fontsize=15)
    plt.xlabel('时段', fontsize=12)
    plt.ylabel('电价', fontsize=12)
    plt.xticks(range(1, 25))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('预测曲线_0320_0330.png', dpi=300)
    print("预测曲线已保存至 '预测曲线_0320_0330.png'")

if __name__ == '__main__':
    train_and_predict()