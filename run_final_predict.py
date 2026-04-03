import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
import optuna
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from advanced_features import load_and_prepare_data, create_features_for_hour

PRICE_MIN = 200

def predict_future_with_lgb():
    print("="*60)
    print("使用 lgb_model.py 的算法逻辑进行 03-20 到 03-30 数据的预测")
    print("="*60)

    # 1. 加载数据
    price_df, wind_df, pv_df, load_df, thermal_df = load_and_prepare_data('汇总.xlsx')
    
    # 找到真实电价非空的行作为训练集
    price_not_null = price_df.notna().all(axis=1)
    train_indices = price_df[price_not_null].index
    
    print(f"\n训练数据范围: {train_indices[0].strftime('%Y-%m-%d')} 至 {train_indices[-1].strftime('%Y-%m-%d')}")
    
    # 预测目标日期
    target_dates = pd.date_range(start='2026-03-20', end='2026-03-30')
    print(f"需要预测的日期: {target_dates[0].strftime('%Y-%m-%d')} 至 {target_dates[-1].strftime('%Y-%m-%d')}\n")
    
    models = {}
    print("正在训练24小时的独立 LightGBM 模型 (使用历史所有有效数据)...")
    for hour in range(24):
        X, y = create_features_for_hour(price_df.loc[train_indices], 
                                        wind_df.loc[train_indices], 
                                        pv_df.loc[train_indices], 
                                        load_df.loc[train_indices], 
                                        thermal_df.loc[train_indices], 
                                        hour)
        
        # 为了高效，直接使用一组通常表现良好的固定参数
        # (如果在生产环境，可以复用 lgb_model.py 中的 optuna 搜索逻辑)
        model = lgb.LGBMRegressor(
            objective='regression',
            metric='mae',
            boosting_type='gbdt',
            n_estimators=150,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=3,
            min_child_samples=10,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)
        models[hour] = model
        
        if (hour + 1) % 6 == 0:
            print(f"已完成 {hour + 1}/24 个小时模型的训练")

    print("\n模型训练完成，开始逐日滚动预测 (自回归)...")
    
    # 准备用于迭代的数据集
    end_date = target_dates[-1]
    all_dates = price_df.index[price_df.index <= end_date]
    
    iter_price_df = price_df.loc[all_dates].copy()
    iter_wind_df = wind_df.loc[all_dates]
    iter_pv_df = pv_df.loc[all_dates]
    iter_load_df = load_df.loc[all_dates]
    iter_thermal_df = thermal_df.loc[all_dates]
    
    for current_date in target_dates:
        print(f"正在预测: {current_date.strftime('%Y-%m-%d')}")
        
        # 截取截至到当前日期的数据
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
            # 取最后一天的特征 (即 current_date)
            X_current = X_all.loc[[current_date]]
            
            pred = models[hour].predict(X_current)[0]
            # 业务下限控制
            pred = max(pred, PRICE_MIN)
            day_preds.append(pred)
            
        # 将当天的预测结果写入迭代用的电价表中，供下一天构建滞后特征
        iter_price_df.loc[current_date] = day_preds

    # 提取最终预测结果
    result_df = iter_price_df.loc[target_dates]
    result_df.columns = [f'{h+1}时' for h in range(24)]
    
    output_excel = 'LightGBM_最终预测_0320_0330.xlsx'
    result_df.to_excel(output_excel, sheet_name='预测电价')
    print(f"\n预测完成！结果已成功保存至 '{output_excel}'")
    
    # 绘图
    plt.figure(figsize=(14, 6))
    for i, row in result_df.iterrows():
        plt.plot(range(1, 25), row.values, marker='o', markersize=4, label=i.strftime('%m-%d'))
        
    plt.axhline(y=PRICE_MIN, color='gray', linestyle=':', alpha=0.5, label=f'下限{PRICE_MIN}')
    
    plt.title('LightGBM 3月20日至3月30日 滚动预测电价走势', fontsize=15)
    plt.xlabel('时段', fontsize=12)
    plt.ylabel('电价', fontsize=12)
    plt.xticks(range(1, 25))
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_png = 'LightGBM_最终预测_0320_0330.png'
    plt.savefig(output_png, dpi=300)
    print(f"预测曲线已成功保存至 '{output_png}'")

if __name__ == '__main__':
    predict_future_with_lgb()