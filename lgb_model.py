import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
import optuna
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)  # 静默optuna输出

# 设置绘图支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from advanced_features import load_and_prepare_data, create_features_for_hour

# ===============================
# 超参数与全局设置
# ===============================
TEST_DAYS = 30  # 测试集天数 (时序分割，最后 N 天)
VAL_DAYS = 15   # 验证集天数 (用于 Optuna 超参数搜索)
N_TRIALS = 15   # 每个小时 Optuna 尝试的次数，权衡时间与精度
PRICE_MIN = 200 # 预测的最低电价限制 (基于业务背景)

def optimize_lgb(X_train, y_train, X_val, y_val):
    """
    使用 Optuna 自动搜索最佳超参数
    """
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    return study.best_params

def train_and_evaluate():
    # 1. 加载基础数据
    price_df, wind_df, pv_df, load_df, thermal_df = load_and_prepare_data('汇总.xlsx')
    
    # 过滤掉标签全是 NaN 的天数（只用有真实电价的数据训练测试）
    price_not_null = price_df.notna().all(axis=1)
    valid_indices = price_df[price_not_null].index
    
    # 获取有效数据（用于训练和测试，忽略未知预测天）
    price_df = price_df.loc[valid_indices]
    wind_df = wind_df.loc[valid_indices]
    pv_df = pv_df.loc[valid_indices]
    load_df = load_df.loc[valid_indices]
    thermal_df = thermal_df.loc[valid_indices]
    
    total_days = len(price_df)
    if total_days <= TEST_DAYS + VAL_DAYS + 10:
        print("数据量太小，无法满足测试集与验证集分割，请减小 TEST_DAYS 或 VAL_DAYS")
        return
    
    print(f"数据加载完成。有效数据天数: {total_days}")
    print(f"测试集划分: 最后 {TEST_DAYS} 天")

    all_test_actuals = []
    all_test_preds = []
    models = {}
    maes, rmses, r2s = [], [], []

    print("\n" + "="*60)
    print("开始训练 24 小时独立 LightGBM 模型 (自动超参数搜索)")
    print("="*60)

    # 2. 遍历24个小时，独立建模
    for hour in range(24):
        # 提取当前小时的高级特征
        X, y = create_features_for_hour(price_df, wind_df, pv_df, load_df, thermal_df, hour)
        
        # 3. 严格的时序分割 (Time-Series Split)
        # 训练集+验证集，以及测试集
        X_train_full = X.iloc[:-TEST_DAYS]
        y_train_full = y.iloc[:-TEST_DAYS]
        X_test = X.iloc[-TEST_DAYS:]
        y_test = y.iloc[-TEST_DAYS:]

        # 从训练集中再剥离出验证集用于 Optuna 调参
        X_train = X_train_full.iloc[:-VAL_DAYS]
        y_train = y_train_full.iloc[:-VAL_DAYS]
        X_val = X_train_full.iloc[-VAL_DAYS:]
        y_val = y_train_full.iloc[-VAL_DAYS:]

        # 4. Optuna 超参数搜索
        best_params = optimize_lgb(X_train, y_train, X_val, y_val)
        best_params['random_state'] = 42
        best_params['verbosity'] = -1
        
        # 5. 用全部已知数据(Train+Val)重新拟合最终模型
        final_model = lgb.LGBMRegressor(**best_params)
        final_model.fit(X_train_full, y_train_full)
        
        # 6. 在测试集上评估
        preds = final_model.predict(X_test)
        
        # 加上业务上的最低电价限制
        preds = np.maximum(preds, PRICE_MIN)
        
        # 计算评估指标
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)
        
        models[hour] = final_model
        all_test_actuals.append(y_test.values)
        all_test_preds.append(preds)
        
        print(f"[{hour+1:02d} 时段] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")

    # ===============================
    # 整体结果分析与输出
    # ===============================
    # 转置矩阵，使得每一行代表一天，每一列代表24个小时
    all_test_actuals = np.array(all_test_actuals).T  # Shape: (TEST_DAYS, 24)
    all_test_preds = np.array(all_test_preds).T      # Shape: (TEST_DAYS, 24)

    overall_mae = mean_absolute_error(all_test_actuals, all_test_preds)
    overall_rmse = np.sqrt(mean_squared_error(all_test_actuals, all_test_preds))
    overall_r2 = r2_score(all_test_actuals.flatten(), all_test_preds.flatten())

    print("\n" + "="*60)
    print("模型整体评估结果 (测试集)")
    print(f"整体 MAE : {overall_mae:.2f}")
    print(f"整体 RMSE: {overall_rmse:.2f}")
    print(f"整体 R²  : {overall_r2:.4f}")
    print("="*60)

    # 获取测试集的真实日期索引
    dates = X.index[-TEST_DAYS:]
    columns = [f'{h+1}时' for h in range(24)]

    actual_df = pd.DataFrame(all_test_actuals, index=dates, columns=columns)
    pred_df = pd.DataFrame(all_test_preds, index=dates, columns=columns)

    # 保存预测结果到 Excel
    output_excel = 'LightGBM_高精度预测结果.xlsx'
    with pd.ExcelWriter(output_excel) as writer:
        actual_df.to_excel(writer, sheet_name='实际值')
        pred_df.to_excel(writer, sheet_name='预测值')
        
        metrics_df = pd.DataFrame({
            '时段': [f'{h+1}时' for h in range(24)],
            'MAE': maes,
            'RMSE': rmses,
            'R²': r2s
        })
        metrics_df.to_excel(writer, sheet_name='各时段评估', index=False)
        
        # 整体评估写入信息表
        info_df = pd.DataFrame({
            '指标': ['整体MAE', '整体RMSE', '整体R²'],
            '数值': [overall_mae, overall_rmse, overall_r2]
        })
        info_df.to_excel(writer, sheet_name='总体评估', index=False)

    print(f"\n结果已成功保存至 '{output_excel}'")

    # ===============================
    # 绘制可视化图表
    # ===============================
    plt.figure(figsize=(14, 6))
    
    # 图 1: 各时段 MAE 趋势
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 25), maes, 'o-', color='orange', linewidth=2, markersize=6, label='LightGBM MAE')
    plt.xlabel('时段', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('LightGBM 各时段预测 MAE', fontsize=14)
    plt.xticks(range(1, 25))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 标记误差最大的几个点以便排查
    for i, m in enumerate(maes):
        plt.text(i+1, m+0.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8)

    # 图 2: 随机选取一天的真实值与预测值对比
    plt.subplot(1, 2, 2)
    random_day_idx = np.random.randint(0, TEST_DAYS)
    plt.plot(range(1, 25), all_test_actuals[random_day_idx], 'b-o', label='实际值', linewidth=2)
    plt.plot(range(1, 25), all_test_preds[random_day_idx], 'r--s', label='预测值', linewidth=2)
    plt.axhline(y=PRICE_MIN, color='gray', linestyle=':', alpha=0.5, label=f'下限{PRICE_MIN}')
    
    random_date_str = dates[random_day_idx].strftime('%Y-%m-%d')
    plt.title(f'单日预测曲线对比 ({random_date_str})', fontsize=14)
    plt.xlabel('时段', fontsize=12)
    plt.ylabel('电价', fontsize=12)
    plt.xticks(range(1, 25))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    output_png = 'LightGBM_评估图表.png'
    plt.savefig(output_png, dpi=300)
    print(f"评估图表已保存至 '{output_png}'")
    # plt.show()

if __name__ == '__main__':
    train_and_evaluate()
