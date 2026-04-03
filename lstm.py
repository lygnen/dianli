import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')
import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def add_price_lag_features(price_df, lag=1):
    """添加前一天电价作为特征"""
    n_days, n_hours = price_df.shape
    lag_features = np.zeros((n_days, n_hours))

    # shift后，前1天的数据会变成NaN
    shifted = price_df.shift(lag).values
    # 将NaN填充为0
    lag_features = np.nan_to_num(shifted, 0)

    return lag_features


def create_lstm_model(input_shape, output_steps):
    """创建LSTM模型"""
    model = Sequential([
        LSTM(32, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dropout(0.2),
        Dense(output_steps)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


if __name__ == '__main__':
    # ====================== 1. 读取数据 ======================
    print("正在读取数据...")

    # 读取所有副表格
    price_df = pd.read_excel('汇总.xlsx', sheet_name='日前电价', index_col=0)
    wind_df = pd.read_excel('汇总.xlsx', sheet_name='风电24', index_col=0)
    pv_df = pd.read_excel('汇总.xlsx', sheet_name='光伏24', index_col=0)
    load_df = pd.read_excel('汇总.xlsx', sheet_name='负荷24', index_col=0)
    thermal_df = pd.read_excel('汇总.xlsx', sheet_name='负载24', index_col=0)

    total_days = len(price_df)
    print(f"数据时间范围: {price_df.index[0]} 到 {price_df.index[-1]}")
    print(f"共 {total_days} 天")

    # ====================== 2. 添加前一天电价特征 ======================
    print("\n正在添加前一天电价特征...")
    price_lag1 = add_price_lag_features(price_df, lag=1)
    print(f"前一天电价特征形状: {price_lag1.shape}")

    # ====================== 3. 时段定义 ======================
    time_periods = {
        '非中午': [(0, 8), (17, 23)],  # 凌晨+晚上：0-8点和17-23点
        '中午': (9, 16)  # 中午：9-16点（8小时）
    }

    period_hours = {
        '非中午': 9 + 7,  # 0-8点(9小时) + 17-23点(7小时) = 16小时
        '中午': 8  # 9-16点(8小时)
    }

    print(f"\n时段划分:")
    print(f"非中午时段: 0-8点 + 17-23点 (共{period_hours['非中午']}小时)")
    print(f"中午时段: 9-16点 (共{period_hours['中午']}小时)")

    # ====================== 4. 构建特征 ======================
    n_hours = 24
    # 原始特征(4) + 前一天电价(1) = 5个特征
    n_features = 4 + 1

    # 初始化特征数组
    feature_array = np.zeros((total_days, n_hours, n_features))

    # 填充原始特征 (索引0-3)
    feature_array[:, :, 0] = wind_df.values
    feature_array[:, :, 1] = pv_df.values
    feature_array[:, :, 2] = load_df.values
    feature_array[:, :, 3] = thermal_df.values

    # 填充前一天电价特征 (索引4)
    feature_array[:, :, 4] = price_lag1

    target_array = price_df.values

    print(f"\n特征数组形状: {feature_array.shape}")
    print(f"特征说明:")
    print(f"  0-3: 风电、光伏、负荷、火电空间")
    print(f"  4: 前一天电价")


    # ====================== 5. 创建序列样本 ======================
    def create_sequences(features, targets):
        """用第i天的特征预测第i+1天的电价"""
        X, y = [], []
        # 从第1天开始，因为第0天的前一天电价特征是0（填充值）
        for i in range(1, len(features) - 1):
            X.append(features[i])
            y.append(targets[i + 1])
        return np.array(X), np.array(y)


    X, y = create_sequences(feature_array, target_array)
    n_samples = len(X)
    print(f"总样本数: {n_samples}")


    # ====================== 6. 为每个时段提取数据 ======================
    def extract_period_data(X, y, hour_ranges):
        """从全天数据中提取指定时段的数据"""
        if isinstance(hour_ranges, list):
            hour_indices = []
            for start, end in hour_ranges:
                hour_indices.extend(range(start, end + 1))
            hour_indices = sorted(hour_indices)
        else:
            start, end = hour_ranges
            hour_indices = range(start, end + 1)

        y_period = y[:, hour_indices]
        return X, y_period, len(hour_indices)


    period_data = {}
    for period_name, hour_ranges in time_periods.items():
        X_period, y_period, output_steps = extract_period_data(X, y, hour_ranges)
        period_data[period_name] = {
            'X': X_period,
            'y': y_period,
            'output_steps': output_steps,
            'hour_ranges': hour_ranges
        }
        print(f"{period_name}时段: 输出步长={output_steps}")

    # ====================== 7. 滚动预测 ======================
    print("\n开始滚动预测...")

    min_train_days = 4  # 最小训练天数（因为只需要前一天电价，4天就够了）
    all_predictions = {period_name: [] for period_name in period_data}
    all_actuals = {period_name: [] for period_name in period_data}
    prediction_dates = []

    for test_idx in range(min_train_days, n_samples):
        print(f"\n正在预测第{test_idx + 1}天 (用前{test_idx}天数据)...")

        for period_name, data in period_data.items():
            X_period = data['X']
            y_period = data['y']
            output_steps = data['output_steps']

            # 划分训练集和测试集
            X_train = X_period[:test_idx]
            y_train = y_period[:test_idx]
            X_test = X_period[test_idx:test_idx + 1]
            y_test = y_period[test_idx]

            # 特征标准化
            n_feat = X_train.shape[2]
            X_train_2d = X_train.reshape(-1, n_feat)
            X_test_2d = X_test.reshape(-1, n_feat)

            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train_2d).reshape(X_train.shape)
            X_test_scaled = scaler_X.transform(X_test_2d).reshape(X_test.shape)

            # 目标值标准化
            y_train_2d = y_train.reshape(-1, 1)
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train_2d).reshape(y_train.shape)

            # 创建并训练模型
            model = create_lstm_model((24, n_feat), output_steps)

            early_stop = EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )

            model.fit(
                X_train_scaled, y_train_scaled,
                epochs=50,
                batch_size=min(16, len(X_train_scaled)),
                callbacks=[early_stop],
                verbose=0
            )

            # 预测
            y_pred_scaled = model.predict(X_test_scaled, verbose=0)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(y_pred_scaled.shape)

            all_predictions[period_name].append(y_pred[0])
            all_actuals[period_name].append(y_test)

        prediction_dates.append(test_idx + 1)

    # ====================== 8. 合并各时段预测结果 ======================
    n_predictions = len(prediction_dates)
    y_pred_full = np.zeros((n_predictions, 24))
    y_actual_full = np.zeros((n_predictions, 24))

    # 处理非中午时段
    non_noon_indices = list(range(0, 9)) + list(range(17, 24))
    for i in range(n_predictions):
        y_pred_full[i, non_noon_indices] = all_predictions['非中午'][i]
        y_actual_full[i, non_noon_indices] = all_actuals['非中午'][i]

    # 处理中午时段
    noon_indices = list(range(9, 17))
    for i in range(n_predictions):
        y_pred_full[i, noon_indices] = all_predictions['中午'][i]
        y_actual_full[i, noon_indices] = all_actuals['中午'][i]


    # ====================== 9. 评估模型 ======================
    def evaluate_model(y_true, y_pred, name):
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        # 移除NaN值
        mask = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
        y_true_flat = y_true_flat[mask]
        y_pred_flat = y_pred_flat[mask]

        if len(y_true_flat) == 0:
            return np.nan, np.nan, np.nan, np.nan

        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        r2 = r2_score(y_true_flat, y_pred_flat)

        mask_mape = y_true_flat > 1
        if np.sum(mask_mape) > 0:
            mape = np.mean(np.abs((y_true_flat[mask_mape] - y_pred_flat[mask_mape]) / y_true_flat[mask_mape])) * 100
        else:
            mape = np.nan

        return mae, rmse, mape, r2


    print("\n" + "=" * 50)
    print("滚动预测整体评估结果")
    print("=" * 50)

    overall_mae, overall_rmse, overall_mape, overall_r2 = evaluate_model(y_actual_full, y_pred_full, "整体")
    print(f"整体: MAE={overall_mae:.2f}, RMSE={overall_rmse:.2f}, MAPE={overall_mape:.2f}%, R²={overall_r2:.4f}")

    print("\n各时段评估:")
    for period_name in period_data:
        y_period_actual = np.array(all_actuals[period_name])
        y_period_pred = np.array(all_predictions[period_name])
        mae, rmse, mape, r2 = evaluate_model(y_period_actual, y_period_pred, period_name)
        print(f"{period_name}: MAE={mae:.2f}, R²={r2:.4f}")

    # ====================== 10. 各小时误差 ======================
    hour_mae = []
    for hour in range(24):
        mae = mean_absolute_error(y_actual_full[:, hour], y_pred_full[:, hour])
        hour_mae.append(mae)

    print("\n各小时MAE:")
    for h in range(24):
        print(f"{h + 1:2d}时: {hour_mae[h]:.2f}")

    # ====================== 11. 可视化 ======================
    # 11.1 预测效果对比
    n_plot = min(10, n_predictions)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
    fig.suptitle('滚动预测结果对比（原始特征+前一天电价）', fontsize=16)

    for i in range(n_plot):
        ax = axes[i]
        hours = range(1, 25)

        ax.plot(hours, y_actual_full[i], 'b-o', label='实际值', markersize=4)
        ax.plot(hours, y_pred_full[i], 'r--s', label='预测值', markersize=4)

        # 用不同背景色标记时段
        ax.axvspan(1, 8, alpha=0.1, color='gray', label='非中午' if i == 0 else '')
        ax.axvspan(9, 16, alpha=0.2, color='red', label='中午' if i == 0 else '')
        ax.axvspan(17, 24, alpha=0.1, color='gray')

        date_idx = prediction_dates[i]
        actual_date = price_df.index[date_idx]
        date_str = actual_date.strftime('%Y-%m-%d')

        ax.set_title(f'预测{date_str}')
        ax.set_xlabel('小时')
        ax.set_ylabel('电价')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('滚动预测对比_简化特征.png', dpi=300)
    plt.show()

    # ====================== 12. 保存结果 ======================
    pred_dates = [price_df.index[i] for i in prediction_dates]

    pred_df = pd.DataFrame(
        y_pred_full,
        index=pred_dates,
        columns=[f'{h}时' for h in range(1, 25)]
    )

    actual_df = pd.DataFrame(
        y_actual_full,
        index=pred_dates,
        columns=[f'{h}时' for h in range(1, 25)]
    )

    output_filename = '滚动预测结果_简化特征.xlsx'

    with pd.ExcelWriter(output_filename) as writer:
        actual_df.to_excel(writer, sheet_name='实际值')
        pred_df.to_excel(writer, sheet_name='预测值')

        # 评估指标
        metrics_df = pd.DataFrame({
            '指标': ['MAE', 'RMSE', 'MAPE(%)', 'R²'],
            '整体': [overall_mae, overall_rmse, overall_mape, overall_r2]
        })
        metrics_df.to_excel(writer, sheet_name='评估指标', index=False)

        # 各小时误差
        hour_mae_df = pd.DataFrame({
            '小时': range(1, 25),
            'MAE': hour_mae
        })
        hour_mae_df.to_excel(writer, sheet_name='各小时误差', index=False)

    print(f"\n预测结果已保存至 '{output_filename}'")
    print(f"\n共进行了 {n_predictions} 次滚动预测")
    print(f"从第{min_train_days + 1}天开始预测，到第{n_samples}天结束")

    # 输出特征说明
    print("\n" + "=" * 50)
    print("特征说明")
    print("=" * 50)
    print("本次预测使用了以下特征：")
    print("1. 原始特征：风电、光伏、负荷、火电空间")
    print("2. 前一天电价")