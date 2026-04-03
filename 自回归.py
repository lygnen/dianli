import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PRICE_MIN = 200  # 电价最低值

if __name__ == '__main__':
    # ====================== 1. 读取数据 ======================
    print("正在读取数据...")

    price_df = pd.read_excel('汇总.xlsx', sheet_name='日前电价', index_col=0)
    wind_df = pd.read_excel('汇总.xlsx', sheet_name='风电24', index_col=0)
    pv_df = pd.read_excel('汇总.xlsx', sheet_name='光伏24', index_col=0)
    load_df = pd.read_excel('汇总.xlsx', sheet_name='负荷24', index_col=0)
    thermal_df = pd.read_excel('汇总.xlsx', sheet_name='负载24', index_col=0)

    total_days = len(price_df)
    print(f"数据时间范围: {price_df.index[0]} 到 {price_df.index[-1]}")
    print(f"共 {total_days} 天")
    print(f"电价下限: {PRICE_MIN}")

    # ====================== 2. 数据准备 ======================
    n_hours = 24

    wind_values = wind_df.values
    pv_values = pv_df.values
    load_values = load_df.values
    thermal_values = thermal_df.values
    price_values = price_df.values

    # ====================== 3. 划分训练集和测试集 (9:1) ======================
    train_size = int(total_days * 0.9)
    test_size = total_days - train_size

    print(f"\n数据集划分:")
    print(f"训练集: 前{train_size}天")
    print(f"测试集: 后{test_size}天")

    # 训练数据
    train_data = {
        'wind': wind_values[:train_size],
        'pv': pv_values[:train_size],
        'load': load_values[:train_size],
        'thermal': thermal_values[:train_size],
        'price': price_values[:train_size]
    }

    # 测试数据
    test_data = {
        'wind': wind_values[train_size:],
        'pv': pv_values[train_size:],
        'load': load_values[train_size:],
        'thermal': thermal_values[train_size:],
        'price': price_values[train_size:]
    }

    # ====================== 4. 定义每个小时的特征配置 ======================
    # 基础特征配置（所有点都有的）
    # 光伏: 用当前小时 (索引hour)
    # 风电: 用当前小时 (索引hour)
    # 负荷: 用当前小时 (索引hour)
    # 负载: 用当前小时 (索引hour)
    # 前一个小时电价: 1点用前一天24点，其他用前一个点预测值
    # 前一天相同时段电价: 所有点都有

    # 特殊点的特征配置（根据相关性分析结果）
    # 注意：小时从1开始，但索引从0开始，所以小时要减1
    special_features = {
        5: {  # 6点 (索引5)
            'pv': [0, 1],  # 用1点和2点的光伏 (索引0,1)
            'wind': [8, 7],  # 用9点和8点的风电 (索引8,7)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [6]  # 用7点的负载 (索引6)
        },
        6: {  # 7点 (索引6)
            'pv': [0, 1],  # 用1点和2点的光伏 (索引0,1)
            'wind': [8, 7],  # 用9点和8点的风电 (索引8,7)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [6]  # 用7点的负载 (索引6)
        },
        7: {  # 8点 (索引7)
            'pv': [1, 12],  # 用1点和2点的光伏 (索引0,1)
            'wind': [0, 23],  # 用9点和8点的风电 (索引8,7)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [7]  # 用7点的负载 (索引6)
        },

        17: {  # 18点 (索引17)
            'pv': [0, 1],  # 用1点和2点的光伏 (索引0,1)
            'wind': [20, 19],  # 用21点和20点的风电 (索引20,19)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [16]  # 用17点的负载 (索引16)
        },
        18: {  # 19点 (索引18)
            'pv': [0, 1],  # 用1点和2点的光伏 (索引0,1)
            'wind': [21, 20],  # 用22点和21点的风电 (索引21,20)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [16]  # 用17点的负载 (索引16)
        },
        23: {  # 24点 (索引23)
            'pv': [0, 1],  # 用1点和2点的光伏 (索引0,1)
            'wind': [22, 21],  # 用23点和22点的风电 (索引22,21)
            'load': [0],  # 用1点的负荷 (索引0)
            'thermal': [16]  # 用17点的负载 (索引16)
        }
    }

    # ====================== 5. 训练24个模型 ======================
    print("\n训练24个线性模型...")

    models = [None] * 24
    scalers = [None] * 24
    train_predictions = np.zeros((train_size, 24))

    # 先训练1点模型（索引0）
    print("\n训练1点模型...")
    X_train_1_list = []
    y_train_1_list = []

    for day in range(1, train_size):
        features = []
        hour = 0

        # 4个原始特征（都用当前小时）
        features.append(train_data['wind'][day, hour])
        features.append(train_data['pv'][day, hour])
        features.append(train_data['load'][day, hour])
        features.append(train_data['thermal'][day, hour])

        # 前一个小时电价（1点用前一天24点）
        features.append(train_data['price'][day - 1, 23])

        # 前一天1点电价
        features.append(train_data['price'][day - 1, 0])

        # 前一天24点-23点变化率
        change_rate = train_data['price'][day - 1, 23] - train_data['price'][day - 1, 22]
        features.append(change_rate)

        X_train_1_list.append(features)
        y_train_1_list.append(train_data['price'][day, 0])

    X_train_1 = np.array(X_train_1_list)
    y_train_1 = np.array(y_train_1_list)

    # 标准化并训练1点模型
    scaler_1 = StandardScaler()
    X_train_1_scaled = scaler_1.fit_transform(X_train_1)
    model_1 = LinearRegression()
    model_1.fit(X_train_1_scaled, y_train_1)

    models[0] = model_1
    scalers[0] = scaler_1

    # 用1点模型预测训练集的所有1点
    for day in range(1, train_size):
        features = []
        hour = 0
        features.append(train_data['wind'][day, hour])
        features.append(train_data['pv'][day, hour])
        features.append(train_data['load'][day, hour])
        features.append(train_data['thermal'][day, hour])
        features.append(train_data['price'][day - 1, 23])
        features.append(train_data['price'][day - 1, 0])
        change_rate = train_data['price'][day - 1, 23] - train_data['price'][day - 1, 22]
        features.append(change_rate)

        X_test = np.array([features])
        X_test_scaled = scaler_1.transform(X_test)
        pred = model_1.predict(X_test_scaled)[0]
        train_predictions[day, 0] = max(PRICE_MIN, pred)

    # 训练2-24点模型
    for hour in range(1, 24):
        print(f"\n训练第{hour + 1}点模型...")

        X_train_list = []
        y_train_list = []

        for day in range(1, train_size):
            features = []

            # 判断是否是特殊点
            if hour in special_features:
                # 使用相关性分析选出的特征
                spec = special_features[hour]

                # 光伏特征（2个指定小时）
                for feat_hour in spec['pv']:
                    features.append(train_data['pv'][day, feat_hour])
                # 风电特征（2个指定小时）
                for feat_hour in spec['wind']:
                    features.append(train_data['wind'][day, feat_hour])
                # 负荷特征（1个指定小时）
                features.append(train_data['load'][day, spec['load'][0]])
                # 负载特征（1个指定小时）
                features.append(train_data['thermal'][day, spec['thermal'][0]])
            else:
                # 普通点：都用当前小时
                features.append(train_data['wind'][day, hour])
                features.append(train_data['pv'][day, hour])
                features.append(train_data['load'][day, hour])
                features.append(train_data['thermal'][day, hour])

            # 前一个小时电价（使用预测值）
            features.append(train_predictions[day, hour - 1])

            # 前一天相同时段电价
            features.append(train_data['price'][day - 1, hour])

            X_train_list.append(features)
            y_train_list.append(train_data['price'][day, hour])

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        print(f"  特征数量: {X_train.shape[1]}")

        # 标准化并训练
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        models[hour] = model
        scalers[hour] = scaler

        # 用当前模型预测训练集的小时
        for day in range(1, train_size):
            features = []
            if hour in special_features:
                spec = special_features[hour]
                for feat_hour in spec['pv']:
                    features.append(train_data['pv'][day, feat_hour])
                for feat_hour in spec['wind']:
                    features.append(train_data['wind'][day, feat_hour])
                features.append(train_data['load'][day, spec['load'][0]])
                features.append(train_data['thermal'][day, spec['thermal'][0]])
            else:
                features.append(train_data['wind'][day, hour])
                features.append(train_data['pv'][day, hour])
                features.append(train_data['load'][day, hour])
                features.append(train_data['thermal'][day, hour])

            features.append(train_predictions[day, hour - 1])
            features.append(train_data['price'][day - 1, hour])

            X_test = np.array([features])
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled)[0]
            train_predictions[day, hour] = max(PRICE_MIN, pred)

    # ====================== 6. 滚动测试集预测 ======================
    print("\n开始滚动测试集预测...")

    test_predictions = []
    test_actuals = []

    # 用于滚动预测的扩展数据
    extended_price = np.vstack([train_data['price'], test_data['price']])
    extended_pv = np.vstack([train_data['pv'], test_data['pv']])
    extended_wind = np.vstack([train_data['wind'], test_data['wind']])
    extended_load = np.vstack([train_data['load'], test_data['load']])
    extended_thermal = np.vstack([train_data['thermal'], test_data['thermal']])

    for test_day in range(test_size):
        day_predictions = np.zeros(24)
        current_day_idx = train_size + test_day

        # 预测1点
        features_1 = []
        hour = 0
        features_1.append(extended_wind[current_day_idx, hour])
        features_1.append(extended_pv[current_day_idx, hour])
        features_1.append(extended_load[current_day_idx, hour])
        features_1.append(extended_thermal[current_day_idx, hour])
        features_1.append(extended_price[current_day_idx - 1, 23])
        features_1.append(extended_price[current_day_idx - 1, 0])
        change_rate = extended_price[current_day_idx - 1, 23] - extended_price[current_day_idx - 1, 22]
        features_1.append(change_rate)

        X_test_1 = np.array([features_1])
        X_test_1_scaled = scalers[0].transform(X_test_1)
        pred_1 = models[0].predict(X_test_1_scaled)[0]
        day_predictions[0] = max(PRICE_MIN, pred_1)

        # 预测2-24点
        for hour in range(1, 24):
            features = []

            if hour in special_features:
                spec = special_features[hour]
                for feat_hour in spec['pv']:
                    features.append(extended_pv[current_day_idx, feat_hour])
                for feat_hour in spec['wind']:
                    features.append(extended_wind[current_day_idx, feat_hour])
                features.append(extended_load[current_day_idx, spec['load'][0]])
                features.append(extended_thermal[current_day_idx, spec['thermal'][0]])
            else:
                features.append(extended_wind[current_day_idx, hour])
                features.append(extended_pv[current_day_idx, hour])
                features.append(extended_load[current_day_idx, hour])
                features.append(extended_thermal[current_day_idx, hour])

            features.append(day_predictions[hour - 1])
            features.append(extended_price[current_day_idx - 1, hour])

            X_test = np.array([features])
            X_test_scaled = scalers[hour].transform(X_test)
            pred = models[hour].predict(X_test_scaled)[0]
            day_predictions[hour] = max(PRICE_MIN, pred)

        test_predictions.append(day_predictions)
        test_actuals.append(test_data['price'][test_day])

    test_predictions = np.array(test_predictions)
    test_actuals = np.array(test_actuals)

    # ====================== 7. 计算各小时MAE ======================
    hour_mae = []
    for hour in range(24):
        mae = mean_absolute_error(test_actuals[:, hour], test_predictions[:, hour])
        hour_mae.append(mae)

    # ====================== 8. 可视化 ======================
    # 8.1 测试集预测曲线对比
    n_plot = min(10, test_size)
    fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
    fig.suptitle('测试集预测结果对比（6,18,19,24点优化）', fontsize=16)

    for i in range(n_plot):
        ax = axes[i]
        hours = range(1, 25)

        ax.plot(hours, test_actuals[i], 'b-o', label='实际值', markersize=4, linewidth=2)
        ax.plot(hours, test_predictions[i], 'r--s', label='预测值', markersize=4, linewidth=2)
        ax.axhline(y=PRICE_MIN, color='gray', linestyle=':', alpha=0.5, label=f'下限{PRICE_MIN}')

        # 标记优化的小时
        ax.axvline(x=6, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=18, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=19, color='red', linestyle='--', alpha=0.3)
        ax.axvline(x=24, color='red', linestyle='--', alpha=0.3)

        actual_date = price_df.index[train_size + i].strftime('%Y-%m-%d')
        ax.set_title(f'{actual_date}')
        ax.set_xlabel('小时')
        ax.set_ylabel('电价')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('测试集预测曲线_优化点.png', dpi=300)
    plt.show()

    # 8.2 各小时MAE
    plt.figure(figsize=(12, 6))

    plt.plot(range(1, 25), hour_mae, 'ro-', linewidth=2, markersize=8, label='MAE')

    # 标记优化的小时
    plt.axvline(x=6, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=18, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=19, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=24, color='red', linestyle='--', alpha=0.5)

    plt.xlabel('小时', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('各小时预测误差', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 25))

    for i, v in enumerate(hour_mae):
        plt.text(i + 1, v + 1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('各小时MAE_优化点.png', dpi=300)
    plt.show()

    # 8.3 输出统计信息
    overall_mae = np.mean(hour_mae)
    print(f"\n测试集整体MAE: {overall_mae:.2f}")
    print(f"各小时MAE范围: {min(hour_mae):.2f} - {max(hour_mae):.2f}")
    print(f"\n各小时MAE详情:")
    for hour in range(24):
        print(f"{hour + 1:2d}时: {hour_mae[hour]:.2f}")

    # ====================== 9. 保存结果 ======================
    test_dates = price_df.index[train_size:]

    pred_df = pd.DataFrame(
        test_predictions,
        index=test_dates,
        columns=[f'{h}时' for h in range(1, 25)]
    )

    actual_df = pd.DataFrame(
        test_actuals,
        index=test_dates,
        columns=[f'{h}时' for h in range(1, 25)]
    )

    output_filename = '滚动测试结果_优化点.xlsx'

    with pd.ExcelWriter(output_filename) as writer:
        actual_df.to_excel(writer, sheet_name='实际值')
        pred_df.to_excel(writer, sheet_name='预测值')

        mae_df = pd.DataFrame({
            '小时': range(1, 25),
            'MAE': hour_mae
        })
        mae_df.to_excel(writer, sheet_name='各小时MAE', index=False)

    print(f"\n预测结果已保存至 '{output_filename}'")