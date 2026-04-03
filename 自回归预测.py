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
    print(f"总天数: {total_days}")

    # ====================== 2. 识别有电价和缺失电价的行 ======================
    # 找出电价不为空的行（训练数据）
    price_not_null = price_df.notna().all(axis=1)  # 每天24小时都不为空
    train_indices = np.where(price_not_null)[0]
    predict_indices = np.where(~price_not_null)[0]

    train_size = len(train_indices)
    predict_size = len(predict_indices)

    print(f"\n有电价的天数: {train_size} 天")
    print(f"需要预测的天数: {predict_size} 天")

    if predict_size == 0:
        print("没有需要预测的数据！")
        exit()

    # ====================== 3. 数据准备 ======================
    n_hours = 24

    wind_values = wind_df.values
    pv_values = pv_df.values
    load_values = load_df.values
    thermal_values = thermal_df.values
    price_values = price_df.values

    # ====================== 4. 定义每个小时的特征配置 ======================
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

    # ====================== 5. 用有电价的数据训练模型 ======================
    print("\n开始训练模型...")

    # 提取训练数据
    train_wind = wind_values[train_indices]
    train_pv = pv_values[train_indices]
    train_load = load_values[train_indices]
    train_thermal = thermal_values[train_indices]
    train_price = price_values[train_indices]

    train_data = {
        'wind': train_wind,
        'pv': train_pv,
        'load': train_load,
        'thermal': train_thermal,
        'price': train_price
    }

    models = [None] * 24
    scalers = [None] * 24
    train_predictions = np.zeros((train_size, 24))

    # 训练1点模型（索引0）
    print("训练1点模型...")
    X_train_1_list = []
    y_train_1_list = []

    for day in range(1, train_size):  # 从第2天开始，因为需要前一天数据
        features = []
        hour = 0

        features.append(train_data['wind'][day, hour])
        features.append(train_data['pv'][day, hour])
        features.append(train_data['load'][day, hour])
        features.append(train_data['thermal'][day, hour])
        features.append(train_data['price'][day - 1, 23])  # 前一天24点
        features.append(train_data['price'][day - 1, 0])  # 前一天1点
        change_rate = train_data['price'][day - 1, 23] - train_data['price'][day - 1, 22]
        features.append(change_rate)

        X_train_1_list.append(features)
        y_train_1_list.append(train_data['price'][day, 0])

    X_train_1 = np.array(X_train_1_list)
    y_train_1 = np.array(y_train_1_list)

    scaler_1 = StandardScaler()
    X_train_1_scaled = scaler_1.fit_transform(X_train_1)
    model_1 = LinearRegression()
    model_1.fit(X_train_1_scaled, y_train_1)

    models[0] = model_1
    scalers[0] = scaler_1

    # 用1点模型预测训练集的所有1点（用于后续训练）
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
        print(f"训练第{hour + 1}点模型...")

        X_train_list = []
        y_train_list = []

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

            features.append(train_predictions[day, hour - 1])  # 前一个小时预测值
            features.append(train_data['price'][day - 1, hour])  # 前一天同时段

            X_train_list.append(features)
            y_train_list.append(train_data['price'][day, hour])

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        models[hour] = model
        scalers[hour] = scaler

        # 用当前模型预测训练集的小时（用于后续训练）
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

    # ====================== 6. 预测缺失的电价 ======================
    print("\n开始预测缺失的电价...")

    # 提取需要预测的数据
    predict_wind = wind_values[predict_indices]
    predict_pv = pv_values[predict_indices]
    predict_load = load_values[predict_indices]
    predict_thermal = thermal_values[predict_indices]

    # 创建完整的电价数组（包含训练集的真实值和预测集的预测值）
    full_price = np.zeros((total_days, 24))
    full_price[train_indices] = train_price  # 训练集用真实值

    # 逐天预测
    for i, day_idx in enumerate(predict_indices):
        print(f"预测第 {day_idx + 1} 天 (索引 {day_idx})...")
        day_predictions = np.zeros(24)

        # 预测1点
        features_1 = []
        hour = 0
        features_1.append(predict_wind[i, hour])
        features_1.append(predict_pv[i, hour])
        features_1.append(predict_load[i, hour])
        features_1.append(predict_thermal[i, hour])

        # 前一天24点电价
        if day_idx > 0:
            features_1.append(full_price[day_idx - 1, 23])
        else:
            features_1.append(PRICE_MIN)  # 如果是第一天，用最小值

        # 前一天1点电价
        if day_idx > 0:
            features_1.append(full_price[day_idx - 1, 0])
        else:
            features_1.append(PRICE_MIN)

        # 前一天24点-23点变化率
        if day_idx > 0:
            change_rate = full_price[day_idx - 1, 23] - full_price[day_idx - 1, 22]
        else:
            change_rate = 0
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
                    features.append(predict_pv[i, feat_hour])
                for feat_hour in spec['wind']:
                    features.append(predict_wind[i, feat_hour])
                features.append(predict_load[i, spec['load'][0]])
                features.append(predict_thermal[i, spec['thermal'][0]])
            else:
                features.append(predict_wind[i, hour])
                features.append(predict_pv[i, hour])
                features.append(predict_load[i, hour])
                features.append(predict_thermal[i, hour])

            features.append(day_predictions[hour - 1])  # 前一个小时预测值

            # 前一天同时段电价
            if day_idx > 0:
                features.append(full_price[day_idx - 1, hour])
            else:
                features.append(PRICE_MIN)

            X_test = np.array([features])
            X_test_scaled = scalers[hour].transform(X_test)
            pred = models[hour].predict(X_test_scaled)[0]
            day_predictions[hour] = max(PRICE_MIN, pred)

        # 保存预测结果
        full_price[day_idx] = day_predictions

    # ====================== 7. 输出预测结果 ======================
    print("\n预测完成！")

    # 创建预测结果的DataFrame
    predict_dates = price_df.index[predict_indices]
    predict_result_df = pd.DataFrame(
        full_price[predict_indices],
        index=predict_dates,
        columns=[f'{h}时' for h in range(1, 25)]
    )

    # 如果有训练数据，可以计算训练集上的MAE作为参考
    if train_size > 1:
        train_mae = mean_absolute_error(
            train_price[1:].reshape(-1),  # 从第2天开始，因为第1天没有前小时
            train_predictions[1:].reshape(-1)
        )
        print(f"\n训练集MAE（参考）: {train_mae:.2f}")

    print(f"\n预测了 {predict_size} 天的电价")
    print(f"预测日期范围: {predict_dates[0]} 到 {predict_dates[-1]}")

    # ====================== 8. 可视化预测结果 ======================
    n_plot = min(10, predict_size)
    if n_plot > 0:
        if n_plot == 1:
            # 只有1天要预测
            fig, ax = plt.subplots(figsize=(14, 6))
            hours = range(1, 25)

            ax.plot(hours, full_price[predict_indices[0]], 'b-o', label='预测值', markersize=4, linewidth=2)
            ax.axhline(y=PRICE_MIN, color='gray', linestyle=':', alpha=0.5, label=f'下限{PRICE_MIN}')

            ax.set_title(f'{predict_dates[0].strftime("%Y-%m-%d")}')
            ax.set_xlabel('小时')
            ax.set_ylabel('电价')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        else:
            # 多天要预测
            fig, axes = plt.subplots(n_plot, 1, figsize=(14, 3 * n_plot))
            fig.suptitle('预测电价结果', fontsize=16)

            for i in range(n_plot):
                ax = axes[i]
                hours = range(1, 25)

                ax.plot(hours, full_price[predict_indices[i]], 'b-o', label='预测值', markersize=4, linewidth=2)
                ax.axhline(y=PRICE_MIN, color='gray', linestyle=':', alpha=0.5, label=f'下限{PRICE_MIN}')

                ax.set_title(f'{predict_dates[i].strftime("%Y-%m-%d")}')
                ax.set_xlabel('小时')
                ax.set_ylabel('电价')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('预测电价结果.png', dpi=300)
        plt.show()

    # ====================== 9. 保存结果到Excel ======================
    # 创建完整的电价表（将预测值填充回原表）
    complete_price_df = price_df.copy()
    for i, day_idx in enumerate(predict_indices):
        complete_price_df.iloc[day_idx] = full_price[day_idx]

    output_filename = '电价预测结果_完整.xlsx'

    with pd.ExcelWriter(output_filename) as writer:
        # 保存完整电价表
        complete_price_df.to_excel(writer, sheet_name='完整电价')

        # 保存预测结果
        predict_result_df.to_excel(writer, sheet_name='预测电价')

        # 保存模型评估信息
        if train_size > 1:
            info_df = pd.DataFrame({
                '项目': ['训练集MAE', '预测天数', '电价下限'],
                '数值': [f'{train_mae:.2f}', predict_size, PRICE_MIN]
            })
            info_df.to_excel(writer, sheet_name='信息', index=False)

    print(f"\n预测结果已保存至 '{output_filename}'")
    print(f"完整电价表已包含预测值")