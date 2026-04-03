import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 超参数
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 150
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_FEATURES = 'sqrt'
ACCURACY_THRESHOLD = 5

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义24个时段的列名
HOUR_COLUMNS = [f'{i}点电价' for i in range(1, 25)]

# 读取数据
df = pd.read_excel("随机森林.xlsx", sheet_name='训练')


def preprocess_data(df, target_hour):
    """
    为特定时段进行数据预处理
    特征中排除所有其他时段的电价
    """
    df_filled = df.copy()

    # 处理缺失值
    for column in df_filled.columns:
        if df_filled[column].isnull().sum() > 0:
            median_val = df_filled[column].median()
            df_filled[column].fillna(median_val, inplace=True)
            print(f"已用中位数 {median_val:.2f} 填充 {column} 的缺失值")

    # 定义特征：排除所有电价列（包括目标列本身）
    # 只保留非电价特征
    feature_columns = [col for col in df_filled.columns
                       if not col.endswith('点电价')]  # 排除所有电价列

    X = df_filled[feature_columns]
    y = df_filled[target_hour]

    print(f"\n{target_hour} 模型使用的特征（共{len(feature_columns)}个）:")
    print(f"特征列表: {feature_columns[:10]}{'...' if len(feature_columns) > 10 else ''}")

    return X, y, df_filled, feature_columns


def train_models_for_all_hours(df, hour_columns):
    """
    为24个时段分别训练模型
    """
    models = {}
    scalers = {}
    results = {}
    feature_importances = {}

    print("=" * 60)
    print("开始为24个时段分别训练模型")
    print("=" * 60)

    for hour in hour_columns:
        print(f"\n正在训练 {hour} 的模型...")
        print("-" * 40)

        # 数据预处理
        X, y, df_filled, feature_names = preprocess_data(df, hour)

        print(f"样本数量: {X.shape[0]}")
        print(f"特征数量: {X.shape[1]}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 训练模型
        rf_model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        rf_model.fit(X_train_scaled, y_train)

        # 预测
        y_train_pred = rf_model.predict(X_train_scaled)
        y_test_pred = rf_model.predict(X_test_scaled)

        # 评估模型
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # 计算平均绝对误差
        train_mae = np.mean(np.abs(y_train - y_train_pred))
        test_mae = np.mean(np.abs(y_test - y_test_pred))

        # 计算正确率
        train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= ACCURACY_THRESHOLD) * 100
        test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= ACCURACY_THRESHOLD) * 100

        print(f"训练集 MSE: {train_mse:.4f}")
        print(f"测试集 MSE: {test_mse:.4f}")
        print(f"训练集 MAE: {train_mae:.4f}")
        print(f"测试集 MAE: {test_mae:.4f}")
        print(f"训练集 R²: {train_r2:.4f}")
        print(f"测试集 R²: {test_r2:.4f}")
        print(f"训练集正确率: {train_accuracy:.2f}%")
        print(f"测试集正确率: {test_accuracy:.2f}%")

        # 保存模型和相关数据
        models[hour] = rf_model
        scalers[hour] = scaler
        results[hour] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'feature_names': feature_names,
            'X_train': X_train,
            'y_train': y_train,
            'y_train_pred': y_train_pred,
            'X_test': X_test,
            'y_test': y_test,
            'y_test_pred': y_test_pred
        }

        # 特征重要性
        importance = rf_model.feature_importances_
        feature_importances[hour] = dict(zip(feature_names, importance))

        # 打印最重要的5个特征
        indices = np.argsort(importance)[::-1][:5]
        print(f"\n{hour} 最重要的5个特征:")
        for i, idx in enumerate(indices):
            print(f"  {i + 1}. {feature_names[idx]}: {importance[idx]:.4f}")

    return models, scalers, results, feature_importances


def predict_new_data_all_hours(models, scalers, hour_columns):
    """
    使用训练好的24个模型预测新数据
    """
    # 读取预测数据
    predict_df = pd.read_excel("随机森林.xlsx", sheet_name='预测')
    original_data = predict_df.copy()

    # 数据预处理（填充缺失值）
    predict_df_filled = predict_df.copy()
    for column in predict_df_filled.columns:
        if predict_df_filled[column].isnull().sum() > 0:
            median_val = predict_df_filled[column].median()
            predict_df_filled[column].fillna(median_val, inplace=True)

    # 获取非电价特征列
    feature_columns = [col for col in predict_df_filled.columns
                       if not col.endswith('点电价')]

    predictions = {}

    print("\n" + "=" * 60)
    print("开始为24个时段分别进行预测")
    print("=" * 60)

    for hour in hour_columns:
        print(f"正在预测 {hour}...")

        # 使用相同的非电价特征进行预测
        X_predict = predict_df_filled[feature_columns]

        # 标准化
        X_predict_scaled = scalers[hour].transform(X_predict)

        # 预测
        y_predict = models[hour].predict(X_predict_scaled)
        predictions[hour] = y_predict

    # 创建结果DataFrame
    result_df = original_data.copy()

    # 添加24个时段的预测结果
    for hour in hour_columns:
        result_df[f'预测_{hour}'] = predictions[hour]

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"24点电价预测结果_{timestamp}.xlsx"
    result_df.to_excel(output_filename, index=False)

    print(f"\n预测完成！结果已保存到 '{output_filename}'")
    print(f"共预测 {len(predict_df)} 条数据，24个时段")

    return result_df, predictions


def save_model_evaluation_summary(results, hour_columns):
    """
    保存24个模型的评估结果汇总
    """
    summary_data = []

    for hour in hour_columns:
        res = results[hour]
        summary_data.append({
            '时段': hour,
            '训练集MSE': res['train_mse'],
            '测试集MSE': res['test_mse'],
            '训练集MAE': res['train_mae'],
            '测试集MAE': res['test_mae'],
            '训练集R²': res['train_r2'],
            '测试集R²': res['test_r2'],
            '训练集正确率(%)': res['train_accuracy'],
            '测试集正确率(%)': res['test_accuracy']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel("24个时段模型评估汇总.xlsx", index=False)
    print("\n模型评估汇总已保存到 '24个时段模型评估汇总.xlsx'")

    # 打印性能最差的5个时段
    worst_performing = summary_df.nlargest(5, '测试集MSE')[['时段', '测试集MSE', '测试集MAE', '测试集R²']]
    print("\n性能最差的5个时段:")
    print(worst_performing.to_string(index=False))

    return summary_df


def save_feature_importance_summary(feature_importances, hour_columns):
    """
    保存24个模型的特征重要性汇总
    """
    # 获取所有特征
    all_features = set()
    for hour, imp_dict in feature_importances.items():
        all_features.update(imp_dict.keys())
    all_features = sorted(list(all_features))

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame(index=all_features)

    for hour in hour_columns:
        hour_imp = feature_importances.get(hour, {})
        importance_df[hour] = [hour_imp.get(feature, 0) for feature in all_features]

    # 计算平均重要性并排序
    importance_df['平均重要性'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('平均重要性', ascending=False)

    importance_df.to_excel("24个时段特征重要性汇总.xlsx")
    print("特征重要性汇总已保存到 '24个时段特征重要性汇总.xlsx'")

    return importance_df


def plot_hourly_performance(results, hour_columns):
    """
    绘制24个时段的模型性能图表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    hours = [h.replace('点电价', '') for h in hour_columns]
    test_mae = [results[h]['test_mae'] for h in hour_columns]
    test_r2 = [results[h]['test_r2'] for h in hour_columns]
    train_accuracy = [results[h]['train_accuracy'] for h in hour_columns]
    test_accuracy = [results[h]['test_accuracy'] for h in hour_columns]

    # 测试集MAE
    axes[0, 0].plot(hours, test_mae, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('时段')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('各时段测试集MAE (平均绝对误差)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 测试集R²
    axes[0, 1].plot(hours, test_r2, 'o-', color='green', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('时段')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('各时段测试集R²')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 正确率对比
    axes[1, 0].plot(hours, train_accuracy, 'o-', label='训练集', linewidth=2, markersize=6)
    axes[1, 0].plot(hours, test_accuracy, 's-', label='测试集', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('时段')
    axes[1, 0].set_ylabel('正确率 (%)')
    axes[1, 0].set_title(f'各时段预测正确率 (偏差 ≤ {ACCURACY_THRESHOLD})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 过拟合程度
    overfit = [results[h]['train_mse'] - results[h]['test_mse'] for h in hour_columns]
    axes[1, 1].bar(hours, overfit, alpha=0.7)
    axes[1, 1].set_xlabel('时段')
    axes[1, 1].set_ylabel('MSE差值 (训练-测试)')
    axes[1, 1].set_title('各时段过拟合程度')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    # plt.savefig('24个时段模型性能分析.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("性能分析图已保存到 '24个时段模型性能分析.png'")


# 主程序执行
if __name__ == "__main__":
    print("=" * 60)
    print("24个时段电价预测系统（已排除电价特征）")
    print("=" * 60)

    # 1. 训练24个模型
    models, scalers, results, feature_importances = train_models_for_all_hours(df, HOUR_COLUMNS)

    # 2. 保存模型评估汇总
    # summary_df = save_model_evaluation_summary(results, HOUR_COLUMNS)
    # print("\n模型评估汇总（前5行）：")
    # print(summary_df.head())

    # 3. 保存特征重要性汇总
    # importance_df = save_feature_importance_summary(feature_importances, HOUR_COLUMNS)
    # print("\n特征重要性前10名：")
    # print(importance_df.head(10))

    # 4. 可视化性能
    plot_hourly_performance(results, HOUR_COLUMNS)

    # 5. 预测新数据
    result_df, predictions = predict_new_data_all_hours(models, scalers, HOUR_COLUMNS)
