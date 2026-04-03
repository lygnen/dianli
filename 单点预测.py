import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 超参数
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 测试集比例
N_ESTIMATORS = 150  # 树的数量
MAX_DEPTH = 10  # 最大深度
MIN_SAMPLES_SPLIT = 2  # 内部节点再划分所需最小样本数
MIN_SAMPLES_LEAF = 1  # 叶子节点最少样本数
MAX_FEATURES = 'sqrt'  # 特征数量策略
ACCURACY_THRESHOLD = 5  # 正确预测的偏差阈值
CV_FOLDS = 5  # 交叉验证折数

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel("9点随机森林.xlsx", sheet_name='训练')


# 数据预处理
def preprocess_data(df):
    # 处理缺失值
    df_filled = df.copy()
    for column in df_filled.columns:
        if df_filled[column].isnull().sum() > 0:
            median_val = df_filled[column].median()
            df_filled[column].fillna(median_val, inplace=True)
            print(f"已用中位数 {median_val:.2f} 填充 {column} 的缺失值")

    # 定义特征和目标变量
    X = df_filled.drop('日前价格', axis=1)
    y = df_filled['日前价格']

    return X, y, df_filled


# 数据预处理
X, y, df_filled = preprocess_data(df)

print(f"样本数量: {X.shape[0]}")

# 数据洗牌和分割
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


# 随机森林模型训练
def train_random_forest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=1
    )

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 预测
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)

    return rf_model, y_train_pred, y_test_pred


# 训练模型
rf_model, y_train_pred, y_test_pred = train_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)


# 计算正确率
def calculate_accuracy(y_true, y_pred, threshold=ACCURACY_THRESHOLD):
    deviations = np.abs(y_true - y_pred)
    correct_predictions = np.sum(deviations <= threshold)
    accuracy = correct_predictions / len(y_true) * 100
    return accuracy, correct_predictions


# 计算训练集和测试集的正确率
train_accuracy, train_correct = calculate_accuracy(y_train, y_train_pred, ACCURACY_THRESHOLD)
test_accuracy, test_correct = calculate_accuracy(y_test, y_test_pred, ACCURACY_THRESHOLD)

# 评估模型
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 计算过拟合程度
overfit_mse = train_mse - test_mse
overfit_r2 = train_r2 - test_r2

print("\n模型评估")
print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")
print(f"训练集 MAE: {train_mae:.4f}")
print(f"测试集 MAE: {test_mae:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")

print(f"\n过拟合分析:")
print(f"MSE过拟合程度: {overfit_mse:.4f}")
print(f"R²过拟合程度: {overfit_r2:.4f}")

print(f"\n正确率分析 (偏差 ≤ {ACCURACY_THRESHOLD} 算正确):")
print(f"训练集正确率: {train_accuracy:.2f}% ({train_correct}/{len(y_train)})")
print(f"测试集正确率: {test_accuracy:.2f}% ({test_correct}/{len(y_test)})")


# 保存训练集预测结果到新表
def save_training_predictions(df_filled, X_train, y_train, y_train_pred, scaler, rf_model):
    # 创建训练集数据的索引
    train_indices = X_train.index

    # 特征、实际值和预测值
    training_predictions_df = df_filled.loc[train_indices].copy()
    training_predictions_df['预测日前价格'] = y_train_pred
    training_predictions_df['预测偏差'] = np.abs(
        training_predictions_df['日前价格'] - training_predictions_df['预测日前价格'])
    training_predictions_df['是否预测正确'] = training_predictions_df['预测偏差'] <= ACCURACY_THRESHOLD

    # 预测不确定性
    tree_predictions = np.array([tree.predict(X_train_scaled) for tree in rf_model.estimators_])
    training_predictions_df['预测标准差'] = np.std(tree_predictions, axis=0)
    training_predictions_df['预测置信度'] = 1 / (1 + training_predictions_df['预测标准差'])

    # 重新排列列的顺序，使关键列在前面
    cols = training_predictions_df.columns.tolist()
    target_col = cols.pop(cols.index('日前价格'))
    pred_col = cols.pop(cols.index('预测日前价格'))
    error_col = cols.pop(cols.index('预测偏差'))
    correct_col = cols.pop(cols.index('是否预测正确'))
    std_col = cols.pop(cols.index('预测标准差'))
    confidence_col = cols.pop(cols.index('预测置信度'))

    new_cols = [target_col, pred_col, error_col, correct_col, std_col, confidence_col] + cols
    training_predictions_df = training_predictions_df[new_cols]

    # 保存到Excel文件
    training_predictions_df.to_excel("训练集预测结果详细表.xlsx", index=True, index_label='原始数据索引')

    return training_predictions_df

training_results = save_training_predictions(df_filled, X_train, y_train, y_train_pred, scaler, rf_model)

# 绘图
fig = plt.figure(figsize=(20, 15))

# 1. 训练集预测对比 (排序后)
plt.subplot(2, 3, 1)
# 按实际值排序，使图表更清晰
sorted_idx = np.argsort(y_train.values)
plt.plot(range(len(y_train)), y_train.values[sorted_idx], 'b-', alpha=0.7, linewidth=1, label='实际值')
plt.plot(range(len(y_train_pred)), y_train_pred[sorted_idx], 'r-', alpha=0.7, linewidth=1, label='预测值')
plt.xlabel('样本索引 (排序后)')
plt.ylabel('电价')
plt.title(f'训练集: 实际值 vs 预测值\nR²={train_r2:.4f}, 正确率={train_accuracy:.2f}%')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 测试集预测对比 (排序后)
plt.subplot(2, 3, 2)
# 按实际值排序，使图表更清晰
sorted_idx_test = np.argsort(y_test.values)
plt.plot(range(len(y_test)), y_test.values[sorted_idx_test], 'b-', alpha=0.7, linewidth=1, label='实际值')
plt.plot(range(len(y_test_pred)), y_test_pred[sorted_idx_test], 'r-', alpha=0.7, linewidth=1, label='预测值')
plt.xlabel('样本索引 (排序后)')
plt.ylabel('电价')
plt.title(f'测试集: 实际值 vs 预测值\nR²={test_r2:.4f}, 正确率={test_accuracy:.2f}%')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. 训练集预测值与实际值散点图
plt.subplot(2, 3, 3)
plt.scatter(y_train, y_train_pred, alpha=0.5, s=10)
# 添加理想预测线 (y=x)
max_val = max(y_train.max(), y_train_pred.max())
min_val = min(y_train.min(), y_train_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测')
plt.xlabel('实际电价')
plt.ylabel('预测电价')
plt.title(f'训练集: 预测值 vs 实际值\nR²={train_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. 测试集预测值与实际值散点图
plt.subplot(2, 3, 4)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=10)
# 添加理想预测线 (y=x)
max_val = max(y_test.max(), y_test_pred.max())
min_val = min(y_test.min(), y_test_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='理想预测')
plt.xlabel('实际电价')
plt.ylabel('预测电价')
plt.title(f'测试集: 预测值 vs 实际值\nR²={test_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. 绝对误差分布
plt.subplot(2, 3, 5)
train_errors = np.abs(y_train - y_train_pred)
test_errors = np.abs(y_test - y_test_pred)
plt.hist(train_errors, bins=50, alpha=0.7, label=f'训练集 (MAE={train_mae:.2f})', color='blue')
plt.hist(test_errors, bins=50, alpha=0.7, label=f'测试集 (MAE={test_mae:.2f})', color='red')
plt.axvline(x=ACCURACY_THRESHOLD, color='green', linestyle='--', linewidth=2,
            label=f'正确阈值 ({ACCURACY_THRESHOLD})')
plt.xlabel('绝对误差')
plt.ylabel('样本数量')
plt.title('绝对误差分布')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. 按电价区间的预测精度
plt.subplot(2, 3, 6)
# 将电价分成几个区间
price_bins = np.linspace(y.min(), y.max(), 10)
train_bin_accuracy = []
test_bin_accuracy = []

for i in range(len(price_bins) - 1):
    # 训练集
    train_mask = (y_train >= price_bins[i]) & (y_train < price_bins[i + 1])
    if train_mask.sum() > 0:
        train_bin_acc, _ = calculate_accuracy(y_train[train_mask], y_train_pred[train_mask])
        train_bin_accuracy.append(train_bin_acc)
    else:
        train_bin_accuracy.append(0)

    # 测试集
    test_mask = (y_test >= price_bins[i]) & (y_test < price_bins[i + 1])
    if test_mask.sum() > 0:
        test_bin_acc, _ = calculate_accuracy(y_test[test_mask], y_test_pred[test_mask])
        test_bin_accuracy.append(test_bin_acc)
    else:
        test_bin_accuracy.append(0)

bin_centers = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(len(price_bins) - 1)]
plt.plot(bin_centers, train_bin_accuracy, 'bo-', label='训练集正确率')
plt.plot(bin_centers, test_bin_accuracy, 'ro-', label='测试集正确率')
plt.axhline(y=70, color='green', linestyle='--', alpha=0.7, label='70%正确率线')
plt.xlabel('电价区间')
plt.ylabel('正确率 (%)')
plt.title('不同电价区间的预测正确率')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
# plt.show()


# 特征重要性分析
def print_feature_importance(model, feature_names, top_n=15):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    print(f"\n特征重要性")
    print("-" * 40)
    for i in range(min(top_n, len(importance))):
        print(f"{i + 1:2d}. {feature_names[indices[i]]:15s} : {importance[indices[i]]:.4f}")


# 打印特征重要性
print_feature_importance(rf_model, X.columns.tolist())


# 交叉验证评估
def accuracy_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    deviations = np.abs(y - y_pred)
    accuracy = np.sum(deviations <= ACCURACY_THRESHOLD) / len(y)
    return accuracy
# print(f"\n进行{CV_FOLDS}折交叉验证...")
# cv_scores = cross_val_score(rf_model, X_train_scaled, y_train,
#                            scoring=accuracy_scorer, cv=CV_FOLDS, n_jobs=1)
# print(f"交叉验证正确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")



# 读取随机森林.xlsx并进行预测
def predict_new_data():
    predict_df = pd.read_excel("9点随机森林.xlsx", sheet_name='预测')#67
    original_data = predict_df.copy()

    # 预处理
    predict_df_filled = predict_df.copy()
    for column in predict_df_filled.columns:
        if predict_df_filled[column].isnull().sum() > 0:
            median_val = predict_df_filled[column].median()
            predict_df_filled[column].fillna(median_val, inplace=True)

    # 提取特征
    X_predict = predict_df_filled.drop('日前价格', axis=1)
    # 标准化
    X_predict_scaled = scaler.transform(X_predict)
    # 预测
    y_predict = rf_model.predict(X_predict_scaled)

    original_data['预测日前价格'] = y_predict
    original_data.to_excel("随机森林_预测结果.xlsx", index=False)

    print(f"\n新数据预测完成，结果已保存到 '随机森林_预测结果.xlsx'")
    print(f"共预测 {len(y_predict)} 条数据")

    return original_data, y_predict


predict_new_data()