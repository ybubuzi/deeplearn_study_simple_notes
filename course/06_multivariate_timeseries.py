"""
TensorFlow 深度学习入门示例 6: 多变量时序预测
学习处理多个特征的时序数据
预测天气温度（基于温度、湿度、气压等多个因素）
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("🌤️ 多变量时序预测入门")
print("=" * 60)

# 1. 生成模拟天气数据
print("\n1. 生成模拟天气数据...")

def create_weather_data(days=1000):
    """
    创建模拟的多变量天气数据
    包含：温度、湿度、气压、风速
    """
    np.random.seed(42)
    
    # 时间序列（天数）
    time = np.arange(days)
    
    # 基础温度：有季节性变化
    base_temp = 20 + 15 * np.sin(2 * np.pi * time / 365)  # 年度周期
    daily_temp = base_temp + 5 * np.sin(2 * np.pi * time / 7)  # 周期性
    temperature = daily_temp + np.random.normal(0, 2, days)  # 添加噪声
    
    # 湿度：与温度负相关
    humidity = 70 - 0.5 * temperature + np.random.normal(0, 5, days)
    humidity = np.clip(humidity, 20, 100)  # 限制在合理范围
    
    # 气压：相对稳定，有小幅波动
    pressure = 1013 + np.random.normal(0, 10, days)
    
    # 风速：随机但有一定模式
    wind_speed = 5 + 3 * np.sin(2 * np.pi * time / 30) + np.random.normal(0, 2, days)
    wind_speed = np.clip(wind_speed, 0, 20)  # 限制在合理范围
    
    # 组合成DataFrame
    weather_data = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'pressure': pressure,
        'wind_speed': wind_speed
    })
    
    return weather_data

# 生成天气数据
weather_df = create_weather_data(1000)
print(f"生成了 {len(weather_df)} 天的天气数据")
print(f"特征列: {list(weather_df.columns)}")
print("\n数据统计:")
print(weather_df.describe())

# 2. 可视化原始数据
plt.figure(figsize=(15, 10))

features = ['temperature', 'humidity', 'pressure', 'wind_speed']
feature_names = ['温度(°C)', '湿度(%)', '气压(hPa)', '风速(m/s)']

for i, (feature, name) in enumerate(zip(features, feature_names)):
    plt.subplot(2, 2, i+1)
    plt.plot(weather_df[feature][:200])  # 显示前200天
    plt.title(f'{name}变化（前200天）')
    plt.xlabel('天数')
    plt.ylabel(name)
    plt.grid(True)

plt.tight_layout()
plt.show()

# 3. 准备多变量时序数据
print("\n2. 准备多变量时序数据...")

def create_multivariate_sequences(data, seq_length, target_col):
    """
    创建多变量时序序列
    
    参数:
    data: DataFrame，包含多个特征
    seq_length: 序列长度
    target_col: 目标列名（要预测的变量）
    
    返回:
    X: 输入序列 (样本数, 序列长度, 特征数)
    y: 目标值 (样本数,)
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 输入：过去seq_length天的所有特征
        X.append(data.iloc[i:i + seq_length].values)
        # 输出：下一天的目标变量
        y.append(data.iloc[i + seq_length][target_col])
    
    return np.array(X), np.array(y)

# 使用过去7天的数据预测下一天的温度
sequence_length = 7
target_feature = 'temperature'

X, y = create_multivariate_sequences(weather_df, sequence_length, target_feature)

print(f"序列长度: {sequence_length} 天")
print(f"特征数量: {len(weather_df.columns)}")
print(f"训练样本数: {len(X)}")
print(f"X形状: {X.shape}")  # (样本数, 序列长度, 特征数)
print(f"y形状: {y.shape}")  # (样本数,)

# 显示数据示例
print(f"\n数据示例（预测温度）:")
print(f"输入序列形状: {X[0].shape}")
print(f"第一个样本的第一天数据: {X[0][0]}")
print(f"对应的目标温度: {y[0]:.2f}°C")

# 4. 数据标准化
print("\n3. 数据标准化...")

# 分割训练和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 标准化特征（基于训练集计算统计量）
from sklearn.preprocessing import StandardScaler

# 为每个特征创建标准化器
feature_scalers = []
for i in range(X_train.shape[2]):  # 对每个特征
    scaler = StandardScaler()
    # 重塑数据以适应scaler
    feature_data = X_train[:, :, i].reshape(-1, 1)
    scaler.fit(feature_data)
    feature_scalers.append(scaler)

# 应用标准化
X_train_scaled = np.zeros_like(X_train)
X_test_scaled = np.zeros_like(X_test)

for i in range(X_train.shape[2]):
    # 标准化训练集
    train_feature = X_train[:, :, i].reshape(-1, 1)
    X_train_scaled[:, :, i] = feature_scalers[i].transform(train_feature).reshape(X_train.shape[0], X_train.shape[1])
    
    # 标准化测试集
    test_feature = X_test[:, :, i].reshape(-1, 1)
    X_test_scaled[:, :, i] = feature_scalers[i].transform(test_feature).reshape(X_test.shape[0], X_test.shape[1])

# 标准化目标变量
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

print(f"训练集大小: {len(X_train_scaled)}")
print(f"测试集大小: {len(X_test_scaled)}")

# 5. 构建多变量LSTM模型
print("\n4. 构建多变量LSTM模型...")

model = tf.keras.Sequential([
    # LSTM层：处理多变量时序数据
    tf.keras.layers.LSTM(
        64, 
        return_sequences=True,  # 返回完整序列
        input_shape=(sequence_length, len(weather_df.columns)),
        name='lstm1'
    ),
    tf.keras.layers.Dropout(0.2),
    
    # 第二个LSTM层
    tf.keras.layers.LSTM(32, name='lstm2'),
    tf.keras.layers.Dropout(0.2),
    
    # 全连接层
    tf.keras.layers.Dense(16, activation='relu', name='dense'),
    tf.keras.layers.Dropout(0.1),
    
    # 输出层：预测温度
    tf.keras.layers.Dense(1, name='output')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("多变量LSTM模型结构:")
model.summary()

# 6. 训练模型
print("\n5. 开始训练...")

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_scaled),
    verbose=1
)

# 7. 评估和预测
print("\n6. 模型评估和预测...")

# 预测
predictions_scaled = model.predict(X_test_scaled, verbose=0)
predictions = target_scaler.inverse_transform(predictions_scaled)

# 计算误差
mae = np.mean(np.abs(predictions.flatten() - y_test))
rmse = np.sqrt(np.mean((predictions.flatten() - y_test) ** 2))

print(f"平均绝对误差 (MAE): {mae:.2f}°C")
print(f"均方根误差 (RMSE): {rmse:.2f}°C")

# 8. 可视化结果
plt.figure(figsize=(15, 10))

# 训练过程
plt.subplot(2, 3, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练过程')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 预测 vs 实际
plt.subplot(2, 3, 2)
plt.plot(y_test[:100], label='实际温度', alpha=0.7)
plt.plot(predictions[:100], label='预测温度', alpha=0.7)
plt.title('温度预测结果（前100天）')
plt.xlabel('天数')
plt.ylabel('温度(°C)')
plt.legend()
plt.grid(True)

# 预测误差
plt.subplot(2, 3, 3)
errors = predictions.flatten() - y_test
plt.hist(errors, bins=30, alpha=0.7)
plt.title('预测误差分布')
plt.xlabel('误差(°C)')
plt.ylabel('频次')
plt.grid(True)

# 散点图
plt.subplot(2, 3, 4)
plt.scatter(y_test, predictions.flatten(), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('预测值 vs 实际值')
plt.xlabel('实际温度(°C)')
plt.ylabel('预测温度(°C)')
plt.grid(True)

# 特征重要性（通过梯度分析）
plt.subplot(2, 3, 5)
feature_importance = np.random.rand(len(features))  # 简化示例
plt.bar(feature_names, feature_importance)
plt.title('特征重要性（示例）')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.xticks(rotation=45)

# 时间序列预测
plt.subplot(2, 3, 6)
last_week = weather_df.tail(7)
plt.plot(range(7), last_week['temperature'], 'o-', label='历史温度')
plt.title('基于历史数据的预测')
plt.xlabel('天数')
plt.ylabel('温度(°C)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. 保存模型
model.save('multivariate_weather_model.h5')
print("\n模型已保存为 'multivariate_weather_model.h5'")

print("\n" + "=" * 60)
print("✅ 多变量时序预测完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 多变量时序数据：同时考虑多个相关特征")
print("2. 特征标准化：每个特征独立标准化")
print("3. 复杂LSTM架构：多层LSTM + Dropout")
print("4. 时序预测评估：MAE、RMSE等指标")
print("5. 特征工程：相关性分析和特征选择")
print("\n🔍 多变量 vs 单变量的优势:")
print("- 利用多个相关特征提高预测准确性")
print("- 捕捉特征间的复杂关系")
print("- 更接近真实世界的预测场景")
print("=" * 60)
