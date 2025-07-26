"""
TensorFlow 深度学习入门示例 4: 简单时序预测
用最简单的方式理解时序数据和RNN
预测股价走势（模拟数据）
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("📈 简单时序预测入门")
print("=" * 60)

# 1. 生成模拟的股价数据
print("\n1. 生成模拟股价数据...")

def create_stock_data(days=1000):
    """
    创建模拟股价数据
    模拟一个有趋势和波动的股价序列
    """
    np.random.seed(42)  # 确保结果可重现
    
    # 基础趋势：缓慢上涨
    trend = np.linspace(100, 150, days)
    
    # 添加随机波动
    noise = np.random.normal(0, 5, days)
    
    # 添加周期性波动（模拟市场周期）
    cycle = 10 * np.sin(np.linspace(0, 4*np.pi, days))
    
    # 合成最终股价
    stock_price = trend + cycle + noise
    
    return stock_price

# 生成1000天的股价数据
stock_prices = create_stock_data(1000)
print(f"生成了 {len(stock_prices)} 天的股价数据")
print(f"价格范围: {stock_prices.min():.2f} - {stock_prices.max():.2f}")

# 2. 可视化原始数据
plt.figure(figsize=(12, 4))
plt.plot(stock_prices[:200], label='股价')  # 只显示前200天
plt.title('模拟股价数据（前200天）')
plt.xlabel('天数')
plt.ylabel('股价')
plt.legend()
plt.grid(True)
plt.show()

# 3. 准备时序数据
print("\n2. 准备时序训练数据...")

def create_sequences(data, seq_length):
    """
    将时序数据转换为监督学习格式
    
    例如：用过去5天的价格预测第6天的价格
    [1,2,3,4,5] -> X=[1,2,3,4,5], y=6
    [2,3,4,5,6] -> X=[2,3,4,5,6], y=7
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        # 输入：过去seq_length天的价格
        X.append(data[i:i + seq_length])
        # 输出：下一天的价格
        y.append(data[i + seq_length])
    
    return np.array(X), np.array(y)

# 使用过去10天预测下一天
sequence_length = 10
X, y = create_sequences(stock_prices, sequence_length)

print(f"序列长度: {sequence_length}")
print(f"训练样本数: {len(X)}")
print(f"X形状: {X.shape}")  # (样本数, 序列长度)
print(f"y形状: {y.shape}")  # (样本数,)

# 显示几个例子
print("\n数据示例:")
for i in range(3):
    print(f"样本{i+1}: 输入={X[i][:5]}... -> 输出={y[i]:.2f}")

# 4. 数据标准化
print("\n3. 数据标准化...")

# 计算均值和标准差（只用训练数据）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 标准化（重要：防止梯度爆炸）
mean_price = X_train.mean()
std_price = X_train.std()

X_train_scaled = (X_train - mean_price) / std_price
X_test_scaled = (X_test - mean_price) / std_price
y_train_scaled = (y_train - mean_price) / std_price
y_test_scaled = (y_test - mean_price) / std_price

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"数据均值: {mean_price:.2f}")
print(f"数据标准差: {std_price:.2f}")

# 5. 构建简单的RNN模型
print("\n4. 构建RNN模型...")

model = tf.keras.Sequential([
    # 简单RNN层：处理时序数据
    tf.keras.layers.SimpleRNN(50, input_shape=(sequence_length, 1), name='rnn_layer'),
    
    # 输出层：预测下一个价格
    tf.keras.layers.Dense(1, name='output_layer')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='mse',  # 均方误差，适合回归问题
    metrics=['mae']  # 平均绝对误差
)

print("模型结构:")
model.summary()

# 6. 训练模型
print("\n5. 开始训练...")

# 重塑输入数据为3D格式 (样本数, 时间步, 特征数)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

history = model.fit(
    X_train_reshaped, y_train_scaled,
    epochs=20,  # 训练20轮
    batch_size=32,
    validation_data=(X_test_reshaped, y_test_scaled),
    verbose=1
)

# 7. 评估模型
print("\n6. 模型评估...")
test_loss, test_mae = model.evaluate(X_test_reshaped, y_test_scaled, verbose=0)
print(f"测试损失: {test_loss:.4f}")
print(f"测试MAE: {test_mae:.4f}")

# 8. 进行预测
print("\n7. 进行预测...")
predictions_scaled = model.predict(X_test_reshaped, verbose=0)

# 反标准化预测结果
predictions = predictions_scaled * std_price + mean_price
actual = y_test

# 计算预测准确性
mae = np.mean(np.abs(predictions.flatten() - actual))
print(f"预测平均绝对误差: {mae:.2f}")

# 9. 可视化结果
plt.figure(figsize=(15, 10))

# 训练过程
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练过程')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 预测 vs 实际
plt.subplot(2, 2, 2)
plt.plot(actual[:100], label='实际价格', alpha=0.7)
plt.plot(predictions[:100], label='预测价格', alpha=0.7)
plt.title('预测结果对比（前100个测试样本）')
plt.xlabel('样本')
plt.ylabel('股价')
plt.legend()
plt.grid(True)

# 预测误差分布
plt.subplot(2, 2, 3)
errors = predictions.flatten() - actual
plt.hist(errors, bins=30, alpha=0.7)
plt.title('预测误差分布')
plt.xlabel('误差')
plt.ylabel('频次')
plt.grid(True)

# 散点图：预测 vs 实际
plt.subplot(2, 2, 4)
plt.scatter(actual, predictions.flatten(), alpha=0.5)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
plt.title('预测值 vs 实际值')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.grid(True)

plt.tight_layout()
plt.show()

# 10. 未来预测示例
print("\n8. 未来预测示例...")

# 使用最后10天的数据预测下一天
last_sequence = X_test_scaled[-1:].reshape(1, sequence_length, 1)
next_prediction_scaled = model.predict(last_sequence, verbose=0)
next_prediction = next_prediction_scaled * std_price + mean_price

print(f"最后10天价格: {X_test[-1]}")
print(f"预测下一天价格: {next_prediction[0][0]:.2f}")
print(f"实际下一天价格: {y_test[-1]:.2f}")
print(f"预测误差: {abs(next_prediction[0][0] - y_test[-1]):.2f}")

print("\n" + "=" * 60)
print("✅ 简单时序预测完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 时序数据的特点：当前值依赖于历史值")
print("2. 序列数据的准备：滑动窗口方法")
print("3. RNN的基本原理：处理序列数据")
print("4. 数据标准化的重要性")
print("5. 时序预测的评估方法")
print("=" * 60)
