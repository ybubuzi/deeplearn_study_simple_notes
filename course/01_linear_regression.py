"""
TensorFlow 深度学习入门示例 1: 线性回归
这是最简单的机器学习模型，用于理解TensorFlow的基本概念
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# 1. 创建训练数据
# 我们要学习的函数是 y = 2x + 1 + 噪声
np.random.seed(42)  # 设置随机种子，确保结果可重现
X_train = np.random.uniform(-10, 10, 100).astype(np.float32)  #  生成 ​100个​ 在区间 [-10, 10) 内的均匀分布**随机浮点数
y_train = 2 * X_train + 1 + np.random.normal(0, 1, 100).astype(np.float32)  # 添加噪声

# 🔧 关键修复：重塑输入数据为2D格式
X_train = X_train.reshape(-1, 1)  # 从 (100,) 变成 (100, 1)

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
print(f"X样本示例: {X_train[:3].flatten()}")
print(f"y样本示例: {y_train[:3]}")

# 2. 构建模型
# 这是一个最简单的神经网络：只有一个神经元，没有激活函数
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), name='linear_layer')  # 1个输入，1个输出
])

# 3. 编译模型
# 优化器：Adam（自适应学习率）
# 损失函数：均方误差（MSE）
# 评估指标：平均绝对误差（MAE）
model.compile(
    optimizer='adam',
    loss='mse',  # 均方误差
    metrics=['mae']  # 平均绝对误差
)

# 4. 查看模型结构
print("\n模型结构:")
model.summary()

# 5. 训练模型
print("\n开始训练...")
print("注意观察损失值是否稳定下降...")

# 🔧 优化训练参数
history = model.fit(
    X_train, y_train,
    epochs=50,         # 减少到50轮，避免过拟合
    batch_size=16,     # 减小批次大小，提高稳定性
    validation_split=0.2,  # 20%的数据用于验证
    verbose=1          # 显示训练过程
)

# 6. 获取训练后的参数
weights, bias = model.layers[0].get_weights()
print(f"\n训练结果:")
print(f"学到的权重(斜率): {weights[0][0]:.4f} (真实值: 2.0)")
print(f"学到的偏置(截距): {bias[0]:.4f} (真实值: 1.0)")

# 7. 进行预测
X_test = np.linspace(-10, 10, 50).astype(np.float32)
# 🔧 关键修复：测试数据也需要重塑为2D
X_test = X_test.reshape(-1, 1)  # 从 (50,) 变成 (50, 1)
y_pred = model.predict(X_test, verbose=0)

# 8. 可视化结果
plt.figure(figsize=(12, 4))

# 绘制训练过程
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练过程')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 绘制预测结果
plt.subplot(1, 2, 2)
# 🔧 修复：展平数据用于绘图
plt.scatter(X_train.flatten(), y_train, alpha=0.6, label='训练数据')
plt.plot(X_test.flatten(), y_pred.flatten(), 'r-', linewidth=2, label=f'预测线 (y={weights[0][0]:.2f}x+{bias[0]:.2f})')
plt.plot(X_test.flatten(), 2*X_test.flatten() + 1, 'g--', linewidth=2, label='真实线 (y=2x+1)')
plt.title('线性回归结果')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. 测试模型
test_values = [0, 5, -3]
print(f"\n模型测试:")
for x in test_values:
    # 🔧 修复：确保输入格式正确
    x_input = np.array([[x]], dtype=np.float32)  # 正确的2D格式
    pred = model.predict(x_input, verbose=0)[0][0]
    true_val = 2 * x + 1
    error = abs(pred - true_val)
    print(f"输入: {x:2}, 预测: {pred:7.4f}, 真实值: {true_val:2}, 误差: {error:.4f}")

# 10. 模型性能评估
final_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
print(f"\n📊 最终性能:")
print(f"训练损失: {final_loss:.6f}")
print(f"验证损失: {final_val_loss:.6f}")
print(f"权重误差: {abs(weights[0][0] - 2.0):.4f}")
print(f"偏置误差: {abs(bias[0] - 1.0):.4f}")

if final_loss < 1.0 and abs(weights[0][0] - 2.0) < 0.1:
    print("✅ 模型训练成功！")
else:
    print("⚠️ 模型可能需要调优")
