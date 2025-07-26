"""
线性回归训练问题诊断脚本
帮助理解为什么模型训练不稳定和误差大的问题
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🔍 线性回归问题诊断")
print("=" * 50)

# 1. 演示错误的数据格式问题
print("\n1. 数据格式问题演示")
print("-" * 30)

# 错误的数据格式
np.random.seed(42)
X_wrong = np.random.uniform(-10, 10, 100).astype(np.float32)  # 1D数组
y_train = 2 * X_wrong + 1 + np.random.normal(0, 1, 100).astype(np.float32)

print(f"❌ 错误格式 - X形状: {X_wrong.shape}")  # (100,)
print(f"   这是1D数组，但TensorFlow期望2D输入")

# 正确的数据格式
X_correct = X_wrong.reshape(-1, 1)  # 转换为2D
print(f"✅ 正确格式 - X形状: {X_correct.shape}")  # (100, 1)
print(f"   这是2D数组，符合TensorFlow要求")

# 2. 演示训练稳定性问题
print("\n2. 训练稳定性对比")
print("-" * 30)

def create_and_train_model(X_data, y_data, title, epochs=50):
    """创建并训练模型"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"\n{title}:")
    try:
        history = model.fit(X_data, y_data, epochs=epochs, verbose=0, batch_size=16)
        
        # 获取最终结果
        weights, bias = model.layers[0].get_weights()
        final_loss = history.history['loss'][-1]
        
        print(f"  最终损失: {final_loss:.6f}")
        print(f"  学到的权重: {weights[0][0]:.4f} (目标: 2.0)")
        print(f"  学到的偏置: {bias[0]:.4f} (目标: 1.0)")
        print(f"  权重误差: {abs(weights[0][0] - 2.0):.4f}")
        
        return model, history, weights[0][0], bias[0]
        
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        return None, None, None, None

# 使用错误格式训练（可能会失败）
print("尝试用1D数据训练...")
try:
    model_wrong, hist_wrong, w_wrong, b_wrong = create_and_train_model(
        X_wrong, y_train, "❌ 1D数据格式"
    )
except:
    print("❌ 1D数据格式: 训练失败（形状不匹配）")
    model_wrong, hist_wrong, w_wrong, b_wrong = None, None, None, None

# 使用正确格式训练
model_correct, hist_correct, w_correct, b_correct = create_and_train_model(
    X_correct, y_train, "✅ 2D数据格式"
)

# 3. 多次训练对比稳定性
print("\n3. 训练稳定性测试（多次运行）")
print("-" * 30)

weights_list = []
bias_list = []
loss_list = []

for i in range(5):
    # 每次使用不同的随机种子
    tf.random.set_seed(i)
    np.random.seed(i)
    
    model_test = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    model_test.compile(optimizer='adam', loss='mse')
    
    history = model_test.fit(X_correct, y_train, epochs=50, verbose=0, batch_size=16)
    weights, bias = model_test.layers[0].get_weights()
    
    weights_list.append(weights[0][0])
    bias_list.append(bias[0])
    loss_list.append(history.history['loss'][-1])
    
    print(f"运行{i+1}: 权重={weights[0][0]:.4f}, 偏置={bias[0]:.4f}, 损失={history.history['loss'][-1]:.6f}")

# 计算稳定性统计
weights_std = np.std(weights_list)
bias_std = np.std(bias_list)
print(f"\n稳定性分析:")
print(f"权重标准差: {weights_std:.6f} (越小越稳定)")
print(f"偏置标准差: {bias_std:.6f} (越小越稳定)")

# 4. 可视化结果
print("\n4. 可视化训练结果")
print("-" * 30)

if model_correct is not None:
    # 生成预测数据
    X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
    y_pred = model_correct.predict(X_test, verbose=0)
    
    plt.figure(figsize=(15, 5))
    
    # 子图1：数据和拟合结果
    plt.subplot(1, 3, 1)
    plt.scatter(X_correct.flatten(), y_train, alpha=0.6, label='训练数据')
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r-', linewidth=2, 
             label=f'预测线 (y={w_correct:.2f}x+{b_correct:.2f})')
    plt.plot(X_test.flatten(), 2*X_test.flatten() + 1, 'g--', linewidth=2, 
             label='真实线 (y=2x+1)')
    plt.title('拟合结果')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # 子图2：训练过程
    plt.subplot(1, 3, 2)
    plt.plot(hist_correct.history['loss'])
    plt.title('训练损失变化')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.grid(True)
    
    # 子图3：多次训练结果分布
    plt.subplot(1, 3, 3)
    plt.scatter(weights_list, bias_list, c=loss_list, cmap='viridis', s=100)
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='目标偏置=1.0')
    plt.axvline(x=2.0, color='r', linestyle='--', alpha=0.7, label='目标权重=2.0')
    plt.xlabel('权重')
    plt.ylabel('偏置')
    plt.title('多次训练结果分布')
    plt.colorbar(label='最终损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 5. 问题总结和解决方案
print("\n" + "=" * 50)
print("📋 问题总结和解决方案")
print("=" * 50)

print("\n🚨 常见问题:")
print("1. 数据形状不匹配:")
print("   - 问题: X是1D数组 (100,)，但Dense层需要2D输入")
print("   - 解决: 使用 X.reshape(-1, 1) 转换为 (100, 1)")

print("\n2. 训练不稳定:")
print("   - 问题: 每次训练结果差异很大")
print("   - 解决: 设置随机种子 np.random.seed(42)")

print("\n3. 误差很大:")
print("   - 问题: 学到的权重偏离目标值2.0很远")
print("   - 解决: 检查数据预处理、调整学习率、增加训练轮数")

print("\n✅ 最佳实践:")
print("1. 总是检查数据形状: print(X.shape)")
print("2. 设置随机种子确保可重现性")
print("3. 监控训练过程中的损失值变化")
print("4. 使用验证集检查过拟合")
print("5. 多次运行检查稳定性")

print("\n🔧 修复后的代码模板:")
print("""
# 正确的线性回归代码
np.random.seed(42)  # 设置随机种子
X_train = np.random.uniform(-10, 10, 100).astype(np.float32)
y_train = 2 * X_train + 1 + np.random.normal(0, 1, 100).astype(np.float32)

# 关键：重塑为2D格式
X_train = X_train.reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model.compile(optimizer='adam', loss='mse')

# 训练
history = model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# 预测时也要保持2D格式
X_test = np.linspace(-10, 10, 50).reshape(-1, 1)
y_pred = model.predict(X_test)
""")

print("\n现在运行修复后的 01_linear_regression.py 应该会得到稳定的结果！")
