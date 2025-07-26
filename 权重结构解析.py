"""
神经网络权重结构详解
解释为什么需要 weights[0][0] 来获取权重值
"""

import tensorflow as tf
import numpy as np

print("🔍 神经网络权重结构解析")
print("=" * 50)

# 1. 创建简单的线性回归模型
print("\n1. 创建模型并查看权重结构")
print("-" * 30)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), name='linear_layer')
])

# 编译模型（这会初始化权重）
model.compile(optimizer='adam', loss='mse')

# 获取权重
weights, bias = model.layers[0].get_weights()

print("📊 权重详细信息:")
print(f"weights 类型: {type(weights)}")
print(f"weights 形状: {weights.shape}")
print(f"weights 内容:\n{weights}")
print(f"weights 维度数: {weights.ndim}")

print(f"\nbias 类型: {type(bias)}")
print(f"bias 形状: {bias.shape}")
print(f"bias 内容: {bias}")
print(f"bias 维度数: {bias.ndim}")

# 2. 解释权重矩阵的结构
print("\n2. 权重矩阵结构解释")
print("-" * 30)

print("🧠 Dense层的数学原理:")
print("Dense层执行的运算: output = input × weights + bias")
print()
print("对于我们的线性回归:")
print("- 输入维度: 1 (一个特征)")
print("- 输出维度: 1 (一个预测值)")
print("- 所以权重矩阵形状: (输入维度, 输出维度) = (1, 1)")
print()
print("权重矩阵结构:")
print("weights = [[w]]  # 2D数组，形状为(1,1)")
print("          ↑")
print("          这就是我们要的权重值")

# 3. 不同网络结构的权重对比
print("\n3. 不同网络结构的权重对比")
print("-" * 30)

# 创建不同结构的网络
models = {
    "1输入→1输出": tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))]),
    "1输入→3输出": tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(1,))]),
    "2输入→1输出": tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(2,))]),
    "2输入→3输出": tf.keras.Sequential([tf.keras.layers.Dense(3, input_shape=(2,))])
}

for name, model in models.items():
    model.compile(optimizer='adam', loss='mse')
    weights, bias = model.layers[0].get_weights()
    print(f"{name}:")
    print(f"  权重形状: {weights.shape}")
    print(f"  偏置形状: {bias.shape}")
    print(f"  权重内容:\n{weights}")
    print()

# 4. 访问权重的不同方式
print("\n4. 访问权重的不同方式")
print("-" * 30)

# 重新获取我们的线性回归模型权重
model_simple = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])
model_simple.compile(optimizer='adam', loss='mse')
weights, bias = model_simple.layers[0].get_weights()

print("权重矩阵:", weights)
print("权重矩阵形状:", weights.shape)
print()

print("🔍 不同的访问方式:")
print(f"weights[0][0] = {weights[0][0]}")  # 标准方式
print(f"weights[0, 0] = {weights[0, 0]}")  # NumPy风格
print(f"weights.item() = {weights.item()}")  # 提取标量值
print(f"weights.flatten()[0] = {weights.flatten()[0]}")  # 展平后取第一个

print("\n💡 为什么用 weights[0][0]:")
print("1. weights 是形状为 (1,1) 的2D数组")
print("2. 第一个 [0] 选择第一行（也是唯一一行）")
print("3. 第二个 [0] 选择第一列（也是唯一一列）")
print("4. 结果得到标量值（单个数字）")

# 5. 实际训练后的权重
print("\n5. 训练后的权重变化")
print("-" * 30)

# 生成训练数据
np.random.seed(42)
X_train = np.random.uniform(-1, 1, 100).reshape(-1, 1)
y_train = 2 * X_train.flatten() + 1 + np.random.normal(0, 0.1, 100)

print("训练前的权重:")
weights_before, bias_before = model_simple.layers[0].get_weights()
print(f"权重矩阵: {weights_before}")
print(f"权重值: {weights_before[0][0]:.6f}")
print(f"偏置值: {bias_before[0]:.6f}")

# 训练模型
print("\n开始训练...")
model_simple.fit(X_train, y_train, epochs=50, verbose=0)

print("\n训练后的权重:")
weights_after, bias_after = model_simple.layers[0].get_weights()
print(f"权重矩阵: {weights_after}")
print(f"权重值: {weights_after[0][0]:.6f} (目标: 2.0)")
print(f"偏置值: {bias_after[0]:.6f} (目标: 1.0)")

# 6. 多层网络的权重结构
print("\n6. 多层网络的权重结构")
print("-" * 30)

# 创建多层网络
multi_layer_model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(2,), name='hidden_layer'),
    tf.keras.layers.Dense(1, name='output_layer')
])
multi_layer_model.compile(optimizer='adam', loss='mse')

print("多层网络结构:")
for i, layer in enumerate(multi_layer_model.layers):
    weights, bias = layer.get_weights()
    print(f"第{i+1}层 ({layer.name}):")
    print(f"  权重形状: {weights.shape}")
    print(f"  偏置形状: {bias.shape}")
    print(f"  权重矩阵:\n{weights}")
    print()

# 7. JavaScript类比理解
print("\n7. JavaScript类比理解")
print("-" * 30)

print("🔗 用JavaScript类比理解:")
print("""
// 如果用JavaScript表示权重矩阵
const weights = [
    [0.5]  // 第一行，第一列
];

// 访问权重值
const weightValue = weights[0][0];  // 0.5

// 类似于Python的
// weight_value = weights[0][0]
""")

print("🎯 总结:")
print("1. Dense层的权重总是2D矩阵，即使只有一个权重")
print("2. 形状为 (输入维度, 输出维度)")
print("3. 对于线性回归: (1, 1) - 1个输入，1个输出")
print("4. weights[0][0] 是访问这个2D矩阵中唯一元素的方式")
print("5. 这种设计保持了矩阵运算的一致性")

# 8. 常见错误演示
print("\n8. 常见错误演示")
print("-" * 30)

print("❌ 常见错误:")
try:
    # 错误：直接把weights当作标量
    wrong_value = weights + 1  # 这会进行数组运算，不是标量运算
    print(f"weights + 1 = {wrong_value}")
    print("这是数组运算，不是我们想要的标量运算")
except Exception as e:
    print(f"错误: {e}")

print("\n✅ 正确做法:")
correct_value = weights[0][0] + 1  # 先提取标量，再运算
print(f"weights[0][0] + 1 = {correct_value}")
print("这是标量运算，符合我们的预期")

print("\n" + "=" * 50)
print("🎓 关键理解:")
print("权重矩阵的设计是为了支持矩阵运算，")
print("即使只有一个权重，也要保持矩阵的结构！")
print("=" * 50)
