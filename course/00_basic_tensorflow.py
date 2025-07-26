"""
TensorFlow 深度学习入门示例 0: 基础TensorFlow操作
这个示例不需要matplotlib，可以直接运行来测试TensorFlow环境

Python语法说明：
- 三引号(\"\"\")：多行字符串，类似于Java的多行注释
- import：导入模块，类似于Java的import或JS的require/import
"""

# Python导入语句详解：
import tensorflow as tf  # 导入TensorFlow库，并给它一个简短的别名tf
                        # 类似于 Java: import tensorflow.* as tf
                        # 类似于 JS: import * as tf from 'tensorflow'

import numpy as np      # 导入NumPy数学计算库，别名np
                        # NumPy是Python中处理数组和矩阵的核心库
                        # 类似于Java中的数学工具类，但功能更强大

# Python字符串操作详解：
print("=" * 60)  # Python字符串乘法：重复字符串60次
                  # 类似于 Java: String.repeat("=", 60)
                  # 类似于 JS: "=".repeat(60)

print("🚀 TensorFlow基础操作示例")  # print()函数：输出到控制台
                                    # 类似于 Java: System.out.println()
                                    # 类似于 JS: console.log()

print("=" * 60)

# 1. 检查TensorFlow版本和GPU
# Python f-string语法详解：
print(f"TensorFlow版本: {tf.__version__}")
# f"..."：格式化字符串，{}内可以放变量或表达式
# 类似于 Java: String.format("TensorFlow版本: %s", tf.__version__)
# 类似于 JS: `TensorFlow版本: ${tf.__version__}`

# Python布尔表达式和函数调用：
print(f"GPU可用: {len(tf.config.list_physical_devices('GPU')) > 0}")
# len()：获取列表长度，类似于Java的list.size()或JS的array.length
# tf.config.list_physical_devices('GPU')：TensorFlow函数，获取GPU设备列表
# > 0：比较运算符，返回True或False

# Python条件语句：
if len(tf.config.list_physical_devices('GPU')) > 0:  # if语句，类似于Java/JS
    # Python缩进：用空格或Tab表示代码块，类似于Java/JS的{}
    print(f"GPU设备: {tf.config.list_physical_devices('GPU')[0].name}")
    # [0]：列表索引，获取第一个元素，类似于Java/JS的数组索引

# Python字符串连接：
print("\n" + "=" * 60)  # \n：换行符，类似于Java的\n或JS的\n
                        # +：字符串连接，类似于Java/JS的字符串拼接

print("📊 基础张量操作")
print("=" * 60)

# 2. 创建张量（Tensor）
print("\n1. 创建张量:")

# Python变量赋值：
a = tf.constant([1, 2, 3, 4])  # 创建TensorFlow常量张量
# Python列表语法：[1, 2, 3, 4] 类似于Java的int[]{1,2,3,4}或JS的[1,2,3,4]
# tf.constant()：TensorFlow函数，创建不可变的张量（类似于Java的final数组）

b = tf.constant([5, 6, 7, 8])  # 创建另一个张量

# 输出张量内容：
print(f"张量a: {a}")  # 张量会显示其值、形状、数据类型等信息
print(f"张量b: {b}")

"""
张量(Tensor)概念解释：
- 张量是多维数组的泛化概念
- 0维张量：标量（单个数字）
- 1维张量：向量（一维数组）[1,2,3,4]
- 2维张量：矩阵（二维数组）[[1,2],[3,4]]
- 3维张量：三维数组（如RGB图像）
- 类似于Java中的多维数组，但功能更强大
"""

# 3. 张量运算
print("\n2. 张量运算:")

# TensorFlow张量运算（类似于数组的元素级运算）：
print(f"a + b = {a + b}")      # 元素对应相加：[1+5, 2+6, 3+7, 4+8] = [6,8,10,12]
print(f"a * b = {a * b}")      # 元素对应相乘：[1*5, 2*6, 3*7, 4*8] = [5,12,21,32]
print(f"a的平方 = {tf.square(a)}")  # tf.square()：TensorFlow函数，计算平方

# 4. 矩阵操作
print("\n3. 矩阵操作:")

# 创建2D张量（矩阵）：
matrix1 = tf.constant([[1, 2], [3, 4]])  # 2x2矩阵
# Python嵌套列表：[[1,2],[3,4]] 表示二维数组
# 类似于 Java: int[][] matrix = {{1,2},{3,4}}
# 类似于 JS: [[1,2],[3,4]]

matrix2 = tf.constant([[5, 6], [7, 8]])  # 另一个2x2矩阵

print(f"矩阵1:\n{matrix1}")  # \n在字符串中表示换行
print(f"矩阵2:\n{matrix2}")

# 矩阵乘法（线性代数运算）：
print(f"矩阵乘法:\n{tf.matmul(matrix1, matrix2)}")
# tf.matmul()：矩阵乘法函数，不是元素对应相乘
# 结果：[[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]

"""
矩阵乘法解释：
matrix1 = [[1,2],    matrix2 = [[5,6],
           [3,4]]               [7,8]]

结果[0][0] = 1*5 + 2*7 = 19
结果[0][1] = 1*6 + 2*8 = 22
结果[1][0] = 3*5 + 4*7 = 43
结果[1][1] = 3*6 + 4*8 = 50
"""

print("\n" + "=" * 60)
print("🧠 简单神经网络示例")
print("=" * 60)

# 5. 创建简单的线性回归模型
print("\n4. 创建线性回归模型:")

# 生成训练数据 y = 2x + 1 + 噪声
np.random.seed(42)
# 生成100个随机数，范围在[-1, 1)之间，然后转换为float32类型，最后reshape为100行1列的矩阵
# reshape函数将矩阵重塑为指定的形状，-1表示自动计算该维度的大小，1表示列数,
# 也就是这将导致x为一个100长度数组，数组内每个元素为长度为1的数组，类似 [[1],[1],[1]...,[1]]
X_train = np.random.uniform(-1, 1, 100).astype(np.float32).reshape(-1, 1)
# flatten函数将矩阵扁平化，[[1, 2], [3, 4]]转为[1, 2, 3, 4]
y_train = 2 * X_train.flatten() + 1 + np.random.normal(0, 0.1, 100).astype(np.float32)

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
print(f"前5个样本: X={X_train[:5].flatten()}, y={y_train[:5]}")

# 6. 构建模型
model = tf.keras.Sequential([ # 创建线性堆叠的神经网络容器
    tf.keras.layers.Dense( # 添加全连接层
        units=1, # 输出空间的维度（神经元数量），此处为单输出;设置本层神经元的输出维度为1
        input_shape=(1,), # 定义输入数据的形状（仅模型首层需指定）声明输入数据的形状为1维向量,如上面声明的x，x就是一维向量
        name='linear_layer' # 本层的自定义名称（便于调试和可视化）
    )
])

# 7. 编译模型
'''
optimizer='adam'：优化算法: 控制模型权重的更新策略，通过梯度下降最小化损失函数
loss='mse'：损失函数: 量化模型预测值与真实值的误差，指导优化方向
metrics=['mae']：评估指标: MAE =  Σ|y - y| / n 绝对误差平均值
'''
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print(f"\n5. 模型结构:")
# 打印模型结构
model.summary()

# 8. 训练模型
print(f"\n6. 开始训练...")
# fit函数 拟合训练，使输出曲线逐渐接近输入数据
history = model.fit(
    X_train, y_train, # 训练数据
    epochs=50,  # 训练轮数
    batch_size=32, # 每次梯度更新所用的样本数,每轮从原始数据集取32个重新训练
    verbose=0  # 不显示训练过程，避免输出太多
)

# 9. 获取训练结果
weights, bias = model.layers[0].get_weights()
print(f"\n7. 训练结果:")
# 其实获取到的是一个kx+b的线性函数，weights[0][0]表示k，bias[0]表示b
print(f"学到的权重(斜率): {weights[0][0]:.4f} (目标值: 2.0)")
print(f"学到的偏置(截距): {bias[0]:.4f} (目标值: 1.0)")
print(f"最终损失: {history.history['loss'][-1]:.6f}")

# 10. 测试预测
print(f"\n8. 模型预测测试:")
test_inputs = [0.0, 0.5, -0.5, 1.0]
for x in test_inputs:
    pred = model.predict([[x]], verbose=0)[0][0]
    true_val = 2 * x + 1
    error = abs(pred - true_val)
    print(f"输入: {x:4.1f}, 预测: {pred:6.3f}, 真实值: {true_val:6.3f}, 误差: {error:.3f}")

print("\n" + "=" * 60)
print("🎯 MNIST手写数字识别(简化版)")
print("=" * 60)

# 11. 加载MNIST数据集
print("\n9. 加载MNIST数据集...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)  # 展平为784维
x_test = x_test.reshape(x_test.shape[0], -1)

print(f"训练集形状: {x_train.shape}")
print(f"测试集形状: {x_test.shape}")
print(f"标签范围: {y_train.min()} - {y_train.max()}")

# 12. 构建神经网络
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

mnist_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n10. MNIST模型结构:")
mnist_model.summary()

# 13. 训练MNIST模型(只训练3轮，节省时间)
print(f"\n11. 训练MNIST模型...")
mnist_history = mnist_model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 14. 评估模型
test_loss, test_accuracy = mnist_model.evaluate(x_test, y_test, verbose=0)
print(f"\n12. MNIST模型评估结果:")
print(f"测试准确率: {test_accuracy:.4f}")
print(f"测试损失: {test_loss:.4f}")

# 15. 预测示例
print(f"\n13. 预测示例:")
predictions = mnist_model.predict(x_test[:5], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

for i in range(5):
    confidence = np.max(predictions[i]) * 100
    print(f"样本{i+1}: 真实标签={y_test[i]}, 预测标签={predicted_classes[i]}, 置信度={confidence:.1f}%")

print("\n" + "=" * 60)
print("✅ 所有示例运行完成!")
print("=" * 60)
print("🎉 恭喜！你的TensorFlow环境配置正确，可以正常运行深度学习模型！")
print("📚 接下来可以运行其他带可视化的示例(需要安装matplotlib)")
print("=" * 60)
