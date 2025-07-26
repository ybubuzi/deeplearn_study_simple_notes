"""
TensorFlow 深度学习入门示例 0: 基础TensorFlow操作（详细注释版）
专为JS和Java开发者设计的Python+深度学习入门

Python基础语法对比：
- Python用缩进表示代码块，不用{}
- Python变量不需要声明类型
- Python函数调用和Java/JS类似
- Python列表用[]，字典用{}
"""

# ==================== 导入库详解 ====================
# Python的import语句类似于Java的import或JS的require/import

import tensorflow as tf  
"""
tensorflow：Google开发的深度学习框架
as tf：给库起个短别名，方便使用
类似于：
- Java: import tensorflow.* as tf (如果Java有这种语法)
- JS: import * as tf from 'tensorflow'
"""

import numpy as np      
"""
numpy：Python科学计算的核心库，处理数组和数学运算
别名np是约定俗成的
类似于：
- Java中的Math类，但功能强大得多
- JS中的数学库，但专门处理多维数组
"""

# ==================== 字符串和输出详解 ====================
print("=" * 60)  
"""
Python字符串操作：
- "=" * 60：字符串重复60次，生成分隔线
- 类似于 Java: String.repeat("=", 60)
- 类似于 JS: "=".repeat(60)
"""

print("🚀 TensorFlow基础操作示例")  
"""
print()函数：输出到控制台
- 类似于 Java: System.out.println()
- 类似于 JS: console.log()
- Python自动换行，不需要\n
"""

print("=" * 60)

# ==================== 变量和函数调用详解 ====================
print(f"TensorFlow版本: {tf.__version__}")  
"""
f-string语法（Python 3.6+）：
- f"..."：格式化字符串字面量
- {}内可以放变量、表达式、函数调用
- 类似于 Java: String.format("版本: %s", tf.__version__)
- 类似于 JS: `版本: ${tf.__version__}`

tf.__version__：
- __version__：Python约定的版本属性
- 双下划线表示特殊属性
"""

# ==================== 列表和布尔运算详解 ====================
gpu_devices = tf.config.list_physical_devices('GPU')  # 获取GPU设备列表
"""
变量赋值：
- Python不需要声明变量类型（动态类型）
- 类似于 JS: let gpu_devices = ...
- Java需要: List<Device> gpu_devices = ...

函数调用：
- tf.config.list_physical_devices('GPU')：TensorFlow函数
- 返回GPU设备的列表（Python list类型）
"""

print(f"GPU可用: {len(gpu_devices) > 0}")
"""
len()函数：获取列表长度
- 类似于 Java: list.size()
- 类似于 JS: array.length

布尔表达式：
- len(gpu_devices) > 0：比较运算，返回True或False
- Python布尔值：True/False（首字母大写）
- Java: true/false，JS: true/false
"""

# ==================== 条件语句详解 ====================
if len(gpu_devices) > 0:  # Python的if语句
    """
    Python条件语句语法：
    - if 条件:（注意冒号）
    - 缩进表示代码块（通常4个空格）
    - 不需要括号和花括号
    
    对比其他语言：
    Java: if (condition) { ... }
    JS: if (condition) { ... }
    Python: if condition: ...（缩进的内容）
    """
    print(f"GPU设备: {gpu_devices[0].name}")
    """
    列表索引：
    - gpu_devices[0]：获取第一个元素
    - 类似于 Java: list.get(0)
    - 类似于 JS: array[0]
    
    属性访问：
    - .name：访问对象的name属性
    - 类似于 Java: object.getName()
    - 类似于 JS: object.name
    """

print("\n" + "=" * 60)  # \n：换行符，+：字符串连接
print("📊 基础张量操作")
print("=" * 60)

# ==================== 张量创建详解 ====================
print("\n1. 创建张量:")

# Python列表语法：
a = tf.constant([1, 2, 3, 4])  
"""
Python列表：
- [1, 2, 3, 4]：Python列表语法
- 类似于 Java: int[] {1, 2, 3, 4} 或 Arrays.asList(1,2,3,4)
- 类似于 JS: [1, 2, 3, 4]

tf.constant()：
- TensorFlow函数，创建常量张量
- 张量：多维数组的泛化，深度学习的基本数据结构
- 类似于创建一个不可变的数组
"""

b = tf.constant([5, 6, 7, 8])

print(f"张量a: {a}")  # 输出张量信息（值、形状、数据类型）
print(f"张量b: {b}")

# ==================== 张量运算详解 ====================
print("\n2. 张量运算:")

print(f"a + b = {a + b}")      
"""
张量运算：
- a + b：元素对应相加
- [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
- 类似于向量运算，不是普通的数组拼接
"""

print(f"a * b = {a * b}")      
"""
元素对应相乘（Hadamard积）：
- [1,2,3,4] * [5,6,7,8] = [5,12,21,32]
- 不是矩阵乘法，是对应位置相乘
"""

print(f"a的平方 = {tf.square(a)}")  
"""
TensorFlow函数：
- tf.square()：计算每个元素的平方
- 类似于数学函数，但作用于整个张量
"""

# ==================== 矩阵操作详解 ====================
print("\n3. 矩阵操作:")

# 嵌套列表创建2D张量：
matrix1 = tf.constant([[1, 2], [3, 4]])  
"""
嵌套列表语法：
- [[1, 2], [3, 4]]：二维列表
- 外层列表包含内层列表
- 类似于 Java: int[][] {{1,2},{3,4}}
- 类似于 JS: [[1,2],[3,4]]

表示矩阵：
[1 2]
[3 4]
"""

matrix2 = tf.constant([[5, 6], [7, 8]])

print(f"矩阵1:\n{matrix1}")  # \n：在输出中换行
print(f"矩阵2:\n{matrix2}")

print(f"矩阵乘法:\n{tf.matmul(matrix1, matrix2)}")
"""
矩阵乘法（线性代数）：
- tf.matmul()：矩阵乘法函数
- 不是元素对应相乘，是线性代数的矩阵乘法
- 结果计算：
  [1 2] × [5 6] = [1×5+2×7  1×6+2×8] = [19 22]
  [3 4]   [7 8]   [3×5+4×7  3×6+4×8]   [43 50]
"""

print("\n" + "=" * 60)
print("🧠 简单神经网络示例")
print("=" * 60)

# ==================== 数据生成详解 ====================
print("\n4. 创建线性回归模型:")

# NumPy随机数生成：
np.random.seed(42)  
"""
随机种子：
- 设置随机数生成器的种子
- 确保每次运行产生相同的随机数序列
- 42是任意选择的数字
- 类似于 Java: Random random = new Random(42)
"""

# 复杂的链式调用详解：
X_train = np.random.uniform(-1, 1, 100).astype(np.float32).reshape(-1, 1)
"""
分解这个复杂语句：

1. np.random.uniform(-1, 1, 100)
   - 生成100个[-1, 1]之间的均匀分布随机数
   - 返回一维数组：[0.1, -0.5, 0.8, ...]

2. .astype(np.float32)
   - 转换数据类型为32位浮点数
   - 类似于 Java: (float)value
   - 深度学习通常用float32节省内存

3. .reshape(-1, 1)
   - 重塑数组形状
   - -1表示自动计算行数，1表示1列
   - [0.1, -0.5, 0.8] -> [[0.1], [-0.5], [0.8]]
   - 从一维变成二维（100行1列）

最终结果：100行1列的矩阵，每行一个数字
"""

y_train = 2 * X_train.flatten() + 1 + np.random.normal(0, 0.1, 100).astype(np.float32)
"""
分解这个语句：

1. X_train.flatten()
   - 将多维数组压平为一维
   - [[0.1], [-0.5]] -> [0.1, -0.5]
   - 类似于Java中将二维数组转为一维

2. 2 * X_train.flatten()
   - 数组广播：每个元素乘以2
   - [0.1, -0.5] -> [0.2, -1.0]

3. + 1
   - 数组广播：每个元素加1
   - [0.2, -1.0] -> [1.2, 0.0]

4. np.random.normal(0, 0.1, 100)
   - 生成100个正态分布随机数（均值0，标准差0.1）
   - 模拟真实数据中的噪声

5. 最终公式：y = 2x + 1 + 噪声
   - 这是一个线性关系，我们让模型学习这个关系
"""

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
"""
.shape属性：
- 返回数组的形状（维度信息）
- X_train.shape = (100, 1)：100行1列
- y_train.shape = (100,)：100个元素的一维数组
- 类似于Java中获取数组长度，但更详细
"""

print(f"前5个样本: X={X_train[:5].flatten()}, y={y_train[:5]}")
"""
数组切片：
- X_train[:5]：取前5个元素
- 类似于 Java: Arrays.copyOfRange(array, 0, 5)
- 类似于 JS: array.slice(0, 5)
- Python切片语法：[start:end]，不包含end
"""

# ==================== 神经网络模型构建详解 ====================
print("\n5. 模型结构:")

# Keras Sequential模型：
model = tf.keras.Sequential([
    """
    tf.keras.Sequential：
    - 顺序模型，层级式堆叠
    - 类似于建筑积木，一层叠一层
    - 类似于 Java 的 Builder 模式
    - 列表[]包含所有层
    """

    tf.keras.layers.Dense(
        """
        Dense层（全连接层）：
        - 最基本的神经网络层
        - 每个输入都连接到每个输出
        - 类似于线性代数中的矩阵乘法 + 偏置
        """
        units=1,                # 输出维度：这层有1个神经元
        input_shape=(1,),       # 输入形状：接受1个特征
        name='linear_layer'     # 层的名称（可选，便于调试）
    )
])

"""
神经网络层的概念：
- 输入层：接收数据
- 隐藏层：处理数据（这里没有）
- 输出层：产生结果（这里的Dense层）

这个模型非常简单：
输入(1个数) -> Dense层(1个神经元) -> 输出(1个数)
实际上就是学习 y = wx + b 这个线性函数
"""

# ==================== 模型编译详解 ====================
model.compile(
    """
    模型编译：设置训练参数
    类似于配置一个学习计划
    """
    optimizer='adam',           # 优化器：如何更新模型参数
    loss='mse',                # 损失函数：如何衡量预测错误
    metrics=['mae']            # 评估指标：训练过程中监控的指标
)

"""
参数详解：

1. optimizer='adam'：
   - Adam优化器：一种智能的参数更新算法
   - 类似于一个聪明的学习策略
   - 会自动调整学习速度

2. loss='mse'：
   - MSE = Mean Squared Error（均方误差）
   - 计算公式：(预测值 - 真实值)² 的平均值
   - 类似于衡量"错得有多离谱"

3. metrics=['mae']：
   - MAE = Mean Absolute Error（平均绝对误差）
   - 计算公式：|预测值 - 真实值| 的平均值
   - 用于监控训练过程，不影响训练
"""

print("模型结构:")
model.summary()  # 显示模型的详细信息
"""
model.summary()：
- 显示模型的层结构、参数数量等
- 类似于打印对象的详细信息
- 帮助理解模型的复杂度
"""

# ==================== 模型训练详解 ====================
print("\n6. 开始训练...")

history = model.fit(
    """
    model.fit()：训练模型的核心函数
    类似于让学生做练习题学习
    """
    X_train, y_train,          # 训练数据：输入和对应的正确答案
    epochs=50,                 # 训练轮数：看数据50遍
    batch_size=32,             # 批次大小：每次处理32个样本
    verbose=0                  # 输出详细程度：0=静默，1=详细
)

"""
训练参数详解：

1. X_train, y_train：
   - 训练数据和标签
   - 类似于练习题和答案

2. epochs=50：
   - 训练轮数，模型看完整数据集的次数
   - 类似于学生复习课本50遍
   - 每轮都会调整模型参数

3. batch_size=32：
   - 批次大小，每次更新参数时使用的样本数
   - 类似于每次做32道题后总结一次
   - 平衡训练速度和稳定性

4. verbose=0：
   - 控制训练过程的输出
   - 0=不输出，1=显示进度条，2=每轮一行

5. 返回值history：
   - 包含训练过程中的损失值、指标等
   - 类似于学习记录，可以用来分析训练效果
"""

# ==================== 获取训练结果详解 ====================
weights, bias = model.layers[0].get_weights()
"""
获取模型参数：

1. model.layers[0]：
   - 获取第一层（我们只有一层）
   - 类似于访问列表的第一个元素

2. .get_weights()：
   - 获取这层的权重参数
   - 返回[权重矩阵, 偏置向量]
   - 对于线性回归：y = wx + b，w是权重，b是偏置

3. weights, bias = ...：
   - Python的多重赋值语法
   - 类似于 Java: Weight w = result[0]; Bias b = result[1];
   - 类似于 JS: const [weights, bias] = result;
"""

print(f"\n7. 训练结果:")
print(f"学到的权重(斜率): {weights[0][0]:.4f} (目标值: 2.0)")
print(f"学到的偏置(截距): {bias[0]:.4f} (目标值: 1.0)")
print(f"最终损失: {history.history['loss'][-1]:.6f}")

"""
结果分析：

1. weights[0][0]：
   - 权重是二维数组，取第一行第一列
   - 应该接近2.0（我们的目标函数是y=2x+1）

2. bias[0]：
   - 偏置是一维数组，取第一个元素
   - 应该接近1.0

3. :.4f：
   - Python字符串格式化
   - 保留4位小数
   - 类似于 Java: String.format("%.4f", value)

4. history.history['loss'][-1]：
   - history.history：训练历史字典
   - ['loss']：获取损失值列表
   - [-1]：Python负索引，取最后一个元素
   - 类似于 JS: array[array.length - 1]
"""
