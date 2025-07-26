"""
TensorFlow 深度学习入门示例 1: 线性回归（详细注释版）
专为JS和Java开发者设计的机器学习入门

线性回归概念：
- 最简单的机器学习算法
- 目标：找到一条直线最好地拟合数据点
- 数学公式：y = wx + b（w是权重，b是偏置）
- 类似于Excel中的趋势线功能
"""

# ==================== 导入库详解 ====================
import tensorflow as tf     # Google的深度学习框架
import numpy as np         # 数值计算库，Python科学计算的基础
import matplotlib.pyplot as plt  # 绘图库，用于数据可视化

"""
库的作用对比：
- tensorflow：类似于Java的深度学习框架，提供神经网络功能
- numpy：类似于Java的数学工具类，但功能强大得多
- matplotlib：类似于前端的图表库（如Chart.js），用于绘制图形
"""

# 设置matplotlib中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

"""
plt.rcParams：
- matplotlib的全局配置字典
- 类似于应用程序的配置文件
- 确保中文字符在图表中正常显示
"""

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# ==================== 数据生成详解 ====================
print("\n1. 创建训练数据")

# 设置随机种子确保结果可重现
np.random.seed(42)  
"""
随机种子的作用：
- 确保每次运行程序产生相同的随机数
- 类似于游戏中的地图种子
- 便于调试和结果对比
- 42是任意选择的数字（程序员的幽默）
"""

# 生成输入数据X
X_train = np.random.uniform(-10, 10, 100).astype(np.float32)
"""
分解这个复杂语句：

1. np.random.uniform(-10, 10, 100)：
   - 生成100个[-10, 10]之间的均匀分布随机数
   - 类似于Java: Random.nextDouble() * 20 - 10 (执行100次)
   - 类似于JS: Math.random() * 20 - 10 (执行100次)

2. .astype(np.float32)：
   - 转换数据类型为32位浮点数
   - 深度学习通常使用float32节省内存
   - 类似于Java的类型转换: (float)value

结果：[3.2, -7.1, 5.8, -2.4, ...] 共100个数字
"""

# 生成目标数据y（我们要学习的函数：y = 2x + 1 + 噪声）
y_train = 2 * X_train + 1 + np.random.normal(0, 1, 100).astype(np.float32)
"""
分解这个公式：

1. 2 * X_train：
   - NumPy广播：每个x值乘以2
   - [3.2, -7.1] -> [6.4, -14.2]

2. + 1：
   - NumPy广播：每个值加1
   - [6.4, -14.2] -> [7.4, -13.2]

3. np.random.normal(0, 1, 100)：
   - 生成100个正态分布随机数（均值0，标准差1）
   - 模拟真实数据中的噪声和测量误差

4. 最终公式：y = 2x + 1 + 噪声
   - 这是我们希望模型学会的关系
   - 模型需要从带噪声的数据中发现这个规律
"""

print(f"训练数据形状: X={X_train.shape}, y={y_train.shape}")
"""
数据形状说明：
- X_train.shape = (100,)：100个输入值的一维数组
- y_train.shape = (100,)：100个目标值的一维数组
- 类似于两个长度为100的数组
"""

# 可视化训练数据
plt.figure(figsize=(10, 6))  # 创建图形，设置大小
plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')  # 散点图
plt.xlabel('X (输入)')
plt.ylabel('y (目标)')
plt.title('线性回归训练数据')
plt.legend()
plt.grid(True)
plt.show()

"""
matplotlib绘图函数：
- plt.figure()：创建新图形 (10, 6) 分布代表宽高，单位是英寸
- plt.scatter()：绘制散点图，可以新增s=，后面的序列表示每个点的绘制直径，要与x，y的数量一致
- alpha=0.6：设置透明度（0-1之间）
- plt.xlabel/ylabel()：设置坐标轴标签
- plt.legend()：显示图例
- plt.grid()：显示网格
- plt.show()：显示图形
"""

# ==================== 模型构建详解 ====================
print("\n2. 构建线性回归模型")

# 创建最简单的神经网络模型
model = tf.keras.Sequential([
    """
    tf.keras.Sequential：
    - 顺序模型，层级式堆叠
    - 类似于搭积木，一层叠一层
    - 适合简单的前馈神经网络
    """
    
    tf.keras.layers.Dense(
        1,                    # units=1：输出维度为1（一个神经元）
        input_shape=(1,),     # 输入形状：接受1个特征
        name='linear_layer'   # 层的名称（可选）
    )
    """
    Dense层（全连接层）：
    - 最基本的神经网络层
    - 实现线性变换：y = Wx + b
    - W是权重矩阵，b是偏置向量
    
    参数说明：
    - units=1：这层有1个神经元（输出1个值）
    - input_shape=(1,)：输入是1维的（接受1个特征）
    - 没有activation参数：默认是线性激活（无激活函数）
    
    这个模型的结构：
    输入(1个数) -> Dense层(1个神经元) -> 输出(1个数)
    实际上就是学习 y = wx + b 这个线性函数
    """
])

# ==================== 模型编译详解 ====================
model.compile(
    optimizer='adam',       # 优化器：控制如何更新模型参数
    loss='mse',            # 损失函数：衡量预测错误的程度
    metrics=['mae']        # 评估指标：训练过程中监控的指标
)

"""
编译参数详解：

1. optimizer='adam'：
   - Adam优化器：一种智能的参数更新算法
   - 会自动调整学习率
   - 类似于一个聪明的学习策略
   - 比传统的梯度下降更高效

2. loss='mse'：
   - MSE = Mean Squared Error（均方误差）
   - 计算公式：Σ(预测值 - 真实值)² / 样本数
   - 衡量预测值与真实值的差距
   - 值越小表示模型越准确

3. metrics=['mae']：
   - MAE = Mean Absolute Error（平均绝对误差）
   - 计算公式：Σ|预测值 - 真实值| / 样本数
   - 用于监控训练过程，不影响参数更新
   - 更直观地反映预测误差
"""

# 查看模型结构
print("模型结构:")
model.summary()
"""
model.summary()输出解释：
- Layer (type)：层的类型
- Output Shape：输出形状
- Param #：参数数量
- 对于Dense(1)层：有2个参数（1个权重 + 1个偏置）
"""

# ==================== 模型训练详解 ====================
print("\n3. 开始训练模型...")

# 重塑输入数据为2D格式（TensorFlow要求）
X_train_reshaped = X_train.reshape(-1, 1)
"""
数据重塑：
- reshape(-1, 1)：转换为N行1列的二维数组
- -1表示自动计算行数
- 从 [1, 2, 3] 变成 [[1], [2], [3]]
- TensorFlow需要2D输入，即使只有1个特征
"""

# 训练模型
history = model.fit(
    X_train_reshaped, y_train,  # 训练数据
    epochs=100,                 # 训练轮数
    batch_size=32,              # 批次大小
    verbose=1                   # 显示训练进度
)

"""
训练参数详解：

1. X_train_reshaped, y_train：
   - 输入特征和目标标签
   - 类似于练习题和标准答案

2. epochs=100：
   - 训练轮数，模型看完整数据集的次数
   - 类似于学生复习课本100遍
   - 每轮都会调整模型参数

3. batch_size=32：
   - 批次大小，每次更新参数时使用的样本数
   - 类似于每次做32道题后总结一次
   - 平衡训练速度和稳定性

4. verbose=1：
   - 控制输出详细程度
   - 1=显示进度条，0=静默，2=每轮一行

5. 返回值history：
   - 包含训练过程中的损失值、指标等
   - 可用于分析训练效果和绘制学习曲线
"""

# ==================== 结果分析详解 ====================
print("\n4. 分析训练结果...")

# 获取模型学到的参数
weights, bias = model.layers[0].get_weights()
"""
获取模型参数：
- model.layers[0]：获取第一层（我们只有一层）
- .get_weights()：获取权重和偏置
- 返回[权重矩阵, 偏置向量]

对于线性回归 y = wx + b：
- weights[0][0]：权重w（斜率）
- bias[0]：偏置b（截距）
"""

print(f"学到的权重(斜率): {weights[0][0]:.4f} (目标值: 2.0)")
print(f"学到的偏置(截距): {bias[0]:.4f} (目标值: 1.0)")
print(f"最终训练损失: {history.history['loss'][-1]:.6f}")

"""
结果解释：
- 权重应该接近2.0（我们的目标函数是y=2x+1）
- 偏置应该接近1.0
- 损失值越小表示模型学得越好
- history.history['loss'][-1]：取最后一轮的损失值
"""

# ==================== 预测和可视化 ====================
print("\n5. 进行预测和可视化...")

# 生成测试数据进行预测
X_test = np.linspace(-10, 10, 100).reshape(-1, 1)  # 等间距的测试点
y_pred = model.predict(X_test, verbose=0)           # 模型预测

"""
预测过程：
- np.linspace(-10, 10, 100)：生成100个等间距的点
- model.predict()：使用训练好的模型进行预测
- verbose=0：不显示预测进度
"""

# 绘制结果
plt.figure(figsize=(12, 8))

# 原始数据和预测结果
plt.subplot(2, 2, 1)  # 2行2列的第1个子图
plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='模型预测')
plt.plot(X_test, 2*X_test + 1, 'g--', linewidth=2, label='真实函数 y=2x+1')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归结果')
plt.legend()
plt.grid(True)

# 训练过程
plt.subplot(2, 2, 2)  # 第2个子图
plt.plot(history.history['loss'], label='训练损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('训练过程')
plt.legend()
plt.grid(True)

# 预测误差分析
plt.subplot(2, 2, 3)  # 第3个子图
train_pred = model.predict(X_train_reshaped, verbose=0)
errors = train_pred.flatten() - y_train
plt.hist(errors, bins=20, alpha=0.7)
plt.xlabel('预测误差')
plt.ylabel('频次')
plt.title('误差分布')
plt.grid(True)

# 预测 vs 实际
plt.subplot(2, 2, 4)  # 第4个子图
plt.scatter(y_train, train_pred, alpha=0.6)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值 vs 实际值')
plt.grid(True)

plt.tight_layout()  # 自动调整子图间距
plt.show()

"""
可视化说明：
- subplot(2,2,1)：创建2x2网格的第1个子图
- plt.plot()：绘制线图
- 'r-'：红色实线，'g--'：绿色虚线
- linewidth：线条粗细
- plt.hist()：绘制直方图
- plt.tight_layout()：自动调整布局避免重叠
"""

print("\n" + "=" * 60)
print("✅ 线性回归训练完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 线性回归：最简单的机器学习算法")
print("2. 神经网络：由层组成的计算图")
print("3. 训练过程：通过数据调整参数")
print("4. 损失函数：衡量预测错误的指标")
print("5. 优化器：更新参数的算法")
print("=" * 60)
