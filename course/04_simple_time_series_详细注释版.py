"""
TensorFlow 深度学习入门示例 4: 简单时序预测（详细注释版）
专为JS和Java开发者设计的时序数据入门

时序数据概念：
- 时序数据：按时间顺序排列的数据
- 特点：当前值依赖于历史值
- 例子：股价、天气、销量等
- 目标：根据历史数据预测未来
"""

# ==================== 导入库详解 ====================
import tensorflow as tf     # 深度学习框架
import numpy as np         # 数值计算库
import matplotlib.pyplot as plt  # 绘图库

"""
matplotlib.pyplot：
- Python的绘图库，类似于Excel图表功能
- plt是约定俗成的别名
- 用于可视化数据和结果
"""

# 设置中文字体（Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

"""
plt.rcParams：
- matplotlib的配置参数
- 类似于全局设置
- 确保中文字符正常显示
"""

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("📈 简单时序预测入门")
print("=" * 60)

# ==================== 函数定义详解 ====================
def create_stock_data(days=1000):
    """
    Python函数定义语法：
    - def 函数名(参数=默认值):
    - 三引号内是文档字符串（docstring）
    - 类似于 Java 的 Javadoc 或 JS 的 JSDoc
    """
    
    # 设置随机种子确保可重现
    np.random.seed(42)  
    
    # 创建基础趋势：从100到150的线性增长
    trend = np.linspace(100, 150, days)
    """
    np.linspace(start, stop, num)：
    - 在start和stop之间生成num个等间距的数
    - 类似于创建一个等差数列
    - 例：np.linspace(0, 10, 5) = [0, 2.5, 5, 7.5, 10]
    """
    
    # 添加随机波动（噪声）
    noise = np.random.normal(0, 5, days)
    """
    np.random.normal(均值, 标准差, 数量)：
    - 生成正态分布（高斯分布）的随机数
    - 均值0，标准差5，生成days个数
    - 模拟真实数据中的随机波动
    """
    
    # 添加周期性波动（模拟市场周期）
    cycle = 10 * np.sin(np.linspace(0, 4*np.pi, days))
    """
    正弦函数创建周期性模式：
    - np.sin()：正弦函数
    - 4*np.pi：4π，表示2个完整周期
    - 乘以10：振幅为10
    - 模拟股市的周期性波动
    """
    
    # 合成最终股价
    stock_price = trend + cycle + noise
    """
    NumPy数组运算：
    - 三个数组对应位置相加
    - trend[i] + cycle[i] + noise[i]
    - 类似于向量加法
    """
    
    return stock_price  # 返回生成的股价数据

# ==================== 数据生成和可视化 ====================
print("\n1. 生成模拟股价数据...")

# 调用函数生成数据
stock_prices = create_stock_data(1000)
"""
函数调用：
- 类似于 Java: double[] prices = createStockData(1000);
- 类似于 JS: const prices = createStockData(1000);
"""

print(f"生成了 {len(stock_prices)} 天的股价数据")
print(f"价格范围: {stock_prices.min():.2f} - {stock_prices.max():.2f}")

"""
数组方法：
- len(stock_prices)：获取数组长度
- stock_prices.min()：最小值
- stock_prices.max()：最大值
- :.2f：格式化为2位小数
"""

# 可视化前200天的数据
plt.figure(figsize=(12, 4))  # 创建图形，设置大小
"""
matplotlib绘图：
- plt.figure()：创建新图形
- figsize=(宽, 高)：设置图形大小（英寸）
"""

plt.plot(stock_prices[:200], label='股价')  # 绘制线图
"""
plt.plot()：绘制线图
- stock_prices[:200]：取前200个数据点
- label='股价'：图例标签
"""

plt.title('模拟股价数据（前200天）')  # 设置标题
plt.xlabel('天数')                    # x轴标签
plt.ylabel('股价')                    # y轴标签
plt.legend()                         # 显示图例
plt.grid(True)                       # 显示网格
plt.show()                          # 显示图形

# ==================== 时序数据预处理详解 ====================
print("\n2. 准备时序训练数据...")

def create_sequences(data, seq_length):
    """
    将时序数据转换为监督学习格式
    
    时序预测的核心思想：
    - 用过去N个时间点的数据预测下一个时间点
    - 类似于根据过去几天的股价预测明天的股价
    """
    
    # 初始化空列表存储结果
    X, y = [], []
    """
    Python列表初始化：
    - []：空列表
    - 类似于 Java: List<Double> X = new ArrayList<>();
    - 类似于 JS: const X = [];
    """
    
    # 遍历数据创建序列
    for i in range(len(data) - seq_length):
        """
        range()函数：
        - range(n)：生成0到n-1的整数序列
        - 类似于 Java: for(int i=0; i<n; i++)
        - 类似于 JS: for(let i=0; i<n; i++)
        """
        
        # 输入：过去seq_length天的价格
        X.append(data[i:i + seq_length])
        """
        列表切片和添加：
        - data[i:i + seq_length]：从i到i+seq_length的数据
        - .append()：添加到列表末尾
        - 类似于 Java: list.add()
        - 类似于 JS: array.push()
        """
        
        # 输出：下一天的价格
        y.append(data[i + seq_length])
    
    # 转换为NumPy数组
    return np.array(X), np.array(y)
    """
    np.array()：
    - 将Python列表转换为NumPy数组
    - NumPy数组支持高效的数学运算
    - 类似于将ArrayList转换为原生数组
    """

# 使用过去10天预测下一天
sequence_length = 10
X, y = create_sequences(stock_prices, sequence_length)

"""
多重赋值：
- Python可以同时赋值多个变量
- 类似于 Java: Pair<X,y> result = ...; X = result.first; y = result.second;
- 类似于 JS: const [X, y] = createSequences(...);
"""

print(f"序列长度: {sequence_length}")
print(f"训练样本数: {len(X)}")
print(f"X形状: {X.shape}")  # (样本数, 序列长度)
print(f"y形状: {y.shape}")  # (样本数,)

"""
数组形状：
- .shape：NumPy数组的形状属性
- X.shape = (990, 10)：990个样本，每个样本10个时间步
- y.shape = (990,)：990个目标值
"""

# 显示数据示例
print("\n数据示例:")
for i in range(3):  # 显示前3个样本
    print(f"样本{i+1}: 输入={X[i][:5]}... -> 输出={y[i]:.2f}")
    """
    字符串格式化：
    - X[i][:5]：取第i个样本的前5个值
    - ...：省略号表示还有更多数据
    - y[i]:.2f：目标值保留2位小数
    """

# ==================== 数据标准化详解 ====================
print("\n3. 数据标准化...")

# 分割训练和测试集
train_size = int(0.8 * len(X))  # 80%用于训练
"""
数据分割：
- int()：转换为整数
- 0.8 * len(X)：80%的数据量
- 类似于机器学习的标准做法
"""

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
"""
数组切片分割：
- X[:train_size]：从开始到train_size
- X[train_size:]：从train_size到结束
- 类似于 Java: Arrays.copyOfRange()
"""

# 计算标准化参数（只用训练数据）
mean_price = X_train.mean()  # 平均值
std_price = X_train.std()    # 标准差
"""
统计函数：
- .mean()：计算平均值
- .std()：计算标准差
- 只用训练数据计算，避免数据泄露
"""

# 应用标准化
X_train_scaled = (X_train - mean_price) / std_price
X_test_scaled = (X_test - mean_price) / std_price
y_train_scaled = (y_train - mean_price) / std_price
y_test_scaled = (y_test - mean_price) / std_price

"""
标准化公式：
- (x - 均值) / 标准差
- 将数据转换为均值0，标准差1的分布
- 帮助神经网络更好地学习
- 类似于将不同单位的数据统一到相同范围
"""

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"数据均值: {mean_price:.2f}")
print(f"数据标准差: {std_price:.2f}")

# ==================== RNN模型构建详解 ====================
print("\n4. 构建RNN模型...")

model = tf.keras.Sequential([
    # SimpleRNN层：处理时序数据的基础层
    tf.keras.layers.SimpleRNN(
        50,                           # 隐藏单元数：RNN的"记忆容量"
        input_shape=(sequence_length, 1),  # 输入形状：(时间步, 特征数)
        name='rnn_layer'
    ),
    """
    SimpleRNN详解：
    - RNN：循环神经网络，专门处理序列数据
    - 50：隐藏单元数，类似于"记忆细胞"的数量
    - input_shape=(10, 1)：10个时间步，每步1个特征
    - 能够记住之前的信息，适合时序预测
    """
    
    # 输出层：预测下一个价格
    tf.keras.layers.Dense(1, name='output_layer')
    """
    Dense层：
    - 全连接层，将RNN的输出转换为最终预测
    - 1：输出1个数值（下一天的股价）
    """
])

# 编译模型
model.compile(
    optimizer='adam',    # Adam优化器
    loss='mse',         # 均方误差损失
    metrics=['mae']     # 平均绝对误差指标
)

print("模型结构:")
model.summary()

"""
RNN模型的工作原理：
1. 输入：过去10天的股价 [价格1, 价格2, ..., 价格10]
2. RNN层：逐步处理每个时间步，累积记忆
3. 输出层：基于累积的记忆预测下一天价格
4. 类似于人类根据历史趋势做判断
"""
