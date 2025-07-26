"""
TensorFlow 深度学习入门示例 5: LSTM文本情感分析（详细注释版）
专为JS和Java开发者设计的文本处理和LSTM入门

文本处理概念：
- 计算机不能直接理解文字，需要转换为数字
- 词嵌入：将单词转换为向量（数字列表）
- LSTM：长短期记忆网络，比RNN更强的记忆能力
- 情感分析：判断文本是正面还是负面情绪
"""

# ==================== 导入库详解 ====================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("🎭 LSTM文本情感分析入门")
print("=" * 60)

# ==================== 数据加载详解 ====================
print("\n1. 加载IMDB电影评论数据...")

# 设置数据参数
max_features = 10000  # 词汇表大小：只使用最常见的10000个单词
max_length = 500      # 序列最大长度：每个评论最多500个单词

"""
参数设置的原因：
- max_features=10000：限制词汇表大小
  * 英语常用词汇约10000个已足够
  * 减少计算复杂度和内存使用
  * 类似于只学习最重要的单词

- max_length=500：限制评论长度
  * 大多数评论不超过500词
  * 统一输入长度，便于批处理
  * 类似于限制文章字数
"""

print("正在下载IMDB数据集...")

# 加载IMDB数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_features  # 只保留最常见的10000个单词
)

"""
IMDB数据集：
- 包含50000条电影评论
- 每条评论已经预处理为数字序列
- 标签：0=负面评论，1=正面评论
- 类似于已经分好类的用户评价数据

数据格式：
- X_train：评论文本（已转换为数字序列）
- y_train：情感标签（0或1）
- 例：[1, 14, 22, 16, ...] 表示一条评论的单词ID序列
"""

print(f"训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"标签: 0=负面评论, 1=正面评论")

# 查看数据结构
print(f"\n数据示例:")
print(f"第一个评论长度: {len(X_train[0])} 个单词")
print(f"第一个评论标签: {y_train[0]} ({'正面' if y_train[0] == 1 else '负面'})")
print(f"评论内容（数字编码）: {X_train[0][:20]}...")

"""
数据解释：
- len(X_train[0])：第一条评论的单词数量
- X_train[0][:20]：显示前20个单词的ID
- 每个数字代表一个单词，如：1=the, 2=and, 3=a 等

条件表达式：
- 'positive' if condition else 'negative'
- 类似于 Java: condition ? "positive" : "negative"
- 类似于 JS: condition ? "positive" : "negative"
"""

# ==================== 数据预处理详解 ====================
print("\n2. 数据预处理...")

# 序列填充：统一长度
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

"""
序列填充的必要性：
- 神经网络需要固定长度的输入
- 不同评论长度不同：有的50词，有的1000词
- 解决方案：统一到500词

填充策略：
- 短评论：前面补0（padding）
  例：[5, 10, 15] -> [0, 0, 0, ..., 5, 10, 15]
- 长评论：截断到500词
  例：[1, 2, 3, ..., 1000词] -> [1, 2, 3, ..., 500词]

类似于：
- Java: 将不同长度的数组统一为固定长度
- JS: 将不同长度的数组padding到相同长度
"""

print(f"填充后的数据形状:")
print(f"X_train: {X_train.shape}")  # (25000, 500)
print(f"X_test: {X_test.shape}")   # (25000, 500)
print(f"y_train: {y_train.shape}") # (25000,)

"""
数据形状解释：
- X_train.shape = (25000, 500)
  * 25000条训练评论
  * 每条评论500个单词
- y_train.shape = (25000,)
  * 25000个标签（一维数组）
"""

# 查看填充后的数据
print(f"\n填充后的第一个评论:")
print(f"前20个词: {X_train[0][:20]}")   # 可能都是0（填充值）
print(f"后20个词: {X_train[0][-20:]}")  # 实际的评论内容

"""
数组索引：
- X_train[0][:20]：第一条评论的前20个词
- X_train[0][-20:]：第一条评论的后20个词
- 负索引：-1表示最后一个，-20表示倒数第20个
"""

# ==================== LSTM模型构建详解 ====================
print("\n3. 构建LSTM模型...")

model = tf.keras.Sequential([
    
    # 词嵌入层：将单词ID转换为密集向量
    tf.keras.layers.Embedding(
        input_dim=max_features,    # 词汇表大小：10000
        output_dim=128,            # 每个词的向量维度：128
        input_length=max_length,   # 输入序列长度：500
        name='embedding'
    ),
    """
    Embedding层详解：
    - 作用：将稀疏的单词ID转换为密集的向量表示
    - 类比：给每个单词一个"身份证"向量
    
    参数说明：
    - input_dim=10000：词汇表大小，支持10000个不同单词
    - output_dim=128：每个单词用128维向量表示
    - input_length=500：输入序列长度
    
    工作原理：
    - 输入：[1, 14, 22, 16] （单词ID序列）
    - 输出：[[0.1,0.3,...,0.8], [0.2,0.1,...,0.5], ...] （向量序列）
    - 类似于查字典，每个ID对应一个向量
    
    类比理解：
    - 类似于Java的Map<Integer, float[]>
    - 类似于JS的对象 {1: [0.1,0.3,...], 14: [0.2,0.1,...]}
    """
    
    # LSTM层：学习序列中的长期依赖关系
    tf.keras.layers.LSTM(64, dropout=0.5, name='lstm'),
    """
    LSTM层详解：
    - LSTM：Long Short-Term Memory（长短期记忆网络）
    - 比简单RNN更强的记忆能力
    
    参数说明：
    - 64：隐藏单元数，LSTM的"记忆容量"
    - dropout=0.5：随机关闭50%的连接，防止过拟合
    
    LSTM vs RNN的优势：
    - 解决梯度消失问题
    - 能记住更长的历史信息
    - 有选择性地记忆和遗忘
    
    工作原理：
    - 输入：词向量序列
    - 处理：逐个读取每个词，更新内部记忆
    - 输出：基于整个序列的特征表示
    
    类比：
    - 像人读文章，边读边记住重要信息
    - 能记住文章开头提到的重要内容
    """
    
    # 输出层：二分类（正面/负面）
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    """
    输出层详解：
    - Dense(1)：全连接层，输出1个数值
    - activation='sigmoid'：sigmoid激活函数
    
    Sigmoid函数：
    - 将任意实数映射到(0,1)区间
    - 输出可以解释为概率
    - 0.5以上为正面，0.5以下为负面
    
    类比：
    - 类似于投票结果的置信度
    - 0.8表示80%确信是正面评论
    - 0.2表示20%确信是正面（即80%确信是负面）
    """
])

# 编译模型
model.compile(
    optimizer='adam',                    # Adam优化器
    loss='binary_crossentropy',         # 二分类交叉熵损失
    metrics=['accuracy']                 # 准确率指标
)

"""
编译参数详解：

1. optimizer='adam'：
   - Adam：自适应学习率优化器
   - 会自动调整学习速度
   - 适合大多数深度学习任务

2. loss='binary_crossentropy'：
   - 二分类交叉熵损失函数
   - 专门用于二分类问题（正面/负面）
   - 衡量预测概率与真实标签的差距

3. metrics=['accuracy']：
   - 准确率：预测正确的样本比例
   - 用于监控训练过程
   - 不影响训练，只用于评估
"""

print("LSTM模型结构:")
model.summary()

"""
模型结构总结：
输入：[1, 14, 22, 16, ...] （500个单词ID）
    ↓
Embedding：[[0.1,0.3,...], [0.2,0.1,...], ...] （500个128维向量）
    ↓
LSTM：[0.5, 0.2, 0.8, ...] （64维特征向量）
    ↓
Dense+Sigmoid：0.75 （正面情感概率）

整个过程类似于：
1. 查字典：将单词转换为向量
2. 理解：LSTM读懂整个句子的含义
3. 判断：输出情感倾向的概率
"""
