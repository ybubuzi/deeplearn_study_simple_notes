"""
TensorFlow 深度学习入门示例 2: 手写数字识别（详细注释版）
专为JS开发者设计的神经网络分类入门

手写数字识别概念：
- 经典的深度学习入门项目
- 目标：识别0-9的手写数字图片
- 输入：28x28像素的灰度图像
- 输出：0-9中的一个数字
- 类似于图像验证码识别
"""

# ==================== 导入库详解 ====================
import tensorflow as tf     # Google的深度学习框架
import numpy as np         # 数值计算库，处理数组和矩阵
import matplotlib.pyplot as plt  # 绘图库，用于显示图像和图表

"""
库的作用类比：
- tensorflow：类似于前端的深度学习框架，提供神经网络API
- numpy：类似于JS的数学工具库，但专门处理多维数组
- matplotlib：类似于Chart.js，用于绘制图表和显示图像
"""

# 设置matplotlib中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

"""
plt.rcParams详解：
- 这是matplotlib的全局配置字典
- 类似于JS中的全局配置对象
- 'font.sans-serif'：设置无衬线字体
- ['SimHei']：黑体字体，确保中文正常显示
- 'axes.unicode_minus'：处理负号显示问题
"""

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# ==================== 数据加载详解 ====================
print("正在加载MNIST数据集...")

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

"""
MNIST数据集详解：
- MNIST：手写数字数据库，包含70000张28x28像素的手写数字图片
- 训练集：60000张图片用于训练模型
- 测试集：10000张图片用于测试模型性能

数据结构：
- X_train：训练图像数据，形状为(60000, 28, 28)
- y_train：训练标签数据，形状为(60000,)，值为0-9
- X_test：测试图像数据，形状为(10000, 28, 28)  
- y_test：测试标签数据，形状为(10000,)

多重赋值语法：
- Python可以同时赋值多个变量
- 类似于JS的解构赋值：const [train, test] = loadData()
- 但Python的语法更简洁
"""

print(f"训练集形状: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签: {y_test.shape}")
print(f"像素值范围: {X_train.min()} - {X_train.max()}")

"""
数据形状解释：
- X_train.shape = (60000, 28, 28)：60000张28x28的图片
- y_train.shape = (60000,)：60000个标签
- 像素值范围0-255：黑色=0，白色=255（标准图像格式）

.min()和.max()方法：
- NumPy数组的方法，找到最小值和最大值
- 类似于JS的Math.min()和Math.max()，但作用于整个数组
"""

# ==================== 数据预处理详解 ====================
print("\n2. 数据预处理...")

# 数据归一化：将像素值从0-255缩放到0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

"""
数据归一化详解：

1. .astype('float32')：
   - 转换数据类型为32位浮点数
   - 原始数据是整数(0-255)，需要转换为浮点数才能除法
   - 类似于JS中的parseInt()转换，但这里是转换为浮点数

2. / 255.0：
   - 将0-255的像素值缩放到0-1范围
   - 255.0是浮点数，确保结果也是浮点数
   - 归一化的好处：帮助神经网络更好地学习

为什么要归一化？
- 神经网络对输入数据的范围敏感
- 0-1范围的数据更容易训练
- 类似于将不同单位的数据统一到相同范围
"""

# 图像展平：将28x28的二维图像转换为784维的一维向量
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

"""
图像展平详解：

1. reshape()方法：
   - 改变数组的形状，但不改变数据
   - 类似于将二维数组转换为一维数组

2. X_train.shape[0]：
   - 获取第一个维度的大小（样本数量）
   - X_train.shape[0] = 60000

3. -1参数：
   - 表示自动计算这个维度的大小
   - 28 × 28 = 784，所以-1会被计算为784

4. 结果：
   - 从(60000, 28, 28)变成(60000, 784)
   - 每张图片从28x28的矩阵变成784个数字的列表

为什么要展平？
- 全连接神经网络需要一维输入
- 类似于将二维表格的数据转换为一行
"""

print(f"展平后训练集形状: {X_train_flat.shape}")

# ==================== 数据可视化详解 ====================
print("\n3. 可视化数据样本...")

plt.figure(figsize=(12, 4))  # 创建图形，设置大小为12x4英寸
"""
plt.figure()详解：
- 创建一个新的图形窗口
- figsize=(宽, 高)：设置图形大小，单位是英寸
- 类似于创建一个画布
"""

for i in range(10):  # 显示前10个样本
    """
    range(10)：
    - 生成0到9的整数序列
    - 类似于JS的for(let i=0; i<10; i++)
    """
    
    plt.subplot(2, 5, i+1)  # 创建2行5列的子图，当前是第i+1个
    """
    plt.subplot(行数, 列数, 位置)：
    - 在一个图形中创建多个子图
    - 2行5列=10个子图
    - i+1：位置从1开始计数（不是0）
    - 类似于网页中的grid布局
    """
    
    plt.imshow(X_train[i], cmap='gray')  # 显示图像
    """
    plt.imshow()详解：
    - 显示图像数据
    - X_train[i]：第i张图片（28x28的数组）
    - cmap='gray'：使用灰度色彩映射
    - 类似于HTML中的<img>标签显示图片
    """
    
    plt.title(f'标签: {y_train[i]}')  # 设置子图标题
    plt.axis('off')  # 隐藏坐标轴
    """
    plt.axis('off')：
    - 隐藏x轴和y轴
    - 让图像显示更清晰
    - 类似于CSS中的display:none隐藏元素
    """

plt.suptitle('MNIST数据集样本')  # 设置整个图形的标题
plt.tight_layout()  # 自动调整子图间距
plt.show()  # 显示图形

"""
布局函数：
- plt.suptitle()：设置整个图形的总标题
- plt.tight_layout()：自动调整布局，避免重叠
- plt.show()：显示图形，类似于渲染到屏幕
"""

# ==================== 神经网络模型构建详解 ====================
print("\n4. 构建神经网络模型...")

model = tf.keras.Sequential([
    """
    Sequential模型：
    - 顺序模型，层级式堆叠
    - 数据从第一层流向最后一层
    - 类似于工厂流水线，每层处理后传给下一层
    """
    
    # 第一个隐藏层
    tf.keras.layers.Dense(
        128,                    # 神经元数量：这层有128个神经元
        activation='relu',      # 激活函数：ReLU（修正线性单元）
        input_shape=(784,),     # 输入形状：784个特征（展平后的图像）
        name='hidden_layer1'    # 层的名称
    ),
    """
    Dense层（全连接层）详解：
    
    1. 128个神经元：
       - 每个神经元接收784个输入
       - 类似于有128个"决策者"，每个都看所有像素
    
    2. activation='relu'：
       - ReLU激活函数：f(x) = max(0, x)
       - 负数变0，正数保持不变
       - 增加非线性，让网络能学习复杂模式
    
    3. input_shape=(784,)：
       - 指定输入数据的形状
       - 784来自28x28=784个像素
       - 只有第一层需要指定输入形状
    
    工作原理：
    - 输入：784个像素值
    - 处理：每个神经元计算加权和 + 偏置，然后应用ReLU
    - 输出：128个特征值
    """
    
    # Dropout层：防止过拟合
    tf.keras.layers.Dropout(0.2, name='dropout_layer'),
    """
    Dropout层详解：
    - 随机"关闭"20%的神经元连接
    - 防止模型过度依赖某些特征
    - 类似于团队中随机缺席一些成员，锻炼其他成员
    - 只在训练时生效，预测时不起作用
    
    过拟合问题：
    - 模型在训练数据上表现很好，但在新数据上表现差
    - 类似于死记硬背，不能举一反三
    - Dropout通过增加随机性来缓解这个问题
    """
    
    # 第二个隐藏层
    tf.keras.layers.Dense(64, activation='relu', name='hidden_layer2'),
    """
    第二个隐藏层：
    - 64个神经元，比第一层少
    - 逐渐压缩特征，提取更高级的模式
    - 类似于从细节到概要的抽象过程
    """
    
    # 输出层
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
    """
    输出层详解：
    
    1. 10个神经元：
       - 对应10个数字类别（0-9）
       - 每个神经元输出该数字的概率
    
    2. activation='softmax'：
       - Softmax激活函数，将输出转换为概率分布
       - 所有输出概率加起来等于1
       - 类似于投票结果的百分比
    
    例如输出：[0.1, 0.05, 0.8, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005]
    表示：数字2的概率是80%，数字0的概率是10%，等等
    """
])

# ==================== 模型编译详解 ====================
model.compile(
    optimizer='adam',                           # 优化器
    loss='sparse_categorical_crossentropy',     # 损失函数
    metrics=['accuracy']                        # 评估指标
)

"""
模型编译参数详解：

1. optimizer='adam'：
   - Adam优化器：自适应学习率优化算法
   - 会自动调整学习速度
   - 类似于一个智能的学习策略

2. loss='sparse_categorical_crossentropy'：
   - 稀疏分类交叉熵损失函数
   - 适用于多分类问题（0-9共10类）
   - "稀疏"表示标签是整数（0,1,2...），不是one-hot编码
   - 衡量预测概率与真实标签的差距

3. metrics=['accuracy']：
   - 准确率：预测正确的样本比例
   - 用于监控训练过程
   - 不影响训练，只用于评估
"""

print("模型结构:")
model.summary()
"""
model.summary()：
- 显示模型的详细结构
- 包括每层的输出形状和参数数量
- 类似于查看对象的属性和方法
"""

# ==================== 模型训练详解 ====================
print("\n5. 开始训练模型...")

history = model.fit(
    X_train_flat, y_train,      # 训练数据：输入和标签
    epochs=10,                  # 训练轮数：看数据10遍
    batch_size=128,             # 批次大小：每次处理128个样本
    validation_split=0.1,       # 验证集比例：10%的训练数据用于验证
    verbose=1                   # 输出详细程度：显示训练进度
)

"""
模型训练参数详解：

1. X_train_flat, y_train：
   - 训练数据和对应的标签
   - X_train_flat：(60000, 784) 展平后的图像数据
   - y_train：(60000,) 对应的数字标签
   - 类似于练习题和标准答案

2. epochs=10：
   - 训练轮数，模型完整看数据集的次数
   - 每轮都会调整模型参数
   - 类似于学生复习课本10遍
   - 更多轮数通常效果更好，但也可能过拟合

3. batch_size=128：
   - 批次大小，每次更新参数时使用的样本数
   - 不是一次处理所有60000个样本，而是分批处理
   - 每批128个样本，共需要60000/128≈469批
   - 类似于分组学习，每组128人

4. validation_split=0.1：
   - 从训练数据中分出10%作为验证集
   - 验证集用于监控训练效果，防止过拟合
   - 实际训练数据：54000个样本
   - 验证数据：6000个样本

5. verbose=1：
   - 控制训练过程的输出详细程度
   - 1：显示进度条和每轮结果
   - 0：静默模式，2：每轮一行

6. 返回值history：
   - 包含训练过程中的损失值、准确率等
   - 可用于分析训练效果和绘制学习曲线
   - 类似于学习记录本
"""

# ==================== 模型评估详解 ====================
print("\n6. 评估模型性能...")

test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
"""
模型评估：
- model.evaluate()：在测试集上评估模型性能
- 返回损失值和准确率
- verbose=0：不显示评估进度
- 测试集是模型从未见过的数据，能真实反映性能
"""

print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集损失: {test_loss:.4f}")

"""
性能指标解释：
- 准确率：预测正确的比例，越高越好（最高1.0）
- 损失值：预测错误的程度，越低越好（最低0.0）
- .4f：格式化为4位小数
"""

# ==================== 预测和结果分析详解 ====================
print("\n7. 进行预测...")

# 对前10个测试样本进行预测
predictions = model.predict(X_test_flat[:10], verbose=0)
"""
模型预测：
- model.predict()：使用训练好的模型进行预测
- X_test_flat[:10]：取前10个测试样本
- 返回每个样本对应10个类别的概率
- predictions形状：(10, 10)
"""

predicted_classes = np.argmax(predictions, axis=1)
"""
获取预测类别：
- np.argmax()：找到最大值的索引
- axis=1：沿着第二个维度（类别维度）找最大值
- 例如：[0.1, 0.05, 0.8, 0.02, ...] -> 2（索引2的值最大）
- predicted_classes：预测的数字类别
"""

print("预测结果示例:")
for i in range(5):  # 显示前5个预测结果
    true_label = y_test[i]
    predicted_label = predicted_classes[i]
    confidence = np.max(predictions[i]) * 100
    print(f"样本{i+1}: 真实={true_label}, 预测={predicted_label}, 置信度={confidence:.1f}%")

"""
结果分析：
- true_label：真实标签
- predicted_label：预测标签
- confidence：预测置信度（最高概率转换为百分比）
- np.max(predictions[i])：该样本预测概率的最大值
"""

# ==================== 可视化训练过程详解 ====================
print("\n8. 可视化训练过程...")

plt.figure(figsize=(15, 5))  # 创建大图形

# 子图1：准确率变化
plt.subplot(1, 3, 1)  # 1行3列的第1个子图
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

"""
训练曲线分析：
- history.history['accuracy']：每轮的训练准确率
- history.history['val_accuracy']：每轮的验证准确率
- 理想情况：两条线都上升且接近
- 如果验证准确率下降，可能是过拟合
"""

# 子图2：损失值变化
plt.subplot(1, 3, 2)  # 第2个子图
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

"""
损失曲线分析：
- 理想情况：两条线都下降且接近
- 训练损失持续下降但验证损失上升：过拟合
- 两条线都不下降：学习率可能太小或模型容量不足
"""

# 子图3：混淆矩阵
plt.subplot(1, 3, 3)  # 第3个子图

# 导入额外的库用于混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns

"""
新导入的库：
- sklearn：机器学习工具库，类似于JS的工具库
- seaborn：基于matplotlib的统计绘图库，让图表更美观
"""

# 对所有测试数据进行预测
y_pred_all = np.argmax(model.predict(X_test_flat, verbose=0), axis=1)
"""
全量预测：
- 对所有10000个测试样本进行预测
- 获取预测的类别标签
"""

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred_all)
"""
混淆矩阵：
- 显示每个真实类别被预测为各个类别的数量
- 对角线：预测正确的数量
- 非对角线：预测错误的数量
- 10x10的矩阵，对应0-9这10个数字
"""

# 绘制热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

"""
热力图参数：
- annot=True：在每个格子中显示数值
- fmt='d'：数值格式为整数
- cmap='Blues'：使用蓝色色彩映射
"""

plt.tight_layout()  # 调整布局
plt.show()

# ==================== 预测结果展示详解 ====================
print("\n9. 展示预测结果...")

plt.figure(figsize=(12, 6))
for i in range(10):  # 显示前10个预测结果
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')  # 显示原始图像

    # 计算置信度
    confidence = np.max(predictions[i]) * 100

    # 设置标题，显示真实值、预测值和置信度
    plt.title(f'真实: {y_test[i]}, 预测: {predicted_classes[i]}\n置信度: {confidence:.1f}%')
    plt.axis('off')

plt.suptitle('预测结果示例')
plt.tight_layout()
plt.show()

# ==================== 模型保存详解 ====================
print("\n10. 保存模型...")

model.save('mnist_model.h5')
"""
模型保存：
- 将训练好的模型保存到文件
- .h5格式：HDF5格式，深度学习模型的标准格式
- 包含模型结构、权重、优化器状态等
- 类似于保存游戏进度
"""

print("模型已保存为 'mnist_model.h5'")

# ==================== 总结详解 ====================
print(f"\n📊 训练总结:")
print(f"- 模型类型: 3层全连接神经网络")
print(f"- 输入层: 784个神经元 (28x28像素)")
print(f"- 隐藏层1: 128个神经元 (ReLU激活)")
print(f"- 隐藏层2: 64个神经元 (ReLU激活)")
print(f"- 输出层: 10个神经元 (Softmax激活)")
print(f"- 最终测试准确率: {test_accuracy:.4f}")

"""
模型架构总结：
输入(784像素) -> 隐藏层1(128神经元) -> Dropout -> 隐藏层2(64神经元) -> 输出层(10类别)

数据流程：
1. 28x28图像 -> 展平为784维向量
2. 归一化到0-1范围
3. 通过神经网络处理
4. 输出10个概率值
5. 选择概率最高的作为预测结果

这个模型能够识别手写数字，准确率通常在97%以上！
"""

print("\n" + "=" * 60)
print("✅ 手写数字识别训练完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 图像分类：将图像归类到不同类别")
print("2. 多层神经网络：通过多层提取特征")
print("3. 激活函数：ReLU和Softmax的作用")
print("4. 过拟合：Dropout的防过拟合机制")
print("5. 模型评估：准确率、损失值、混淆矩阵")
print("=" * 60)
