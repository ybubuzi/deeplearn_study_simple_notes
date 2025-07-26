"""
TensorFlow 深度学习入门示例 3: 卷积神经网络CNN（详细注释版）
专为JS开发者设计的计算机视觉入门

CNN和图像分类概念：
- CNN：卷积神经网络，专门处理图像的神经网络
- 特点：能够自动提取图像特征（边缘、纹理、形状等）
- CIFAR-10：包含10类物体的32x32彩色图像数据集
- 目标：识别图像中的物体类别
- 类似于人眼识别物体的过程
"""

# ==================== 导入库详解 ====================
import tensorflow as tf     # 深度学习框架
import numpy as np         # 数值计算库
import matplotlib.pyplot as plt  # 绘图库

"""
CNN需要的库和之前相同，但处理的数据更复杂：
- 从1D数据（线性回归）到2D图像（MNIST）再到3D彩色图像（CIFAR-10）
- 数据复杂度：标量 < 向量 < 矩阵 < 3D张量
"""

# 设置matplotlib中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# ==================== 数据加载详解 ====================
print("正在加载CIFAR-10数据集...")

# 加载CIFAR-10数据集
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

"""
CIFAR-10数据集详解：
- CIFAR-10：Canadian Institute For Advanced Research的10类数据集
- 包含60000张32x32像素的彩色图像
- 10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 训练集：50000张，测试集：10000张

数据结构对比：
- MNIST：(样本数, 28, 28) 灰度图像
- CIFAR-10：(样本数, 32, 32, 3) 彩色图像
- 第4个维度3表示RGB三个颜色通道
"""

# 定义类别名称（中文）
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

"""
类别标签：
- 数据集中的标签是数字0-9
- 我们用中文名称数组来对应这些数字
- 类似于JS中的枚举或映射对象
- class_names[0] = '飞机', class_names[1] = '汽车', ...
"""

print(f"训练集形状: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签: {y_test.shape}")
print(f"图像尺寸: 32x32x3 (彩色)")
print(f"类别数量: {len(class_names)}")

"""
数据形状解释：
- X_train.shape = (50000, 32, 32, 3)：50000张32x32的彩色图片
- y_train.shape = (50000, 1)：50000个标签（注意是2D数组）
- 3个颜色通道：Red(红), Green(绿), Blue(蓝)
- 每个像素有3个值，分别表示RGB强度（0-255）
"""

# ==================== 数据预处理详解 ====================
print("\n2. 数据预处理...")

# 归一化：将像素值从0-255缩放到0-1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

"""
彩色图像归一化：
- 原始像素值：0-255（8位整数）
- 归一化后：0.0-1.0（32位浮点数）
- 对RGB三个通道都进行相同的归一化
- 帮助CNN更好地学习颜色特征
"""

# 标签展平：从2D转换为1D
y_train = y_train.flatten()
y_test = y_test.flatten()

"""
标签处理：
- 原始形状：(50000, 1) - 二维数组
- 展平后：(50000,) - 一维数组
- flatten()：将多维数组压平为一维
- 类似于JS的array.flat()方法
- TensorFlow的损失函数需要一维标签
"""

print(f"预处理后训练集形状: {X_train.shape}")
print(f"预处理后标签形状: {y_train.shape}")

# ==================== 数据可视化详解 ====================
print("\n3. 可视化数据样本...")

plt.figure(figsize=(12, 8))  # 创建较大的图形
for i in range(20):  # 显示前20个样本
    plt.subplot(4, 5, i+1)  # 4行5列的网格
    
    plt.imshow(X_train[i])  # 显示彩色图像
    """
    显示彩色图像：
    - X_train[i]：第i张图片，形状为(32, 32, 3)
    - plt.imshow()自动识别RGB格式
    - 不需要cmap参数（灰度图像才需要）
    """
    
    plt.title(f'{class_names[y_train[i]]}')  # 显示类别名称
    """
    标题设置：
    - y_train[i]：第i个样本的标签（0-9的数字）
    - class_names[y_train[i]]：对应的中文类别名
    - 例如：y_train[0]=3，则显示class_names[3]='猫'
    """
    
    plt.axis('off')  # 隐藏坐标轴

plt.suptitle('CIFAR-10数据集样本')
plt.tight_layout()
plt.show()

# ==================== CNN模型构建详解 ====================
print("\n4. 构建CNN模型...")

model = tf.keras.Sequential([
    """
    CNN架构设计思路：
    1. 卷积层：提取局部特征（边缘、纹理）
    2. 池化层：降低分辨率，减少计算量
    3. 重复卷积+池化：提取更高级特征
    4. 展平层：转换为1D向量
    5. 全连接层：进行最终分类
    
    类似于人眼看图的过程：
    细节 -> 局部特征 -> 整体形状 -> 物体识别
    """
    
    # 第一个卷积块
    tf.keras.layers.Conv2D(
        32,                      # 过滤器数量：32个特征检测器
        (3, 3),                  # 卷积核大小：3x3
        activation='relu',       # 激活函数：ReLU
        input_shape=(32, 32, 3), # 输入形状：32x32的彩色图像
        name='conv_layer1'
    ),
    """
    Conv2D层详解：
    
    1. 32个过滤器：
       - 每个过滤器检测不同的特征（如边缘、角点）
       - 类似于32个不同的"特征检测器"
       - 输出32个特征图（feature maps）
    
    2. (3, 3)卷积核：
       - 3x3的小窗口在图像上滑动
       - 每次计算窗口内像素的加权和
       - 类似于用放大镜逐块检查图像
    
    3. 卷积操作原理：
       - 输入：(32, 32, 3)
       - 卷积后：(30, 30, 32)
       - 尺寸减小：32-3+1=30（无padding）
       - 通道增加：3->32（特征图数量）
    
    4. activation='relu'：
       - 增加非线性，让网络能学习复杂模式
       - 负值变0，正值保持不变
    """
    
    tf.keras.layers.MaxPooling2D((2, 2), name='pool_layer1'),
    """
    MaxPooling2D层详解：
    
    1. (2, 2)池化窗口：
       - 在2x2区域内取最大值
       - 将图像尺寸减半
       - (30, 30, 32) -> (15, 15, 32)
    
    2. 池化的作用：
       - 降低分辨率，减少计算量
       - 增加平移不变性（物体稍微移动仍能识别）
       - 保留最重要的特征
    
    3. 类比理解：
       - 像缩略图，保留主要信息但减少细节
       - 类似于图像压缩，但保留关键特征
    """
    
    # 第二个卷积块
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer2'),
    """
    第二个卷积层：
    - 64个过滤器：比第一层更多，提取更复杂特征
    - 输入：(15, 15, 32)
    - 输出：(13, 13, 64)
    - 特征越来越抽象：边缘 -> 纹理 -> 形状
    """
    
    tf.keras.layers.MaxPooling2D((2, 2), name='pool_layer2'),
    """
    第二个池化层：
    - (13, 13, 64) -> (6, 6, 64)
    - 继续减少空间尺寸
    """
    
    # 第三个卷积块
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer3'),
    """
    第三个卷积层：
    - 64个过滤器：提取最高级特征
    - 输入：(6, 6, 64)
    - 输出：(4, 4, 64)
    - 此时特征已经很抽象，接近物体的整体形状
    """
    
    # 展平层：2D -> 1D
    tf.keras.layers.Flatten(name='flatten_layer'),
    """
    Flatten层详解：
    - 将2D特征图展平为1D向量
    - (4, 4, 64) -> (1024,)
    - 4 × 4 × 64 = 1024个特征
    - 类似于将立体积木摊平成一排
    - 为全连接层准备输入
    """
    
    # 全连接层
    tf.keras.layers.Dense(64, activation='relu', name='dense_layer'),
    """
    全连接层：
    - 64个神经元
    - 接收1024个特征，输出64个高级特征
    - 类似于MNIST中的全连接层
    - 进行特征的最终整合
    """
    
    tf.keras.layers.Dropout(0.5, name='dropout_layer'),
    """
    Dropout层：
    - 随机关闭50%的连接
    - 防止过拟合（CNN容易过拟合）
    - 比MNIST的0.2更强，因为CNN参数更多
    """
    
    # 输出层
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
    """
    输出层：
    - 10个神经元对应10个类别
    - Softmax激活：输出概率分布
    - 例如：[0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005]
    - 表示第2类（汽车）的概率是80%
    """
])

"""
完整的CNN架构总结：
输入(32×32×3) 
-> Conv2D(32) -> (30×30×32) 
-> MaxPool -> (15×15×32)
-> Conv2D(64) -> (13×13×64) 
-> MaxPool -> (6×6×64)
-> Conv2D(64) -> (4×4×64)
-> Flatten -> (1024)
-> Dense(64) -> (64)
-> Dropout(0.5)
-> Dense(10) -> (10)
-> Softmax -> 概率分布

特征提取过程：
原始图像 -> 边缘检测 -> 纹理识别 -> 形状提取 -> 物体分类
"""

# ==================== 模型编译详解 ====================
model.compile(
    optimizer='adam',                           # Adam优化器
    loss='sparse_categorical_crossentropy',     # 稀疏分类交叉熵损失
    metrics=['accuracy']                        # 准确率指标
)

"""
CNN模型编译：
- 参数与MNIST相同，但处理的问题更复杂
- Adam优化器：适合CNN的复杂参数空间
- 损失函数：多分类问题的标准选择
- CNN通常比全连接网络更难训练，需要更多轮次
"""

print("CNN模型结构:")
model.summary()

"""
模型参数分析：
- Conv2D层参数：(卷积核大小 × 输入通道 + 1) × 输出通道
- 例如第一层：(3×3×3 + 1) × 32 = 896个参数
- CNN的参数主要在全连接层：1024×64 + 64×10
- 总参数数量比全连接网络少，但表达能力更强
"""

# ==================== 数据增强详解 ====================
print("\n5. 设置数据增强...")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,        # 随机旋转：±20度
    width_shift_range=0.2,    # 水平移动：±20%
    height_shift_range=0.2,   # 垂直移动：±20%
    horizontal_flip=True,     # 水平翻转：50%概率
    zoom_range=0.2           # 随机缩放：±20%
)

"""
数据增强详解：

数据增强的目的：
- 增加训练数据的多样性
- 提高模型的泛化能力
- 防止过拟合
- 模拟真实世界中图像的变化

各种增强方式：

1. rotation_range=20：
   - 随机旋转图像±20度
   - 模拟拍照时的角度偏差
   - 让模型对旋转不敏感

2. width_shift_range=0.2：
   - 水平移动图像±20%
   - 模拟物体在画面中的位置变化
   - 0.2表示最多移动图像宽度的20%

3. height_shift_range=0.2：
   - 垂直移动图像±20%
   - 类似于水平移动

4. horizontal_flip=True：
   - 50%概率水平翻转图像
   - 对于某些物体（如汽车、飞机）很有用
   - 注意：不是所有数据都适合翻转（如文字）

5. zoom_range=0.2：
   - 随机缩放±20%
   - 模拟距离远近的变化
   - 让模型对物体大小不敏感

类比理解：
- 就像给同一张照片拍出不同角度、距离的版本
- 类似于PS中的各种变换操作
- 让AI见过更多样的图像，提高识别能力
"""

# ==================== 模型训练详解 ====================
print("\n6. 开始训练CNN...")

# 注意：这里我们先用普通训练，不使用数据增强
# 实际项目中可以使用 datagen.fit(X_train) 和 model.fit_generator()
history = model.fit(
    X_train, y_train,         # 训练数据
    epochs=10,                # 训练轮数（实际项目中通常需要50-100轮）
    batch_size=32,            # 批次大小
    validation_split=0.2,     # 验证集比例
    verbose=1                 # 显示训练进度
)

"""
CNN训练特点：

1. 训练时间：
   - CNN比全连接网络慢很多
   - 卷积操作计算量大
   - 建议使用GPU加速

2. epochs=10：
   - 为了演示只用10轮
   - 实际项目中CNN通常需要50-100轮
   - CIFAR-10比MNIST更难，需要更多训练

3. batch_size=32：
   - CNN对内存要求高
   - 32是常用的批次大小
   - GPU内存不足时可以减小到16

4. validation_split=0.2：
   - 20%数据用于验证
   - 监控过拟合情况
   - CNN容易过拟合，需要密切监控

训练过程监控：
- 训练准确率应该逐渐上升
- 验证准确率应该跟随训练准确率
- 如果验证准确率停止上升或下降，可能是过拟合
"""

# ==================== 模型评估详解 ====================
print("\n7. 评估模型性能...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集损失: {test_loss:.4f}")

"""
CNN性能评估：

准确率期望：
- CIFAR-10比MNIST难很多
- 简单CNN：60-70%
- 复杂CNN：80-90%
- 最先进模型：95%+

性能对比：
- 随机猜测：10%（10个类别）
- 传统机器学习：40-50%
- 简单全连接网络：50-60%
- CNN：70%+

影响因素：
- 网络深度：更深的网络通常效果更好
- 数据增强：能提高5-10%的准确率
- 训练轮数：充分训练很重要
- 正则化：Dropout、BatchNorm等
"""

# ==================== 训练过程可视化详解 ====================
print("\n8. 可视化训练过程...")

plt.figure(figsize=(12, 4))

# 子图1：准确率变化
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率', marker='o')
plt.plot(history.history['val_accuracy'], label='验证准确率', marker='s')
plt.title('CNN模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

"""
准确率曲线分析：

理想情况：
- 两条线都上升
- 验证准确率跟随训练准确率
- 最终趋于稳定

常见问题：
1. 过拟合：
   - 训练准确率持续上升
   - 验证准确率开始下降
   - 解决：早停、Dropout、数据增强

2. 欠拟合：
   - 两条线都很低且平缓
   - 解决：增加模型复杂度、更多训练

3. 训练不稳定：
   - 曲线震荡很大
   - 解决：降低学习率、增加批次大小
"""

# 子图2：损失值变化
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失', marker='o')
plt.plot(history.history['val_loss'], label='验证损失', marker='s')
plt.title('CNN模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

"""
损失曲线分析：

理想情况：
- 两条线都下降
- 验证损失跟随训练损失
- 最终趋于稳定

损失值含义：
- 交叉熵损失衡量预测概率与真实标签的差距
- 值越小表示预测越准确
- 通常比准确率更敏感，能更早发现问题
"""

plt.tight_layout()
plt.show()

# ==================== 预测结果展示详解 ====================
print("\n9. 展示预测结果...")

# 对前16个测试样本进行预测
predictions = model.predict(X_test[:16], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

"""
预测过程：
- model.predict()：获取每个类别的概率
- np.argmax()：找到概率最高的类别
- predictions形状：(16, 10)
- predicted_classes形状：(16,)
"""

plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_test[i])

    # 获取预测结果
    true_label = class_names[y_test[i]]
    predicted_label = class_names[predicted_classes[i]]
    confidence = np.max(predictions[i]) * 100

    # 设置颜色：正确=绿色，错误=红色
    color = 'green' if y_test[i] == predicted_classes[i] else 'red'

    plt.title(f'真实: {true_label}\n预测: {predicted_label}\n置信度: {confidence:.1f}%',
              color=color, fontsize=8)
    plt.axis('off')

plt.suptitle('CNN预测结果示例（绿色=正确，红色=错误）')
plt.tight_layout()
plt.show()

"""
结果分析技巧：

1. 置信度分析：
   - 高置信度且正确：模型很确信
   - 高置信度但错误：模型过度自信，需要改进
   - 低置信度：模型不确定，可能需要更多训练

2. 错误类型分析：
   - 相似物体混淆：猫和狗、汽车和卡车
   - 图像质量问题：模糊、光线不好
   - 标注错误：数据集本身的问题

3. 改进方向：
   - 数据增强：增加训练数据多样性
   - 网络加深：更多卷积层
   - 正则化：防止过拟合
   - 预训练模型：使用ImageNet预训练权重
"""

# ==================== 模型保存详解 ====================
model.save('cnn_cifar10_model.h5')
print("\n模型已保存为 'cnn_cifar10_model.h5'")

"""
CNN模型保存：
- 包含完整的网络结构和权重
- .h5格式是HDF5，深度学习标准格式
- 文件较大（几MB到几GB）
- 可以用于后续的预测或迁移学习
"""

print("\n" + "=" * 60)
print("✅ CNN图像分类训练完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 卷积神经网络：专门处理图像的神经网络")
print("2. 卷积层：提取局部特征（边缘、纹理）")
print("3. 池化层：降低分辨率，减少计算量")
print("4. 特征提取：从像素到高级语义特征")
print("5. 数据增强：提高模型泛化能力")
print("6. 过拟合：CNN的常见问题及解决方法")
print("=" * 60)

"""
CNN vs 全连接网络对比：

优势：
1. 参数共享：卷积核在整个图像上共享，参数少
2. 局部连接：只关注局部区域，符合图像特性
3. 平移不变性：物体位置变化不影响识别
4. 层次特征：从低级到高级特征的自动提取

应用场景：
- 图像分类：识别图像中的物体
- 目标检测：找到物体的位置
- 图像分割：像素级别的分类
- 人脸识别：特定的图像分类任务
- 医学影像：X光、CT等图像分析

这就是为什么CNN是计算机视觉的基础！
"""
