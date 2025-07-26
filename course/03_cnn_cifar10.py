"""
TensorFlow 深度学习入门示例 3: 卷积神经网络 (CNN) - CIFAR-10图像分类
学习使用CNN处理彩色图像，这是计算机视觉的基础
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# 1. 加载CIFAR-10数据集
print("正在加载CIFAR-10数据集...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# CIFAR-10的类别名称
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

print(f"训练集形状: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签: {y_test.shape}")
print(f"图像尺寸: 32x32x3 (彩色)")
print(f"类别数量: {len(class_names)}")

# 2. 数据预处理
# 归一化像素值到0-1之间
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 将标签转换为一维数组
y_train = y_train.flatten()
y_test = y_test.flatten()

# 3. 可视化一些样本
plt.figure(figsize=(12, 8))
for i in range(20):
    plt.subplot(4, 5, i+1)
    plt.imshow(X_train[i])
    plt.title(f'{class_names[y_train[i]]}')
    plt.axis('off')
plt.suptitle('CIFAR-10数据集样本')
plt.tight_layout()
plt.show()

# 4. 构建CNN模型
model = tf.keras.Sequential([
    # 第一个卷积块
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv_layer1'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool_layer1'),

    # 第二个卷积块
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer2'),
    tf.keras.layers.MaxPooling2D((2, 2), name='pool_layer2'),

    # 第三个卷积块
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv_layer3'),

    # 展平层：将2D特征图转换为1D向量
    tf.keras.layers.Flatten(name='flatten_layer'),

    # 全连接层
    tf.keras.layers.Dense(64, activation='relu', name='dense_layer'),
    tf.keras.layers.Dropout(0.5, name='dropout_layer'),  # 防止过拟合

    # 输出层
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

# 5. 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. 查看模型结构
print("\nCNN模型结构:")
model.summary()

# 7. 数据增强（可选，提高模型泛化能力）
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,      # 随机旋转
    width_shift_range=0.2,  # 水平移动
    height_shift_range=0.2, # 垂直移动
    horizontal_flip=True,   # 水平翻转
    zoom_range=0.2         # 随机缩放
)

# 8. 训练模型
print("\n开始训练CNN...")
# 使用较少的epochs以节省时间，实际项目中可以增加到50-100
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 9. 评估模型
print("\n评估模型性能...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集损失: {test_loss:.4f}")

# 10. 可视化训练过程
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 11. 预测和可视化结果
predictions = model.predict(X_test[:16], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_test[i])
    confidence = np.max(predictions[i]) * 100
    true_label = class_names[y_test[i]]
    pred_label = class_names[predicted_classes[i]]
    
    # 如果预测正确，用绿色；错误用红色
    color = 'green' if y_test[i] == predicted_classes[i] else 'red'
    plt.title(f'真实: {true_label}\n预测: {pred_label}\n({confidence:.1f}%)', color=color)
    plt.axis('off')

plt.suptitle('CNN预测结果')
plt.tight_layout()
plt.show()

# 12. 分析模型性能
from sklearn.metrics import classification_report
y_pred_all = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_all, target_names=class_names))

# 13. 保存模型
model.save('cnn_cifar10_model.h5')
print("\n模型已保存为 'cnn_cifar10_model.h5'")

print(f"\nCNN模型总结:")
print(f"- 使用了3个卷积层提取图像特征")
print(f"- 使用了2个池化层减少参数数量")
print(f"- 使用了Dropout防止过拟合")
print(f"- 最终测试准确率: {test_accuracy:.4f}")
print(f"- CNN比全连接网络更适合处理图像数据")
