"""
TensorFlow 深度学习入门示例 2: 手写数字识别 (MNIST)
这是深度学习的经典入门项目，使用全连接神经网络识别0-9的手写数字
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("GPU可用:", len(tf.config.list_physical_devices('GPU')) > 0)

# 1. 加载MNIST数据集
print("正在加载MNIST数据集...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"训练集形状: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集形状: {X_test.shape}, 标签: {y_test.shape}")
print(f"像素值范围: {X_train.min()} - {X_train.max()}")

# 2. 数据预处理
# 将像素值从0-255缩放到0-1之间（归一化）
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 将28x28的图像展平为784维的向量
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"展平后训练集形状: {X_train_flat.shape}")

# 3. 可视化一些样本
plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f'标签: {y_train[i]}')
    plt.axis('off')
plt.suptitle('MNIST数据集样本')
plt.tight_layout()
plt.show()

# 4. 构建神经网络模型
model = tf.keras.Sequential([
    # 输入层：784个神经元（28x28像素）
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,), name='hidden_layer1'),
    # 添加Dropout防止过拟合
    tf.keras.layers.Dropout(0.2, name='dropout_layer'),
    # 第二个隐藏层
    tf.keras.layers.Dense(64, activation='relu', name='hidden_layer2'),
    # 输出层：10个神经元（0-9数字）
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

# 5. 编译模型
model.compile(
    optimizer='adam',                    # Adam优化器
    loss='sparse_categorical_crossentropy',  # 多分类交叉熵损失
    metrics=['accuracy']                 # 准确率指标
)

# 6. 查看模型结构
print("\n模型结构:")
model.summary()

# 7. 训练模型
print("\n开始训练...")
history = model.fit(
    X_train_flat, y_train,
    epochs=10,                # 训练10轮
    batch_size=128,           # 每批128个样本
    validation_split=0.1,     # 10%数据用于验证
    verbose=1
)

# 8. 评估模型
print("\n评估模型性能...")
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集损失: {test_loss:.4f}")

# 9. 进行预测
predictions = model.predict(X_test_flat[:10], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# 10. 可视化训练过程和预测结果
plt.figure(figsize=(15, 5))

# 训练过程
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 预测结果展示
plt.subplot(1, 3, 3)
# 显示混淆矩阵的简化版本
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_all = np.argmax(model.predict(X_test_flat, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

plt.tight_layout()
plt.show()

# 11. 展示一些预测结果
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i], cmap='gray')
    confidence = np.max(predictions[i]) * 100
    plt.title(f'真实: {y_test[i]}, 预测: {predicted_classes[i]}\n置信度: {confidence:.1f}%')
    plt.axis('off')
plt.suptitle('预测结果示例')
plt.tight_layout()
plt.show()

# 12. 保存模型
model.save('mnist_model.h5')
print("\n模型已保存为 'mnist_model.h5'")

print(f"\n总结:")
print(f"- 训练了一个3层神经网络")
print(f"- 输入层: 784个神经元 (28x28像素)")
print(f"- 隐藏层1: 128个神经元 (ReLU激活)")
print(f"- 隐藏层2: 64个神经元 (ReLU激活)")
print(f"- 输出层: 10个神经元 (Softmax激活)")
print(f"- 最终测试准确率: {test_accuracy:.4f}")
