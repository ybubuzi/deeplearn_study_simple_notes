"""
TensorFlow 深度学习入门示例 5: LSTM文本情感分析
学习LSTM处理序列数据的强大能力
分析电影评论的情感（正面/负面）
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("🎭 LSTM文本情感分析入门")
print("=" * 60)

# 1. 加载IMDB电影评论数据集
print("\n1. 加载IMDB电影评论数据...")

# 只使用最常见的10000个单词
max_features = 10000
# 每个评论最多使用500个单词
max_length = 500

print("正在下载IMDB数据集...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(
    num_words=max_features
)

print(f"训练样本数: {len(X_train)}")
print(f"测试样本数: {len(X_test)}")
print(f"标签: 0=负面评论, 1=正面评论")

# 查看数据结构
print(f"\n数据示例:")
print(f"第一个评论长度: {len(X_train[0])} 个单词")
print(f"第一个评论标签: {y_train[0]} ({'正面' if y_train[0] == 1 else '负面'})")
print(f"评论内容（数字编码）: {X_train[0][:20]}...")

# 2. 数据预处理
print("\n2. 数据预处理...")

# 将序列填充到相同长度
# 短的序列用0填充，长的序列截断
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_length)

print(f"填充后的数据形状:")
print(f"X_train: {X_train.shape}")  # (样本数, 序列长度)
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")

# 查看填充后的数据
print(f"\n填充后的第一个评论:")
print(f"前20个词: {X_train[0][:20]}")
print(f"后20个词: {X_train[0][-20:]}")

# 3. 构建LSTM模型
print("\n3. 构建LSTM模型...")

model = tf.keras.Sequential([
    # 词嵌入层：将单词ID转换为密集向量
    # 类似于给每个单词一个"身份证"向量
    tf.keras.layers.Embedding(
        input_dim=max_features,  # 词汇表大小
        output_dim=128,          # 每个词的向量维度
        input_length=max_length, # 输入序列长度
        name='embedding'
    ),
    
    # LSTM层：学习序列中的长期依赖关系
    tf.keras.layers.LSTM(64, dropout=0.5, name='lstm'),
    
    # 输出层：二分类（正面/负面）
    tf.keras.layers.Dense(1, activation='sigmoid', name='output')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # 二分类交叉熵
    metrics=['accuracy']
)

print("LSTM模型结构:")
model.summary()

# 4. 训练模型
print("\n4. 开始训练LSTM...")

history = model.fit(
    X_train, y_train,
    epochs=5,  # 训练5轮（LSTM训练较慢）
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. 评估模型
print("\n5. 模型评估...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试准确率: {test_accuracy:.4f}")
print(f"测试损失: {test_loss:.4f}")

# 6. 进行预测
print("\n6. 进行预测...")
predictions = model.predict(X_test[:10], verbose=0)

print("预测结果示例:")
for i in range(10):
    prob = predictions[i][0]
    predicted = "正面" if prob > 0.5 else "负面"
    actual = "正面" if y_test[i] == 1 else "负面"
    print(f"样本{i+1}: 预测={predicted}({prob:.3f}), 实际={actual}")

# 7. 可视化结果
plt.figure(figsize=(12, 8))

# 训练过程
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('LSTM模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('LSTM模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 预测概率分布
plt.subplot(2, 2, 3)
all_predictions = model.predict(X_test, verbose=0)
plt.hist(all_predictions, bins=50, alpha=0.7)
plt.title('预测概率分布')
plt.xlabel('预测概率')
plt.ylabel('频次')
plt.axvline(x=0.5, color='red', linestyle='--', label='分类阈值')
plt.legend()
plt.grid(True)

# 混淆矩阵
plt.subplot(2, 2, 4)
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred = (all_predictions > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')

plt.tight_layout()
plt.show()

# 8. 文本解码示例（可选）
print("\n7. 文本解码示例...")

# 获取单词索引字典
word_index = tf.keras.datasets.imdb.get_word_index()
# 创建反向字典（从索引到单词）
reverse_word_index = {value: key for key, value in word_index.items()}

def decode_review(encoded_review):
    """将数字编码的评论转换回文本"""
    # 注意：索引偏移了3位，因为0,1,2是保留的特殊标记
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])

# 显示一个原始评论
print("原始评论示例:")
sample_review = decode_review(X_test[0])
print(f"评论内容: {sample_review[:200]}...")
print(f"实际标签: {'正面' if y_test[0] == 1 else '负面'}")
print(f"预测概率: {all_predictions[0][0]:.3f}")
print(f"预测标签: {'正面' if all_predictions[0][0] > 0.5 else '负面'}")

# 9. 保存模型
model.save('lstm_sentiment_model.h5')
print("\n模型已保存为 'lstm_sentiment_model.h5'")

print("\n" + "=" * 60)
print("✅ LSTM文本情感分析完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 词嵌入(Embedding)：将单词转换为向量")
print("2. LSTM：处理长序列的记忆网络")
print("3. 序列填充：统一输入长度")
print("4. 二分类：sigmoid激活函数")
print("5. 文本预处理：编码和解码")
print("\n🔍 LSTM vs 简单RNN的优势:")
print("- 能记住更长的历史信息")
print("- 解决了梯度消失问题")
print("- 更适合处理长文本序列")
print("=" * 60)
