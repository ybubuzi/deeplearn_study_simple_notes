"""
TensorFlow 深度学习入门示例 7: 序列到序列模型 (Seq2Seq)
学习编码器-解码器架构
简单的数字序列翻译（例如：将数字序列反转）
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("TensorFlow版本:", tf.__version__)
print("=" * 60)
print("🔄 序列到序列模型入门")
print("=" * 60)

# 1. 生成序列翻译数据
print("\n1. 生成序列翻译数据...")

def create_sequence_data(num_samples=10000, seq_length=10):
    """
    创建序列翻译数据
    任务：将输入序列反转
    例如：[1,2,3,4,5] -> [5,4,3,2,1]
    """
    np.random.seed(42)
    
    # 词汇表大小（数字0-9）
    vocab_size = 10
    
    input_sequences = []
    target_sequences = []
    
    for _ in range(num_samples):
        # 生成随机序列
        seq = np.random.randint(1, vocab_size, seq_length)
        
        # 输入序列
        input_sequences.append(seq)
        
        # 目标序列（反转）
        target_sequences.append(seq[::-1])
    
    return np.array(input_sequences), np.array(target_sequences), vocab_size

# 生成数据
seq_length = 8
input_seqs, target_seqs, vocab_size = create_sequence_data(10000, seq_length)

print(f"生成了 {len(input_seqs)} 个序列对")
print(f"序列长度: {seq_length}")
print(f"词汇表大小: {vocab_size}")
print(f"输入形状: {input_seqs.shape}")
print(f"目标形状: {target_seqs.shape}")

# 显示数据示例
print(f"\n数据示例:")
for i in range(5):
    print(f"输入: {input_seqs[i]} -> 目标: {target_seqs[i]}")

# 2. 数据预处理
print("\n2. 数据预处理...")

# 分割训练和测试集
train_size = int(0.8 * len(input_seqs))
X_train, X_test = input_seqs[:train_size], input_seqs[train_size:]
y_train, y_test = target_seqs[:train_size], target_seqs[train_size:]

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 为目标序列添加开始和结束标记
# 0: 填充符, 10: 开始标记, 11: 结束标记
START_TOKEN = vocab_size
END_TOKEN = vocab_size + 1
vocab_size_extended = vocab_size + 2

# 创建解码器输入和目标
def prepare_decoder_data(target_sequences):
    """
    准备解码器的输入和目标数据
    解码器输入: [START, 目标序列]
    解码器目标: [目标序列, END]
    """
    decoder_input = np.zeros((len(target_sequences), seq_length + 1))
    decoder_target = np.zeros((len(target_sequences), seq_length + 1))
    
    for i, seq in enumerate(target_sequences):
        # 解码器输入：开始标记 + 目标序列
        decoder_input[i, 0] = START_TOKEN
        decoder_input[i, 1:seq_length+1] = seq
        
        # 解码器目标：目标序列 + 结束标记
        decoder_target[i, :seq_length] = seq
        decoder_target[i, seq_length] = END_TOKEN
    
    return decoder_input, decoder_target

decoder_input_train, decoder_target_train = prepare_decoder_data(y_train)
decoder_input_test, decoder_target_test = prepare_decoder_data(y_test)

print(f"解码器输入形状: {decoder_input_train.shape}")
print(f"解码器目标形状: {decoder_target_train.shape}")

# 3. 构建Seq2Seq模型
print("\n3. 构建Seq2Seq模型...")

# 模型参数
embedding_dim = 64
hidden_units = 128

# 编码器
encoder_inputs = tf.keras.Input(shape=(seq_length,), name='encoder_inputs')
encoder_embedding = tf.keras.layers.Embedding(vocab_size_extended, embedding_dim)(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(hidden_units, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# 解码器
decoder_inputs = tf.keras.Input(shape=(seq_length + 1,), name='decoder_inputs')
decoder_embedding = tf.keras.layers.Embedding(vocab_size_extended, embedding_dim)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(hidden_units, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(vocab_size_extended, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# 完整模型
model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_model')

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Seq2Seq模型结构:")
model.summary()

# 4. 训练模型
print("\n4. 开始训练...")

history = model.fit(
    [X_train, decoder_input_train],
    decoder_target_train,
    epochs=20,
    batch_size=64,
    validation_data=([X_test, decoder_input_test], decoder_target_test),
    verbose=1
)

# 5. 构建推理模型
print("\n5. 构建推理模型...")

# 编码器推理模型
encoder_model = tf.keras.Model(encoder_inputs, encoder_states, name='encoder_inference')

# 解码器推理模型
decoder_state_input_h = tf.keras.Input(shape=(hidden_units,))
decoder_state_input_c = tf.keras.Input(shape=(hidden_units,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_single_input = tf.keras.Input(shape=(1,))
decoder_single_embedding = decoder_embedding(decoder_single_input)
decoder_single_outputs, state_h, state_c = decoder_lstm(
    decoder_single_embedding, initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]
decoder_single_outputs = decoder_dense(decoder_single_outputs)

decoder_model = tf.keras.Model(
    [decoder_single_input] + decoder_states_inputs,
    [decoder_single_outputs] + decoder_states,
    name='decoder_inference'
)

# 6. 序列生成函数
def decode_sequence(input_seq):
    """
    使用训练好的模型生成目标序列
    """
    # 编码输入序列
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # 初始化解码器输入
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = START_TOKEN
    
    # 生成序列
    decoded_sequence = []
    
    for _ in range(seq_length + 1):
        # 预测下一个词
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)
        
        # 选择概率最高的词
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        # 如果遇到结束标记，停止生成
        if sampled_token_index == END_TOKEN:
            break
            
        decoded_sequence.append(sampled_token_index)
        
        # 更新解码器输入
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # 更新状态
        states_value = [h, c]
    
    return decoded_sequence

# 7. 测试模型
print("\n6. 测试模型...")

# 测试几个例子
test_samples = 10
correct_predictions = 0

print("测试结果:")
for i in range(test_samples):
    input_sequence = X_test[i:i+1]
    target_sequence = y_test[i]
    
    # 生成预测序列
    predicted_sequence = decode_sequence(input_sequence)
    
    # 检查预测是否正确
    is_correct = np.array_equal(predicted_sequence, target_sequence)
    if is_correct:
        correct_predictions += 1
    
    print(f"输入: {X_test[i]} -> 预测: {predicted_sequence} -> 目标: {target_sequence} {'✓' if is_correct else '✗'}")

accuracy = correct_predictions / test_samples
print(f"\n序列级准确率: {accuracy:.2f}")

# 8. 可视化结果
plt.figure(figsize=(12, 8))

# 训练过程
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('Seq2Seq模型准确率')
plt.xlabel('训练轮数')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('Seq2Seq模型损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.legend()
plt.grid(True)

# 预测准确性分析
plt.subplot(2, 2, 3)
# 测试更多样本
test_accuracies = []
for i in range(0, min(100, len(X_test)), 10):
    batch_correct = 0
    for j in range(10):
        if i + j < len(X_test):
            pred_seq = decode_sequence(X_test[i+j:i+j+1])
            if np.array_equal(pred_seq, y_test[i+j]):
                batch_correct += 1
    test_accuracies.append(batch_correct / 10)

plt.plot(test_accuracies, 'o-')
plt.title('批次预测准确率')
plt.xlabel('批次')
plt.ylabel('准确率')
plt.grid(True)

# 序列长度分析
plt.subplot(2, 2, 4)
lengths = [len(decode_sequence(X_test[i:i+1])) for i in range(min(50, len(X_test)))]
plt.hist(lengths, bins=range(seq_length + 3), alpha=0.7)
plt.title('生成序列长度分布')
plt.xlabel('序列长度')
plt.ylabel('频次')
plt.grid(True)

plt.tight_layout()
plt.show()

# 9. 保存模型
model.save('seq2seq_model.h5')
print("\n模型已保存为 'seq2seq_model.h5'")

print("\n" + "=" * 60)
print("✅ 序列到序列模型完成！")
print("=" * 60)
print("📚 学到的概念:")
print("1. 编码器-解码器架构：将输入序列编码，然后解码为输出序列")
print("2. 状态传递：编码器的最终状态作为解码器的初始状态")
print("3. 教师强制：训练时使用真实目标作为解码器输入")
print("4. 推理模式：逐步生成序列，每次预测一个词")
print("5. 特殊标记：START和END标记控制序列生成")
print("\n🔍 Seq2Seq的应用场景:")
print("- 机器翻译：英语 -> 中文")
print("- 文本摘要：长文本 -> 摘要")
print("- 对话系统：问题 -> 回答")
print("- 代码生成：描述 -> 代码")
print("=" * 60)
