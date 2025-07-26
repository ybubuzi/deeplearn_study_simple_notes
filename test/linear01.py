"""
这第一个线性回归模型训练
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置随机数种子
np.random.seed(10086)

total = 1000
# 生成10000个[-169.169)的随机数
x_datas = np.random.uniform(-10086, 10086, total).astype(np.float32)

# 生成2x+1+噪声的随机数
y_datas = 2 * x_datas + 1 + np.random.uniform(-50, 50, total).astype(np.float32)

# # 学习点： 归一化
# 函数mean,计算数列中的平均值，函数std标准差，数列中(每个数值与均值的差的平方)和除以总数再开方
# x_mean,x_std = x_datas.mean(), x_datas.std()
# y_mean,y_std = y_datas.mean(), y_datas.std()
#
# x_datas = (x_datas - x_mean) / x_std
# y_datas = (y_datas - y_mean) / y_std

s_datas = 1 * np.random.rand(total)

plt.figure(figsize=(12, 6))
plt.scatter(x_datas, y_datas, s=s_datas, alpha=0.6, label='训练数据')
plt.xlabel('X (输入)')
plt.ylabel('y (目标)')
plt.title('线性回归训练数据v1')
# plt.legend(loc='lower right', title='模型对比')
plt.legend()
plt.grid(True)
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,), name='1linear_layer')
])



# model.compile(optimizer='adam',  # 优化器：控制如何更新模型参数
#               loss='mse',  # 损失函数：衡量预测错误的程度
#               metrics=['mae']  # 评估指标：训练过程中监控的指标
#               )

# 学习点，学习率的作用
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae']
              )
model.summary()
# 将一维数值转为二维，从 [1, 2, 3] 变成 [[1], [2], [3]]
x_datas_reshaped = x_datas.reshape(-1, 1)

# 开始训练
history = model.fit(x_datas_reshaped, y_datas, epochs=100, batch_size=64, verbose=1)
# 获取参数
weights, bias = model.layers[0].get_weights()

x_test = np.linspace(-10, 10, 100).reshape(-1, 1)
y_pred = model.predict(x_test, verbose=1)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(x_test, y_pred, alpha=0.6, label='训练数据')
plt.plot(x_test, y_pred, 'r-', label='预测数据')
plt.plot(x_test, 2 * x_test + 1, 'g--', label='真实数据 y=2x+1')
# plt.subplot(2,2,1)
plt.title('线性回归结果')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.xlabel('训练轮数')
plt.ylabel('损失值')
plt.title('训练过程')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
# 基于原始数据进行预测
train_pred = model.predict(x_datas_reshaped, verbose=1)
# 用预测值减去真实值得到误差
errors = train_pred.flatten() - y_datas
plt.hist(errors, bins=20, alpha=0.7)
plt.xlabel('预测误差')
plt.ylabel('频次')
plt.title('误差分布')
plt.grid(True)

# 实际值对比预测值
plt.subplot(2, 2, 4)
plt.scatter(y_datas, train_pred, alpha=0.6)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('预测值 vs 实际值')
plt.grid(True)

plt.tight_layout()  # 自动调整子图间距

plt.show()
"""
关于为什么要weights[0][0]才能获取斜率，
我们使用layers[0]明确到神经网络的层级
但是一层网络有多个神经元，但是我们这里在构建模型时只设置了一个神经元
一个神经元会接受多个数据，我们这里的输入只有一个一维的输入，也就是一次只会有一个输入
因此我这里layers[0]指定神经网络层级，weights[0]定位该层级的输入维度，我们这里的输入x_data是只有一个维度的数据，因此为0
weights[0][0]这里的第二个0表示的是该层的第x个神经元，其对应的值是该神经元面对该维度输入时所响应的权重
"""
print(f"最终拟合的函数是 {weights[0][0]:.4} * x + {bias[0]:.4}")
