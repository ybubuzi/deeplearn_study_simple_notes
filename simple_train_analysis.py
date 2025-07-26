"""
简化版火车状态分析 - 快速入门版本
专门用于理解数据结构和基础模型训练
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("🚂 简化版火车运行状态分析")
print("=" * 50)

def load_train_data(file_path):
    """加载火车日志数据"""
    print("📊 正在加载数据...")
    
    # 手动解析数据文件
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 获取标题行
    headers = lines[0].strip().split()
    print(f"数据列: {headers}")
    
    # 解析数据行
    for i, line in enumerate(lines[1:], 1):
        parts = line.strip().split()
        if len(parts) >= 11:  # 确保有足够的列
            # 提取主要字段
            try:
                row = {
                    '序号': int(parts[0]) if parts[0].isdigit() else i,
                    '记录名称': parts[1],
                    '日期时间': parts[2],
                    '公里标': float(parts[3]) if parts[3].replace('.', '').isdigit() else 0,
                    '距离': int(parts[4]) if parts[4].isdigit() else 0,
                    '车站名': parts[5] if len(parts) > 5 else '',
                    '信号灯': parts[6] if len(parts) > 6 else '',
                    '速度': int(parts[7]) if parts[7].isdigit() else 0,
                    '限速': int(parts[8]) if parts[8].isdigit() else 0,
                    '空挡': parts[9] if len(parts) > 9 else '',
                    '前后': parts[10] if len(parts) > 10 else '',
                    '管压': int(parts[11]) if len(parts) > 11 and parts[11].isdigit() else 0,
                    '其他': ' '.join(parts[12:]) if len(parts) > 12 else ''
                }
                data.append(row)
            except:
                continue
    
    df = pd.DataFrame(data)
    print(f"成功加载 {len(df)} 条记录")
    return df

def create_simple_features(df):
    """创建简单特征"""
    print("⚙️ 正在创建特征...")
    
    # 创建运行状态标签（简化版）
    def get_status(row):
        speed = row['速度']
        signal = str(row['信号灯'])
        record = str(row['记录名称'])
        
        if '开机' in record or '进入' in record:
            return 0  # 初始化
        elif speed == 0:
            return 1  # 停车
        elif speed > 0 and speed <= 30:
            return 2  # 低速
        elif speed > 30 and speed <= 60:
            return 3  # 中速
        elif speed > 60:
            return 4  # 高速
        else:
            return 5  # 其他
    
    df['状态'] = df.apply(get_status, axis=1)
    
    # 状态名称映射
    status_names = {0: '初始化', 1: '停车', 2: '低速', 3: '中速', 4: '高速', 5: '其他'}
    
    # 显示状态分布
    print("状态分布:")
    for status, count in df['状态'].value_counts().sort_index().items():
        print(f"  {status_names[status]}: {count}条")
    
    # 选择数值特征
    features = ['速度', '限速', '管压', '距离', '公里标']
    
    # 创建特征矩阵
    X = df[features].values
    y = df['状态'].values
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"标签数组形状: {y.shape}")
    
    return X, y, status_names, df

def build_simple_model(input_dim, num_classes):
    """构建简单的神经网络模型"""
    print("🧠 构建神经网络模型...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,), name='hidden1'),
        tf.keras.layers.Dropout(0.3, name='dropout1'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden2'),
        tf.keras.layers.Dropout(0.3, name='dropout2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("模型结构:")
    model.summary()
    return model

def train_and_evaluate(X, y, status_names):
    """训练和评估模型"""
    print("🚀 开始训练模型...")
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"训练集: {X_train.shape[0]}条, 测试集: {X_test.shape[0]}条")
    
    # 构建和训练模型
    model = build_simple_model(X_train.shape[1], len(status_names))
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # 评估模型
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 最终结果:")
    print(f"测试准确率: {test_accuracy:.4f}")
    print(f"测试损失: {test_loss:.4f}")
    
    return model, history, scaler, (X_test, y_test)

def visualize_results(df, history, model, test_data, status_names):
    """可视化分析结果"""
    print("📈 生成可视化图表...")
    
    plt.figure(figsize=(15, 10))
    
    # 1. 训练过程
    plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型训练过程 - 准确率')
    plt.xlabel('训练轮数')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型训练过程 - 损失')
    plt.xlabel('训练轮数')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 2. 状态分布
    plt.subplot(2, 3, 3)
    status_counts = df['状态'].value_counts().sort_index()
    status_labels = [status_names[i] for i in status_counts.index]
    plt.pie(status_counts.values, labels=status_labels, autopct='%1.1f%%')
    plt.title('运行状态分布')
    
    # 3. 速度变化趋势
    plt.subplot(2, 3, 4)
    sample_size = min(500, len(df))
    plt.plot(df['速度'][:sample_size], label='速度', alpha=0.7)
    plt.title(f'速度变化趋势（前{sample_size}条记录）')
    plt.xlabel('时间序列')
    plt.ylabel('速度 (km/h)')
    plt.legend()
    plt.grid(True)
    
    # 4. 管压变化趋势
    plt.subplot(2, 3, 5)
    plt.plot(df['管压'][:sample_size], label='管压', color='orange', alpha=0.7)
    plt.title(f'管压变化趋势（前{sample_size}条记录）')
    plt.xlabel('时间序列')
    plt.ylabel('管压')
    plt.legend()
    plt.grid(True)
    
    # 5. 预测结果分析
    plt.subplot(2, 3, 6)
    X_test, y_test = test_data
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    # 计算每个类别的准确率
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    categories = [status_names[i] for i in sorted(status_names.keys()) if str(i) in report]
    accuracies = [report[str(i)]['f1-score'] for i in sorted(status_names.keys()) if str(i) in report]
    
    plt.bar(categories, accuracies)
    plt.title('各状态预测准确率')
    plt.xlabel('运行状态')
    plt.ylabel('F1分数')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict_example(model, scaler, X, y, status_names, num_examples=5):
    """预测示例"""
    print(f"\n🔮 预测示例（随机选择{num_examples}个样本）:")
    
    # 随机选择样本
    indices = np.random.choice(len(X), num_examples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = X[idx:idx+1]
        sample_scaled = scaler.transform(sample)
        
        # 预测
        prediction = model.predict(sample_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        true_class = y[idx]
        
        print(f"样本 {i+1}:")
        print(f"  输入特征: 速度={sample[0][0]}, 限速={sample[0][1]}, 管压={sample[0][2]}")
        print(f"  真实状态: {status_names[true_class]}")
        print(f"  预测状态: {status_names[predicted_class]}")
        print(f"  预测置信度: {confidence:.1f}%")
        print(f"  预测{'✅正确' if predicted_class == true_class else '❌错误'}")
        print()

def main():
    """主函数"""
    try:
        # 1. 加载数据
        df = load_train_data('test.log')
        
        # 2. 特征工程
        X, y, status_names, df = create_simple_features(df)
        
        # 3. 训练模型
        model, history, scaler, test_data = train_and_evaluate(X, y, status_names)
        
        # 4. 可视化结果
        visualize_results(df, history, model, test_data, status_names)
        
        # 5. 预测示例
        predict_example(model, scaler, X, y, status_names)
        
        # 6. 保存模型
        model.save('simple_train_model.h5')
        print("✅ 模型已保存为 'simple_train_model.h5'")
        
        print("\n" + "=" * 50)
        print("🎉 火车状态分析完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        print("请检查数据文件是否存在且格式正确")

if __name__ == "__main__":
    main()
