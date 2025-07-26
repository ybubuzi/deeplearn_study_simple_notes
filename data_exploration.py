"""
火车日志数据探索脚本
帮助理解数据结构和特征分布
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def explore_train_data(file_path):
    """探索火车日志数据"""
    print("🔍 开始探索火车日志数据")
    print("=" * 60)
    
    # 读取原始数据
    print("📖 读取数据文件...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    print(f"标题行: {lines[0].strip()}")
    
    # 解析数据
    data = []
    headers = lines[0].strip().split()
    
    for i, line in enumerate(lines[1:], 1):
        parts = line.strip().split()
        if len(parts) >= 11:
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
    print(f"成功解析 {len(df)} 条记录")
    
    # 基本信息
    print("\n📊 数据基本信息:")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 数值列统计
    print("\n📈 数值列统计:")
    numeric_cols = ['速度', '限速', '管压', '距离', '公里标']
    for col in numeric_cols:
        if col in df.columns:
            print(f"{col}:")
            print(f"  范围: {df[col].min()} - {df[col].max()}")
            print(f"  平均值: {df[col].mean():.2f}")
            print(f"  非零值: {(df[col] != 0).sum()}条")
    
    # 分类列分析
    print("\n🏷️ 分类列分析:")
    categorical_cols = ['记录名称', '信号灯', '空挡', '前后', '车站名']
    for col in categorical_cols:
        if col in df.columns:
            unique_values = df[col].value_counts()
            print(f"{col} (共{len(unique_values)}种):")
            # 显示前5个最常见的值
            for value, count in unique_values.head().items():
                print(f"  {value}: {count}次")
            if len(unique_values) > 5:
                print(f"  ... 还有{len(unique_values)-5}种其他值")
    
    # 时间分析
    print("\n⏰ 时间序列分析:")
    try:
        # 尝试解析时间
        df['时间'] = pd.to_datetime(df['日期时间'], format='%Y-%m-%d%H:%M:%S', errors='coerce')
        valid_times = df['时间'].dropna()
        if len(valid_times) > 0:
            print(f"时间范围: {valid_times.min()} 到 {valid_times.max()}")
            print(f"时间跨度: {valid_times.max() - valid_times.min()}")
        else:
            print("无法解析时间格式")
    except:
        print("时间解析失败")
    
    # 状态变化分析
    print("\n🔄 状态变化分析:")
    
    # 速度变化
    speed_changes = df['速度'].diff().dropna()
    print(f"速度变化:")
    print(f"  加速次数: {(speed_changes > 0).sum()}")
    print(f"  减速次数: {(speed_changes < 0).sum()}")
    print(f"  最大加速: {speed_changes.max()}")
    print(f"  最大减速: {speed_changes.min()}")
    
    # 信号灯变化
    signal_changes = df['信号灯'] != df['信号灯'].shift(1)
    print(f"信号灯变化次数: {signal_changes.sum()}")
    
    # 可视化
    visualize_data(df)
    
    return df

def visualize_data(df):
    """可视化数据"""
    print("\n📊 生成数据可视化...")
    
    plt.figure(figsize=(20, 15))
    
    # 1. 速度时序图
    plt.subplot(3, 4, 1)
    sample_size = min(500, len(df))
    plt.plot(df['速度'][:sample_size], alpha=0.7)
    plt.title(f'速度变化（前{sample_size}条记录）')
    plt.xlabel('记录序号')
    plt.ylabel('速度 (km/h)')
    plt.grid(True, alpha=0.3)
    
    # 2. 管压时序图
    plt.subplot(3, 4, 2)
    plt.plot(df['管压'][:sample_size], color='orange', alpha=0.7)
    plt.title(f'管压变化（前{sample_size}条记录）')
    plt.xlabel('记录序号')
    plt.ylabel('管压')
    plt.grid(True, alpha=0.3)
    
    # 3. 速度分布
    plt.subplot(3, 4, 3)
    plt.hist(df['速度'], bins=30, alpha=0.7, edgecolor='black')
    plt.title('速度分布')
    plt.xlabel('速度 (km/h)')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 4. 管压分布
    plt.subplot(3, 4, 4)
    plt.hist(df['管压'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('管压分布')
    plt.xlabel('管压')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 5. 记录名称分布（前10）
    plt.subplot(3, 4, 5)
    top_records = df['记录名称'].value_counts().head(10)
    plt.barh(range(len(top_records)), top_records.values)
    plt.yticks(range(len(top_records)), top_records.index)
    plt.title('最常见记录类型（前10）')
    plt.xlabel('频次')
    
    # 6. 信号灯分布
    plt.subplot(3, 4, 6)
    signal_counts = df['信号灯'].value_counts()
    plt.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%')
    plt.title('信号灯状态分布')
    
    # 7. 速度vs管压散点图
    plt.subplot(3, 4, 7)
    sample_indices = np.random.choice(len(df), min(1000, len(df)), replace=False)
    plt.scatter(df.iloc[sample_indices]['速度'], df.iloc[sample_indices]['管压'], alpha=0.5)
    plt.title('速度 vs 管压关系')
    plt.xlabel('速度 (km/h)')
    plt.ylabel('管压')
    plt.grid(True, alpha=0.3)
    
    # 8. 前后方向分布
    plt.subplot(3, 4, 8)
    direction_counts = df['前后'].value_counts()
    plt.bar(direction_counts.index, direction_counts.values)
    plt.title('前后方向分布')
    plt.xlabel('方向')
    plt.ylabel('频次')
    
    # 9. 空挡状态分布
    plt.subplot(3, 4, 9)
    gear_counts = df['空挡'].value_counts()
    plt.bar(gear_counts.index, gear_counts.values)
    plt.title('空挡状态分布')
    plt.xlabel('状态')
    plt.ylabel('频次')
    
    # 10. 速度变化率
    plt.subplot(3, 4, 10)
    speed_diff = df['速度'].diff().dropna()
    plt.hist(speed_diff, bins=30, alpha=0.7, edgecolor='black')
    plt.title('速度变化率分布')
    plt.xlabel('速度变化 (km/h)')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 11. 管压变化率
    plt.subplot(3, 4, 11)
    pressure_diff = df['管压'].diff().dropna()
    plt.hist(pressure_diff, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.title('管压变化率分布')
    plt.xlabel('管压变化')
    plt.ylabel('频次')
    plt.grid(True, alpha=0.3)
    
    # 12. 相关性热力图
    plt.subplot(3, 4, 12)
    numeric_cols = ['速度', '限速', '管压', '距离']
    corr_data = df[numeric_cols].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0)
    plt.title('数值特征相关性')
    
    plt.tight_layout()
    plt.show()

def analyze_patterns(df):
    """分析数据模式"""
    print("\n🔍 分析数据模式:")
    
    # 1. 识别不同的运行阶段
    print("\n1. 运行阶段识别:")
    
    # 根据速度划分阶段
    def get_phase(speed):
        if speed == 0:
            return "停车"
        elif speed <= 20:
            return "启动/低速"
        elif speed <= 50:
            return "中速运行"
        elif speed <= 80:
            return "高速运行"
        else:
            return "超高速"
    
    df['运行阶段'] = df['速度'].apply(get_phase)
    phase_counts = df['运行阶段'].value_counts()
    
    for phase, count in phase_counts.items():
        percentage = count / len(df) * 100
        print(f"  {phase}: {count}条记录 ({percentage:.1f}%)")
    
    # 2. 异常检测
    print("\n2. 异常数据检测:")
    
    # 速度异常
    speed_q99 = df['速度'].quantile(0.99)
    speed_anomalies = df[df['速度'] > speed_q99]
    print(f"  高速异常 (>{speed_q99}km/h): {len(speed_anomalies)}条")
    
    # 管压异常
    pressure_q99 = df['管压'].quantile(0.99)
    pressure_anomalies = df[df['管压'] > pressure_q99]
    print(f"  高管压异常 (>{pressure_q99}): {len(pressure_anomalies)}条")
    
    # 3. 状态转换分析
    print("\n3. 状态转换分析:")
    
    # 信号灯转换
    signal_transitions = []
    for i in range(1, len(df)):
        if df.iloc[i]['信号灯'] != df.iloc[i-1]['信号灯']:
            transition = f"{df.iloc[i-1]['信号灯']} → {df.iloc[i]['信号灯']}"
            signal_transitions.append(transition)
    
    transition_counts = Counter(signal_transitions)
    print("  常见信号灯转换:")
    for transition, count in transition_counts.most_common(5):
        print(f"    {transition}: {count}次")
    
    return df

def main():
    """主函数"""
    try:
        # 探索数据
        df = explore_train_data('test.log')
        
        # 分析模式
        df = analyze_patterns(df)
        
        print("\n" + "=" * 60)
        print("✅ 数据探索完成！")
        print("💡 建议:")
        print("1. 可以根据速度、信号灯、管压等特征预测运行状态")
        print("2. 时序特征很重要，建议使用LSTM等时序模型")
        print("3. 注意处理数据中的异常值和缺失值")
        print("4. 可以考虑创建滑动窗口特征来捕捉趋势")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 探索过程出错: {e}")

if __name__ == "__main__":
    main()
