"""
火车运行状态分析 - 深度学习模型
基于时序数据预测火车运行状态

这个程序的目标：
1. 读取火车运行日志数据（类似于web服务器的访问日志）
2. 清洗和预处理数据（就像处理JSON数据一样）
3. 使用深度学习模型预测火车的运行状态
4. 可视化分析结果
"""

# 导入必要的库（类似于Java中的import或JS中的require/import）
import pandas as pd           # 数据处理库，类似于Excel操作，用于读取和处理表格数据
import numpy as np            # 数学计算库，处理数组和矩阵运算（类似于Java中的数组操作）
import tensorflow as tf       # 深度学习框架，Google开发的机器学习库
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 数据预处理工具
from sklearn.model_selection import train_test_split           # 数据分割工具
import matplotlib.pyplot as plt  # 绘图库，用于生成图表（类似于前端的图表库）
import seaborn as sns            # 高级绘图库，基于matplotlib
from datetime import datetime   # 时间处理库
import warnings                  # 警告处理库
warnings.filterwarnings('ignore')  # 忽略警告信息，让输出更清洁

# 设置matplotlib绘图库的中文字体支持（否则中文会显示为方块）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

print("=" * 60)
print("🚂 火车运行状态分析系统")
print("=" * 60)

# 定义火车状态分析器类（类似于Java中的class）
class TrainStatusAnalyzer:
    """
    火车状态分析器类
    这个类封装了所有的数据处理和机器学习功能
    类似于Java中的一个Service类，包含了完整的业务逻辑
    """

    def __init__(self, log_file_path):
        """
        构造函数（类似于Java的构造器）
        初始化分析器的各种属性

        参数:
        log_file_path: 日志文件路径（字符串类型）
        """
        # 实例变量（类似于Java中的成员变量）
        self.log_file_path = log_file_path    # 日志文件路径
        self.df = None                        # 原始数据表格（DataFrame，类似于二维数组）
        self.processed_df = None              # 处理后的数据表格
        self.label_encoders = {}              # 标签编码器字典（用于将文字转换为数字）
        self.scaler = StandardScaler()        # 数据标准化器（将数据缩放到相同范围）
        self.model = None                     # 深度学习模型对象
        
    def load_and_clean_data(self):
        """
        加载和清洗数据的方法
        这个方法负责从文件中读取数据并进行初步清洗
        类似于从API获取数据并进行格式化处理
        """
        print("\n📊 正在加载和清洗数据...")

        # 读取日志文件（类似于读取CSV文件或解析JSON）
        try:
            # 第一种方法：使用pandas直接读取
            # pandas.read_csv() 类似于 JavaScript 的 CSV.parse() 或 Java 的 CSVReader
            self.df = pd.read_csv(
                self.log_file_path,           # 文件路径
                sep='\s+',                    # 分隔符：使用空白字符（空格、制表符等）
                header=0,                     # 第一行作为列名
                encoding='utf-8',             # 文件编码
                engine='python'               # 使用Python引擎解析
            )
        except:
            # 如果自动解析失败，手动解析文件（类似于手动解析文本文件）
            print("自动解析失败，使用手动解析...")

            # 打开文件并读取所有行（类似于Java的BufferedReader或JS的fs.readFileSync）
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()  # 读取所有行到列表中

            # 解析数据
            data = []  # 存储解析后的数据（二维列表）
            headers = lines[0].strip().split()  # 第一行作为列名

            # 遍历每一行数据（从第二行开始，跳过标题行）
            for line in lines[1:]:
                # 分割每行数据，去除首尾空白字符
                parts = line.strip().split()

                if len(parts) >= len(headers):
                    # 如果数据列数超过标题列数，合并最后的"其他"列
                    if len(parts) > len(headers):
                        # 将多余的部分合并到最后一列（处理"其他"字段包含空格的情况）
                        other_data = ' '.join(parts[len(headers)-1:])
                        row = parts[:len(headers)-1] + [other_data]
                    else:
                        row = parts
                    data.append(row)
                else:
                    # 如果列数不够，用空字符串补齐缺失的列
                    row = parts + [''] * (len(headers) - len(parts))
                    data.append(row)

            # 创建DataFrame对象（类似于创建一个二维表格）
            # DataFrame 可以理解为类似于 JavaScript 的对象数组或 Java 的 List<Map<String, Object>>
            self.df = pd.DataFrame(data, columns=headers)

        # 打印数据基本信息
        print(f"原始数据形状: {self.df.shape}")  # shape 返回 (行数, 列数) 的元组
        print(f"列名: {list(self.df.columns)}")   # 显示所有列名

        # 调用数据清洗方法
        self._clean_data()
        
    def _clean_data(self):
        """
        清洗数据的私有方法（方法名前的下划线表示私有，类似于Java的private）
        这个方法负责处理脏数据、缺失值、数据类型转换等
        """
        print("\n🧹 正在清洗数据...")

        # 处理数值列（将文本转换为数字）
        # 定义哪些列应该是数值类型
        numeric_columns = ['序号', '距离', '速度', '限速', '管压']

        for col in numeric_columns:
            if col in self.df.columns:  # 检查列是否存在
                # pd.to_numeric() 类似于 JavaScript 的 parseInt() 或 Java 的 Integer.parseInt()
                # errors='coerce' 表示无法转换的值设为 NaN（Not a Number）
                # fillna(0) 将 NaN 值填充为 0
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        # 处理时间列（时间字符串转换为时间对象）
        if '日期时间' in self.df.columns:
            try:
                # pd.to_datetime() 类似于 JavaScript 的 new Date() 或 Java 的 SimpleDateFormat.parse()
                # format 参数指定时间格式：%Y=年，%m=月，%d=日，%H=时，%M=分，%S=秒
                self.df['日期时间'] = pd.to_datetime(self.df['日期时间'], format='%Y-%m-%d%H:%M:%S')
            except:
                print("⚠️ 时间格式解析失败，使用序号作为时间序列")
                # 如果时间解析失败，创建一个递增的序列作为时间轴
                # range(len(self.df)) 生成 0, 1, 2, 3, ... 的序列
                self.df['时间序列'] = range(len(self.df))

        # 处理缺失值（将所有 NaN 或 None 值填充为空字符串）
        # fillna('') 类似于 JavaScript 的 || '' 或 Java 的 Optional.orElse("")
        self.df = self.df.fillna('')

        # 创建状态标签（这是机器学习的目标变量）
        self._create_status_labels()

        print(f"清洗后数据形状: {self.df.shape}")
        
    def _create_status_labels(self):
        """
        根据事件类型和参数创建状态标签
        这是机器学习中的"标签工程"，将原始数据转换为可以预测的目标类别
        类似于给每条记录打上一个"标签"，告诉模型这条记录代表什么状态
        """
        print("\n🏷️ 正在创建状态标签...")

        def classify_status(row):
            """
            状态分类函数（内部函数，类似于JavaScript的内部函数或Java的内部方法）
            根据记录名称、速度、信号灯等信息分类状态

            参数:
            row: DataFrame的一行数据（类似于一个对象，包含所有列的值）

            返回:
            字符串类型的状态标签
            """
            # 提取关键字段（类似于从对象中取值）
            record_name = str(row['记录名称']).lower()  # 转换为小写便于匹配
            speed = float(row['速度']) if row['速度'] != '' else 0  # 安全转换为浮点数
            signal = str(row['信号灯'])

            # 定义状态分类规则（类似于一系列if-else判断）
            # 这些规则是基于业务逻辑制定的，类似于业务规则引擎
            if '开机' in record_name or '进入' in record_name:
                return '初始化'
            elif '自检' in record_name or '制动' in record_name:
                return '检测状态'
            elif speed == 0 and ('停车' in record_name or '停' in record_name):
                return '停车状态'
            elif speed > 0 and speed <= 30:
                return '低速运行'
            elif speed > 30 and speed <= 60:
                return '中速运行'
            elif speed > 60:
                return '高速运行'
            elif '红' in signal:
                return '停车信号'
            elif '黄' in signal:
                return '减速信号'
            elif '绿' in signal:
                return '正常运行'
            elif '按键操作' in record_name:
                return '操作状态'
            elif '管压变化' in record_name or '速度变化' in record_name:
                return '参数变化'
            else:
                return '其他状态'

        # 应用分类函数到每一行数据
        # df.apply() 类似于 JavaScript 的 array.map() 或 Java 的 stream().map()
        # axis=1 表示按行应用函数（axis=0是按列）
        self.df['运行状态'] = self.df.apply(classify_status, axis=1)

        # 统计各状态的分布情况
        # value_counts() 类似于 JavaScript 的 reduce() 计数或 Java 的 groupingBy() + counting()
        status_counts = self.df['运行状态'].value_counts()
        print("状态分布:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}条记录")
            
    def feature_engineering(self):
        """
        特征工程方法
        这是机器学习中最重要的步骤之一，将原始数据转换为模型可以理解的特征
        类似于前端开发中的数据预处理，或者后端开发中的DTO转换
        """
        print("\n⚙️ 正在进行特征工程...")

        # 选择基础数值特征列（这些是可以直接用于训练的数值型数据）
        feature_columns = ['速度', '限速', '管压', '距离']

        # 定义需要编码的分类特征（文本类型的数据需要转换为数字）
        # 机器学习模型只能处理数字，不能直接处理文字
        categorical_features = ['记录名称', '信号灯', '空挡', '前后', '车站名']

        # 复制原始数据，避免修改原始数据（类似于深拷贝）
        processed_data = self.df.copy()

        # 对分类特征进行标签编码（将文字转换为数字）
        for col in categorical_features:
            if col in processed_data.columns:
                # LabelEncoder 类似于创建一个字典映射
                # 例如：{'红灯': 0, '绿灯': 1, '黄灯': 2}
                le = LabelEncoder()

                # 确保所有值都是字符串类型
                processed_data[col] = processed_data[col].astype(str)

                # 进行编码转换
                # fit_transform() 相当于先学习映射关系，再应用转换
                processed_data[f'{col}_encoded'] = le.fit_transform(processed_data[col])

                # 保存编码器，以便后续预测时使用相同的映射
                self.label_encoders[col] = le

                # 将编码后的列添加到特征列表
                feature_columns.append(f'{col}_encoded')

        # 创建时间特征（从时间戳中提取有用的时间信息）
        if '日期时间' in processed_data.columns:
            # 提取小时和分钟作为特征（类似于从Date对象中提取时间部分）
            processed_data['小时'] = processed_data['日期时间'].dt.hour
            processed_data['分钟'] = processed_data['日期时间'].dt.minute
            feature_columns.extend(['小时', '分钟'])
        else:
            # 如果没有时间列，使用序号创建时间特征
            processed_data['时间序列'] = range(len(processed_data))
            feature_columns.append('时间序列')

        # 创建滑动窗口特征（时序数据的重要特征）
        # 滑动窗口可以捕捉数据的趋势和模式
        window_size = 5  # 窗口大小：使用前5个数据点

        for col in ['速度', '管压']:
            if col in processed_data.columns:
                # 滑动平均：计算过去5个时间点的平均值
                # rolling() 类似于创建一个滑动窗口
                # 例如：[1,2,3,4,5] 的滑动平均(窗口=3) = [1, 1.5, 2, 3, 4]
                processed_data[f'{col}_滑动平均'] = processed_data[col].rolling(
                    window=window_size, min_periods=1
                ).mean()

                # 变化率：计算相对于前一个时间点的变化百分比
                # pct_change() 类似于 (current - previous) / previous
                processed_data[f'{col}_变化率'] = processed_data[col].pct_change().fillna(0)

                # 添加到特征列表
                feature_columns.extend([f'{col}_滑动平均', f'{col}_变化率'])

        # 选择最终的特征列（只保留存在的列）
        available_features = [col for col in feature_columns if col in processed_data.columns]

        # 创建最终的处理后数据集（包含特征和目标变量）
        self.processed_df = processed_data[available_features + ['运行状态']].copy()

        # 处理异常值：无穷大和NaN值
        # replace() 将无穷大值替换为NaN，然后fillna(0)将NaN替换为0
        self.processed_df = self.processed_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"特征工程完成，最终特征数: {len(available_features)}")
        print(f"特征列表: {available_features}")

        """
        特征工程后的数据结构示例：

        原始数据：
        | 记录名称 | 速度 | 信号灯 | 运行状态 |
        |---------|------|--------|----------|
        | 开机    | 0    | 红灯   | 初始化   |
        | 加速    | 30   | 绿灯   | 低速运行 |

        处理后数据：
        | 速度 | 记录名称_encoded | 信号灯_encoded | 速度_滑动平均 | 速度_变化率 | 运行状态 |
        |------|------------------|----------------|---------------|-------------|----------|
        | 0    | 0                | 0              | 0             | 0           | 初始化   |
        | 30   | 1                | 1              | 15            | inf         | 低速运行 |
        """
        
    def prepare_sequences(self, sequence_length=10):
        """
        准备时序数据的方法
        将普通的表格数据转换为时序序列数据，供LSTM等时序模型使用

        时序数据的概念：
        - 普通数据：每一行是独立的样本
        - 时序数据：每个样本包含连续的多个时间点的数据

        例如：预测第11个时间点的状态，需要看前10个时间点的数据
        类似于根据过去10天的股价预测明天的股价
        """
        print(f"\n📈 正在准备时序数据，序列长度: {sequence_length}")

        # 分离特征和标签（类似于分离输入和输出）
        # 特征：用于预测的输入数据（X）
        # 标签：要预测的目标数据（y）
        feature_cols = [col for col in self.processed_df.columns if col != '运行状态']
        X = self.processed_df[feature_cols].values  # .values 将DataFrame转换为numpy数组

        # 编码目标变量（将状态文字转换为数字）
        # 例如：['初始化', '低速运行', '高速运行'] -> [0, 1, 2]
        le_target = LabelEncoder()
        y = le_target.fit_transform(self.processed_df['运行状态'])
        self.label_encoders['target'] = le_target  # 保存编码器

        # 标准化特征（将所有特征缩放到相同范围，通常是0-1或-1到1）
        # 这很重要，因为不同特征的数值范围可能差异很大
        # 例如：速度(0-100) vs 管压(0-500)，需要标准化到相同范围
        X_scaled = self.scaler.fit_transform(X)

        # 创建时序序列
        # 这是时序建模的核心：将连续的数据点组合成序列
        X_sequences = []  # 存储输入序列
        y_sequences = []  # 存储对应的标签

        # 从第sequence_length个数据点开始，因为前面的点用作历史数据
        for i in range(sequence_length, len(X_scaled)):
            # 取前sequence_length个时间点作为输入序列
            # 例如：i=10时，取索引0-9的数据作为输入，预测索引10的标签
            X_sequences.append(X_scaled[i-sequence_length:i])
            y_sequences.append(y[i])

        # 转换为numpy数组（深度学习框架需要的格式）
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)

        print(f"序列数据形状: X={X_sequences.shape}, y={y_sequences.shape}")

        """
        数据形状说明：
        X_sequences.shape = (样本数, 序列长度, 特征数)
        例如：(1000, 10, 15) 表示：
        - 1000个样本
        - 每个样本包含10个时间步
        - 每个时间步有15个特征

        这类似于一个三维数组：
        [
          [ [特征1, 特征2, ...], [特征1, 特征2, ...], ... ],  # 样本1的10个时间步
          [ [特征1, 特征2, ...], [特征1, 特征2, ...], ... ],  # 样本2的10个时间步
          ...
        ]
        """

        return X_sequences, y_sequences
        
    def build_model(self, input_shape, num_classes):
        """
        构建LSTM深度学习模型
        LSTM (Long Short-Term Memory) 是一种特殊的神经网络，擅长处理时序数据
        类似于人的记忆：能记住重要的历史信息，忘记不重要的信息
        """
        print(f"\n🧠 正在构建LSTM模型...")
        print(f"输入形状: {input_shape}, 类别数: {num_classes}")

        # 使用Sequential模型（层级式堆叠，类似于搭积木）
        model = tf.keras.Sequential([

            # 第一个LSTM层
            # LSTM层是模型的"记忆单元"，能够学习时序模式
            # 64: 神经元数量（类似于大脑中的神经元数量）
            # return_sequences=True: 输出完整序列（为下一层LSTM提供输入）
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape, name='lstm1'),

            # Dropout层：防止过拟合（类似于随机"关闭"一些神经元）
            # 0.2表示随机关闭20%的神经元，类似于正则化
            tf.keras.layers.Dropout(0.2, name='dropout1'),

            # 第二个LSTM层
            # 32: 更少的神经元，逐渐压缩信息
            # return_sequences=False: 只输出最后一个时间步的结果
            tf.keras.layers.LSTM(32, return_sequences=False, name='lstm2'),
            tf.keras.layers.Dropout(0.2, name='dropout2'),

            # 全连接层（Dense层）
            # 将LSTM的输出进一步处理，类似于传统的神经网络层
            # 32: 神经元数量
            # activation='relu': 激活函数，类似于开关，决定神经元是否激活
            tf.keras.layers.Dense(32, activation='relu', name='dense1'),
            tf.keras.layers.Dropout(0.3, name='dropout3'),

            # 输出层
            # num_classes: 输出类别数（有多少种状态就有多少个神经元）
            # activation='softmax': 将输出转换为概率分布（所有概率加起来=1）
            tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
        ])

        # 编译模型（设置训练参数）
        model.compile(
            # 优化器：控制模型如何学习（类似于学习策略）
            optimizer='adam',  # Adam是一种自适应学习率的优化器

            # 损失函数：衡量预测与真实值的差距（类似于考试评分标准）
            loss='sparse_categorical_crossentropy',  # 适用于多分类问题

            # 评估指标：训练过程中监控的指标
            metrics=['accuracy']  # 准确率：预测正确的比例
        )

        print("模型结构:")
        model.summary()  # 显示模型的详细结构

        """
        模型结构解释：

        输入: (batch_size, 10, 15) - 批次大小 x 时间步数 x 特征数
        ↓
        LSTM1(64): 学习时序模式，输出 (batch_size, 10, 64)
        ↓
        Dropout(0.2): 随机关闭20%神经元
        ↓
        LSTM2(32): 进一步学习，输出 (batch_size, 32)
        ↓
        Dropout(0.2): 防止过拟合
        ↓
        Dense(32): 全连接层，输出 (batch_size, 32)
        ↓
        Dropout(0.3): 更强的正则化
        ↓
        Dense(num_classes): 输出层，输出 (batch_size, num_classes)
        ↓
        Softmax: 转换为概率分布
        """

        return model

    def train_model(self, X, y, test_size=0.2, epochs=50):
        """
        训练深度学习模型
        这是机器学习的核心步骤：让模型从数据中学习规律
        类似于让学生通过做练习题来学习知识

        参数:
        X: 输入特征数据（三维数组）
        y: 目标标签数据（一维数组）
        test_size: 测试集比例（0.2表示20%用于测试）
        epochs: 训练轮数（模型看数据的次数）
        """
        print(f"\n🚀 开始训练模型...")

        # 分割数据集（类似于将题目分为练习题和考试题）
        # train_test_split 类似于随机分配，确保训练和测试数据的代表性
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,                    # 要分割的数据
            test_size=test_size,     # 测试集比例
            random_state=42,         # 随机种子，确保结果可重现
            stratify=y               # 分层抽样，确保各类别比例一致
        )

        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")

        # 构建模型
        num_classes = len(np.unique(y))  # 计算类别数量
        self.model = self.build_model(X_train.shape[1:], num_classes)

        # 训练模型（这是最重要的步骤）
        print("开始训练过程...")
        history = self.model.fit(
            X_train, y_train,           # 训练数据
            epochs=epochs,              # 训练轮数（模型看数据的次数）
            batch_size=32,              # 批次大小（每次处理32个样本）
            validation_data=(X_test, y_test),  # 验证数据（用于监控训练效果）
            verbose=1                   # 显示训练进度
        )

        """
        训练过程解释：

        Epoch（轮次）: 模型完整看一遍所有训练数据算一轮
        - 类似于学生完整复习一遍所有课本

        Batch（批次）: 每次处理的样本数量
        - 类似于学生每次做32道题，而不是一次做完所有题

        Loss（损失）: 模型预测错误的程度
        - 数值越小表示预测越准确
        - 类似于考试的错误率

        Accuracy（准确率）: 预测正确的比例
        - 数值越大表示模型越好
        - 类似于考试的正确率

        Validation（验证）: 用测试数据检查模型是否过拟合
        - 类似于用模拟考试检查学习效果
        """

        # 评估模型性能
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n📊 模型评估结果:")
        print(f"测试准确率: {test_accuracy:.4f}")  # .4f表示保留4位小数
        print(f"测试损失: {test_loss:.4f}")

        # 返回训练历史和测试数据（用于后续分析）
        return history, (X_test, y_test)

    def predict_status(self, sequence_data):
        """
        预测运行状态的方法
        使用训练好的模型对新数据进行预测
        类似于学生学会知识后，用来解答新题目

        参数:
        sequence_data: 输入的时序数据（需要预测的序列）

        返回:
        predicted_statuses: 预测的状态名称（如"低速运行"）
        predictions: 原始预测概率（每个类别的概率值）
        """
        # 检查模型是否已训练
        if self.model is None:
            print("❌ 模型未训练，请先训练模型")
            return None

        # 预处理输入数据
        # 确保输入数据的形状正确（需要是三维：批次 x 时间步 x 特征）
        if len(sequence_data.shape) == 2:
            # 如果是二维数据，添加批次维度
            # 例如：(10, 15) -> (1, 10, 15)
            sequence_data = sequence_data.reshape(1, *sequence_data.shape)

        # 进行预测
        # model.predict() 返回每个类别的概率
        predictions = self.model.predict(sequence_data, verbose=0)

        # 获取概率最高的类别索引
        # np.argmax() 找到最大值的索引
        # 例如：[0.1, 0.7, 0.2] -> 1（第二个元素最大）
        predicted_classes = np.argmax(predictions, axis=1)

        # 将数字标签转换回文字标签
        # 使用之前保存的编码器进行反向转换
        # 例如：1 -> "低速运行"
        le_target = self.label_encoders['target']
        predicted_statuses = le_target.inverse_transform(predicted_classes)

        """
        预测结果解释：

        predictions: 原始概率输出
        例如：[[0.1, 0.7, 0.15, 0.05]]
        表示：
        - 初始化状态: 10%概率
        - 低速运行: 70%概率  <- 最高概率
        - 中速运行: 15%概率
        - 高速运行: 5%概率

        predicted_classes: [1]  (概率最高的类别索引)
        predicted_statuses: ["低速运行"]  (转换后的状态名称)
        """

        return predicted_statuses, predictions

    def visualize_results(self, history, test_data=None):
        """可视化结果"""
        print("\n📈 正在生成可视化结果...")

        plt.figure(figsize=(15, 10))

        # 训练历史
        plt.subplot(2, 3, 1)
        plt.plot(history.history['accuracy'], label='训练准确率')
        plt.plot(history.history['val_accuracy'], label='验证准确率')
        plt.title('模型准确率')
        plt.xlabel('训练轮数')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title('模型损失')
        plt.xlabel('训练轮数')
        plt.ylabel('损失值')
        plt.legend()
        plt.grid(True)

        # 状态分布
        plt.subplot(2, 3, 3)
        status_counts = self.df['运行状态'].value_counts()
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
        plt.title('运行状态分布')

        # 时序特征分析
        plt.subplot(2, 3, 4)
        if '速度' in self.df.columns:
            plt.plot(self.df['速度'][:200], label='速度')
            plt.title('速度变化趋势（前200条记录）')
            plt.xlabel('时间序列')
            plt.ylabel('速度 (km/h)')
            plt.legend()
            plt.grid(True)

        plt.subplot(2, 3, 5)
        if '管压' in self.df.columns:
            plt.plot(self.df['管压'][:200], label='管压', color='orange')
            plt.title('管压变化趋势（前200条记录）')
            plt.xlabel('时间序列')
            plt.ylabel('管压')
            plt.legend()
            plt.grid(True)

        # 混淆矩阵
        if test_data is not None:
            X_test, y_test = test_data
            y_pred = np.argmax(self.model.predict(X_test, verbose=0), axis=1)

            plt.subplot(2, 3, 6)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('混淆矩阵')
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')

        plt.tight_layout()
        plt.show()

    def save_model(self, model_path='train_status_model.h5'):
        """保存模型"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"✅ 模型已保存到: {model_path}")
        else:
            print("❌ 没有可保存的模型")

    def load_model(self, model_path='train_status_model.h5'):
        """加载模型"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ 模型已从 {model_path} 加载")
        except:
            print(f"❌ 无法加载模型: {model_path}")


def main():
    """
    主函数 - 程序的入口点
    这个函数协调整个机器学习流程，类似于项目的主控制器
    按照标准的机器学习流程执行：数据处理 -> 特征工程 -> 模型训练 -> 评估 -> 预测
    """
    print("🚀 开始火车运行状态分析项目")
    print("这是一个完整的深度学习项目，包含以下步骤：")
    print("1. 数据加载和清洗")
    print("2. 特征工程")
    print("3. 时序数据准备")
    print("4. 模型构建和训练")
    print("5. 结果可视化")
    print("6. 模型保存和预测")

    # 第一步：创建分析器实例
    # 类似于创建一个项目管理器，负责整个分析流程
    analyzer = TrainStatusAnalyzer('test.log')

    # 第二步：数据处理流程
    print("\n" + "="*50)
    print("📊 数据处理阶段")
    print("="*50)

    # 加载和清洗原始数据
    analyzer.load_and_clean_data()

    # 特征工程：将原始数据转换为机器学习可用的特征
    analyzer.feature_engineering()

    # 第三步：准备时序数据
    print("\n" + "="*50)
    print("🔄 时序数据准备阶段")
    print("="*50)

    # 将表格数据转换为时序序列数据
    # sequence_length=10 表示用过去10个时间点预测下一个时间点
    X, y = analyzer.prepare_sequences(sequence_length=10)

    # 第四步：模型训练
    print("\n" + "="*50)
    print("🧠 模型训练阶段")
    print("="*50)

    # 训练LSTM深度学习模型
    # epochs=30 表示模型看数据30遍（可以根据需要调整）
    history, test_data = analyzer.train_model(X, y, epochs=30)

    # 第五步：结果可视化
    print("\n" + "="*50)
    print("📈 结果分析阶段")
    print("="*50)

    # 生成训练过程图表和性能分析图
    analyzer.visualize_results(history, test_data)

    # 第六步：保存模型
    print("\n" + "="*50)
    print("💾 模型保存阶段")
    print("="*50)

    # 保存训练好的模型，以便后续使用
    analyzer.save_model()

    # 第七步：示例预测
    print("\n" + "="*50)
    print("🔮 预测演示阶段")
    print("="*50)

    print("进行状态预测示例...")

    # 取第一个序列作为预测示例
    sample_sequence = X[0:1]  # 形状：(1, 10, 特征数)

    # 使用训练好的模型进行预测
    predicted_status, confidence = analyzer.predict_status(sample_sequence)

    print(f"预测状态: {predicted_status[0]}")
    print(f"各状态置信度分布:")

    # 显示每个状态的预测概率
    le_target = analyzer.label_encoders['target']
    for i, prob in enumerate(confidence[0]):
        status_name = le_target.inverse_transform([i])[0]
        print(f"  {status_name}: {prob:.3f} ({prob*100:.1f}%)")

    # 项目完成总结
    print("\n" + "=" * 60)
    print("✅ 火车运行状态分析项目完成！")
    print("=" * 60)
    print("📋 项目成果:")
    print("1. ✅ 成功处理了火车运行日志数据")
    print("2. ✅ 构建了LSTM时序预测模型")
    print("3. ✅ 实现了运行状态的自动分类")
    print("4. ✅ 生成了详细的分析报告和图表")
    print("5. ✅ 保存了可重用的预测模型")
    print("\n💡 下一步可以:")
    print("- 使用更多历史数据提高模型准确性")
    print("- 调整模型参数优化性能")
    print("- 部署模型到生产环境进行实时预测")
    print("=" * 60)


# Python程序入口点
# 当直接运行这个文件时，会执行main()函数
# 类似于Java的public static void main()或C的int main()
if __name__ == "__main__":
    main()
