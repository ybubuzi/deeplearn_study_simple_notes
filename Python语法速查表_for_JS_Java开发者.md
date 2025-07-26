# Python语法速查表 - 专为JS和Java开发者

## 🎯 基础语法对比

### 1. 变量声明和类型

| 概念 | Java | JavaScript | Python |
|------|------|------------|--------|
| 变量声明 | `int x = 5;` | `let x = 5;` | `x = 5` |
| 常量 | `final int X = 5;` | `const X = 5;` | `X = 5` (约定大写) |
| 字符串 | `String s = "hello";` | `let s = "hello";` | `s = "hello"` |
| 列表/数组 | `int[] arr = {1,2,3};` | `let arr = [1,2,3];` | `arr = [1,2,3]` |

### 2. 函数定义

```python
# Python函数
def function_name(param1, param2=default_value):
    """文档字符串"""
    return result

# 对比：
# Java: public int functionName(int param1, int param2) { return result; }
# JS: function functionName(param1, param2 = defaultValue) { return result; }
```

### 3. 条件语句

```python
# Python
if condition:
    # 注意：用缩进表示代码块，不用{}
    do_something()
elif another_condition:
    do_another()
else:
    do_default()

# 对比Java/JS：
# if (condition) {
#     doSomething();
# } else if (anotherCondition) {
#     doAnother();
# } else {
#     doDefault();
# }
```

### 4. 循环语句

```python
# for循环
for i in range(10):        # 0到9
    print(i)

for item in list:          # 遍历列表
    print(item)

# while循环
while condition:
    do_something()

# 对比：
# Java: for(int i=0; i<10; i++) { System.out.println(i); }
# JS: for(let i=0; i<10; i++) { console.log(i); }
```

## 🔧 数据结构对比

### 1. 列表 (List) - 类似数组

```python
# 创建列表
arr = [1, 2, 3, 4]
arr = []  # 空列表

# 常用操作
arr.append(5)        # 添加元素 - 类似 JS: arr.push(5)
arr.insert(0, 0)     # 在索引0插入 - 类似 Java: list.add(0, 0)
arr.remove(3)        # 删除值为3的元素
arr.pop()            # 删除最后一个元素
len(arr)             # 长度 - 类似 JS: arr.length, Java: arr.size()

# 切片 (Slicing) - Python特有
arr[1:3]             # 索引1到2 - 类似 JS: arr.slice(1,3)
arr[:5]              # 前5个元素
arr[5:]              # 从索引5到末尾
arr[-1]              # 最后一个元素
```

### 2. 字典 (Dict) - 类似对象/Map

```python
# 创建字典
obj = {'name': 'John', 'age': 30}
obj = {}  # 空字典

# 访问和修改
obj['name']          # 获取值 - 类似 JS: obj.name 或 obj['name']
obj['city'] = 'NYC'  # 添加/修改
del obj['age']       # 删除键值对

# 遍历
for key in obj:                    # 遍历键
for value in obj.values():         # 遍历值
for key, value in obj.items():     # 遍历键值对
```

## 📚 常用库和函数

### 1. NumPy - 数值计算

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4])        # 一维数组
matrix = np.array([[1, 2], [3, 4]]) # 二维数组

# 常用操作
arr.shape           # 形状：(4,)
matrix.shape        # 形状：(2, 2)
arr.mean()          # 平均值
arr.std()           # 标准差
arr.min(), arr.max() # 最小值，最大值

# 数学运算
arr + 5             # 每个元素加5
arr * 2             # 每个元素乘2
arr1 + arr2         # 对应元素相加
```

### 2. TensorFlow/Keras - 深度学习

```python
import tensorflow as tf

# 创建张量
tensor = tf.constant([1, 2, 3, 4])

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32)
```

## 🎨 字符串操作

```python
# 字符串格式化
name = "John"
age = 30

# f-string (推荐，Python 3.6+)
message = f"Hello {name}, you are {age} years old"

# format方法
message = "Hello {}, you are {} years old".format(name, age)

# % 格式化 (旧式)
message = "Hello %s, you are %d years old" % (name, age)

# 对比：
# Java: String.format("Hello %s, you are %d years old", name, age)
# JS: `Hello ${name}, you are ${age} years old`
```

## 🔍 文件操作

```python
# 读取文件
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()        # 读取全部
    lines = f.readlines()     # 读取所有行

# 写入文件
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write("Hello World")
    f.writelines(["line1\n", "line2\n"])

# with语句自动关闭文件，类似于Java的try-with-resources
```

## 🧮 异常处理

```python
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
finally:
    print("总是执行")

# 对比：
# Java: try { ... } catch (Exception e) { ... } finally { ... }
# JS: try { ... } catch (e) { ... } finally { ... }
```

## 📦 模块和包

```python
# 导入整个模块
import math
result = math.sqrt(16)

# 导入特定函数
from math import sqrt, pi
result = sqrt(16)

# 导入并重命名
import numpy as np
import tensorflow as tf

# 对比：
# Java: import java.util.List;
# JS: import { sqrt, pi } from 'math';
```

## 🎯 深度学习常用模式

### 1. 数据预处理模式

```python
# 1. 加载数据
data = load_data()

# 2. 数据清洗
data = data.dropna()  # 删除空值
data = data.fillna(0)  # 填充空值

# 3. 数据分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. 数据标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. 模型训练模式

```python
# 1. 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 2. 编译模型
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3. 训练模型
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 4. 评估模型
test_loss, test_accuracy = model.evaluate(X_test, y_test)
```

## 💡 Python特有的优雅语法

### 1. 列表推导式

```python
# 传统方式
squares = []
for i in range(10):
    squares.append(i**2)

# Python方式
squares = [i**2 for i in range(10)]

# 带条件
even_squares = [i**2 for i in range(10) if i % 2 == 0]
```

### 2. 多重赋值

```python
# 交换变量
a, b = b, a

# 函数返回多个值
def get_name_age():
    return "John", 30

name, age = get_name_age()
```

### 3. 链式比较

```python
# Python
if 0 < x < 10:
    print("x在0到10之间")

# 其他语言需要
# if (x > 0 && x < 10)
```

## 🚀 学习建议

1. **从语法开始**：先熟悉基本语法差异
2. **多练习**：运行每个示例代码
3. **查文档**：遇到不懂的函数就查官方文档
4. **写注释**：理解每行代码的作用
5. **对比学习**：将Python概念与Java/JS对比

记住：Python的哲学是"简洁优雅"，很多在Java/JS中需要多行的代码，Python可能一行就搞定！
