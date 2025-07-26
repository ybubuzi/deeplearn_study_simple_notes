"""
TensorFlow深度学习示例运行脚本
这个脚本可以帮你选择运行哪个示例
"""

import os
import sys

def main():
    print("=" * 60)
    print("🚀 TensorFlow深度学习入门示例")
    print("=" * 60)
    print()
    print("请选择要运行的示例:")
    print("1. 线性回归 (最简单，理解基本概念)")
    print("2. 手写数字识别 (经典深度学习项目)")
    print("3. CNN图像分类 (卷积神经网络)")
    print("4. 简单时序预测 (股价预测，RNN入门)")
    print("5. LSTM文本情感分析 (处理文本序列)")
    print("6. 多变量时序预测 (天气预测)")
    print("7. 序列到序列模型 (Seq2Seq翻译)")
    print("8. 运行所有基础示例 (1-3)")
    print("9. 运行所有时序示例 (4-7)")
    print("0. 退出")
    print()
    
    while True:
        try:
            choice = input("请输入选择 (0-9): ").strip()
            
            if choice == "0":
                print("再见！")
                break
            elif choice == "1":
                print("\n🔥 运行线性回归示例...")
                os.system("python 01_linear_regression.py")
            elif choice == "2":
                print("\n🔥 运行手写数字识别示例...")
                os.system("python 02_mnist_classification.py")
            elif choice == "3":
                print("\n🔥 运行CNN图像分类示例...")
                os.system("python 03_cnn_cifar10.py")
            elif choice == "4":
                print("\n🔥 运行简单时序预测示例...")
                os.system("python 04_simple_time_series.py")
            elif choice == "5":
                print("\n🔥 运行LSTM文本情感分析示例...")
                os.system("python 05_lstm_sentiment.py")
            elif choice == "6":
                print("\n🔥 运行多变量时序预测示例...")
                os.system("python 06_multivariate_timeseries.py")
            elif choice == "7":
                print("\n🔥 运行序列到序列模型示例...")
                os.system("python 07_seq2seq_translation.py")
            elif choice == "8":
                print("\n🔥 运行所有基础示例...")
                print("注意：这将花费较长时间！")
                confirm = input("确认运行所有基础示例？(y/n): ").strip().lower()
                if confirm == 'y':
                    os.system("python 01_linear_regression.py")
                    os.system("python 02_mnist_classification.py")
                    os.system("python 03_cnn_cifar10.py")
            elif choice == "9":
                print("\n🔥 运行所有时序示例...")
                print("注意：这将花费很长时间！")
                confirm = input("确认运行所有时序示例？(y/n): ").strip().lower()
                if confirm == 'y':
                    os.system("python 04_simple_time_series.py")
                    os.system("python 05_lstm_sentiment.py")
                    os.system("python 06_multivariate_timeseries.py")
                    os.system("python 07_seq2seq_translation.py")
            else:
                print("❌ 无效选择，请重新输入")
                continue
                
            print("\n" + "=" * 60)
            print("示例运行完成！")
            print("=" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    main()
