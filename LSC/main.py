import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from filter import filter_data  # 导入 filter.py 中的 filter_data 函数

def read_data(file_path):
    # 定义列名
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # 读取数据
    data = pd.read_csv(file_path, names=column_names)
    return data

def plot_data(X_train, X_test, y_train, y_test, w_lsc, b_lsc, w_fld, b_fld):
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    
    # 分别绘制训练集与测试集
    for species in [0, 1]:
        subset_train = X_train[y_train == species]
        subset_test = X_test[y_test == species]
        plt.scatter(subset_train[:, 0], subset_train[:, 1], label=f'Train Class {species}', alpha=0.5)
        plt.scatter(subset_test[:, 0], subset_test[:, 1], label=f'Test Class {species}', marker='x')
    
    # 添加标题、标签
    plt.title('Sepal Length vs Sepal Width (Train and Test Data)')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    
    # 绘制 LSC 决策边界
    x_values = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
    y_values_lsc = (-w_lsc[0] * x_values - b_lsc) / w_lsc[1]
    plt.plot(x_values, y_values_lsc, color='red', linestyle='--', label='LSC Decision Boundary')
    
    # 绘制 FLD 决策边界
    y_values_fld = (-w_fld[0] * x_values - b_fld) / w_fld[1]
    plt.plot(x_values, y_values_fld, color='blue', linestyle='-.', label='FLD Decision Boundary')
    
    plt.legend()
    plt.show()

def least_squares_classifier(X, y):
    # 添加偏置项
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # 计算权重向量 w
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # 返回权重向量 w 和偏置项 b
    return w[1:], w[0]

def fisher_linear_discriminant(X, y):
    # 分离两个类别的数据
    X1 = X[y == 1]
    X2 = X[y == 0]
    
    # 计算每个类别的均值向量
    mean1 = np.mean(X1, axis=0)
    mean2 = np.mean(X2, axis=0)
    
    # 计算类内散布矩阵 S_W
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    S_W = S1 + S2
    
    # 计算最优投影方向 w
    w = np.linalg.inv(S_W) @ (mean1 - mean2)
    
    # 计算偏置项 b
    b = -0.5 * w.T @ (mean1 + mean2)
    
    return w, b

def evaluate_model(X_train, X_test, y_train, y_test, w, b):
    # 预测测试集的标签
    y_pred = (X_test @ w + b > 0).astype(int)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

if __name__ == "__main__":
    # 定义输入输出路径
    input_path = 'iris/iris.data'
    filtered_path = 'iris/iris_filter.data'
    
    # 调用 filter_data 函数筛选数据并保存到新文件
    filter_data(input_path, filtered_path)
    print(f"筛选后的数据已保存到 {filtered_path}")

    # 读取筛选的数据
    filtered_data = read_data(filtered_path)
    
    # 提取特征矩阵 X 和标签向量 y
    X = filtered_data[['sepal_length', 'sepal_width']].values
    y = (filtered_data['class'] == filtered_data['class'].unique()[0]).astype(int)  # 将类别转换为 0 和 1
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练 LSC 模型
    w_lsc, b_lsc = least_squares_classifier(X_train, y_train)
    
    # 训练 FLD 模型
    w_fld, b_fld = fisher_linear_discriminant(X_train, y_train)
    
    # 评估模型性能
    lsc_accuracy = evaluate_model(X_train, X_test, y_train, y_test, w_lsc, b_lsc)
    fld_accuracy = evaluate_model(X_train, X_test, y_train, y_test, w_fld, b_fld)
    
    print(f"LSC Accuracy: {lsc_accuracy:.2f}")
    print(f"FLD Accuracy: {fld_accuracy:.2f}")
    
    # matplolib绘图
    plot_data(X_train, X_test, y_train, y_test, w_lsc, b_lsc, w_fld, b_fld)