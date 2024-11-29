# main.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_data(file_path):
    # 定义列名
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    # 读取数据
    data = pd.read_csv(file_path, names=column_names)
    return data

def plot_data(data):
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    
    # 分别绘制每种鸢尾花的数据
    for species in data['class'].unique():
        subset = data[data['class'] == species]
        plt.scatter(subset['sepal_length'], subset['sepal_width'], label=species)
    
    # 添加标题和标签
    plt.title('Sepal Length vs Sepal Width')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    
    # 调用最小二乘分类器函数并绘制分类界线
    w_lsc, b_lsc = least_squares_classifier(data)
    x_values = np.array([data['sepal_length'].min(), data['sepal_length'].max()])
    y_values_lsc = (-w_lsc[0] * x_values - b_lsc) / w_lsc[1]
    plt.plot(x_values, y_values_lsc, color='red', linestyle='--', label='LSC Decision Boundary')
    
    # 调用 Fisher 判别分析函数并绘制分类界线
    w_fld, b_fld = fisher_linear_discriminant(data)
    y_values_fld = (-w_fld[0] * x_values - b_fld) / w_fld[1]
    plt.plot(x_values, y_values_fld, color='blue', linestyle='-.', label='FLD Decision Boundary')
    
    plt.legend()
    plt.show()

def least_squares_classifier(data):
    # 提取特征矩阵 X 和标签向量 y
    X = data[['sepal_length', 'sepal_width']].values
    y = (data['class'] == data['class'].unique()[0]).astype(int)  # 将类别转换为 0 和 1
    
    # 添加偏置项
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    
    # 计算权重向量 w
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    
    # 返回权重向量 w 和偏置项 b
    return w[1:], w[0]

def fisher_linear_discriminant(data):
    # 提取特征矩阵 X 和标签向量 y
    X = data[['sepal_length', 'sepal_width']].values
    y = (data['class'] == data['class'].unique()[0]).astype(int)  # 将类别转换为 0 和 1
    
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

if __name__ == "__main__":
    # 读取筛选后的数据
    filtered_data = read_data("iris/iris_filter.data")
    
    # 绘制数据
    plot_data(filtered_data)