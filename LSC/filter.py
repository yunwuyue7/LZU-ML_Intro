# filter.py

import pandas as pd

def filter_data(input_path, filtered_path):
    # 定义列名
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # 读取数据
    iris_data = pd.read_csv(input_path, names=column_names)
    
    # 找出数据最多的两种鸢尾花
    top_classes = iris_data['class'].value_counts().head(2).index.tolist()
    
    # 筛选出这些数据
    filtered_data = iris_data[iris_data['class'].isin(top_classes)]
    
    # 保存筛选后的数据到新的文件
    filtered_data.to_csv(filtered_path, index=False, header=False)
    
    # 打印保存路径
    print(f"Filtered data has been saved to: {filtered_path}")

if __name__ == "__main__":
    # 如果直接运行 filter.py，可以在这里设置输入和输出路径
    input_path = 'iris/iris.data'
    filtered_path = 'iris/iris_filter.data'
    filter_data(input_path, filtered_path)