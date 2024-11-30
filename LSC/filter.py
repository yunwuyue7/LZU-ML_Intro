import pandas as pd

def filter_data(input_path, filtered_path):
    # 定义列名
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    # 读取数据
    iris_data = pd.read_csv(input_path, names=column_names)
    
    # 找出样本最多的两种鸢尾花（实际上样本量相同）
    top_classes = iris_data['class'].value_counts().head(2).index.tolist()
    
    # 筛选数据
    filtered_data = iris_data[iris_data['class'].isin(top_classes)]
    
    # 保存筛选数据到新文件
    filtered_data.to_csv(filtered_path, index=False, header=False)
    
    # 输出保存路径
    print(f"Filtered data has been saved to: {filtered_path}")

if __name__ == "__main__":
    # 只运行 filter.py时，在这里设置输入和输出路径
    input_path = 'iris/iris.data'
    filtered_path = 'iris/iris_filter.data'
    filter_data(input_path, filtered_path)