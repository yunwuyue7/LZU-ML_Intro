import pandas as pd

# 定义列名
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# 指定文件路径
data_path = 'iris/iris.data'

# 读取数据
iris_data = pd.read_csv(data_path, names=column_names)

# 找出数据最多的两种鸢尾花
top_classes = iris_data['class'].value_counts().head(2).index.tolist()

# 筛选出这些数据
filtered_data = iris_data[iris_data['class'].isin(top_classes)]

# 保存筛选后的数据到新的文件
output_path = 'iris/iris_filter'
filtered_data.to_csv(output_path, index=False, header=False)

print(f"Filtered data has been saved to: {output_path}")