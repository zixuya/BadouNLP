import pandas as pd

# 示例数据：包含字典的列表
data = [
    {"original_name": "Alice", "original_age": 30, "original_city": "New York"},
    {"original_name": "Bob", "original_age": 25, "original_city": "Los Angeles"},
    {"original_name": "Charlie", "original_age": 35, "original_city": "Chicago"}
]

# 指定列的顺序和别名
column_aliases = ['Name', 'Age', 'City']

# 使用Pandas的DataFrame构造函数创建DataFrame对象，并指定列的顺序和别名
df = pd.DataFrame(data, columns=['original_name', 'original_age', 'original_city'])
df.columns = column_aliases  # 为列指定别名

# 指定输出的CSV文件名
csv_filename = 'output_with_aliases.csv'

# 使用to_csv()方法将DataFrame导出为CSV文件
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')