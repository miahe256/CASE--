import pandas as pd

# 读取Excel文件
file_path = 'policy_data.xlsx'
data = pd.read_excel(file_path)

# 获取前5行数据
first_five_rows = data.head(5)

# 打印前5行数据
print(first_five_rows)