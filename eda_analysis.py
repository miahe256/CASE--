import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 请根据你的系统调整字体路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = 'policy_data.xlsx'
data = pd.read_excel(file_path)

# 数据概览
print("数据概览：")
print(data.info())
print("\n数据描述：")
print(data.describe())

# 缺失值分析
print("\n缺失值分析：")
print(data.isnull().sum())

# 数值型数据分布
data.hist(bins=30, figsize=(10, 8))
plt.suptitle('数值型数据分布')
plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/numeric_distribution.png')
plt.show()

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=data)
plt.title('数值型数据箱线图')
plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/boxplot.png')
plt.show()

# 类别型数据分析
print("\n类别型数据分布：")
for column in data.select_dtypes(include=['object']).columns:
    print(f"\n{column}分布：")
    print(data[column].value_counts())
    plt.figure(figsize=(10, 6))
    sns.countplot(y=column, data=data, order=data[column].value_counts().index)
    plt.title(f'{column}分布')
    plt.savefig(f'E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/{column}_distribution.png')
    plt.show()

# 相关性分析（仅数值型数据）
plt.figure(figsize=(10, 8))
numeric_data = data.select_dtypes(include=['number'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性')
plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/correlation_heatmap.png')
plt.show()