import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import font_manager
import os

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 请根据你的系统调整字体路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取Excel文件
file_path = 'policy_data.xlsx'
data = pd.read_excel(file_path)

# 准备输入特征和目标变量
# 假设renewal是二分类目标（Yes/No）
data['renewal'] = data['renewal'].map({'Yes': 1, 'No': 0})

# 选择输入特征
X = data.drop('renewal', axis=1)
y = data['renewal']

# 区分数值型特征和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 移除不适合作为特征的列，如id、日期等
if 'policy_id' in numeric_features:
    numeric_features.remove('policy_id')
if 'policy_start_date' in X.columns:
    X = X.drop('policy_start_date', axis=1)
if 'policy_end_date' in X.columns:
    X = X.drop('policy_end_date', axis=1)

# 数据预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 创建模型管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
print(f'准确率: {accuracy_score(y_test, y_pred):.4f}')
print('\n分类报告:')
print(classification_report(y_test, y_pred))
print('\n混淆矩阵:')
print(confusion_matrix(y_test, y_pred))

# 获取特征名称
# 获取转换后的特征名称
feature_names = []
feature_names.extend(numeric_features)

# 为分类特征创建名称
cat_features = []
for cat_feat in categorical_features:
    unique_values = data[cat_feat].unique()
    for value in unique_values[1:]:  # 跳过第一个类别（因为我们使用了drop='first'）
        cat_features.append(f"{cat_feat}_{value}")

feature_names.extend(cat_features)

# 获取系数
coefficients = model.named_steps['classifier'].coef_[0]

# 如果特征名称和系数长度不匹配（通常会发生这种情况），我们使用索引
if len(feature_names) != len(coefficients):
    print(f"警告：特征名称数量({len(feature_names)})与系数数量({len(coefficients)})不匹配。将使用索引代替。")
    feature_names = [f"特征 {i}" for i in range(len(coefficients))]

# 创建系数数据框
coef_df = pd.DataFrame({'特征': feature_names[:len(coefficients)], '系数': coefficients})

# 按系数绝对值排序
coef_df = coef_df.sort_values(by='系数', key=abs, ascending=False)

# 根据系数的正负设置颜色
colors = ['red' if c < 0 else 'green' for c in coef_df['系数']]

# 创建visualizations文件夹
visualizations_path = 'visualizations'
os.makedirs(visualizations_path, exist_ok=True)

# 可视化系数（只展示前20个最重要的特征）
top_n = min(20, len(coef_df))
plt.figure(figsize=(10, 12))
bars = plt.barh(coef_df['特征'][:top_n], coef_df['系数'][:top_n], color=colors[:top_n])
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.title('逻辑回归系数（绿色为正向影响，红色为负向影响）')
plt.xlabel('系数值')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_path, 'logistic_regression_coefficients.png'))
plt.show()

# 打印并保存逻辑回归模型的结果
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印结果
print(f'准确率: {accuracy:.4f}')
print('\n分类报告:')
print(classification_rep)
print('\n混淆矩阵:')
print(conf_matrix)

# 输出特征重要性排名
print('\n特征重要性排名（按系数绝对值）:')
print(coef_df[['特征', '系数']].head(top_n))

# 生成解释文档
explanation_path = '逻辑回归解释.md'
with open(explanation_path, 'w', encoding='utf-8') as f:
    f.write('# 逻辑回归模型结果解释\n\n')
    f.write(f'## 准确率\n准确率为: {accuracy:.4f}\n\n')
    f.write('## 分类报告\n')
    f.write(classification_rep + '\n')
    f.write('## 混淆矩阵\n')
    f.write(str(conf_matrix) + '\n\n')
    f.write('## 特征重要性排名（按系数绝对值）\n')
    f.write(coef_df[['特征', '系数']].head(top_n).to_string(index=False) + '\n\n')
    f.write('### 解释\n')
    f.write('1. **准确率**：模型在测试集上的预测准确率。\n')
    f.write('2. **分类报告**：包括精确率、召回率和F1分数等指标。\n')
    f.write('3. **混淆矩阵**：显示预测结果的分布情况。\n')
    f.write('4. **特征重要性**：根据逻辑回归系数的绝对值，评估特征对模型的影响。\n')