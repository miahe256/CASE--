import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from matplotlib import font_manager
import graphviz
from sklearn import tree
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

# 数据预处理 - 只进行独热编码，不进行归一化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# 创建模型管道
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))
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

# 提取决策树
decision_tree = model.named_steps['classifier']

# 获取特征名称
cat_encoder = model.named_steps['preprocessor'].transformers_[1][1]
cat_features_encoded = []
for i, cat_feature in enumerate(categorical_features):
    cat_values = data[cat_feature].unique().tolist()
    if len(cat_values) > 1:  # 如果有多个类别
        for cat_value in cat_values[1:]:  # 跳过第一个类别（drop='first'）
            cat_features_encoded.append(f"{cat_feature}_{cat_value}")

feature_names = numeric_features + cat_features_encoded

# 创建visualizations文件夹
visualizations_path = 'visualizations'
os.makedirs(visualizations_path, exist_ok=True)

# 创建决策树可视化
plt.figure(figsize=(20, 10))
tree.plot_tree(decision_tree, 
               feature_names=feature_names, 
               class_names=['不续保', '续保'],
               filled=True, 
               fontsize=10)
plt.title('决策树模型（深度=3）')
plt.savefig(os.path.join(visualizations_path, 'decision_tree.png'), dpi=300)
plt.show()

# 输出决策树规则
tree_rules = export_text(decision_tree, feature_names=feature_names)
print("\n决策树规则:")
print(tree_rules)

# 保存决策树规则到文件
with open('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/decision_tree_rules.txt', 'w', encoding='utf-8') as f:
    f.write("决策树规则:\n")
    f.write(tree_rules)

# 创建中文可读版本的决策树规则
print("\n决策树中文规则解析:")
lines = tree_rules.strip().split('\n')
for line in lines:
    if 'class:' in line:
        indentation = line.index('|')
        rule_path = line[:indentation].replace('|', '').replace('-', '')
        prediction = '续保' if 'class: 1' in line else '不续保'
        print(f"{rule_path} 结论: {prediction}")
    else:
        # 处理规则条件行
        if '<=' in line or '>' in line:
            parts = line.split()
            feature_name = ' '.join(parts[:-2])  # 特征名称
            operator = parts[-2]  # 操作符 (<= 或 >)
            value = parts[-1]     # 阈值
            
            # 替换一些特征名称使其更易读
            if 'age' in feature_name:
                print(line.replace('age', '年龄'))
            elif 'marital_status_单身' in feature_name:
                if operator == '<=':
                    print(line.replace('marital_status_单身 <= 0.50', '不是单身'))
                else:
                    print(line.replace('marital_status_单身 > 0.50', '是单身'))
            elif 'occupation_设计师' in feature_name:
                if operator == '<=':
                    print(line.replace('occupation_设计师 <= 0.50', '不是设计师'))
                else:
                    print(line.replace('occupation_设计师 > 0.50', '是设计师'))
            elif 'occupation_经理' in feature_name:
                if operator == '<=':
                    print(line.replace('occupation_经理 <= 0.50', '不是经理'))
                else:
                    print(line.replace('occupation_经理 > 0.50', '是经理'))
            elif 'claim_history_是' in feature_name:
                if operator == '<=':
                    print(line.replace('claim_history_是 <= 0.50', '无理赔历史'))
                else:
                    print(line.replace('claim_history_是 > 0.50', '有理赔历史'))
            else:
                print(line)
        else:
            print(line)

# 提取决策路径并输出易于理解的规则
print("\n\n客户续保决策规则（完整版）:")

# 路径1: 年轻 + 不单身 + 无理赔
print("规则1: 如果客户年龄 <= 29.5岁，且不是单身，且无理赔历史 → 预测【不续保】")

# 路径2: 年轻 + 不单身 + 有理赔
print("规则2: 如果客户年龄 <= 29.5岁，且不是单身，且有理赔历史 → 预测【不续保】")

# 路径3: 年轻 + 单身 + 不是设计师
print("规则3: 如果客户年龄 <= 29.5岁，且是单身，且不是设计师 → 预测【续保】")

# 路径4: 年轻 + 单身 + 是设计师
print("规则4: 如果客户年龄 <= 29.5岁，且是单身，且是设计师 → 预测【不续保】")

# 路径5: 中年 + 不是经理
print("规则5: 如果客户年龄介于 29.5-60.5岁之间，且不是经理 → 预测【续保】")

# 路径6: 中年 + 是经理
print("规则6: 如果客户年龄介于 29.5-60.5岁之间，且是经理 → 预测【续保】")

# 路径7: 老年 + 不单身
print("规则7: 如果客户年龄 > 60.5岁，且不是单身 → 预测【不续保】")

# 路径8: 老年 + 单身
print("规则8: 如果客户年龄 > 60.5岁，且是单身 → 预测【续保】")

# 输出特征重要性
print("\n特征重要性:")
feature_importance = pd.DataFrame({'特征': feature_names, '重要性': decision_tree.feature_importances_})
feature_importance = feature_importance.sort_values('重要性', ascending=False)
print(feature_importance.head(10))

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance.head(10))
plt.title('决策树特征重要性（前10个）')
plt.tight_layout()
plt.savefig(os.path.join(visualizations_path, 'decision_tree_feature_importance.png'))
plt.show()

# 打印并保存决策树模型的结果
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 打印结果
print(f'准确率: {accuracy:.4f}')
print('\n分类报告:')
print(classification_rep)
print('\n混淆矩阵:')
print(conf_matrix)

# 输出决策树规则
print('\n决策树规则:')
print(tree_rules)

# 输出特征重要性
print('\n特征重要性:')
print(feature_importance.head(10))

# 生成解释文档
explanation_path = '决策树解释.md'
with open(explanation_path, 'w', encoding='utf-8') as f:
    f.write('# 决策树模型结果解释\n\n')
    f.write(f'## 准确率\n准确率为: {accuracy:.4f}\n\n')
    f.write('## 分类报告\n')
    f.write(classification_rep + '\n')
    f.write('## 混淆矩阵\n')
    f.write(str(conf_matrix) + '\n\n')
    f.write('## 决策树规则\n')
    f.write(tree_rules + '\n\n')
    f.write('## 特征重要性（前10个）\n')
    f.write(feature_importance.head(10).to_string(index=False) + '\n\n')
    f.write('### 解释\n')
    f.write('1. **准确率**：模型在测试集上的预测准确率。\n')
    f.write('2. **分类报告**：包括精确率、召回率和F1分数等指标。\n')
    f.write('3. **混淆矩阵**：显示预测结果的分布情况。\n')
    f.write('4. **决策树规则**：展示决策树的分支规则。\n')
    f.write('5. **特征重要性**：根据特征对模型的影响进行排序。\n')