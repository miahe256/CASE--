# 客户续保预测分析项目

本项目使用机器学习方法对保险客户的续保行为进行预测和分析，旨在帮助保险公司识别潜在的流失客户，并提供针对性的客户维护策略。

## 项目内容

本项目包含以下主要组件：

1. **数据探索分析 (EDA)**：对客户数据进行可视化和统计分析
2. **逻辑回归模型**：构建逻辑回归模型预测客户续保行为
3. **决策树模型**：构建决策树模型预测客户续保行为并提供可解释的规则

## 使用指南

### 环境准备

在使用本项目前，请确保安装了以下Python库：

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 数据要求

本项目需要使用`policy_data.xlsx`文件，该文件应包含以下字段：
- `policy_id`：保单ID
- `age`：客户年龄
- `gender`：客户性别
- `birth_region`：出生地区
- `insurance_region`：投保地区
- `income_level`：收入水平
- `education_level`：教育水平
- `occupation`：职业
- `marital_status`：婚姻状态
- `family_members`：家庭成员数量
- `policy_type`：保单类型
- `policy_term`：保险期限
- `premium_amount`：保费金额
- `policy_start_date`：保单开始日期
- `policy_end_date`：保单结束日期
- `claim_history`：理赔历史
- `renewal`：续保状态（Yes/No）

### 使用步骤

#### 1. 数据探索分析

运行`eda_analysis.py`文件进行数据探索分析：

```bash
python eda_analysis.py
```

该脚本将生成以下结果：
- 数据概览信息
- 缺失值分析
- 数值型数据的分布图
- 类别型数据的分布图
- 相关性分析图

所有可视化结果将保存在`visualizations`文件夹中。

#### 2. 逻辑回归模型

运行`logistic_regression_model.py`文件构建逻辑回归模型：

```bash
python logistic_regression_model.py
```

该脚本将：
- 构建逻辑回归模型预测客户续保行为
- 输出模型性能评估（准确率、精确率、召回率等）
- 可视化模型系数，展示各特征对续保的影响
- 生成逻辑回归模型的解释文档`逻辑回归解释.md`


#### 3. 决策树模型

运行`decision_tree_model.py`文件构建决策树模型：

```bash
python decision_tree_model.py
```

该脚本将：
- 构建决策树模型（最大深度为3）预测客户续保行为
- 输出模型性能评估
- 可视化决策树结构
- 生成决策树规则文本`decision_tree_rules.txt`
- 提供易于理解的中文决策规则

### 解释文档的生成

项目中提供了两个模型的解释文档：

1. **逻辑回归解释文档**
   
   逻辑回归模型脚本会自动生成`逻辑回归解释.md`文件，该文件包含：
   - 输出模型性能评估（准确率、精确率、召回率等）
   - 可视化模型系数，展示各特征对续保的影响

   - 提示：你可以将解释文档内容放在AI对话大模型中，进行分析。（提示词：这是Cursor建立逻辑回归模型输出的，顾客是否续保的逻辑回归结果。请你将模型性能，以及特征重要性（最重要的正向因素，重要的负向因素，解释逻辑回归的特征，告诉我哪些人愿意续保，哪些人不需要续保），并提供业务优化建议）

2. **决策树解释文档**
   
   决策树模型会生成`decision_tree_rules.txt`文件，但要生成详细的解释文档，需要运行以下命令：

   ```bash
   python decision_tree_model.py
   ```

   然后，程序会输出决策树的易读规则，并生成`决策树解释.md`文件。该文件包含：
   - 决策树规则图
   - 客户续保意愿分析
   - 关键影响因素分析

   - 提示：你可以将解释文档内容放在AI对话大模型中，进行分析。（提示词：这是Cursor建立决策树模型输出的，顾客似乎否续保的决策树结果。请你将模型性能，以及特征重要性告诉我，并提供业务优化建议。）

### 自定义分析

如果你想使用这些模型进行自定义分析，可以按照以下步骤操作：

#### 针对新数据进行预测

可以修改脚本中的数据路径，加载新的数据文件：

```python
# 读取新的Excel文件
file_path = '你的新数据文件.xlsx'
data = pd.read_excel(file_path)
```

#### 调整模型参数

可以根据需要调整模型参数：

**逻辑回归模型**：
```python
# 修改逻辑回归参数
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
])
```

**决策树模型**：
```python
# 修改决策树参数
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
])
```

#### 创建新的解释文档

可以根据自己的分析结果，创建新的解释文档：

```python
# 生成新的解释文档
with open('新的解释文档.md', 'w', encoding='utf-8') as f:
    f.write("# 模型分析结果\n\n")
    f.write("## 1. 模型性能\n\n")
    # 添加你的分析内容
    f.write(f"准确率：{accuracy_score:.4f}\n")
    # ...更多内容
```

## 示例输出

### 逻辑回归模型输出示例

```
准确率: 0.9233

分类报告:
              precision    recall  f1-score   support
           0       0.91      0.68      0.78        60
           1       0.95      0.99      0.97       240
    accuracy                           0.92       300
   macro avg       0.93      0.83      0.88       300
weighted avg       0.94      0.92      0.93       300

混淆矩阵:
[[ 41  19]
 [  4 236]]
```

### 决策树模型输出示例

```
准确率: 0.9067

分类报告:
              precision    recall  f1-score   support
           0       0.78      0.75      0.76        60
           1       0.94      0.95      0.94       240
    accuracy                           0.91       300
   macro avg       0.86      0.85      0.85       300
weighted avg       0.91      0.91      0.91       300

混淆矩阵:
[[ 45  15]
 [ 13 227]]
```

## 注意事项

1. 请确保数据格式正确，尤其是目标变量`renewal`应为分类变量（Yes/No）
2. 逻辑回归模型对数据进行了标准化处理，而决策树模型则不需要
3. 所有可视化结果默认保存在`visualizations`文件夹中
4. 解释文档中的业务建议仅供参考，实际应用中应结合具体业务情况

## 贡献与反馈

如果你对项目有任何改进建议或发现了问题，欢迎提交反馈。