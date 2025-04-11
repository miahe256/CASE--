import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from matplotlib import font_manager

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 请根据你的系统调整字体路径
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 先训练模型并保存，因为我们之前没有保存模型
def train_and_save_model():
    print("训练和保存模型...")
    # 读取原始训练数据
    train_data = pd.read_excel('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/policy_data.xlsx')
    
    # 准备输入特征和目标变量
    train_data['renewal'] = train_data['renewal'].map({'Yes': 1, 'No': 0})
    
    # 选择输入特征
    X = train_data.drop('renewal', axis=1)
    y = train_data['renewal']
    
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
    
    # 保存模型
    joblib.dump(model, 'E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/decision_tree_model.joblib')
    
    return model, X.columns, list(X.dtypes)

# 加载模型（如果存在）或训练新模型
try:
    print("尝试加载已有模型...")
    model = joblib.load('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/decision_tree_model.joblib')
    
    # 因为我们需要原始数据的列信息，所以还是要加载原始数据
    train_data = pd.read_excel('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/policy_data.xlsx')
    X_columns = train_data.drop('renewal', axis=1).columns
    X_dtypes = list(train_data.drop('renewal', axis=1).dtypes)
    
    print("模型加载成功!")
except:
    print("未找到已保存的模型，训练新模型...")
    model, X_columns, X_dtypes = train_and_save_model()
    print("新模型训练完成!")

# 读取测试数据集
print("加载测试数据集...")
test_data = pd.read_excel('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/policy_test.xlsx')

# 查看测试数据结构
print("\n测试数据基本信息:")
print(test_data.info())

# 检查测试数据是否包含目标变量
if 'renewal' in test_data.columns:
    has_target = True
    print("\n测试数据包含目标变量'renewal'，将进行模型评估")
    
    # 将目标变量转换为数值
    if test_data['renewal'].dtype == 'object':
        test_data['renewal'] = test_data['renewal'].map({'Yes': 1, 'No': 0})
    
    y_test = test_data['renewal']
    X_test = test_data.drop('renewal', axis=1)
else:
    has_target = False
    print("\n测试数据不包含目标变量'renewal'，将只进行预测")
    X_test = test_data

# 预处理测试数据（确保数据格式与训练数据一致）
print("\n预处理测试数据...")
# 移除日期列
if 'policy_start_date' in X_test.columns:
    X_test = X_test.drop('policy_start_date', axis=1)
if 'policy_end_date' in X_test.columns:
    X_test = X_test.drop('policy_end_date', axis=1)

# 进行预测
print("\n进行预测...")
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# 如果测试数据包含目标变量，评估模型性能
if has_target:
    print("\n模型评估结果:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('决策树模型在测试集上的混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/test_confusion_matrix.png')
    plt.show()

# 创建结果数据框
results = X_test.copy()
results['预测_续保概率'] = y_pred_proba[:, 1]
results['预测_续保'] = y_pred
if has_target:
    results['实际_续保'] = y_test
    results['预测正确'] = y_test == y_pred

# 保存预测结果
results.to_excel('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/prediction_results.xlsx', index=False)
print("\n预测结果已保存到 'prediction_results.xlsx'")

# 分析预测错误的案例（如果有目标变量）
if has_target:
    print("\n分析预测错误的案例:")
    wrong_predictions = results[results['预测正确'] == False]
    print(f"预测错误的案例数: {len(wrong_predictions)}")
    
    if len(wrong_predictions) > 0:
        # 分析错误预测的类型分布
        print("\n错误预测的类型分布:")
        error_types = pd.DataFrame({
            '实际值': wrong_predictions['实际_续保'],
            '预测值': wrong_predictions['预测_续保']
        })
        print(pd.crosstab(error_types['实际值'], error_types['预测值']))
        
        # 分析主要特征的分布
        print("\n错误预测中的关键特征分析:")
        
        # 年龄分析
        if 'age' in wrong_predictions.columns:
            print("\n年龄分布:")
            print(wrong_predictions['age'].describe())
        
        # 婚姻状态分析
        if 'marital_status' in wrong_predictions.columns:
            print("\n婚姻状态分布:")
            print(wrong_predictions['marital_status'].value_counts())
        
        # 职业分析
        if 'occupation' in wrong_predictions.columns:
            print("\n职业分布:")
            print(wrong_predictions['occupation'].value_counts())
        
        # 理赔历史分析
        if 'claim_history' in wrong_predictions.columns:
            print("\n理赔历史分布:")
            print(wrong_predictions['claim_history'].value_counts())

print("\n模型评估完成!")