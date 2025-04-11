import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 加载预测结果
prediction_results = pd.read_excel('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/prediction_results.xlsx')

# 打印预测结果的基本信息
print("预测结果基本信息:")
print(prediction_results.info())

# 查看预测的分布
print("\n预测续保的分布:")
print(prediction_results['预测_续保'].value_counts())
print(f"续保率: {prediction_results['预测_续保'].mean() * 100:.2f}%")

# 查看预测概率的分布
plt.figure(figsize=(10, 6))
plt.hist(prediction_results['预测_续保概率'], bins=20)
plt.title('预测续保概率分布')
plt.xlabel('续保概率')
plt.ylabel('客户数量')
plt.axvline(x=0.5, color='r', linestyle='--')
plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/prediction_probability_distribution.png')
plt.show()

# 分析哪些特征与续保预测相关
print("\n各特征与续保预测的关联分析:")

# 年龄与续保的关系
plt.figure(figsize=(10, 6))
plt.scatter(prediction_results['age'], prediction_results['预测_续保概率'], alpha=0.5)
plt.title('年龄与续保概率的关系')
plt.xlabel('年龄')
plt.ylabel('续保概率')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/age_vs_renewal_probability.png')
plt.show()

# 婚姻状态与续保的关系
if 'marital_status' in prediction_results.columns:
    plt.figure(figsize=(10, 6))
    renewal_by_marital = prediction_results.groupby('marital_status')['预测_续保'].mean().sort_values()
    renewal_by_marital.plot(kind='bar')
    plt.title('婚姻状态与续保率的关系')
    plt.xlabel('婚姻状态')
    plt.ylabel('续保率')
    plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/marital_status_vs_renewal_rate.png')
    plt.show()

# 职业与续保的关系
if 'occupation' in prediction_results.columns:
    plt.figure(figsize=(12, 6))
    renewal_by_occupation = prediction_results.groupby('occupation')['预测_续保'].mean().sort_values()
    renewal_by_occupation.plot(kind='bar')
    plt.title('职业与续保率的关系')
    plt.xlabel('职业')
    plt.ylabel('续保率')
    plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/occupation_vs_renewal_rate.png')
    plt.show()

# 查看高续保概率和低续保概率的客户特征
high_prob = prediction_results[prediction_results['预测_续保概率'] > 0.8]
low_prob = prediction_results[prediction_results['预测_续保概率'] < 0.2]

print("\n高续保概率(>0.8)客户数量:", len(high_prob))
print("低续保概率(<0.2)客户数量:", len(low_prob))

# 分析高续保概率客户的特征
if len(high_prob) > 0:
    print("\n高续保概率客户的特征:")
    
    # 年龄分布
    print("\n年龄分布:")
    print(high_prob['age'].describe())
    
    # 婚姻状态分布
    if 'marital_status' in high_prob.columns:
        print("\n婚姻状态分布:")
        print(high_prob['marital_status'].value_counts(normalize=True) * 100)
    
    # 职业分布
    if 'occupation' in high_prob.columns:
        print("\n职业分布:")
        print(high_prob['occupation'].value_counts(normalize=True) * 100)

# 分析低续保概率客户的特征
if len(low_prob) > 0:
    print("\n低续保概率客户的特征:")
    
    # 年龄分布
    print("\n年龄分布:")
    print(low_prob['age'].describe())
    
    # 婚姻状态分布
    if 'marital_status' in low_prob.columns:
        print("\n婚姻状态分布:")
        print(low_prob['marital_status'].value_counts(normalize=True) * 100)
    
    # 职业分布
    if 'occupation' in low_prob.columns:
        print("\n职业分布:")
        print(low_prob['occupation'].value_counts(normalize=True) * 100)

# 手动标注功能
print("\n由于测试数据集中没有实际的续保标签，您可以手动标注一些样本进行评估")
print("请在prediction_results.xlsx文件中添加一列'手动标注_续保'，填入实际的续保情况（1表示续保，0表示不续保）")
print("然后重新运行此脚本进行评估")

# 检查是否已有手动标注
if '手动标注_续保' in prediction_results.columns:
    print("\n检测到手动标注数据，进行模型评估...")
    y_true = prediction_results['手动标注_续保']
    y_pred = prediction_results['预测_续保']
    
    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n模型准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_true, y_pred))
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('模型在手动标注数据上的混淆矩阵')
    plt.colorbar()
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.xticks([0, 1], ['不续保', '续保'])
    plt.yticks([0, 1], ['不续保', '续保'])
    
    # 在混淆矩阵中显示数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.savefig('E:/知乎-AIGC-工程师/1.主干课/@4.10/3.作业/CASE-客户续保预测/visualizations/manual_confusion_matrix.png')
    plt.show()
else:
    print("\n未检测到手动标注数据，无法计算准确率")

print("\n分析完成!")