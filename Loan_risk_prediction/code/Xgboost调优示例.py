#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Xgboost调优示例.py
@time: 2018/2/24 17:09
@desc:
"""
'''
   Xgboost 调参详解
   参考博客: http://blog.csdn.net/han_xiaoyang/article/details/52665396
   
   GridSearchCV 博客:
   http://blog.csdn.net/cherdw/article/details/54970366
'''
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt

from matplotlib.pylab import rcParams
from xgboost import XGBClassifier
from sklearn import svm, grid_search, datasets
from sklearn.model_selection import GridSearchCV  # 网格搜索

# %matplotlib inline
rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('E:\py_workspace\TIANCHI_Project\Loan_risk_prediction\data\Train_nyOWmfK.csv')
target = 'Disbursed'
IDcol = 'ID'

# 先定义一个函数，帮助我们建立XGBoost models 并进行交叉验证
def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=0.5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain)

#
# 第一步: 确定学习速率和tree_based参数  调优的估计器的数目
#
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
    learning_rate= 0.1,
    n_estimators= 1000,
    max_depth= 5,          # 树的最大深度。这个值也是用来避免过拟合的。max_depth越大。模型会学到更具体更局部的样本。
                           # 需要使用cv函数来进行调优。
                           # 典型值 3-10
    min_child_weight= 1,   # 决定最小叶子节点样本权重和。这个参数用于避免过拟合，当他的值较大时，可以避免
                           # 模型学习到局部的特殊样本
                           # 但是如果这个值过高，会导致欠拟合。这个参数需要使用cv来调整
                           # 默认是 1
    gamma= 0,              # 在节点分裂时，只有分裂后损失函数的值下降，才会分裂这个节点。
                           # gamma指定了节点分裂所需的最小损失函数下降值
                           # 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关。
    subsample= 0.8,        # 这个参数控制对于每棵树随机采样的比例
                           # 减少这个参数的值，算法会更加保守，避免过拟合。
                           # 但是，如果这个值设置的越小，他可能会导致欠拟合。
                           # 典型值 0.5-1
    colsample_bytree= 0.8, # 用来控制树的每一级的每一次分裂，对列数的采样的占比
                           # subsample和colsample_bytree可以起到相同的作用。
    objective= 'binary:logistic', # 学习目标参数 二分类的逻辑回归，返回预测的概率 (不是类别)
           #   ‘multi:softmax’      使用softmax的多分类器，返回预测的类别 (不是概率)
                                  # 这种情况下需要多加一个参数 : num_class （类别数目）
    nthread= 4,
    scale_pos_weight= 1,   # 默认为1
                           # 在各类样本十分不平衡时，把这个参数设置为一个正值，可以使算法更快收敛
    seed= 27               # 随机数种子
                           # 设置他可以复现随机数据的结果，也可以用于调整参数
)


# 第二部:max_depth 和 min_weight 参数调优
# grid_search 参考:
# http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html
# http://blog.csdn.net/abcjennifer/article/details/23884761

# 网格搜索scoring=’roc_auc’只支持二分类，多分类需要修改scoring(默认支持多分类)
param_test1 = {
    'max_depth':range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

param_test2 = {
    'max_depth':[4,5,6],
    'min_child_weoght':[4,5,6]
}
# Deprecated since version 0.18: This module will be removed in 0.20.
# Use sklearn.model_selection.GridSearchCV instead.
# GridSearchCV 他存在的意义就是自动调参，只要把参数传进去，就能给出最优化的结果和参数，但是这个方法只适合小数据集。
# 一旦数据量上去了，很难得出结果。
# 数据量大的时候可以使用一个快速调优的方法-坐标下降，它其实是一种贪心算法，拿当前对模型影响最大的参数调优，直到最优化
# 再拿下一个影响最大的参数调优，如此下去，直到所有的参数调整完毕。
# 这个方法的缺点就是可能会调到局部最优而不是全局最优，但是省时省力
# 后续可用bagging优化
gsearch1 = GridSearchCV(
    estimator= XGBClassifier(       # 确定所使用的分类器，每一个分类器都需要一个score参数，或者score方法
        learning_rate=0.1,
        n_estimators=140,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27
    ),
    param_grid=param_test1,     # 值为字典或者列表，即需要优化的参数的值
    scoring='roc_auc',          # 准确度评价标准，默认为None，这时需要使用score函数
    n_jobs=4,                   # 并行数， int: 个数 -1 跟cpu核数一致， 1 默认值
    iid=False,                  # 默认为True，为True时，默认为各个样本fold概率分布一致，误差估计为所有样本之和，而非各个fold的平均
    cv=5,                       # 交叉验证参数，默认为None
    verbose= 2,                 # 日志冗长度，int, 0:不输出训练过程， 1: 偶尔输出， >1:对每个子模型都输出
    refit=True                  # 默认为True，程序将会以交叉验证训练集得到的最佳参数，重新对所有可用的训练集与开发集进行，
                                # 作为最终用于性能评估的最佳模型参数。即在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集。
)
gsearch1.fit(train[predictors],train[target])
gsearch1.grid_scores_,
gsearch1.best_params_,
gsearch1.best_score_





if __name__ == '__main__':
    pass