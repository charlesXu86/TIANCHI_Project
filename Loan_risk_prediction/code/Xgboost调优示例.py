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
'''
from xgboost import XGBClassifier

#
# 第一步: 确定学习速率和tree_based参数  调优的估计器的数目
#
xgb1 = XGBClassifier(
    learning_rate= 0.1,
    n_estimators= 1000,
    max_depth= 5,
    min_child_weight= 1,
    gamma= 0,
    subsample= 0.8,

)




if __name__ == '__main__':
    pass