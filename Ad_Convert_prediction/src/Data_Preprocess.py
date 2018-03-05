#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Data_Preprocess.py
@time: 2018/3/2 13:15
@desc: 阿里妈妈广告点击转化数据预处理
"""

import pandas as pd
import numpy as np

# 读取数据
test_set = pd.read_csv('E:\dataset\TIANCHI_ad\\test.txt',sep=' ')
train_set = pd.read_csv('E:\dataset\TIANCHI_ad\\train.txt', sep=' ')
# print(test_set.info())
# print(train_set.info())

train_set['dayofweek'] = (train_set['context_timestamp']/(60*60*24)).apply(np.floor) % 7
train_set['hourofday'] = (train_set['context_timestamp']/(60*60)).apply(np.floor)%24
train_set['minofday'] = (train_set['context_timestamp']/(60)).apply(np.floor)%(24*60)


test_set['is_trade'] = -1
test_set['dayofweek'] = (test_set['context_timestamp']/(60*60*24)).apply(np.floor)%7
test_set['hourofday'] = (test_set['context_timestamp']/(60*60)).apply(np.floor)%24
test_set['minofday'] = (test_set['context_timestamp']/(60)).apply(np.floor)%(24*60)

print((train_set['context_timestamp']/(60*60*24)).apply(np.floor).max())




# if __name__ == '__main__':
#     pass