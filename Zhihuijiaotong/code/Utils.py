#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Utils.py
@time: 2018/2/27 14:19
@desc: 智慧交通数据处理工具类
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bucket_data(lines):
    bucket = {}
    for line in lines:
        time_series = line[-2]
        bucket[time_series] = []
    for line in lines:
        time_series, y1 = line[-2:]
        line = np.delete(line, -2, axis=0)
        bucket[time_series].append(line)
    return bucket

def cross_valid(regressor, bucket, lagging):
    '''
     交叉验证
    :param regressor:
    :param bucket:
    :param lagging:
    :return:
    '''
    valid_loss = []
    last = [[] for i in range(len(bucket[bucket.keys()[0]]))]
    for time_series in sorted(bucket.keys(), key=float):
        if time_series >= 120:
            if int(time_series) in range(120, 120 + lagging, 2):
                last = np.concatenate((last, np.array(bucket[time_series], dtype=float)[:, -1].reshape(-1, 1)), axis=1)
            else:
                batch = np.array(bucket[time_series], dtype=float)
                y = batch[:, -1]
                batch = np.delete(batch, -1, axis=1)
                batch = np.concatenate((batch, last), axis=1)
                y_pre = regressor.predict(batch)
                last = np.delete(last, 0, axis=1)
                last = np.concatenate((last, y_pre.reshape(-1, 1)), axis=1)
                loss = np.mean(abs(np.expm1(y) - np.expm1(y_pre)) / np.expm1(y))
                valid_loss.append(loss)
    return np.mean(valid_loss)

def mape_in(y, d):
    '''

    :param y:
    :param d:
    :return:
    '''
    c = d.get_label()
    result = np.sum(np.abs(np.expm1(y) - np.expm1(c)) / np.expm1(c)) / len(c)
    return 'mape', result

def feature_vis(regressor, train_feature):
    importances = regressor.feature_importances_     #
    indices = np.argsort(importances)[::-1] # 将元祖或列表的内容反转
    selected_features = [train_feature[e] for e in indices]
    plt.figure(figsize=(20, 10))
    plt.title("train_feature importances")
    plt.bar(range(len(train_feature)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(selected_features)), selected_features, rotation=70)
    plt.show()


# --- #
def submission()


