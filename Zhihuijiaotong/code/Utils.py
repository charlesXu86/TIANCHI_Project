#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Ad_Utils.py
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
def submission(train_feature, regressor, df, file1, file2, file3, file4):
    test_df = df.loc[((df['time_interval_begin'].dt.year == 2017) & (df['time_interval_begin'].dt.month == 7)
                      & (df['time_interval_begin'].dt.hour.isin([7, 14, 17])) & (df['time_interval_begin'].dt.minite == 58))].copy()
    test_df['lagging5'] = test_df['lagging4']
    test_df['lagging4'] = test_df['lagging3']
    test_df['lagging3'] = test_df['lagging2']
    test_df['lagging2'] = test_df['lagging1']
    test_df['lagging1'] = test_df['travel_time']

    with open(file1, 'w'):
        pass
    with open(file2, 'w'):
        pass
    with open(file3, 'w'):
        pass
    with open(file4, 'w'):
        pass

    for i in range(30):
        test_X = test_df[train_feature]
        y_prediction = regressor.predict(test_X.values)

        test_df['lagging5'] = test_df['lagging4']
        test_df['lagging4'] = test_df['lagging3']
        test_df['lagging3'] = test_df['lagging2']
        test_df['lagging2'] = test_df['lagging1']
        test_df['lagging1'] = y_prediction

        test_df['predicted'] = np.expm1(y_prediction)
        test_df['time_interval_begin'] = test_df['time_interval_begin'] + pd.DateOffset(minutes=2)
        test_df['time_interval'] = test_df['time_interval_begin'].map(
            lambda x: '[' + str(x) + ',' + str(x + pd.DateOffset(minutes=2)) + ')')
        test_df.time_interval = test_df.time_interval.astype(object)
        if i < 7:
            test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file1, mode='a', header=False,
                                                                              index=False,
                                                                              sep=';')
        elif (7 <= i) and (i < 14):
            test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file2, mode='a', header=False,
                                                                              index=False,
                                                                              sep=';')
        elif (14 <= i) and (i < 22):
            test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file3, mode='a', header=False,
                                                                              index=False,
                                                                              sep=';')
        else:
            test_df[['link_ID', 'date', 'time_interval', 'predicted']].to_csv(file4, mode='a', header=False,
                                                                              index=False,
                                                                              sep=';')



