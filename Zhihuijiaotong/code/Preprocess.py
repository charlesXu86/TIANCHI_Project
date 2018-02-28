#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Preprocess.py
@time: 2018/2/27 13:14
@desc: 数据预处理（类型转换，缺失值处理，特征提取）
"""

import xgboost as xgb

from Utils import *

from sklearn.model_selection import train_test_split
from scipy.interpolate import UnivariateSpline  # 一维平滑样条拟合一组给定的数据点
from sklearn import linear_model

pd.options.display.float_format = '{:, .2f}'.format()
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def cast_log_outliers(to_file):
    df = pd.read_csv('E:\dataset\data\quaterfinal_gy_cmp_training_traveltime.txt', delimiter=';', dtype={'link_ID':object})
    df['time_interval_begin'] = pd.to_datetime(df['time_interval'].map(lambda x: x[1:20]))

    df2 = pd.read_csv('E:\dataset\data\gy_contest_traveltime_training_data_second.txt', delimiter=';', dtype={'linkID':object})
    df2 = df2.rename(columns={'linkID': 'link_ID'})
    df2['time_interval_begin'] = pd.to_datetime(df2['time_interval'].map(lambda x: x[1:20]))
    df2 = df2.loc[(df2['time_interval_begin'] >= pd.to_datetime('2017-03-01'))
                    & (df2['time_interval_begin'] <= pd.to_datetime('2017-03-31'))]

    df = pd.concat([df, df2])
    df = df.drop(['time_interval'], axis=1)
    df['travel_time'] = np.log1p(df['travel_time'])  # log1p   log(1 + x)

    def quantile_clip(group):
        # group.plt()
        group[group < group.quantile(.05)] = group.quantile(.05)   # 分位数
        group[group > group.quantile(.05)] = group.quantile(.95)
        return group

    df['travel_time'] = df.groupby(['link_ID', 'date'])['travel_time'].transform(quantile_clip)
    df = df.loc[(df['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]

    print(df.count())
    df.to_csv(to_file, header=True, index=None, sep=';', mode='W')

def imputation_prepre(file, to_file):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID':object})








if __name__ == '__main__':
    pass