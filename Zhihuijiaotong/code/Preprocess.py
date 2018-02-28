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
    link_df = pd.read_csv('E:\dataset\data\gy_contest_link_info.txt', delimiter=';', dtype={'link_ID':object})
    date_range = pd.date_range('2017-03-01 00:00:00', '2017-07-31 23:58:00', freq='2min')
    new_index = pd.MultiIndex.from_product([link_df['link_ID'].unique(), date_range],
                                              names=['link_ID', 'time_interval_begin'])
    new_df = pd.DataFrame(index=new_index).reset_index()
    df2 = pd.merge(new_df, df, on=['link_ID', 'time_interval_begin'], how='left')

    df2 = df2.loc[(df2['time_interval_begin'].dt.hour.isin([6, 7, 8, 13, 14, 15, 16, 17, 18]))]
    df2 = df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 7) &
                  (df2['time_interval_begin'].dt.hour.isin([8, 15, 18])))]
    df2 =df2.loc[~((df2['time_interval_begin'].dt.year == 2017) & (df2['time_interval_begin'].dt.month == 3) &
                   (df2['time_interval_begin'].dt.day == 31))]

    df2['date'] = df2['time_interval_begin'].dt.strftime('%Y-%m-%d')

    print(df2.count())

    df2.to_csv(to_file, header=True, index=None, sep=';', mode='w')

def imputation_with_model(file, to_file):
    '''

    :param file:
    :param to_file:
    :return:
    '''
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID':object})
    print(df.describe())

    link_infos = pd.read_csv('E:\dataset\data\gy_contest_link_info.txt', delimiter=';', dtype={'link_ID': object})
    link_tops = pd.read_csv('E:\dataset\data\gy_contest_link_top.txt', delimiter=';', dtype={'link_ID': object})
    link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on='link_ID', how='left')
    link_infos['area'] = link_infos['length'] * link_infos['width']
    link_infos['links_num'] = link_infos['in_links'].astype('str') + "," + link_infos['out_links'].astype('str')
    df = pd.merge(df, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')

    df.loc[df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df.loc[~df['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    df['hour'] = df['time_interval_begin'].dt.hour
    df['week_day'] = df['time_interval_begin'].map(lambda x: x.weekday() + 1)
    df['month'] = df['time_interval_begin'].dt.month
    df['year'] = df['time_interval_begin'].df.year

    df = pd.get_dummies(df, columns=['vocation', 'links_num', 'hour', 'week_day', 'month', 'year'])     # 将分类变量转换为虚拟/指示符变量

    def mean_time(group):
        group['link_ID_en'] = group['travel_time'].mean()
        return group

    df = df.groupby('link_ID').apply(mean_time)
    sorted_link = np.sort(df['link_ID_en'].unique())
    df['link_ID_en'] = df['link_ID_en'].map(lambda x: np.argmin(x >= sorted_link))

    train_df = df.loc[~df['travel_time'].isnull()]
    test_df = df.loc[df['travel_time'].isnull()].copy()

    feature = df.columns.values.tolist()
    train_feature = [x for x in feature if
                     x not in ['link_ID', 'time_intervel_begin', 'travel_time', 'date']]
    x = train_df[train_feature].values
    y = test_df['travel_time'].values

    print(train_feature)

    params={
        'learning_rate': 0.2,
        'n_estimators': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 7,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'gamma': 0
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(
        learning_rate= params['learning_rate'],
        n_estimators= params['n_estimators'],
        booster= 'gbtree',
        objective= 'reg:linear',
        n_jobs= 1,
        subsample= params['subsample'],
        max_depth= params['max_depth'],
        gamma= params['gamma'],
        min_child_weight= params['min_child_weight'],
        reg_alpha= params['reg_alpha']
    )
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10,    # verbose=True   verbose表示详细信息，verbose=FALSE，意思就是设置运行的时候不显示详细信息。
                  eval_metric=mape_in, eval_set=eval_set)
    feature_vis = (regressor, train_feature)

    test_df['prediction'] = regressor.predict(test_df[train_feature].values)
    df = pd.merge(df, test_df[['link_ID', 'time_interval_begin', 'travel_time', 'imputation1']],
                  on=['link_ID', 'time_interval_begin'], how='left')
    print(df[['travel_time', 'prediction']].describe())

    df['imputation1'] = df['travel_time'].isnull()
    df['travel_time'] = df['travel_time'].fillna(value=df['prediction'])
    df[['link_ID', 'date', 'time_interval_begin', 'travel_time', 'imputation1']].to_csv(to_file, header=True, index=None, sep=';', mode='w')

def create_lagging(df, df_original, i):
    df1 = df_original.copy()
    df1['time_interval_begin'] = df1['time_interval_begin'] + pd.DateOffset(minutes = i * 2)
    df1 = df1.rename(columns={'travel_time': 'lagging' + str(i)})
    df2 = pd.merge(df, df1[['link_ID', 'time_interval_begin', 'lagging' + str(i)]],
                       on=['link_ID', 'time_interval_begin'], how='left')
    return df2

def create_feture(file, to_file, lagging=5):
    df = pd.read_csv(file, delimiter=';', parse_dates=['time_interval_begin'], dtype={'link_ID':object})

    # lagging feature
    df1 = create_lagging(df, df, 1)
    for i in range(2, lagging + 1):
        df1 = create_lagging(df1, df, i)

    # length, width feature
    link_infos = pd.read_csv('E:\dataset\data\gy_contest_link_info.txt', delimiter=';', dtype={'link_ID':object})
    link_tops = pd.read_csv('E:\dataset\data\gy_contest_link_top.txt', delimiter=';', dtype={'link_ID':object})
    link_tops['in_links'] = link_tops['in_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops['out_links'] = link_tops['out_links'].str.len().apply(lambda x: np.floor(x / 19))
    link_tops = link_tops.fillna(0)
    link_infos = pd.merge(link_infos, link_tops, on='link_ID', how='left')
    link_infos['links_num'] = link_infos['in_links'].astype('str') + "," + link_infos['out_links'].astype('str')
    link_infos['area'] = link_infos['length'] * link_infos['width']
    df2 = pd.merge(df1, link_infos[['link_ID', 'length', 'width', 'links_num', 'area']], on=['link_ID'], how='left')
    # df.boxplot(by=['width'], column='travel_time')
    # plt.show()
    # df.boxplot(by=['length'], column='travel_time')
    # plt.show()

    # links_num feature
    df2.loc[df2['links_num'].isin(['0.0,2.0', '2.0,0.0', '1.0,0.0']), 'links_num'] = 'other'
    # df.boxplot(by=['links_num'], column='travel_time')
    # plt.show()

    # vacation feture
    df2.loc[df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 1
    df2.loc[~df2['date'].isin(
        ['2017-04-02', '2017-04-03', '2017-04-04', '2017-04-29', '2017-04-30', '2017-05-01',
         '2017-05-28', '2017-05-29', '2017-05-30']), 'vacation'] = 0

    # minute_series for CV
    df2.loc[df2['time_interval_begin'].dt.hour.isin([6, 7, 8]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 6) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([13, 14, 15]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 13) * 60

    df2.loc[df2['time_interval_begin'].dt.hour.isin([16, 17, 18]), 'minute_series'] = \
        df2['time_interval_begin'].dt.minute + (df2['time_interval_begin'].dt.hour - 16) * 60



if __name__ == '__main__':
    create_feture('data/com_training.txt', 'data/training.txt', lagging=5)