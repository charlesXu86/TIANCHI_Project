#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Gen_user_click_features.py
@time: 2018/3/8 15:35
@desc: 生成用户点击的特征
"""

import os
import pickle
import gc
import pandas as pd
import numpy as np

from tqdm import tqdm
from Ad_Utils import load_pickle, dump_pickle, raw_data_path, feature_data_path
from Feature_joint import addTime, addAd, addPosition, addAppCategories, addUserInfo

def gen_click_stats(data, col):
    click_user_day = pd.DataFrame(data.groupby(['userID', col])['clickTime'].count())
    click_user_day.rename(columns={'clickTime': col + '_m'}, inplace=True)
    click_user_day.reset_index(inplace=True)
    click_user_day_m = pd.DataFrame(click_user_day.groupby(['userID'])[col + '_m'].mean()).rename(columns={col+'_m':col+'_mean'}).reset_index()
    clicks_user_day_ma = pd.DataFrame(click_user_day.groupby(['userID'])[col + '_m'].max()).rename(columns={col + '_m': col + '_max'}).reset_index()
    clicks_user_day_mi = pd.DataFrame(click_user_day.groupby(['userID'])[col + '_m'].min()).rename(columns={col + '_m': col + '_min'}).reset_index()
    stats_columns = [col + '_max', col+ '_mean', col+'_min']
    data = pd.merge(data, click_user_day_m, how='left', on='userID')
    data = pd.merge(data, clicks_user_day_ma, how='left', on='userID')
    data = pd.merge(data, clicks_user_day_mi, how='left', on='userID')
    return data

def gen_user_day_click():
    feature_path = feature_data_path + 'user_day_clicks.pkl'
    if os.path.exists(feature_path):
        print('Found' + feature_path)
    else:
        print('Generating' + feature_path)
        train = load_pickle(raw_data_path + 'train.pkl')
        test = load_pickle(raw_data_path + 'test.pkl')
        all_data = train.append(test)
        all_data = addTime(all_data)
        user_click_day = pd.DataFrame(all_data.groupby(['clickDay', 'userID']).size()).reset_index().rename(columns={0:'user_click_day'})
        dump_pickle(user_click_day, feature_path)

def gen_user_day_click_count(update=False):
    '''
     生成所有数据的每天点击统计量
     拼接键['ID_name', 'clickDay']
    :param update:
    :return:
    '''
    train = load_pickle(raw_data_path + 'train.pkl')
    test = load_pickle(raw_data_path + 'test.pkl')
    data = train.append(test)
    data = addTime(data)
    data = addAd(data)
    data = addAppCategories(data)
    data = addPosition(data)

    ads_feature = ['advertiserID', 'camgaignID', 'adID', 'creativeID', 'appID', 'appCategory', ]
    context_feature = ['positionID', 'sitesetID', ]
    stats_feature = ads_feature + context_feature

    for feature in tqdm(stats_feature):
        feature_path = feature_data_path + 'user_' + feature + '_click_day.pkl'
        if os.path.exists(feature_path) and update==False:
            print('Found' + feature_path)
        else:
            print('Generating' + feature_path)
            user_feature_click_day = data.groupby(['userID','clickDay',feature]).size().reset_index().rename(columns={0:'user_'+feature+'_click_day'})
            dump_pickle(user_feature_click_day, feature_path)

def add_user_day_hour_account(data,feature_list=['positionID','advertiserID','camgaignID','adID','creativeID','appID']):
    '''
        data必须包含clickHour等字段，可以通过addTime，addAD,addPostion,addAppCategories添加
    :param data:
    :param feature_list:
    :return:
    '''
    ads_feature = ['advertiserID','camgaignID','adID','creativeID','appID']
    context_feature = ['positionID', 'sitesetID']
    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_' + feature + '_click_hour.pkl'
        feature_day_click = load_pickle(feature_path)
        data = pd.merge(data, feature_day_click, 'left', [feature, 'clickDay', 'clickHour', 'userID'])
    return data

def gen_user_hour_click_count(update=False):
    '''
     生成所有数据的每天没小时点击统计量
     拼接键['ID_name', 'clickDay', 'clickHour']
    :param update:
    :return:
    '''
    train = load_pickle(raw_data_path + 'train.pkl')
    test = load_pickle(raw_data_path + 'test.pkl')
    data = train.append(test)
    data = addTime(data)
    data = addAd(data)
    data = addPosition(data)
    data = addAppCategories(data)

    ads_feature = ['advertiserID','camgaignID','adID','creativeID','appID','appCategory']
    context_feature = ['positionID', 'sitesetID']
    state_feature = ads_feature + context_feature

    for feature in tqdm(state_feature):
        feature_path = feature_data_path + 'user_' + feature + '_click_hour.pkl'
        if os.path.exists(feature_path):
            print('Found' + feature_path)
        else:
            print('Generation' + feature_path)
            user_feature_click_day = data.groupby(['userID','clickDay','clickHour',feature]).size().reset_index().rename(columns={0:'user_'+feature+'_click_hour'})
            dump_pickle(user_feature_click_day, feature_path)

def add_user_day_click(data):
    '''
     添加用户当天的点击总数
    :param data:
    :return:
    '''
    feature_path = feature_data_path + 'user_day_click.pkl'
    if not os.path.exists(feature_path):
        gen_user_day_click()
    user_click_day = load_pickle(feature_path)
    data = pd.merge(data, user_click_day, 'left', ['clickDay', 'userID'])
    return data

def add_user_day_click_count(data, feature_list=['positionID','advertiserID','camgaignID','adID','creativeID','appID','sitesetID']):
    '''
    当天点击数量统计
    :param data:
    :param feature_list:
    :return:
    '''
    ads_feature = ['advertiserID','camgaignID','adID','creativeID','appID']
    context_feature = ['positionID','sitesetID']
    stats_feature = ads_feature + context_feature
    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_' + feature + '_click_day.pkl'
        feature_day_click = load_pickle(feature_path)
        data = pd.merge(data, feature_day_click, 'left', [feature, 'clickDay', 'userID'])
    return data

def generate_stats_feature():
    '''
    输入train和test，进行concat后，添加用户点击数据的统计特征
    :return:
    '''
    feature_path = feature_data_path + 'UserClickStats.pkl'
    if os.path.exists(feature_path):
        print('Found', feature_path)
    else:
        train = load_pickle(raw_data_path + 'train.pkl')
        test = load_pickle(raw_data_path + 'test.pkl')
        data = train.append(test)
        del train, test
        gc.collect()
        data = addTime(data)
        data = addAd(data)
        data = addPosition(data)
        data = addAppCategories(data)
        data = add_user_day_click(data)
        data = add_user_day_click_count(data, feature_list=['camgaignID', 'adID', 'appID', 'sitesetID'])
        # data = add_user_day_hour_count(data)
        # train_origin_features = train.columns.values.tolist()
        # test_origin_features = test.columns.values.tolist()

        feature_names = [
            'user_adID_click_day_mean',  # 有些统计特征没包括进来
            'user_adID_click_day_min',
            'user_camgaignID_click_day_min',
            'user_appID_click_day_mean',
            'user_appID_click_day_max',
            'user_appID_click_day_min',
            'user_sitesetID_click_day_mean',
            'user_sitesetID_click_day_max',
            'user_sitesetID_click_day_min',
            'user_click_day_mean',
            'user_click_day_max',
            'user_click_day_min'
        ]

        print('Generating', feature_path)
        columns_day = ['user_adID_click_day','user_camgaignID_click_day','user_appID_click_day','user_sitesetID_click_day',
                   'user_click_day']
        columns_hour = ['user_adID_click_hour','user_camgaignID_click_hour','user_appID_click_hour',
                        'user_sitesetID_click_hour']
        sub_feature = ['userID', 'clickTime']
        # data = pd.concat([train[sub_feature+columns_day+columns_hour],test[sub_feature+columns_day+columns_hour]])
        for col in tqdm(columns_day):
            data = gen_click_stats(data, col)
        # for col in tqdm(columns_day):
        #     data = add
        data = data[feature_names + ['userID']].drop_duplicates(['userID'])
        dump_pickle(data, feature_path)

def add_user_click_stats(data):
    train_click_stats = load_pickle(feature_data_path + 'UserClickStats.pkl')
    data = pd.merge(data, train_click_stats, on='userID', how='left')
    return data

if __name__ == '__main__':
    gen_user_day_click()
    gen_user_day_click_count()
    gen_user_hour_click_count()
    generate_stats_feature()
    print('All Done')