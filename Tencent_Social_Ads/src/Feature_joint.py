#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Feature_joint.py
@time: 2018/3/8 15:41
@desc: 特征拼接
"""

import os
import pickle
import gc
import pandas as pd
import numpy as np

from tqdm import tqdm
from Ad_Utils import load_pickle,dump_pickle,raw_data_path

def addAd(data):
    '''
     拼接原始ad特征
    :param data:
    :return:
    '''
    feature_path = raw_data_path + 'ad.pkl'
    ad_feature = ['adID', 'camgaignID', 'creativeID', 'advertiserID', 'appID', 'appPlatform'] #ad.csv的所有字段
    if os.path.exists(feature_path):
        ad = load_pickle(feature_path)
    else:
        ad = pd.read_csv(raw_data_path + 'ad.csv')
        dump_pickle(ad, feature_path)
    return pd.merge(data, ad[ad_feature], on='creativeID', how='left')

def addPosition(data):
    '''
     拼接原始position特征
    :param data:
    :return:
    '''
    feature_path = raw_data_path + 'position.pkl'
    position_feature = ['positionID', 'sitesetID', 'positionType']
    if os.path.exists(feature_path):
        position = load_pickle(feature_path)
    else:
        position = pd.read_csv(raw_data_path + 'position.csv')
        dump_pickle(position, feature_path)

    return pd.merge(data, position[position_feature], on='positionID', how='left')

def addAppCategories(data):
    '''
     拼接原始app_categories特征
    :param data:
    :return:
    '''
    app = pd.read_csv(raw_data_path + 'app_categories.csv')
    app['cate_a'] = app['appCategory'].apply(lambda x: x//100 if x > 100 else x)
    return pd.merge(data, app, on='appID', how='left')

def addUserInfo(data):
    '''
     添加用户信息，以及将居住地按省份和城市提取
    :param data:
    :return:
    '''
    user_info = pd.read_csv(raw_data_path + 'user.csv')
    data = pd.merge(data, user_info, on='userID', how='left')
    data['ht_province'] = data['hometown']//100
    data['ht_province'] = data['residence']//100
    return data

def addTime(data):
    '''
    添加一些时间转换的信息
    :param data:
    :return:
    '''
    data['clickDay'] = data['clickTime']//1000000
    data['clickDay'] = data['clickDay'].astype(int)
    data['clickHour'] = (data['clickTime']//10000%100).astype(int)
    # data['clickMin'] = (data['clickTime']%10000//100).astype(int)
    # data['clickSec'] = (data['clickTime']%100).astype(int)
    return data
