#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: gen_user_click_features.py
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
from feature_joint import addTime, addAd, addPosition, addAppCategories, addUserInfo


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


def gen_user_day_click_count(data, feature_list=['positionID', 'advertiserID', 'camgaignID', 'adID', 'creativeID', 'appID', 'sitesetID']):
    '''
     data必须包含clickHour字段，可以通过addTime，addAD,addPostion,addAppCategories添加
    :param data:
    :param feature_list:
    :return:
    '''
    ads_feature = ['advertiserID','camgaignID','adID','creativeID','appID',]
    context_feature = ['positionID', 'sitesetID']
    stats_feature = ads_feature + context_feature
    for feature in tqdm(feature_list):
        feature_path = feature_data_path + 'user_' + feature + '_click_hour.pkl'
        feature_day_click = load_pickle(feature_path)








if __name__ == '__main__':
    gen_user_day_click()
    gen_user_day_click_count()