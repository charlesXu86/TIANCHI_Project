#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Gen_app_install_features.py
@time: 2018/3/12 14:13
@desc: 生成第一天之前各app被记录的安装数量（根据install）和每天之前各app被记录安装数量（根据action）
"""

import os
import pickle
import gc
import pandas as pd
import numpy as np

from tqdm import tqdm
from Ad_Utils import load_pickle, dump_pickle, raw_data_path, feature_data_path
from Feature_joint import addTime, addPosition, addAppCategories, addAd, addUserInfo

def gen_app_start_installed():
    '''
    记录第一天之前各个appID被记录的安装数量
    拼接键['appID']
    :return:
    '''
    feature_path = feature_data_path + 'app_start_installed.pkl'
    if os.path.exists(feature_path):
        print('Found:' + feature_path)
    else:
        print('Generating ' + feature_path)
        user_install = load_pickle(raw_data_path + 'user_installedapps.pkl')
        app_start_sum = user_install.groupby('appID').size().reset_index().rename(columns={0:'app_start_install_num'})
        del user_install
        gc.collect()
        dump_pickle(app_start_sum, feature_path)

def add_app_start_installed(data):
    feature_path = feature_data_path + 'app_start_installed.pkl'
    app_start_installed = load_pickle(feature_data_path)
    data = pd.merge(data, app_start_installed, on='appID', how='left')
    return data

def gen_app_hist_install():
    '''
     记录截至clickDay前一天，各个appID根据action表统计出的安装量
    :return:
    '''
    user_action = pd.read_csv(raw_data_path + 'user_app_actions.csv')
    feature_path = feature_data_path + 'app_hist_intall.pkl'
    if os.path.exists(feature_path):
        print('Found ' + feature_path)
    else:
        user_action['installDay'] = user_action['installTime']//1000000
        app_hist_install = user_action.groupby(['installDay', 'appID']).size().reset_index()
        app_hist_install.rename(columns={0:'app_day_install'},inplace=True)
        app_hist_install['app_hist_install'] = 0
        data_set = None
        for day in tqdm(app_hist_install.installDay.unique()):
            # df = app_hist_install[app_hist_install.installDay==day]
            last_day_install = app_hist_install[app_hist_install.installDay<day].groupby('appID').size().reset_index()
            last_day_install.rename(columns={0:'app_hist_install'}, inplace=True)
            last_day_install['Day'] = day
            if data_set is None:
                data_set = last_day_install
            else:
                data_set = pd.concat([data_set, last_day_install])
        # 添加最后一天的相关信息
        last_day_install = app_hist_install[app_hist_install.installDay<31].groupby('appID').size().reset_index()
        last_day_install.rename(columns={0: 'app_hist_install'}, inplace=True)
        pickle.dump(data_set, open(feature_path, 'wb'))

def add_app_hist_install(data):
    feature_path = feature_data_path + 'app_hist_install.pkl'
    app_hist_install = load_pickle(feature_path)
    data = pd.merge(data, app_hist_install, on='left', how=['appId', 'clickDay'])
    app_hist_install['app_hist_install'] = app_hist_install['app_hist_install'] / (app_hist_install['clickDay']-1)
    return data

def gen_user_start_installed_cateA():
    '''
    计算用户初始安装的各大类的app的数量
    拼接键['userID']
    :return:
    '''
    user_install = load_pickle(raw_data_path + 'user_installedapps.pkl' )
    app_cate = pd.read_csv(raw_data_path + 'app_categories.csv')
    app_cate['cate_a'] = app_cate.appCategory.apply(lambda x:x//100 if x>100 else x)
    user_install = user_install.merge(app_cate, on='appID', how='left')
    for cate_a in tqdm(app_cate.cate_a.unique()):
        feature_path = feature_data_path + 'user_start_installed_cate_' + str(cate_a) + '.pkl'
        if os.path.exists(feature_path):
            print('Found ' + feature_path)
        else:
            print('Generating ' + feature_path)
            user_install_cate = user_install[user_install.cate_a == cate_a][['userID', 'cate_a']]
            user_install_cate.rename(columns={'cate_a':'user_start_install_cate_' + str(cate_a)}, inplace=True)
            user_install_cate = user_install_cate.groupby('userID', as_index=False).sum()
            dump_pickle(user_install_cate, feature_path)

def add_user_start_installed_cateA(data):
    for cate in tqdm([0,1,2,3,4,5]):
        feature_path = feature_data_path + 'user_start_install_cate_' + str(cate) + '.pkl'
        user_start_installed_cateA = load_pickle(feature_path)
        data = pd.merge(data, user_start_installed_cateA, 'left', 'userID')
    return data

def gen_user_hist_install_cateA():
    pass













if __name__ == '__main__':
    pass