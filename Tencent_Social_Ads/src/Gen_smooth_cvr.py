#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Gen_smooth_cvr.py
@time: 2018/3/15 11:22
@desc: 广告转化率特征提取
       构造转化率特征，使用全局和滑动窗口等方式计算单特征转化率，组合特征转化率，使用均值填充，层级填充，贝叶斯平滑，拉普拉斯平滑等方式对转化率进行修正。
"""

import numpy as np
import pandas as pd
import gc
import os

from Smooth import BayesianSmoothing
from tqdm import tqdm
from Ad_Utils import raw_data_path, feature_data_path,load_pickle, dump_pickle
from  Feature_joint import addAd, addTime, addPosition


def gen_hist_cvr_smooth(start_day, end_day, key, alpha=0.25):
    # train_data = pd.read_csv(raw_data_path, 'train.csv')
    train_data = load_pickle(raw_data_path + 'train.pkl')
    test_data = load_pickle(raw_data_path + 'test.pkl')
    data = train_data.append(test_data)
    del train_data, test_data
    gc.collect()
    data = addTime(data)
    data = addAd(data)
    data = addPosition(data)
    ID_hist_cvr = None
    for day in tqdm(np.arange(start_day, end_day+1)):
        feature_path = feature_data_path + key + '_histcvr_smooth_day_'+str(day)+'.pkl'
        if os.path.exists(feature_path):
            print('found ' + feature_path)
        else:
            print('generating ' + feature_path)
            dfCvr = data[data.clickDay < day]
            dfCvr = pd.get_dummies(dfCvr, columns=['label'], prefix='label')
            dfCvr = dfCvr.groupby([key], as_index=False).sum()
            dfCvr[key + '_cvr'] = (dfCvr['label_1'] + alpha) / (dfCvr['label_0'] + dfCvr['label_0'] + alpha * 2)
            # dfCvr['clickDay'] = day
            sub_data = pd.merge(data.loc[data.clickDay==day,['clickDay',key]],dfCvr[[key,key+'_cvr']],'left',on=[key,])
            sub_data.drop_duplicates(['clickDay', key], inplace=True)
            sub_data.sort_values(['clickDay', key], inplace=True)
            dump_pickle(sub_data[['clickDay', key, key + '_cvr']], feature_path)

def add_hist_cvr_smooth(data, key):
    hist_cvr_smooth = None
    for day in tqdm((data.clickTime // 1000000).unique()):
        feature_path = feature_data_path + key + '_histcvr_smooth_day_' + str(day) + '.pkl'
        day_cvr_smooth = load_pickle(feature_path)
        if hist_cvr_smooth is None:
            hist_cvr_smooth = day_cvr_smooth
        else:
            hist_cvr_smooth = pd.concat([hist_cvr_smooth, day_cvr_smooth], axis=0)
    data = pd.merge(data, hist_cvr_smooth, 'left', ['clickDay', key])
    return data


def gen_positionID_cvr_smooth(test_day):
    feature_path = feature_data_path + 'positionID_cvr_smooth_day_' + str(test_day) + '.pkl'
    if os.path.exists(feature_path):
        print('found ' + feature_path)
    else:
        print('generating ' + feature_path)
        data = load_pickle(raw_data_path + 'train.pkl')
        data = addTime(data)
        positionID_cvr = data[data.clickDay < test_day]
        I = positionID_cvr.groupby('positionID')['label'].size().reset_index()
        I.columns = ['positionID', 'I']
        C = positionID_cvr.groupby('positionID')['label'].sum().reset_index()
        C.columns = ['positionID', 'C']
        positionID_cvr = pd.concat([I, C['C']], axis=1)
        hyper = BayesianSmoothing(1, 1)
        hyper.update(positionID_cvr['I'].values, positionID_cvr['C'].values, 10000, 0.00000001)
        alpha = hyper.alpha
        beta = hyper.beta
        positionID_cvr['positionID_cvr_smooth'] = (positionID_cvr['C'] + alpha) / (positionID_cvr['I'] + alpha + beta)
        dump_pickle(positionID_cvr[['positionID', 'positionID_cvr_smooth']], feature_path)

def add_smooth_pos_cvr(data, test_day):
    feature_path = feature_data_path + 'positionID_cvr_smooth_day_'+str(test_day)+'.pkl'
    smooth_pos_cvr = load_pickle(feature_path)
    data = pd.merge(data,smooth_pos_cvr,'left','positionID')
    return data

if __name__ == '__main__':
    gen_hist_cvr_smooth(23, 31, 'userID', )
    gen_hist_cvr_smooth(23, 31, 'creativeID', )
    gen_hist_cvr_smooth(23, 31, 'adID', )
    gen_hist_cvr_smooth(23, 31, 'appID', )

    gen_positionID_cvr_smooth(27)
    gen_positionID_cvr_smooth(31)