#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Utils.py
@time: 2018/3/7 16:27
@desc: 工具类
"""

import pickle
import pandas as pd
import numpy as np
import scipy.stats as sps

from tqdm import tqdm

# file_path
raw_data_path = '../data/'
feature_data_path ='../features/'
cache_pkl_path = '../cache_pkl/'
result_path = '../result/'

def load_pickle(path):
    return pickle.load(open(path, 'rb'))

def dump_pickle(obj, path, protocol=None):
    pickle.dump(obj, open(path, 'wb'), protocol=protocol)

def analyse(data, field):
    a = data.groupby(field).size()
    b = data.groupby(field)['label'].sum()
    c = pd.DataFrame({'conversion':b, 'click':a})
    c.reset_index(inplace=True)
    c['prob'] = c['conversion'] / c['click']
    return c.sort_values('prob', ascending=False)

def generate_file(valid_y, pred_prob):
    pass


if __name__ == '__main__':
    pass