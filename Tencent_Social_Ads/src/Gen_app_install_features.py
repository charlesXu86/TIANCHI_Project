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

from sklearn.feature_extraction.text import TfidfTransformer  #该类会统计每个词语的tf-idf权值

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
    app_start_installed = load_pickle(feature_path)
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
        last_day_install['Day'] = 31
        data_set = pd.concat([data_set, last_day_install])
        data_set.rename(columns={'Day':'clickDay'}, inplace=True)
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
    '''
    记录截至clickDay前一天，用户安装的各个大类app总量，根据action表统计
    拼接键['userID', 'clickDay']
    :return:
    '''
    feature_path = feature_data_path + 'user_hist_install_cateA'
    if os.path.exists(feature_path):
        print('Found ' + feature_path)
    else:
        print('Generating ' + feature_path)
        user_action = pd.read_csv(raw_data_path + 'user_app_actions.csv')
        app_cate = pd.read_csv(raw_data_path + 'app_categories.csv')
        app_cate['cate_a'] = app_cate.appCategory.apply(lambda x:x//100 if x>100 else x)
        user_action = user_action.merge(app_cate[['appID', 'cate_a']], on='appID', how='left')
        user_action['installDay'] = user_action['installTime']//1000
        user_action = pd.get_dummies(user_action[['userID', 'cate_a', 'installDay']],
                                     prefix='user_hist_install_cateA', columns=['cate_a'])
        stats_columns = ['user_hist_install_cateA_' + str(i) for i in range(0,6)]
        user_hist_install_cateA = None
        for clickday in tqdm(range(17, 32)):
            last_day_acc_install = user_action[user_action.installDay < clickday][['userID'] + stats_columns]
            last_day_acc_install = last_day_acc_install.groupby('userID', as_index=False).sum()
            last_day_acc_install['clickDay'] = clickday
            if user_hist_install_cateA is None:
                user_hist_install_cateA = last_day_acc_install
            else:
                user_hist_install_cateA = pd.concat([user_hist_install_cateA, last_day_acc_install], axis=0)
        dump_pickle(user_hist_install_cateA, feature_path)

def add_user_hist_install_cateA(data):
    raise NotImplementedError('NotImplementedError')

def gen_CountVector_appID_user_installed(appID_describe_feature_names=['age_cut','gender','education','marriageStatus','haveBaby','hometown_province','residence_province']):
    '''
    生成根据install表计算的appID计数描述向量
    :param appID_describe_feature_names:
    :return:
    '''
    user_install = load_pickle(raw_data_path + 'user_installedapps.pkl')
    user_info = pd.read_csv(raw_data_path + 'user.csv')
    user_info['age_cut'] = pd.cut(user_info['age'], bins=[-1, 0, 18, 25, 35, 45, 55, 65, np.inf], labels=False)
    user_info['hometown_province'] = user_info['hometown'].apply(lambda x:x//100)
    user_info['residence_province'] = user_info['residence'].apply(lambda x:x//100)

    for feature in tqdm(appID_describe_feature_names):
        feature_path = feature_data_path + 'CountVector_appID_user_installed_' + feature + '.pkl'
        if os.path.exists(feature_path):
            print('Found ' + feature_path)
        else:
            print('Generating ' + feature_path)
            sub_user_info = pd.get_dummies(user_info[['userID', feature]], columns=[feature], prefix='appID_installed_' + feature)  # 进行独热编码
            user_install = pd.merge(user_install, sub_user_info, on='userID', how='left')
            dummy_features = sub_user_info.columns.tolist()
            dummy_features.remove('userID')
            app_describe_feature = None
            for dummy_feature in tqdm(dummy_features):
                app_feature_installed = user_install[['appID', dummy_feature]].groupby('appID', as_index=False).sum()
                if app_describe_feature is None:
                    app_describe_feature = app_feature_installed
                else:
                    app_describe_feature = pd.concat([app_describe_feature, app_feature_installed[[dummy_feature]]], axis=1)
                user_install.drop(dummy_feature, inplace=True, axis=1)
            dump_pickle(app_describe_feature, feature_path)


def getConcatedAppIDCountVector(concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby']):
    '''
     拼接键['appID']
    :param concated_list:
    :return:
    '''
    concated_countvec = None
    for feature in tqdm(concated_list):
        feature_path = feature_data_path + 'CountVector_appID_user_installed_' + feature + 'pkl'
        if os.path.exists(feature_path):
            count_vec = load_pickle(feature_path)
        else:
            gen_CountVector_appID_user_installed(concated_list)
            count_vec = load_pickle(feature_path)
        if concated_countvec is None:
            concated_countvec = count_vec
        else:
            concated_countvec = pd.merge(concated_countvec, count_vec, on='appID', how='left')
    return concated_countvec

def get_ConcatedAppIDTfidfVector_userinstalled(concated_list=['age_cut', 'gender', 'education', 'marriageStatus', 'haveBaby'], mode='local', norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
    assert mode in ['global', 'local'], 'mode must be global or local'
    tfidf_vec = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf)
    if mode == 'global':
        concated_countvec = getConcatedAppIDCountVector(concated_list)
        concated_countvec.set_index('appID', inplace=True)
        vec_columns = concated_countvec.columns
        global_tfidf_vec = tfidf_vec.fit_transform(concated_countvec).todense()
        global_tfidf_vec = pd.DataFrame(global_tfidf_vec, columns=vec_columns, index=concated_countvec.index).reset_index()
        return global_tfidf_vec
    else:
        concated_tfidf_vec = None
        for feature in tqdm(concated_list):
            feature_path = feature_data_path + 'CountVector_appID_user_installed_' + feature + 'pkl'
            if os.path.exists(feature_path):
                count_vec = load_pickle(feature_path)
            else:
                gen_CountVector_appID_user_installed(concated_list)
                count_vec = load_pickle(feature_path)
            count_vec.set_index('appID', inplace=True)
            vec_columns = count_vec.columns
            local_tfidf_vec = tfidf_vec.fit_transform(count_vec).todense()
            local_tfidf_vec = pd.DataFrame(local_tfidf_vec, columns=vec_columns, index=count_vec.index).reset_index()
            if concated_tfidf_vec is None:
                concated_tfidf_vec = local_tfidf_vec
            else:
                concated_tfidf_vec = pd.merge(concated_tfidf_vec, local_tfidf_vec, on='appID', how='left')
        return concated_tfidf_vec

def gen_CountVector_appCategory_user_installed(appCategory_describe_feature_names=['age_cut','gender','education','marriageStatus','haveBaby','hometown_province','residence_province']):
    '''
    生成根据install表统计的appid计数
    :param appCategory_describe_feature_names:
    :return:
    '''
    pass

def gen_CountVector_appCategory_user_action_hour():
    '''
    拼接键['appcategory']
    :return:
    '''
    feature_path = raw_data_path + 'CountVector_appCategory_actionHour.pkl'
    if os.path.exists(feature_path):
        print('Found ' + feature_path)
    else:
        print('Generating ' + feature_path)
        user_action = pd.read_csv(raw_data_path + 'user_app_actions.csv')
        app_cate = pd.read_csv(raw_data_path + 'app_categories.csv')
        user_action = pd.merge(user_action, app_cate, 'left', 'appID')
        user_action['installHour'] = user_action['installTime'] % 1000000 // 10000
        user_action = pd.get_dummies(user_action[['appCategory', 'installHour']],columns=['installHour'])
        user_action = user_action.groupby('appCategory', as_index=False).sum()
        dump_pickle(user_action, feature_path)

def get_TfidfVector_appCategory_user_action_hour(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

    feature_path = feature_data_path + 'CountVector_appCategory_actionHour.pkl'
    if not os.path.exists(feature_path):
        gen_CountVector_appCategory_user_action_hour()
    count_vec = load_pickle(feature_path)
    count_vec.set_index('appCategory', inplace=True)
    col_name = count_vec.columns
    tfidf_vec = pd.DataFrame(tfidf.fit_transform(count_vec).todense(), columns=col_name,
                             index=count_vec.index).reset_index()
    return tfidf_vec


"""
def gen_user_hist_install():
    train = load_pickle(raw_data_path+'train.pkl')
    train_user_hist_install = load_pickle(feature_data_path+'train_user_hist_install')
    train = addTime(train)
    train_user_hist_install.index = train.index
    train = pd.concat([train,train_user_hist_install],axis=1)
    train_user_hist_install = train[['clickDay','userID','user_hist_install']].drop_duplicates()
    dump_pickle(train_user_hist_install,feature_data_path+'train_user_hist_install.pkl')

    test_user_hist_install = load_pickle(feature_data_path+'test_user_hist_install')
    dump_pickle(test_user_hist_install,feature_data_path+'test_user_hist_install.pkl')
def add_user_hist_install(data,mode):
    assert mode in ['train','test']
    if mode == 'train':
        train_user_hist_install = load_pickle(feature_data_path+'train_user_hist_install.pkl')
        data = pd.merge(data,train_user_hist_install,'left',['clickDay','userID'])
    elif mode=='test':
        test_user_hist_install = load_pickle(feature_data_path+'test_user_hist_install.pkl')
        test_user_hist_install.index = test_user_hist_install.index
        data = pd.concat([data,test_user_hist_install],axis=1)
    return data
"""

def gen_user_hist_install():
    '''
    记录截至clickDay前一天，用户安装的app总量，根据action表统计
    :return:
    '''
    feature_path = feature_data_path + 'user_hist_install.pkl'
    if os.path.exists(feature_path):
        print('found ' + feature_path)
    else:
        print('generating ' + feature_path)
        user_action = pd.read_csv(raw_data_path + 'user_app_actions.csv')
        user_action['installDay'] = user_action['installTime'] // 1000000
        user_hist_install = None
        for clickday in tqdm(range(17, 32)):
            last_day_acc_install = user_action[user_action.installDay < clickday][['userID', 'appID']].groupby('userID',
                                                                                                               as_index=False).count()
            last_day_acc_install['clickDay'] = clickday
            last_day_acc_install.rename(columns={'appID': 'user_hist_install'}, inplace=True)
            last_day_acc_install['user_hist_install'] = last_day_acc_install['user_hist_install'] / clickday  # 对天数做平滑
            if user_hist_install is None:
                user_hist_install = last_day_acc_install
            else:
                user_hist_install = pd.concat([user_hist_install, last_day_acc_install], axis=0)
        pd.to_pickle(user_hist_install, feature_path)

def add_user_hist_install(data, mode):
    feature_path = feature_data_path + 'user_hist_install.pkl'
    user_hist_installed = load_pickle(feature_path)
    data = pd.merge(data, user_hist_installed, 'left', ['userID', 'clickDay'])
    return data


# 计算个人群的初始安装的appID的量
def gen_user_group_install():
    user_install = load_pickle(raw_data_path+'user_installedapps.pkl')
    user_info = load_pickle(raw_data_path+'user.pkl')
    user_info['age_cut_small']=pd.cut(user_info['age'],bins=[-1,0,18,25,35,45,55,np.inf],labels=False)
    user_info['education_new'] = user_info['education']
    user_info.loc[user_info.education_new==7,'education_new'] = 6
    user_info_comb = user_info[['age_cut_small','gender','education_new',]].drop_duplicates()
    user_info_comb['user_group'] = np.arange(0,user_info_comb.shape[0])
    user_info = pd.merge(user_info,user_info_comb,'left',['age_cut_small','gender','education_new',])
    user_install = pd.merge(user_install,user_info[['userID','user_group','age_cut_small','gender','education_new',]],'left','userID')
    def update_dict(row,dic):
        dic[row['appID']] += 1
    user_group_install = None
    for i,u_g in tqdm(enumerate(user_install.user_group.unique())):
        sub_install = user_install[user_install.user_group==u_g]
        install_dict = dict((k,0) for k in user_install.appID.unique())
        install_dict['user_group'] = u_g
        install_dict['age_cut_small'] = sub_install['age_cut_small'].iloc[0]
        install_dict['gender'] = sub_install['gender'].iloc[0]
        install_dict['education_new'] = sub_install['education_new'].iloc[0]
        sub_install.apply(update_dict, args=(install_dict,),axis=1,)
        if user_group_install is None:
            user_group_install = pd.DataFrame(install_dict,index=[i,])
        else:
            user_group_install = pd.concat([user_group_install,pd.DataFrame(install_dict,index=[i,])])
    dump_pickle(user_group_install,feature_data_path+'user_group_install.pkl')

if __name__ == '__main__':
    gen_user_start_installed_cateA()
    gen_user_hist_install()
    # gen_user_hist_install_cateA()A
    # gen_app_start_installed()
    # gen_app_hist_install()
    # gen_CountVector_appID_user_installed(['age_cut','gender','education','marriageStatus','haveBaby'])
    # gen_CountVector_appCategory_user_installed(['age_cut','gender','education','marriageStatus','haveBaby'])
    # gen_CountVector_appCategory_user_action_hour()

    print('All done')
