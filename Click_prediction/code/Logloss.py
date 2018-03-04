#-*- coding:utf-8 _*-  
""" 
@author:charlesXu
@file: Logloss.py 
@desc: 腾讯算法大赛logloss求法
@time: 2018/03/04 
"""

import scipy as sp

def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * - 1.0 / len(act)
    return ll