#-*- coding:utf-8 _*-

"""
@version: 
@author: CharlesXu
@license: Q_S_Y_Q 
@file: Smooth.py
@time: 2018/3/14 9:42
@desc: 贝叶斯平滑  参考博客:
       http://blog.csdn.net/mytestmy/article/details/19088519
"""

'''
  主要思想是在分子分母各加一个比较大的数
'''

import numpy as np
import random
import scipy.special as special   # 排列组合与阶乘

from tqdm import tqdm

np.random.seed(0)

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)   # Draw samples from a Beta distribution.
        I = []
        C = []
        for clk_rt in sample:
            # imp = random.random() * imp_upperbound
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in tqdm(range(iter_num)):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.beta) < epsilon and abs(new_beta - self.alpha) < epsilon:
                break
            print(new_alpha, new_beta, i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        '''
          参数估计的几种方法之一:  fixed-point iteration
          首先构造出似然函数，然后利用fixed-point iteration来求得似然函数的最大值
            1）首先给出参数的一个初始值。
            2）在初始值处，构造似然函数的一个紧的下界函数。这个下界函数可以用closed-form的方式计算其最大值处的参数值，将此参数值作为新的参数估计。
            3）不断重复上述（2）的步骤，直至收敛。此时便可到达似然函数的stationary point。
            其实fixed-point iteration（不动点迭代）的思想与EM类似。
        :param imps:
        :param clks:
        :param alpha:
        :param beta:
        :return:
        '''
        numerater_alpha = 0.0
        numerater_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerater_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerater_beta += (special.digamma(imps[i] + beta) - special.digamma(beta))   # 计算psi值
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))
        return alpha * (numerater_alpha / denominator), beta * (numerater_beta / denominator)

def test():
    bs = BayesianSmoothing(1, 1)
    I, C = bs.sample(500, 500, 1000, 1000)
    print(I, C)
    bs.update(I, C, 1000, 0.0000000001)
    print(bs.alpha, bs.beta)


if __name__ == '__main__':
    pass