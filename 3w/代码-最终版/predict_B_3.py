# 关键在于喂给我现有最优模型最优的数据，让它自己去拟合就行了.B场景数据是必不可少的，但是必须准确。
# A场景也是必须准确才能选为fit数据
# 所以从理论上来说我的现有结果肯定是基于A榜的，不可能超过A榜，只能尽可能去拟合A榜，因为数据信息都是从A榜中来的.
# 从最终B集预测结果来看，这是一个类不平衡，即使它不是类不平衡，也证明我的算法过于关注负类.所以抽样必不可少
# 第一轮去预测B的时候，就出现了类不平衡的导致大部分都是训练集负类的问题，因为我们先验假设不可能B场景训练集都是负标签.
# 信用卡场景下有风险者确实会降低,但是都是负标签我们假设不会存在，我们将从预测分布中找到可以接受误差的阈值
# 我们通过观察trainB集的预测结果，设置正负类分类的阈值0.62.即使这个标签是错的，这个数据样本内我们也能接受
# 直到没有该类样本再被训练出来.
# 问题在于每轮迭代去再次学习的时候还是不平衡，数据量也少.所以需要抽样
# 随着轮数变多，0.62下数据会变多，正类会变多.所以需要设置迭代次数
# 这个阈值可以自动根据先验分布来



import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
# tomek可以有效去掉噪声
from imblearn.combine import SMOTETomek


def update_pre(xa, ya, xb, threshold, model):
    '''
       更新预测结果
       input:
           xa: A集训练集属性集
           ya: A集标签
           xb: B集训练集属性集
           threshold: 正负类判断阈值
       return:
           pred,prob: B集预测标签以及概率
           bre: B集是否为空
    '''
    modeler = model.fit(xa, ya)
    # update,
    pred = modeler.predict(xb)
    prob = modeler.predict_proba(xb)
    # 按照阈值更新pred，只需更新正类
    # 如果已经没有有效的xb传入则退出
    bre = False
    if prob[:, 0].size != 0:
        pred[np.where(prob[:, 0] < threshold)] = 1
    else:
        bre = True
    return pred, prob, bre


def get_index(prob, p_index, rp, rn):
    '''
            得到本轮加入A集的B集索引以及本轮需预测的B集索引
            input:
                prob: 初始化概率
                p_index: B集上一轮索引
                (rp,rn):标签信息有效性判断范围，概率在该范围内表示有效
            return:
                f_index: 本轮加入A集的B集索引
                p_index: 本轮B集索引
        '''
    # proba 指定标签
    proba = prob.loc[prob[2].isin(p_index)]
    # all == p_index
    all = proba[2].values
    # 正类
    ft_index = proba.loc[proba[0] < rp][2].values
    # 负类
    fb_index = proba.loc[proba[0] > rn][2].values
    # 选取的加入A集的索引
    f_index = np.append(ft_index, fb_index)
    # 新的B集索引
    p_index = list(set(all).difference(set(f_index)))
    return f_index, p_index


class trans_scene(BaseEstimator):

    def __init__(self, xa, ya, rp=0.62, rn=0.7,threshold = 0.62, N=5,
                 model=XGBClassifier(booster='dart', eval_metric='auc', scale_pos_weight=0.5, subsample=0.9,
                                     colsample_bytree=0.5,
                                     random_state=111, gamma=0),
                 model_p=XGBClassifier(booster='dart', eval_metric='auc', subsample=0.9,
                                       colsample_bytree=0.5,
                                       random_state=111, gamma=0)):
        '''
            初始化
            input:
                xa: A集训练集属性集
                ya: A集标签
                (rp,rn):标签信息有效性判断范围，概率在该范围内表示有效
                threshold: 正负类判断阈值
                N: 迭代次数，过小造成B场景数据作用小，过大可能造成对误判正类的学习
                model:auc最优模型
                model_p:概率最优模型
        '''
        self.xa = xa
        self.ya = ya
        self.rp = rp
        self.rn = rn
        self.threshold = threshold
        self.N = N
        self.model = model
        self.model_p = model_p
        self.modeler = None

    def fit(self, xb):
        '''
        :param xb: B场景训练集，无标签
        :return:
        '''
        xa = self.xa
        ya = self.ya
        # 更新预测结果
        pred, prob, bre = update_pre(xa, ya, xb, self.threshold, self.model_p)
        ##初始化
        # preda 初始化概率表，后期只根据索引更改提取
        preda = pred
        proba = pd.DataFrame(prob)
        proba[2] = proba.index
        # 抽样器初始化
        stkor = SMOTETomek()
        # f_index 加入A集的B集数据(有标签)索引
        # p_index 本轮预测的B集数据索引
        f_index, p_index = get_index(proba, proba.index, self.rp, self.rn)
        for i in range(self.N):
            if bre:
                break
            # 更新A集，加入有标签的B集
            f_xb = xb[f_index]
            f_yb = preda[f_index]
            # 更新B集
            p_xb = xb[p_index]
            # 对f_xb,f_yb 过抽样
            # 过抽样过程增加B集信息
            # 如果在update时直接过抽样.A的信息可能会干扰住B信息
            # 其次破坏了A集最优模型后，结果偏差无法预测，无法向最优成绩拟合,不利于得到最准确概率.
            f_xb, f_yb = stkor.fit_sample(f_xb, f_yb)
            # B集信息利用最大化以及对未知分布稳定
            xa = np.concatenate((f_xb, xa), axis=0)
            ya = np.concatenate((f_yb, ya), axis=0)
            # 用更新的A集，预测更新B集数据
            pred, prob, bre = update_pre(xa, ya, p_xb, self.threshold, self.model_p)
            # 更新维护的概率列表以及标签列表
            preda[p_index] = pred
            proba.loc[proba[2].isin(p_index), [0, 1]] = prob
            f_index, p_index = get_index(proba, p_index, self.rp, self.rn)

        #用最优auc模型学习最终数据
        self.modeler = self.model.fit(xa, ya)
        return self

    def predict_proba(self, xb):
        '''

        :param xb: 传入B集测试集
        :return:
        '''
        return self.modeler.predict_proba(xb)
