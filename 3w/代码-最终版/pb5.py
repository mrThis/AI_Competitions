# 第二版思路，设置一个阈值，当阈值接近0.7,0.6的时候...就停在这儿.不去学习B，因为不确定性太大
# 第二版思路问题，这个阈值的设定不好设定，太高了容易造成完全不迭代全部是负值，完全丧失掉正类信息
# 关键在于喂给我现有最优模型最优的数据，让它自己去拟合就行了.B场景数据是必不可少的，但是必须准确。
# A场景也是必须准确才能选为fit数据
# 所以从理论上来说我的现有结果肯定是基于A榜的，不可能超过A榜，只能尽可能去拟合A榜，因为数据信息都是从A榜中来的.


# 从最终B集预测结果来看，这是一个类不平衡，即使它不是类不平衡，也证明我的算法过于关注负类.所以抽样必不可少

# 第一轮去预测B的时候，就出现了类不平衡的导致大部分都是训练集负类的问题，因为我们先验假设不可能B场景训练集都是负标签.
# 信用卡场景下有风险者确实会降低,但是都是负标签我们假设不会存在，我们将从预测分布中找到可以接受误差的阈值

# 为了解决这个问题，我们设置两个阈值，我们默认以0列概率0.7下的为正类，0.85以上的为负类，其余的舍弃，去迭代学习.选出每轮满足这个阈值的样本
# 即使0.7的标签是错的，这个数据样本内我们也能接受
# 直到没有该类样本再被训练出来.
# 问题在于每轮迭代去再次学习的时候还是不平衡.所以需要抽样
# 随着轮数变多，0.7下数据会变多，正类会变多.所以需要设置迭代次数
# 这个阈值可以自动根据分布来，但是分布未知，就设置为magic number去调参了


# 测试方案:我们将B场景的预测结果，全都假设为0和1，然后将结果加权平均..


# 第二版更新，加抽样方法.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
# Tmklinks是都删
# EditedNearestNeighbours是只删多数类
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN
# tomek可以有效去掉噪声
from imblearn.combine import SMOTETomek


# 抽样方法
# def up_sample(df):
#     df1=df[df['target']==1]#正例
#     df2=df[df['target']==0]##负例
#     df3=pd.concat([df1,df1,df1,df1,df1],ignore_index=True)
#     return pd.concat([df2,df3],ignore_index=True)


# 更新预测结果，xa,ya,xb分别为A场景训练集，与B场景属性集
# rp,小于rp为正类(在负类里那一列)
# rn, 大于rn为负类（同上）
# bre ,是否跳出循环
def update_pre(xa, ya, xb, threshold, model):
    modeler = model.fit(xa, ya)
    # update,
    pred = modeler.predict(xb)
    prob = modeler.predict_proba(xb)
    # 按照阈值更新pred，只需更新正类
    # r如果已经没了退出
    bre = False
    if prob[:, 0].size != 0:
        pred[np.where(prob[:, 0] < threshold)] = 1
    else:
        bre = True
    return pred, prob, bre


# 得到指定index上的，下一轮标签
# prob 维护的proba
# p_index 上一轮预测结束的标签
# rp,小于rp为正类概率(在负类里那一列)
# rn, 大于rn为负类概率（同上）
def get_index(prob, p_index, rp, rn):
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
    # 新的B集
    p_index = list(set(all).difference(set(f_index)))
    return f_index, p_index


class trans_scene(BaseEstimator):
    # xa A场景训练集
    # ya A场景标签
    # N,迭代上界,(防止学习的误分正类过多)
    # rp,正类概率(在负类里那一列)
    # rn, 负类概率（同上）

    def __init__(self, xa, ya, rp=0.62, rn=0.7,threshold = 0.62, N=5,
                 model=XGBClassifier(booster='dart', eval_metric='auc', scale_pos_weight=0.5, subsample=0.9,
                                     colsample_bytree=0.5,
                                     random_state=111, gamma=0),
                 model_p=XGBClassifier(booster='dart', eval_metric='auc', subsample=0.9,
                                       colsample_bytree=0.5,
                                       random_state=111, gamma=0)):
        self.xa = xa
        self.ya = ya
        self.rp = rp
        self.rn = rn
        self.threshold = threshold
        self.N = N
        self.model = model
        self.model_p = model_p
        self.modeler = None

    # xb 为B场景训练集
    def fit(self, xb):
        xa = self.xa
        ya = self.ya
        # 更新预测结果
        pred, prob, bre = update_pre(xa, ya, xb, self.threshold, self.model_p)
        # 去维护的一个prob，第2列中放真正index
        # 初始化
        preda = pred
        proba = pd.DataFrame(prob)
        proba[2] = proba.index
        stkor = SMOTETomek()
        # f_index 加入A集有标签列的B集数据索引
        # p_index 下一轮预测的B集数据索引
        f_index, p_index = get_index(proba, proba.index, self.rp, self.rn)
        for i in range(self.N):
            if bre:
                break
            # 更新A集，加入上一轮加标签的B集
            f_xb = xb[f_index]
            f_yb = preda[f_index]
            p_xb = xb[p_index]
            # 对f_xb,f_yb 过抽样
            # 过抽样过程增加B集信息
            # 如果在update时直接过抽样.A的信息可能会干扰住B信息
            # 其次破坏了A集最优模型后，结果偏差无法预测，无法向最优成绩拟合,不利于得到最准确概率.
            f_xb, f_yb = stkor.fit_sample(f_xb, f_yb)
            # A集信息利用最大化以及干扰信息稳定

            xa = np.concatenate((f_xb, xa), axis=0)
            ya = np.concatenate((f_yb, ya), axis=0)
            # 用更新的A集，预测剩余B集未加标签数据，在正真学习的时候应该组成新A集后一起预测？
            pred, prob, bre = update_pre(xa, ya, p_xb, self.threshold, self.model_p)
            # 更新维护的概率列表以及标签列表
            preda[p_index] = pred
            proba.loc[proba[2].isin(p_index), [0, 1]] = prob
            f_index, p_index = get_index(proba, p_index, self.rp, self.rn)

        # self.modeler = self.model.fit(xb,preda)
        # or
        # 新的xa，ya中的B的信息已近不少，为了稳定(未知的B集分布)，使用过抽样，填充多数类并且利用SMOTETomek去掉A，B中可能重合的点
        # or
        self.modeler = self.model.fit(xa, ya)
        return self

    # xb 为B场景测试集
    def predict_proba(self, xb):
        return self.modeler.predict_proba(xb)
