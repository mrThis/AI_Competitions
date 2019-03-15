import warnings
import sys
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import os
import itertools
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations


# 工具函数序列化与反序列化

def read_p(name):
    return pd.read_pickle('../data/input/' + name + '.p')


def save_p(var, name):
    var.to_pickle('../data/input/' + name + '.p')


# 工具函数 cv分数
def auc_cv(model, train_data, label):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_data.values)
    r_auc = cross_val_score(model, train_data, label, scoring="roc_auc", cv=kf, verbose=5)  # auc评价
    return (r_auc)

# 处理函数 将0设置为缺失值
if True: #TODO:del it
    def set_nan(df):
        tem = len(df)
        for x in df:
            ratio = df[df[x] == 0].count()[x] / tem
            # 1 0.93102
            # 0.9降低一个千分位
            if ratio > 0.8:
                df[x]=df[x].agg(lambda x: np.nan if x == 0 else x)
        return df

# 处理函数， 数据预处理
def preprocess(x):
    # 处理FTR51
    '''
    :param df: train+test
    :return:  经过特征工程后的数据
    '''

    def deal_FTR51(train):
        '''
        :param train: test+train
        :return mj: 处理FTR51后得到的特征
        '''

        def quarter(x):
            if x.month in [3, 4, 5, 6, 7, 8]:
                return 1
            else:
                return 0

        def cate(arr):
            return len(arr.unique())

        def mean(arr):
            return pd.value_counts(arr).mean()

        def max(arr):
            return pd.value_counts(arr).max()

        def min(arr):
            return pd.value_counts(arr).min()

        # TODO：没有对这部分统计量统计skew and kurt
        a = train[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',',
                                                                             expand=True).unstack().dropna().reset_index().drop(
            'level_0', axis=1).rename(columns={0: 'FTR51'})
        b = train[['PERSONID', 'APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
        c = train[['APPLYNO', 'CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()
        x = pd.merge(a, b, on='APPLYNO')
        x = pd.merge(x, c, on='APPLYNO')
        # TODO:添加了min
        # m_sum
        m_sum = x.groupby('PERSONID')['FTR51'].count().rename('FTR51m_sum')
        # m_cate_sum
        m_cate_sum = x.groupby('PERSONID')['FTR51'].agg(cate).rename('FTR51m_cate_sum')
        # m_freq1,max1,min1
        mj1 = x.groupby('PERSONID')['APPLYNO'].agg({'FTR51m_freq1': mean, 'FTR51max1': max, 'FTR51min1': min})
        # m_freq2,max2,min2
        mj2 = x.groupby('PERSONID')['CREATETIME'].agg({'FTR51m_freq2': mean, 'FTR51max2': max, 'FTR51min2': min})
        # m_freq3,max3,min3
        x['MONTH'] = pd.to_datetime(x.CREATETIME).apply(lambda x: x.month)
        mj3 = x.groupby('PERSONID')['MONTH'].agg({'FTR51m_freq3': mean, 'FTR51max3': max, 'FTR51min3': min})
        if False:  # TODO:季度,过拟合风险较大，B榜时尝试
            # m_freq4,max4,max4
            x['Quarter'] = pd.to_datetime(x.CREATETIME).apply(quarter)
            mj4 = x.groupby('PERSONID')['Quarter'].agg({'FTR51m_freq4': mean, 'FTR51max4': max, 'FTR51min4': min})
        mj = mj1.join(mj2).join(mj3).join(m_sum).join(m_cate_sum)  # .join(mj4)

        # 将FTR51前频数15做特征
        temp = train[['PERSONID', 'FTR51']].copy()
        flag = list(temp['FTR51'].value_counts()[:15].index)
        temp1 = pd.DataFrame(index=mj.index)
        for item in flag:
            temp_select = temp[temp['FTR51'] == item]
            df_count = temp_select.groupby('PERSONID').size().to_frame()
            df_count.columns = ['FTR51' + item + '_count']
            temp1 = temp1.join(df_count).fillna(0)
        mj = mj.join(temp1)
        if True:  # TODO: FTR51单个药品频数？
            pass
        return mj

    def compare(s):
        # TODO?
        gb = s.groupby('season')
        if gb.ngroups == 1:
            return pd.Series(index=s.columns)
        else:
            sum = gb.sum()
            return sum.loc[1.00] / (sum.loc[0.25])
    #特征融合
    def comb_feature(data):
        corr = data.corr()
        d = []
        for x in corr:
            if x in d or corr[x].dropna().empty:
                continue
            # 相关性大的特征
            # 0.9的时候会有，sum-1980 .9298
            # 0.95 sum-1720 .9285
            de = list(corr[corr[x] > 0.9].index)
            # 特征列除了自己的1还有其它的
            if (len(de) > 1):
                # 其它的与自己进行特征融合
                for dde in list(set(de) - set([x])):
                    print(dde)
                    print(x)
                    # 融合特征
                    c1 = data[dde] / data[x]
                    c1.name = dde + '/' + x
                    c2 = data[dde] - data[x]
                    c2.name = dde + '-' + x
                    data = data.join(c1).join(c2)
            # 将融合过的特征加入不再融合列
            print('round' + str(x) + '-------------------------------------')
            print(de)
            d = d + de
        print(len(d))
        return data

    # 初始化返回值
    dff = pd.DataFrame(index=x.PERSONID.sort_values().unique())
    l = []

    # 去掉完全重复数据
    x = x.drop(['FTR20'], axis=1)
    #TODO：将0 置为nan
    x = set_nan(x)
    #特征融合
    x = comb_feature(x)
    # TODO:未删除
    # 相关性大的相减构建新特征
    # 按季度（依照数据量分季度）统计#TODO？
    q = pd.to_datetime(x.CREATETIME).apply(lambda x: 1 if x.month in [3, 4, 5, 6, 7, 8] else 0.25)
    countadj = x.join(q.rename('adj')).groupby('PERSONID')['adj'].sum()
    incresing = x.join(q.rename('season')).groupby('PERSONID').apply(compare).unstack().drop(
        ['APPLYNO', 'CREATETIME', 'PERSONID', 'season'], axis=1)

    # 按时间特点新增列#TODO 添加yfy特征，但不知有无重合
    x['CREATETIME'] = pd.to_datetime(x['CREATETIME'])
    x["year"] = x["CREATETIME"].dt.year
    x['month'] = x["CREATETIME"].dt.month
    x['day'] = x["CREATETIME"].dt.day
    x['weekday'] = x["CREATETIME"].dt.weekday + 1
    x['week'] = x["CREATETIME"].dt.week
    x['is_weekend'] = x['weekday'].apply(lambda x: 1 if x in (6, 7) else 0)

    # 聚合，取统计量
    counts = x.groupby('PERSONID')['APPLYNO'].count().rename('count')
    lasts = x.groupby('PERSONID').last().drop(['CREATETIME', 'FTR51', 'APPLYNO'], axis=1)
    avg = x.groupby('PERSONID').mean()
    max = x.groupby('PERSONID').max().drop(['CREATETIME', 'FTR51', 'APPLYNO'], axis=1)
    std = x.groupby('PERSONID').std()
    skew = x.groupby('PERSONID').apply(lambda x: x.skew())
    kurt = x.groupby('PERSONID').apply(lambda x: x.kurt())
    sum = x.groupby('PERSONID').sum()
    l += ['countadj', 'incresing', 'counts', 'lasts', 'avg', 'max', 'std', 'skew', 'kurt', 'sum']
    lastdm = (pd.to_datetime('2016-03-01') - pd.to_datetime(x.groupby('PERSONID').CREATETIME.last())) \
        .apply(lambda x: x.days).rename('lastdm')
    countd = x.groupby('PERSONID').apply(lambda k: k.CREATETIME.nunique()).rename('countdate')
    avgd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum().mean())
    maxd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum().max())
    stdd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum().std())
    skewd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum().skew())
    kurtd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum().kurt())
    weekday = x.set_index('PERSONID').CREATETIME.apply(lambda x: x.weekday() >= 5).groupby(
        'PERSONID').mean().rename('weekday')
    l += ['lastdm', 'countd', 'avgd', 'maxd', 'stdd', 'skewd', 'kurtd']

    # 拼接聚合后结果#TODO
    for colname in l:
        col = eval(colname)
        if type(col) is pd.Series:
            dff = dff.join(col.rename(colname))
        else:
            dff = dff.join(col, rsuffix=colname)

    # 针对缺失率补充特征

    # 返回拼接FTR51后最终结果
    return dff.join(deal_FTR51(x)).fillna(0)


# 处理函数 特征选择
def feature_eng():
    pass


# 处理数据并序列化
if True:
    # 读取数据
    test_x = pd.read_csv('../data/input/test_A.tsv', sep='\t')
    train_y = pd.read_csv('../data/input/train_id.tsv', sep='\t')
    train_x = pd.read_csv('../data/input/train.tsv', sep='\t')
    train_y = train_y.set_index('PERSONID').sort_index().LABEL
    train_i = train_x.PERSONID.unique()
    test_i = test_x.PERSONID.unique()
    # x test+train
    all_x = train_x.append(test_x)

    # 数据预处理
    all_x = preprocess(all_x)
    train_x = all_x.loc[train_i, :].sort_index()
    test_x = all_x.loc[test_i, :].sort_index()

    # 观察处理后结果
    train_x.to_csv('../data/output/train_x_1.csv', index=False)
    test_x.to_csv('../data/output/test_x_1.csv', index=False)

    # 序列化
    save_p(train_x, 'train_x_1')
    save_p(test_x, 'test_x_1')
    save_p(train_y, 'train_y_1')

# 反序列化
train_x = read_p('train_x_1')
test_x = read_p('test_x_1')
train_y = read_p('train_y')


#TODO:添加次数新特征
train_x_imp=read_p('train_x_imp')
test_x_imp=read_p('test_x_imp')
train_x = train_x.join(train_x_imp)
test_x = test_x.join(test_x_imp)
#TODO：添加众数新特征

# 特征选择
feature_eng()

# 建模
model_lgb = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 77, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)


if True:

    # TODO：将缺失率大的值(0.9)置为缺失，使用lgb处理,降低1-2个千分位
    if False:
        train_i = train_x.index
        test_i = test_x.index
        all_x = train_x.append(test_x)

        #直为nan
        all_x=set_nan(all_x)

        train_x = all_x.loc[train_i, :].sort_index()
        test_x = all_x.loc[test_i, :].sort_index()

    #观察输出level，and ratio
    if False:
        def describe_more(df):
            tem = len(df)
            var = []
            l = []
            r = []
            for x in df:
                var.append(x)
                l.append(len(pd.value_counts(df[x])))
                r.append(df[df[x] == 0].count()[x] / tem)
            levels = pd.DataFrame({'Variable': var, 'Levels': l, 'ratio': r})
            levels.sort_values(by='Levels', inplace=True)
            return levels


        train = train_x.append(test_x)
        describe_more(train).to_csv('../data/output/4.csv', index=False)

    #得到重要性
    if False:
        def getip(m, x, y):  # 计算重要性并输出
            ip = pd.DataFrame(m.fit(x, y).feature_importances_, index=x.columns)
            return ip
        getip(model_lgb, train_x, train_y).to_csv('../data/output/6.csv')

    #得到相关性
    if False:
        pass

    if True:
        # 评估
        lgb_auc = auc_cv(model_lgb, train_x, train_y)
        print('model_lgb auc : ', lgb_auc)
        print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
        #BASE 0.93104 0.01
        #set_NAN 0.9 0.9314 0.01
        #add countm and countr 0.929
        # 输出结果
        if False:
            model_lgb.fit(train_x, train_y)
            pred_lgb = model_lgb.predict_proba(test_x)[:, 1]
            lgb_res = pd.DataFrame()
            lgb_res['id'] = test_x.index
            lgb_res['score'] = pred_lgb
            lgb_res.to_csv('../data/output/r1.csv', index=None, header=None, sep='\t')