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


# 处理函数
# 将0设置为缺失值

if True:  # TODO:del it
    def set_nan(df):
        tem = len(df)
        for x in df:
            ratio = df[df[x] == 0].count()[x] / tem
            # 1 0.93102
            # 0.9降低一个千分位
            if ratio > 0.8:
                df[x] = df[x].agg(lambda x: np.nan if x == 0 else x)
        return df


# 根据缺失率 删除特征
def delete_r(df):
    l = []
    tem = len(df)
    for x in df:
        ratio = df[df[x] == 0].count()[x] / tem
        if ratio >= 0.999:
            l = l + [x]
    return l


# 根据相关性

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
        gby = x.groupby('PERSONID')
        m_sum = gby['FTR51'].count().rename('FTR51m_sum')
        # m_cate_sum
        m_cate_sum = gby['FTR51'].agg(cate).rename('FTR51m_cate_sum')
        # m_freq1,max1,min1
        mj1 = gby['APPLYNO'].agg({'FTR51m_freq1': mean, 'FTR51max1': max, 'FTR51min1': min})
        # m_freq2,max2,min2
        mj2 = gby['CREATETIME'].agg({'FTR51m_freq2': mean, 'FTR51max2': max, 'FTR51min2': min})
        # m_freq3,max3,min3
        x['MONTH'] = pd.to_datetime(x.CREATETIME).apply(lambda x: x.month)
        mj3 = gby['MONTH'].agg({'FTR51m_freq3': mean, 'FTR51max3': max, 'FTR51min3': min})
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

    def comb_feature(data):
        corr = data.corr()
        d = []
        for x in corr:
            print('round' + str(x) + '-------------------------------------')
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
                    c1.fillna(0)
                    c2 = data[dde] - data[x]
                    c2.name = dde + '-' + x
                    c2.fillna(0)
                    print(c1)
                    print(c2)
                    data = data.join(c1).join(c2)
            # 将融合过的特征加入不再融合列
            print(de)
            d = d + de
        print(len(d))
        return data
    # 初始化返回值
    dff = pd.DataFrame(index=x.PERSONID.sort_values().unique())
    l = []

    # 去掉完全重复数据
    x = x.drop(['FTR20'], axis=1)

    # TODO：将0 置为nan
    x = comb_feature(x)

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
    gdp = x.groupby('PERSONID')
    counts = gdp['APPLYNO'].count().rename('count')
    lasts = gdp.last().drop(['CREATETIME', 'FTR51', 'APPLYNO'], axis=1)
    avg = gdp.mean()
    max = gdp.max().drop(['CREATETIME', 'FTR51', 'APPLYNO'], axis=1)
    std = gdp.std()
    skew = gdp.apply(lambda x: x.skew())
    kurt = gdp.apply(lambda x: x.kurt())
    sum = gdp.sum()
    mode = gdp.apply(lambda s: s.mode().iloc[0])
    l += ['countadj', 'incresing', 'counts', 'lasts', 'avg', 'max', 'std', 'skew', 'kurt', 'sum', 'mode']

    lastdm = (pd.to_datetime('2016-03-01') - pd.to_datetime(x.groupby('PERSONID').CREATETIME.last())) \
        .apply(lambda x: x.days).rename('lastdm')
    countd = x.groupby('PERSONID').apply(lambda k: k.CREATETIME.nunique()).rename('countdate')

    # TODO：？
    gdpd = x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum()).groupby('PERSONID')
    avgd = gdpd.apply(lambda k: k.mean())
    maxd = gdpd.apply(lambda k: k.max())
    stdd = gdpd.apply(lambda k: k.std())
    skewd = gdpd.apply(lambda k: k.skew())
    kurtd = gdpd.apply(lambda k: k.kurt())
    l += ['lastdm', 'countd', 'avgd', 'maxd', 'stdd', 'skewd', 'kurtd']
    # TODO：？
    # 新想法
    weekmean = pd.to_datetime(x.set_index('PERSONID').CREATETIME).groupby('PERSONID').apply(
        lambda t: (t.dt.week + 1).mean())
    year2015 = pd.to_datetime(x.set_index('PERSONID').CREATETIME).apply(lambda x: 1 if x.year == 2015 else 0).groupby(
        'PERSONID').mean().rename('year2015')
    weekday = pd.get_dummies(pd.to_datetime(x.set_index('PERSONID').CREATETIME).apply(lambda x: x.weekday())).groupby(
        'PERSONID').mean().rename(columns={i: 'weekday' + str(i) for i in range(7)})
    l += ['weekmean', 'year2015', 'weekday']

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
    test_x = pd.read_csv('../data/input/test_A.tsv', sep='\t',nrows=1000)
    train_y = pd.read_csv('../data/input/train_id.tsv', sep='\t')
    train_x = pd.read_csv('../data/input/train.tsv', sep='\t',nrows=1000)
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
# train_x = read_p('train_x_1')
# test_x = read_p('test_x_1')
# train_y = read_p('train_y_1')
# train_x.drop(['APPLYNO', 'CREATETIME', 'FTR51mode'], axis=1, inplace=True)
# test_x.drop(['APPLYNO', 'CREATETIME', 'FTR51mode'], axis=1, inplace=True)
# # TODO:添加次数新特征
# train_x_imp = read_p('train_x_imp')
# test_x_imp = read_p('test_x_imp')
# train_x = train_x.join(train_x_imp)
# test_x = test_x.join(test_x_imp)
# pd.options.mode.use_inf_as_na = True
# # 特征选择
# feature_eng()



# 建模
model_lgb = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 77, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)

# 评估及数据分析
if False:
    # 处理一下数据
    ##删掉缺失率大的
    ##删掉相关性大的
    if False:
        # 缺失率大于0.99的特征
        '''
        FTR51,FTR19skew,FTR19kurt,FTR19,FTR11std,FTR11skew,FTR11kurt,FTR19sum,FTR19stdd,FTR19std,FTR19skewd,FTR19mode,FTR19maxd,FTR19max,FTR19lasts,FTR19kurtd,FTR19avgd,FTR19avg,FTR19_rcount,FTR9mode,FTR43mode,FTR24skew,FTR24kurt,FTR26kurt,FTR24std,FTR24,FTR26skew,FTR13sum,FTR13stdd,FTR13std,FTR13skewd,FTR13skew,FTR13mode,FTR13maxd,FTR13max,FTR13lasts,FTR13kurtd,FTR13kurt,FTR13avgd,FTR13avg,FTR13_rcount,FTR13,FTR26std,FTR37-FTR46kurt,FTR11,FTR6skew,FTR6std,FTR6kurt,FTR6,FTR22sum,FTR22stdd,FTR22std,FTR22skewd,FTR22skew,FTR22mode,FTR22maxd,FTR22max,FTR22lasts,FTR22kurtd,FTR22kurt,FTR22avgd,FTR22avg,FTR22_rcount,FTR22,FTR6skewd,FTR24sum,FTR24stdd,FTR24skewd,FTR24mode,FTR24maxd,FTR24max,FTR24lasts,FTR24kurtd,FTR24avgd,FTR24avg,FTR24_rcount,FTR6stdd,FTR6kurtd,FTR6sum,FTR6mode,FTR6maxd,FTR6max,FTR6lasts,FTR6avgd,FTR6avg,FTR6_rcount,FTR37-FTR46maxd,FTR3kurt,FTR1kurt,FTR7-FTR4kurtd,FTR11sum,FTR11stdd,FTR11skewd,FTR11mode,FTR11maxd,FTR11max,FTR11lasts,FTR11kurtd,FTR11avgd,FTR11avg,FTR11_rcount,FTR37-FTR46skew,FTR3skew,FTR1skew,FTR49,FTR3std,FTR1std,FTR7-FTR4skewd,FTR26,FTR7-FTR4skew,FTR43-FTR9mode,FTR3,FTR1,FTR49kurt,FTR49skew,FTR31kurt,FTR7-FTR4kurt,FTR26sum,FTR26stdd,FTR26skewd,FTR26mode,FTR26maxd,FTR26max,FTR26lasts,FTR26kurtd,FTR26avgd,FTR26avg,FTR26_rcount,FTR49std,FTR3sum,FTR3stdd,FTR3skewd,FTR3mode,FTR3maxd,FTR3max,FTR3lasts,FTR3kurtd,FTR3avgd,FTR3avg,FTR3_rcount,FTR49sum,FTR49stdd,FTR49skewd,FTR49mode,FTR49maxd,FTR49max,FTR49lasts,FTR49kurtd,FTR49avgd,FTR49avg,FTR49_rcount,FTR1sum,FTR1stdd,FTR1skewd,FTR1mode,FTR1maxd,FTR1max,FTR1lasts,FTR1kurtd,FTR1avgd,FTR1avg,FTR1_rcount,FTR31skew,FTR37-FTR46kurtd,FTR15kurt
        '''
        str = 'FTR51,FTR19skew,FTR19kurt,FTR19,FTR11std,FTR11skew,FTR11kurt,FTR19sum,FTR19stdd,FTR19std,FTR19skewd,FTR19mode,FTR19maxd,FTR19max,FTR19lasts,FTR19kurtd,FTR19avgd,FTR19avg,FTR19_rcount,FTR9mode,FTR43mode,FTR24skew,FTR24kurt,FTR26kurt,FTR24std,FTR24,FTR26skew,FTR13sum,FTR13stdd,FTR13std,FTR13skewd,FTR13skew,FTR13mode,FTR13maxd,FTR13max,FTR13lasts,FTR13kurtd,FTR13kurt,FTR13avgd,FTR13avg,FTR13_rcount,FTR13,FTR26std,FTR37-FTR46kurt,FTR11,FTR6skew,FTR6std,FTR6kurt,FTR6,FTR22sum,FTR22stdd,FTR22std,FTR22skewd,FTR22skew,FTR22mode,FTR22maxd,FTR22max,FTR22lasts,FTR22kurtd,FTR22kurt,FTR22avgd,FTR22avg,FTR22_rcount,FTR22,FTR6skewd,FTR24sum,FTR24stdd,FTR24skewd,FTR24mode,FTR24maxd,FTR24max,FTR24lasts,FTR24kurtd,FTR24avgd,FTR24avg,FTR24_rcount,FTR6stdd,FTR6kurtd,FTR6sum,FTR6mode,FTR6maxd,FTR6max,FTR6lasts,FTR6avgd,FTR6avg,FTR6_rcount,FTR37-FTR46maxd,FTR3kurt,FTR1kurt,FTR7-FTR4kurtd,FTR11sum,FTR11stdd,FTR11skewd,FTR11mode,FTR11maxd,FTR11max,FTR11lasts,FTR11kurtd,FTR11avgd,FTR11avg,FTR11_rcount,FTR37-FTR46skew,FTR3skew,FTR1skew,FTR49,FTR3std,FTR1std,FTR7-FTR4skewd,FTR26,FTR7-FTR4skew,FTR43-FTR9mode,FTR3,FTR1,FTR49kurt,FTR49skew,FTR31kurt,FTR7-FTR4kurt,FTR26sum,FTR26stdd,FTR26skewd,FTR26mode,FTR26maxd,FTR26max,FTR26lasts,FTR26kurtd,FTR26avgd,FTR26avg,FTR26_rcount,FTR49std,FTR3sum,FTR3stdd,FTR3skewd,FTR3mode,FTR3maxd,FTR3max,FTR3lasts,FTR3kurtd,FTR3avgd,FTR3avg,FTR3_rcount,FTR49sum,FTR49stdd,FTR49skewd,FTR49mode,FTR49maxd,FTR49max,FTR49lasts,FTR49kurtd,FTR49avgd,FTR49avg,FTR49_rcount,FTR1sum,FTR1stdd,FTR1skewd,FTR1mode,FTR1maxd,FTR1max,FTR1lasts,FTR1kurtd,FTR1avgd,FTR1avg,FTR1_rcount,FTR31skew,FTR37-FTR46kurtd,FTR15kurt'
        lr = str.split(',')
        train_x.drop(lr, axis=1, inplace=True)
        test_x.drop(lr, axis=1, inplace=True)
    if False:
        train_i = train_x.index
        test_i = test_x.index
        all_x = train_x.append(test_x)
        all_x = all_x.drop(delete_r(all_x), axis=1)
        train_x = all_x.loc[train_i, :].sort_index()
        test_x = all_x.loc[test_i, :].sort_index()

    # 观察输出level，and ratio
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
        describe_more(train).to_csv('../data/output/base_info.csv', index=False)

    # 得到重要性
    if False:
        def getip(m, x, y):  # 计算重要性并输出
            ip = pd.DataFrame(m.fit(x, y).feature_importances_, index=x.columns)
            return ip


        getip(model_lgb, train_x, train_y).to_csv('../data/output/importance.csv')

    # 得到相关性
    if False:
        data = pd.concat([train_x, test_x], axis=0, ignore_index=True)
        data.corr().to_csv('../data/output/corr.csv')
    # 评估
    if True:
        # 评估
        lgb_auc = auc_cv(model_lgb, train_x, train_y)
        print('model_lgb auc : ', lgb_auc)
        print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
        # BASE 0.93104 0.01
        # set_NAN 0.9 0.9314 0.01
        # add countm 0.929
        # new feature
        # 输出结果
        if False:
            model_lgb.fit(train_x, train_y)
            pred_lgb = model_lgb.predict_proba(test_x)[:, 1]
            lgb_res = pd.DataFrame()
            lgb_res['id'] = test_x.index
            lgb_res['score'] = pred_lgb
            lgb_res.to_csv('../data/output/r1.csv', index=None, header=None, sep='\t')

