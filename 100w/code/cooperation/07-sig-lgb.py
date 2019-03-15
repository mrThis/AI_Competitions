# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 21:06:31 2018

@author: Thinkpad
"""

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


# 定义通用函数
def gr_agg(df, by_name, col_name, *functions):
    gr = df.groupby(by_name)
    mapper = lambda x: col_name + '_' + x if x != by_name else by_name  # col_name_sum x是函数名称（sum、mean……）
    return gr[col_name].agg(functions).reset_index().rename(columns=mapper)


train = pd.read_csv('D:/kesci/biaoxian/data/train.tsv', sep='\t')
train_target = pd.read_csv('D:/kesci/biaoxian/data/train_id.tsv', sep='\t')
test = pd.read_csv('D:/kesci/biaoxian/data/test_A.tsv', sep='\t')

# train['FTR52']=train[['FTR'+str(i) for i in range(52)]].sum(axis=1)
# test['FTR52']=test[['FTR'+str(i) for i in range(52)]].sum(axis=1)

train = train.drop(['APPLYNO', 'FTR6', 'FTR9', 'FTR22'], axis=1)
test = test.drop(['APPLYNO', 'FTR6', 'FTR9', 'FTR22'], axis=1)

data = pd.concat([train, test], axis=0, ignore_index=True)
data['CREATETIME'] = pd.to_datetime(data['CREATETIME'])
#新增年月日周及周末特征
data["year"] = data["CREATETIME"].dt.year
data['month'] = data["CREATETIME"].dt.month
data['day'] = data["CREATETIME"].dt.day
data['weekday'] = data["CREATETIME"].dt.weekday + 1
data['week'] = data["CREATETIME"].dt.week
data['is_weekend'] = data['weekday'].apply(lambda x: 1 if x in (6, 7) else 0)
data['FTR51_len'] = data['FTR51'].apply(lambda x: len(x.split(',')))

feature = [item for item in train.columns.tolist() if item not in ['PERSONID', 'FTR51', 'CREATETIME']]

v = gr_agg(data, 'PERSONID', 'PERSONID', 'count')
#根据personid聚合所有数据，取具有统计意义数据
##将week，weekday，isweekend等也取了该统计意义特征
###对于类别型来说平均值是不是不如众数有统计意义
for item in feature:
    temp = gr_agg(data, 'PERSONID', item, 'sum', 'mean', 'max', 'std', 'skew', 'min')
    # temp[item+'_range']=temp[item+'_max']-temp[item+'_min']
    v = pd.merge(v, temp, on='PERSONID', how='left')

data_day = gr_agg(data, 'PERSONID', 'day', 'mean', 'max', 'min')
data_week = gr_agg(data, 'PERSONID', 'week', 'mean', 'max', 'min')
data_is_weekend = gr_agg(data, 'PERSONID', 'is_weekend', 'sum')

data_FTR = gr_agg(data, 'PERSONID', 'FTR51', 'nunique')
data_FTR1 = gr_agg(data, 'PERSONID', 'FTR51_len', 'sum', 'mean', 'max', 'min', 'std', 'skew')

#在年月周求和
df_temp = data[['PERSONID', 'year', 'month', 'weekday']].copy()
df_temp = pd.get_dummies(df_temp, columns=['year', 'month', 'weekday'])
df_temp = df_temp.groupby('PERSONID').sum().reset_index()

#将FTR51前频数15做特征
temp = data[['PERSONID', 'FTR51']].copy()
flag = list(temp['FTR51'].value_counts()[:15].index)
temp1 = pd.DataFrame()
temp1['PERSONID'] = v['PERSONID']
for item in flag:
    temp_select = temp[temp['FTR51'] == item]
    df_count = pd.DataFrame(temp_select.groupby('PERSONID').size()).reset_index()
    df_count.columns = ['PERSONID'] + [item + '_count']
    temp1 = pd.merge(temp1, df_count, on='PERSONID', how='left').fillna(0)

for df in [data_day, data_week, data_is_weekend, data_FTR, data_FTR1, df_temp, temp1]:
    v = pd.merge(v, df, 'left', 'PERSONID')

#删除变量
del data_day, data_week, data_is_weekend, data_FTR, df_temp, temp1
del df, df_count, feature, flag, item, temp, temp_select



feature = [item for item in list(v.columns) if item not in ['PERSONID']]

# 求标准差>0的特征
v_feature_std = [v[item].std() for item in feature]
num_feature_std = pd.DataFrame()
num_feature_std['num_feature_name'] = list(feature)
num_feature_std['std'] = v_feature_std
num_feature_std = num_feature_std.sort_values(by='std', ascending=False)

feature = list(num_feature_std.loc[num_feature_std['std'] > 0, 'num_feature_name'])

v = v[['PERSONID'] + feature]
v = v.fillna(0)
del num_feature_std, v_feature_std

train = pd.merge(train_target, v, on='PERSONID', how='left')

train_id = train['PERSONID']
y = train['LABEL']
del train['PERSONID'], train['LABEL']

test1 = pd.DataFrame()
test1['PERSONID'] = list(test['PERSONID'].unique())

test = pd.merge(test1, v, on='PERSONID', how='left')
test_id = test['PERSONID']
del test['PERSONID'], test1

print(train.shape)
print(test.shape)

n_folds = 5


def auc_cv(model, train_data, label):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_data.values)
    r_auc = cross_val_score(model, train_data, label, scoring="roc_auc", cv=kf, verbose=5)  # auc评价
    return (r_auc)


model_lgb = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 77, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)
'''
lgb_auc = auc_cv(model_lgb,train,y)
print('model_lgb auc : ',lgb_auc)
print('model_lgb auc : ',lgb_auc.mean())

'''
model_lgb.fit(train, y)
pred_lgb = model_lgb.predict_proba(test)[:, 1]
lgb_res = pd.DataFrame()
lgb_res['id'] = test_id
lgb_res['score'] = pred_lgb
lgb_res.to_csv('D:/kesci/biaoxian/result/lgb_07171.csv', index=None, header=None, sep='\t')
