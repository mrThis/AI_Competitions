# index PERSONID
# columns  m_sum,m_cate_sum,m_freq1,m_freq2,m_freq3,max1,max2,max3,min1,min2,min3

# m_sum  报销项目总数(可能有重复项目)
# m_cate_sum 报销项目种类数目
# m_freq1 报销项目总数/报销笔数
# m_freq2 单日报销项目数平均值
# m_freq3 单月报销项目数平均值
# m_freq4 单季度报销项目数平均值  #现在提供的
# max1 按每笔报销项目数最大值
# min1 按每笔报销项目数最小值
# max2 按单日报销项目数最大值
# min2 按单日报销项目数最小值
# max3 按单月报销项目数最大值
# min3 按单月报销项目数最小值
# max4 按季度报销项目数最大值
# min4 按季度报销项目数最小值
# #TODO：并没有对重复药项目出现次数进行分析，freq取第一高是不够的，应该取第二高
# question:
# 分布.数据量比较少,难以实现.
# 存不存在单日报销重复项目
import pandas as pd
import numpy as np


def quarter(x):
    if x.month in [2, 4, 10]:
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


# columns: PERSONID,APPLYNO,FTR51,CREATETIME
x = pd.read_csv('../data/output/implement.csv')
gpb = x.groupby('PERSONID')
# m_sum
m_sum = gpb['FTR51'].count().rename('m_sum')
# m_cate_sum
m_cate_sum = gpb['FTR51'].agg(cate).rename('m_cate_sum')
# m_freq1,max1,min1
m1 = gpb['APPLYNO'].agg({'m_freq1': mean, 'max1': max, 'min1': min})
# m_freq2,max2,min2
m2 = gpb['CREATETIME'].agg({'m_freq2': mean, 'max2': max, 'min2': min})
# m_freq3,max3,min3
x['MONTH'] = pd.to_datetime(x.CREATETIME).apply(lambda x: x.month)
m3 = gpb['MONTH'].agg({'m_freq3': mean, 'max3': max, 'min3': min})
# m_freq4,max4,max4
x['Quarter'] = pd.to_datetime(x.CREATETIME).apply(quarter)
m4 = gpb['Quarter'].agg({'m_freq4': mean, 'max4': max, 'min4': min})
m = m1.join(m2).join(m3).join(m4)

m.to_csv('../data/output/del.csv')
