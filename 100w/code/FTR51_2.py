# 关联分析


# s;支持度 出现次数/总次数
# c0,c1;置信度 出现在0or1次数/0or1总次数
# 在一定支持度下，找到置信度满足的magic 特征

# a 分析一个医疗项目与0,1的关系
# b 分析一笔报销id下所有药品与0,1的关系 单纯分析每笔没有太大意义
# c 分析一个病人id下所有药品与0,1的关系 具体到每个病人

import pandas as pd
import orangecontrib.associate.fpgrowth as oaf


def cc0(arr):
    return arr.apply(lambda x: 1 if x == 0 else 0).sum()


def cc1(arr):
    return arr.apply(lambda x: 1 if x == 1 else 0).sum()


test = pd.read_csv('../data/input/train_id.tsv', sep='\t', header=0)
implement = pd.read_csv('../data/output/implement.csv')
a = implement[['PERSONID', 'FTR51']]
a = pd.merge(a, test, on='PERSONID').drop('PERSONID', axis=1)
# s_a,index,FTR51

t = a.groupby('LABEL').count().values
sum0 = t[0][0]
sum1 = t[1][0]
sum = sum0 + sum1
s_a = a.groupby('FTR51').count().LABEL.rename('s_a').apply(lambda x: x / sum)
print(s_a.to_frame())
c_a = a.groupby('FTR51')['LABEL'].agg({'c0_a': cc0, 'c1_a': cc1})
c_a.c0_a = c_a.c0_a.apply(lambda x: x / sum0)
c_a.c1_a = c_a.c1_a.apply(lambda x: x / sum1)
a = s_a.to_frame().join(c_a)
print(a.shape)
a.reset_index().to_csv('../data/output/a.csv', index=False)
a.to_pickle('../data/output/a.p')
