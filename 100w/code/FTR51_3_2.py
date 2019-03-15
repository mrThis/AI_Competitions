import pandas as pd


def describe_more(df):
    var = []
    l = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
    levels = pd.DataFrame({'Variable': var, 'Levels': l})
    levels.sort_values(by='Levels', inplace=True)
    return levels

def freq(df):
    return pd.value_counts(df).max() / pd.value_counts(df).sum()

def max_cate(df):
    return pd.value_counts(df).index[0]

def max_count(df):
    return pd.value_counts(df).max()

def cate(arr):
    return len(arr.unique())

def min_cate(df):
    return pd.value_counts(df).index[-1]

def min_count(df):
    return pd.value_counts(df).min()


implement = pd.read_csv('../data/output/implement.csv')
# 单独的药品项目类目
# shape (4079851, 2)
temp = implement[['FTR51']].FTR51.str.split(pat='[ABCDE]', expand=True).drop(0,
                                                                             axis=1).rename(
    columns=lambda x: 'FTR51_' + str(x))
print(temp.head(10))
implement = implement.join(temp)
implement.set_index('PERSONID', inplace=True)
implement.drop('FTR51',inplace=True,axis=1)
print(implement.head(5))
# 对各个医疗项目的编码进行进一步挖掘
# 这里我们假设得到的implement，下的abcde都全是类别，并且其做组合得到了现在的大类别

# 类别数目的count
# 各个类出现最多的类别
# 各个类最多类别的出现次数
# 求最多类别出现次数的频数 没有意义，因为大部分人还是挂号
# 各个类出现最少的类别
# 各个类最少类别的出现次数
#

# TODO:按笔数统计

# TODO:按照时间滑窗
# 同一时间段出现最多的类别，以及出现次数

#根据笔数统计

#先按照PERSONID和APPLYNO聚合
gdp = implement[['FTR51_1', 'FTR51_2', 'FTR51_3', 'FTR51_4', 'FTR51_5']].groupby('PERSONID','APPLYNO')
c_cate = gdp.agg(cate)
freq = gdp.agg(freq)
max1 = gdp.agg(max_count)
max2 = gdp.agg(max_cate)
min1 = gdp.agg(min_count)
min2 = gdp.agg(min_cate)
print(c_cate.head(5))

# 重命名
# c_cate.rename(columns=lambda x: x + 'c_cate', inplace=True)
# freq.rename(columns=lambda x: x + 'freq', inplace=True)
# max1.rename(columns=lambda x: x + 'max1', inplace=True)
# max2.rename(columns=lambda x: x + 'max2', inplace=True)
# min1.rename(columns=lambda x: x + 'min1', inplace=True)
# min2.rename(columns=lambda x: x + 'min2', inplace=True)

# 拼接
# dff = pd.DataFrame(index=implement.index.sort_values().unique())
# dff = dff.join(c_cate).join(max1).join(max2).join(min1).join(min2).join(freq)
#
# 观察
# print(dff.shape)
# print(dff.head(5))

# 输出
# 输出FTR51_3,没有实现上述的
# 3_1新实现了频率
# 3_2新实现了按照笔数统计
# 输出FTR51_4，实现了TODO
# dff.to_pickle('../data/input/FTR51_3_2.p')
#