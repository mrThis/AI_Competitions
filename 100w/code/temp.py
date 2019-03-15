import pandas as pd

# 统计高频数出现个数
# 统计剩余频数出现个数
'''
# 添加众数
str1 = 'FTR16,FTR47,FTR30,FTR36,FTR41,FTR28,FTR33,FTR34,FTR35,FTR44,FTR48,FTR0,FTR17,FTR23,FTR42,FTR18,FTR32'

# 50% 000023659591303665
# 50% 0000230510467704935
# 95% 0.05
# 88% 0.55
str2 = 'FTR16,FTR47,FTR35,FTR44,FTR48,FTR0,FTR17,FTR23,FTR42,FTR18,FTR32,FTR34'

# 90% 0
str6 = 'FTR19,FTR11,FTR24,FTR6,FTR3,FTR1,FTR13,FTR26,FTR49,FTR22,FTR15,FTR31,FTR46,FTR37,FTR45,FTR27,FTR40,FTR20,FTR25,FTR14,FTR2,FTR29,FTR8,FTR50'
# 80% 0
str7 = 'FTR10,FTR12,FTR38,FTR21,FTR7,FTR4,FTR39,FTR5'
'''


# 统计频数
def out(str):
    return train[['PERSONID'] + str.split(',')]


def mcount(df):
    return pd.value_counts(df).max()


def rcount(df):
    return len(df) - pd.value_counts(df).max()


def result(train):
    train = train.drop('APPLYNO', axis=1)
    # 得到新特征
    ##统计高频数出现个数
    out2_m = train.groupby('PERSONID').agg(mcount)
    out2_m.rename(columns=lambda x: x + '_mcount', inplace=True)
    ##统计剩余频数出现个数
    out2_r = train.groupby('PERSONID').agg(rcount)
    out2_r.rename(columns=lambda x: x + '_rcount', inplace=True)
    out2 = out2_m.join(out2_r)
    return out2


# 读取数据
test_x = pd.read_csv('../data/input/test_A.tsv', sep='\t')
train_x = pd.read_csv('../data/input/train.tsv', sep='\t')

# 处理数据
train_x = result(train_x)
test_x = result(test_x)
# 输出补充结果
train_x.to_pickle('../data/input/train_x_imp.p')
test_x.to_pickle('../data/input/test_x_imp.p')

train_x.to_csv('../data/output/train_x_imp.csv')
test_x.to_csv('../data/output/test_x_imp.csv')
