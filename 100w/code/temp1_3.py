import pandas as pd

# 统计高频数出现个数
# 统计剩余频数出现个数



# 统计频数
def out(str):
    return train[['PERSONID'] + str.split(',')]


def mcount(df):
    return pd.value_counts(df).max()


def rcount(df):
    return len(df) - pd.value_counts(df).max()


def result(train):
    train = train[['PERSONID','FTR16', 'FTR47', 'FTR34', 'FTR35', 'FTR18', 'FTR44', 'FTR0','FTR48','FTR42','FTR17','FTR23','FTR32','FTR5']]
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
train_x.to_pickle('../data/input/train_x_imp_2.p')
test_x.to_pickle('../data/input/test_x_imp_2.p')

