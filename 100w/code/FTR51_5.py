import pandas as pd

# 补充每个用户,买单种药的频率，只取频率排名前5，并加和取一列
# TODO:取前3
# TODO：取前2 不信它没有用系列
# TODO：统计最前1 ，真的tM没用
# freq1，freq2，freq3,freq4,freq5,sum
train = pd.read_csv('../data/output/implement.csv')
test = pd.read_csv('../data/output/implement_test.csv')
train = train[['PERSONID', 'FTR51']].set_index('PERSONID')
test = test[['PERSONID', 'FTR51']].set_index('PERSONID')


# 数据
# 取前几频率
def ftr51_freq(data, n):
    def freq(ser, i):
        t = pd.value_counts(ser)
        if i >= len(t):
            return 0
        return t[i] / len(ser)

    gpb = train.groupby('PERSONID')
    df = pd.DataFrame(index=data.index.unique())
    for i in range(n):
        t = gpb.FTR51.agg(freq, i=i)
        t.rename('FTR51_repeat_freq' + str(i), inplace=True)
        df = df.join(t)
    df['FTR51_repeat_freq_sum'] = df.apply(lambda x: x.sum(), axis=1)
    ##TODO :如果统计大于1 ，注释这一行
    df.drop('FTR51_repeat_freq_sum',axis=1,inplace=True)
    return df


# 得到新的train补充集

train_imp = ftr51_freq(train, 1)
print(train_imp.head(5))
print(train_imp.shape)
test_imp = ftr51_freq(test, 1)
print(test_imp.shape)
train_imp.to_pickle('../data/input/FTR51_10train.p')
test_imp.to_pickle('../data/input/FTR51_10test.p')
