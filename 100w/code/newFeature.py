import pandas as pd
# 增加纵向非0计数

# 具体到每笔，统计后，取最大值
# 具体到个人，统计后，取和
# 具体到每天，统计后，取最大值
#TODO：没有意义,确实： 具体到每月，统计后，取最大值


# 滑窗，cui..

#读取数据
train = pd.read_csv('../data/input/train.tsv', sep='\t')
test =  pd.read_csv('../data/input/test_A.tsv', sep='\t')
train.set_index('PERSONID',inplace=True)
test.set_index('PERSONID',inplace=True)

train.drop(['APPLYNO','CREATETIME','FTR51'],axis=1,inplace=True)
test.drop(['APPLYNO','CREATETIME','FTR51'],axis=1,inplace=True)
#处理数据
def handle(data):
    def count(ser):
        return ser[ser == 0].count()
    dff = pd.DataFrame(index=data.index.unique())
    # data['CREATETIME'] = pd.to_datetime(data['CREATETIME'])
    # data['day'] = data["CREATETIME"].dt.day
    gby = data.groupby('PERSONID')
    data['new_count'] = data.apply(count,axis=1)
    t1 = gby['new_count'].sum()
    t1.rename('new_count_personid',inplace=True)
    t2 = gby['new_count'].max()
    t2.rename('new_count_applyno', inplace=True)
    return dff.join(t1).join(t2)


train_imp_new = handle(train)
test_imp_new = handle(test)

train_imp_new.to_pickle('../data/input/train_imp_new.p')
test_imp_new.to_pickle('../data/input/test_imp_new.p')