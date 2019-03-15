import pandas as pd
import numpy as np


def extend(df):
    df_extend = pd.DataFrame(columns=['PERSONID', 'APPLYNO', 'FTR51', 'CREATETIME'])
    i = 0
    for index, row in df.iterrows():
        id = row['PERSONID']
        str = row['FTR51']
        time = row['CREATETIME']
        strs = str.split(',')
        for s in strs:
            list = [id] + [index] + [s] + [time]
            df_extend.loc[i] = list
            i = i + 1
    return df_extend

# x
# index: APPLYNO(重复)
# columns: PERSONID,FTR51,CREATETIME
train = pd.read_csv('../data/input/train.tsv', sep='\t', header=0,nrows=100000)
print(train.shape)
# x = train[['PERSONID', 'FTR51', 'CREATETIME']]
# x = extend(x)
# x.to_csv('../data/output/imp.csv',index=False)



a=train[['APPLYNO','FTR51']].set_index('APPLYNO').FTR51.str.split(',',expand=True).unstack() .dropna().reset_index().drop('level_0',axis=1).rename(columns={0:'FTR51'})
b=train[['PERSONID','APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
c=train[['APPLYNO','CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()

x = pd.merge(a,b,on='APPLYNO')
x = pd.merge(x,c,on='APPLYNO')

x.to_csv('../data/output/imp2.csv',index=False)

print(a.shape)
print(b.shape)
print(c.shape)
print(x.head(5))
print(x.shape)
print(x.isnull().any())