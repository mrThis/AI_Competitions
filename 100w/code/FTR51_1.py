import pandas as pd


train = pd.read_csv('../data/input/train.tsv', sep='\t')
a=train[['APPLYNO','FTR51']].set_index('APPLYNO').FTR51.str.split(',',expand=True).unstack() .dropna().reset_index().drop('level_0',axis=1).rename(columns={0:'FTR51'})
b=train[['PERSONID','APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
c=train[['APPLYNO','CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()

x = pd.merge(a,b,on='APPLYNO')
x = pd.merge(x,c,on='APPLYNO')

print(len(x.FTR51.unique()))#(20157, 1)
print(x.shape)
x.to_csv('../data/outputb/implement.csv',index=False)