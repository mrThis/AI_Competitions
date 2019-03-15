# 无法查得真实意义,已询问医保人员
# 40w数据量，4w重复
# 序列固定,a b c e d
# 每个值处理成5个特征，若一行有多个字符串，先处理为一对多
# 聚合方式,更具level决定

import pandas as pd


# 扩展为唯一id，对应唯一字符串
def extend(df):
    df_extend = pd.DataFrame(columns=['PERSONID', 'FTR51_a', 'FTR51_b', 'FTR51_c', 'FTR51_e', 'FTR51_d'])
    i = 0
    for index, row in df.iterrows():
        id = row['PERSONID']
        str = row['FTR51']
        strs = str.split(',')
        for s in strs:
            list = [id] + discrete(s)
            print(list)
            df_extend.loc[i] = list
            i = i + 1
    return df_extend


# 处理字符串
def discrete(str):
    import re
    return list(map(lambda x: int(x), re.split('[ABCDE]', str)[1:]))




train = pd.read_csv('../data/input/F_train.tsv', sep='\t', header=0)
FTR51 = pd.DataFrame({'PERSONID': train.PERSONID, 'FTR51': train.FTR51})
FTR51 = extend(FTR51)
FTR51.to_csv('../data/output/F_FTR51.csv', index=False)



