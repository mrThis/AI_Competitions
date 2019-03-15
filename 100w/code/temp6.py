#给cui特征加强特
import pandas as pd

def handel(data):

    temp = data[['PERSONID', 'FTR51']]
    flag = list(temp['FTR51'].value_counts()[:15].index)
    temp1 = pd.DataFrame(index=data.PERSONID.unique())
    for item in flag:
        temp_select = temp[temp['FTR51'] == item]
        df_count = temp_select.groupby('PERSONID').size().to_frame()
        df_count.columns = ['FTR51' + item + '_count']
        temp1 = temp1.join(df_count).fillna(0)
    return temp1

train = pd.read_csv('../data/input/train.tsv', sep='\t')
test = pd.read_csv('../data/input/test_A.tsv', sep='\t')

yfytrain_imp = handel(train)
yfytest_imp = handel(test)
print(yfytrain_imp.shape)
yfytrain_imp.to_pickle('../data/input/yfytrain_imp.p')
yfytest_imp.to_pickle('../data/input/yfytest_imp.p')
