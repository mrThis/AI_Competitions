import pandas as pd
from sklearn.model_selection import StratifiedKFold

X = pd.read_csv('../data/input/train.tsv', sep='\t')
y = pd.read_pickle('../data/input/train_y.p')
X.set_index('PERSONID',inplace=True)
all = X.join(y)

tt = all[['FTR50','FTR5','FTR14','FTR43','FTR8','FTR9','LABEL']]
tt.to_excel('temp.xlsx')

