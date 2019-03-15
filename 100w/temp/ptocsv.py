import pandas as pd
from skfeature.function.statistical_based import CFS


def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui2\\' + name + '.p')
train_x = read_p('train_x')
test_x = read_p('test_x')
train_y = read_p('train_y')
FTR33train = read_p('FTR33train')
FTR33test = read_p('FTR33test')
train_x = train_x.join(FTR33train)
test_x = test_x.join(FTR33test)

print(train_y.shape)
train_x.to_csv('train_x.csv')
test_x.to_csv('test_x.csv')
train_y.to_csv('train_y.csv')