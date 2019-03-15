import pandas as pd

train = pd.read_csv('C:/Users/jary_/Desktop/yfy/train.csv')
test = pd.read_csv('C:/Users/jary_/Desktop/yfy/test.csv')
imp = pd.read_csv('C:/Users/jary_/Desktop/yfy/imp.csv')
train.set_index('PERSONID', inplace=True)
test.set_index('PERSONID', inplace=True)
feature = list(imp.name)[:75]
train = train[feature]
test = test[feature]


print(train.shape)
print(test.shape)
train.to_pickle(r'C:\Users\jary_\Desktop\cui3\yfy75train.p')
test.to_pickle(r'C:\Users\jary_\Desktop\cui3\yfy75test.p')