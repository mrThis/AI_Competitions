import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
import lightgbm as lgb


def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui4\\' + name + '.p')


# yfy的数据
c_train=read_p('c_train')
c_test=read_p('c_test')

f33_166_train=read_p( 'f33_166_train')
f33_166_test=read_p( 'f33_166_test')

f33_166_fill_train=read_p( 'f33_166_fill_train')
f33_166_fill_test=read_p( 'f33_166_fill_test')



if False:
    # 读取当前1700特征-cui's 是否住院特征
    train = pd.read_csv('C:/Users/jary_/Desktop/yfy/train.csv')

    train.set_index('PERSONID', inplace=True)

    train_y = train['LABEL']
    train_x = train.drop(['id', 'LABEL'], axis=1)
    pd.options.mode.use_inf_as_na = True
    train_x = train_x.fillna(0)
    print(train_x.shape)
modeler = lgb.LGBMClassifier(
    boosting_type='gbdt', is_unbalance=True, objective='binary',
    n_jobs=-1, verbose=-1)

rfecv = RFECV(estimator=modeler, step=5, cv=5, scoring='roc_auc')  # 5-fold cross-validation
rfecv = rfecv.fit(train_x.values, train_y)
print('Optimal number of features :', rfecv.n_features_)
print('Best features :', train_x.columns[rfecv.support_])
tt =  train_x.columns[rfecv.support_].to_frame()
tt.to_csv('t.csv',index=False)

import matplotlib.pyplot as plt

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
