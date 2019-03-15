import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb


def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui1\\' + name + '.p')

train_x = read_p('train_x')
test_x = read_p('test_x')
train_y = read_p('train_y')
#划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.4, random_state=42)

#建立模型
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 120, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)

#得到auc
#训练
lgb_model.fit(X_train,y_train)
#概率
pred_lgb = lgb_model.predict_proba(X_test)[:, 1]
#反向概率
f_pred_lgb = lgb_model.predict_proba(X_test)[:, 0]
auc = roc_auc_score(y_test,pred_lgb)
f_auc = roc_auc_score(y_test,f_pred_lgb)

print('auc',auc)
print('fauc',f_auc)
print(1-f_auc)
print(auc==(1-f_auc))
print(auc-(1-f_auc))

ff_pred_lgb = 1-pred_lgb
ff_auc = roc_auc_score(y_test,ff_pred_lgb)
print('ffauc',ff_auc)
print(f_auc==ff_auc)
print(auc==(1-ff_auc))
