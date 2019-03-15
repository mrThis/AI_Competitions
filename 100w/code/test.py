import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui1\\' + name + '.p')
def auc_cv(model, train_data, label):
    skf = StratifiedKFold(5, shuffle=True, random_state=42).get_n_splits(train_data.values)
    r_auc = cross_val_score(model, train_data.values, label, scoring="roc_auc", cv=skf, verbose=5)  # auc评价
    return (r_auc)



# 反序列化
train_x = read_p('train_x')
test_x = read_p('test_x')
train_y = read_p('train_y')
pd.options.mode.use_inf_as_na = True
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)


# TODO：添加众数新特征

# 建模
from model1_3 import CURmodel
# 不聚合 baseline  mean= 0.931549,std= 0.009323
# 不聚合 增大正类权重.
#1，删除的数据一定不能太多，并且你是随机删除...有必要删除吗...

#聚合 2,0.5 mean= 0.928414,std= 0.007927
# 聚合 3,0.5 mean= 0.928235,std= 0.008523
# 3 0.5 mean= 0.930146,std= 0.006996
# 3 0.8 mean= 0.929719,std= 0.007111
# 聚合 9,0.5 mean= 0.929056,std= 0.007257
# 聚合 12 0.5 mean= 0.929183,std= 0.007918

# 聚合 12，0.6 mean= 0.929556,std= 0.007347
# 聚合 12，0.8 mean=0.929012 ,std =0.006
# 聚合 12, 0.9 mean= 0.928968,std= 0.008367


# 聚合 12，0.8

from model1_2 import CURmodel2
# modeler = CURmodel2(12,0.8)
modeler = CURmodel(2)
# modeler = lgb.LGBMClassifier(
#     boosting_type='gbdt', scale_pos_weight=1423 / 120, num_leaves=64,
#     reg_alpha=0.0, reg_lambda=10,
#     max_depth=-1, n_estimators=750, objective='binary',
#     subsample=0.8, colsample_bytree=0.6,
#     learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
# )
#自评估函数
if False:
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    current_param_auc=[]
    for train_index, test_index in skf.split(train_x.values, train_y):
        print(len(train_index))
        X_train = train_x[train_index]
        X_test = train_x[test_index]
        y_train = train_y[train_index]
        y_test = train_y[test_index]
        modeler.fit(X_train,y_train)
        predictions = modeler.predict_proba(X_test)
        auc = roc_auc_score(y_test, predictions[:, 1])
        current_param_auc.append(auc)
    print('model_lgb auc : ', current_param_auc)
    print('mean= %f,std= %f' % (np.mean(np.array(current_param_auc)),np.std(np.array(current_param_auc))))
if True:
    # 评估
    lgb_auc = auc_cv(modeler, train_x, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
    # BASE 0.9319 0.01

