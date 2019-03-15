# out第一弹

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import roc_auc_score
from model1 import CURmodel
import lightgbm as lgb
def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui2\\' + name + '.p')
def auc_cv(model, train_data, label):
    skf = StratifiedKFold(5, shuffle=True, random_state=42).get_n_splits(train_data)
    r_auc = cross_val_score(model, train_data, label, scoring="roc_auc", cv=skf, verbose=5)  # auc评价
    return (r_auc)



# 反序列化
train_x = read_p('train_x')
test_x = read_p('test_x')
train_y = read_p('train_y')
FTR33train = read_p('FTR33train')
FTR33test = read_p('FTR33test')
train_x = train_x.join(FTR33train)
test_x = test_x.join(FTR33test)


#增加频数新特征
###########################################
train_x_imp = read_p('train_x_imp_2')
test_x_imp = read_p('train_x_imp_2')
train_x = train_x.join(train_x_imp)
test_x = test_x.join(test_x_imp)


modeler = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 77, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)

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
    lgb_auc = auc_cv(modeler, train_x.values, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
    # BASE 0.9319 0.01


#输出结果

# modeler.fit(train_x, train_y)
# pred_lgb = modeler.predict_proba(test_x)[:, 1]
# lgb_res = pd.DataFrame()
# lgb_res['id'] = test_x.index
# lgb_res['score'] = pred_lgb
# lgb_res.to_csv('../data/output/result3_3.csv', index=None, header=None, sep='\t')
#result 1是代表最好结果，线下 0.9339 0.00837
#result 3_1是加了我频数特征后的结果，线下 mean= 0.933608,std= 0.009698
#result 3_2是删去一部分特征（cui）后的结果，mean= 0.933665,std= 0.009675 # 删特征是有帮助的，但如果明天结果没超过清华，也没有意义
#result 3_3是我自选的一部分特征频数后的结果，mean= 0.933399,std= 0.009471
#result 2是代表一般结果（我的CURBOOST）结果降低，两个千分位.
#result 3理论最强,加上了该加的 mean= 0.935943,std= 0.008829
#result 4理论应该更强，但是新加特征可能降分,感觉线上表现会优于result3






