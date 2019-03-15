#改变环境，数据后的测试，跟上团队baseline 7-27

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import roc_auc_score
from model1_3 import CURmodel
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
FTR33train_new = read_p('FTR33train_new')
FTR33test_new = read_p('FTR33test_new')
#cui-1 baseline mean= 0.934163,std= 0.009121
train_x = train_x.join(FTR33train)
test_x = test_x.join(FTR33test)

#加入cui2
cols_to_use = list(set(FTR33train_new.columns)-set(FTR33train.columns))

train_x.join(FTR33train_new[cols_to_use])
test_x.join(FTR33test_new[cols_to_use])

#cui-2 baseline

#放入玄学特征 mean= 0.934895,std= 0.009963 #实际线上降了，原因不明
if False:
    FTR51train = read_p('FTR51train')
    FTR51test = read_p('FTR51test')
    train_x = train_x.join(FTR51train)
    test_x = test_x.join(FTR51test)

#放入置信度，和高支持度特征 : mean= 0.933430,std= 0.009223
if False:
    FTR51_2train = read_p('FTR51_2train')
    FTR51_2test = read_p('FTR51_2test')
    train_x = train_x.join(FTR51_2train)
    test_x = test_x.join(FTR51_2test)


#放入少数类中高支持度特征 : mean= 0.933355,std= 0.009595
if False:
    FTR51_3train = read_p('FTR51_3train')
    FTR51_3test = read_p('FTR51_3test')
    train_x = train_x.join(FTR51_3train)
    test_x = test_x.join(FTR51_3test)

#放入频数特征: mean= 0.933146,std= 0.009076
if False:
    FTR51_4train = read_p('FTR51_4train')
    FTR51_4test = read_p('FTR51_4test')
    train_x = train_x.join(FTR51_4train)
    test_x = test_x.join(FTR51_4test)

#pindata>0.01 mean= 0.933986,std= 0.008867
if False:
    FTR51_5train = read_p('FTR51_5train')
    FTR51_5test = read_p('FTR51_5test')
    train_x = train_x.join(FTR51_5train)
    test_x = test_x.join(FTR51_5test)

#pindata>0.005 mean= 0.933496,std= 0.008954
if False:
    FTR51_5train = read_p('FTR51_6train')
    FTR51_5test = read_p('FTR51_6test')
    train_x = train_x.join(FTR51_5train)
    test_x = test_x.join(FTR51_5test)
#pindata >0.001
if False:
    FTR51_5train = read_p('FTR51_7train')
    FTR51_5test = read_p('FTR51_7test')
    train_x = train_x.join(FTR51_5train)
    test_x = test_x.join(FTR51_5test)

#频率前3的，和它的sum
#频率前2的，和它的sum mean= 0.934001,std= 0.008832
if False:
    FTR51train = read_p('FTR51_10train')
    FTR51test = read_p('FTR51_10test')
    train_x = train_x.join(FTR51train)
    test_x = test_x.join(FTR51test)

#纵向计数新特征 mean= 0.933912,std= 0.008763
if False:
    train_imp_new = read_p('train_imp_new')
    test_imp_new = read_p('test_imp_new')
    train_x = train_x.join(train_imp_new)
    test_x = test_x.join(test_imp_new)


#选择前75 mean= 0.938478,std= 0.008458
#调整深度为1000 mean= 0.938548,std= 0.008409
#前100 mean= 0.939976,std= 0.006315
#选择前738 mean= 0.933779,std= 0.009096


#cui new feature mean= 0.932638,std= 0.009068
#100  mean= 0.938689,std= 0.007005]
if False:
    f1 = pd.read_csv('../data/output/importance_new_1.csv')
    f1.sort_values(by='importance', ascending=False, inplace=True)
    feature = list(f1.feature)[:75]  # 75
    train_x = train_x[feature]
    test_x = test_x[feature]


modeler=lgb.LGBMClassifier(
        boosting_type='gbdt', is_unbalance=True, num_leaves=64,
        reg_alpha=0.0, reg_lambda=10,
        max_depth=-1, n_estimators=800, objective='binary',
        subsample=0.8, colsample_bytree=0.6,
        learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1,verbose=-1)
#重要性


if False:
    def getip(m, x, y):  # 计算重要性并输出
        ip = pd.DataFrame(m.fit(x, y).feature_importances_, index=x.columns)
        return ip
    getip(modeler, train_x, train_y).to_csv('../data/output/importance_new_1.csv')



#cui1,baseline
#jary1
#jary2
#jary3

if True:
    # 评估
    lgb_auc = auc_cv(modeler, train_x.values, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))


#输出结果
# modeler.fit(train_x.values, train_y)
# pred_lgb = modeler.predict_proba(test_x.values)[:, 1]
# lgb_res = pd.DataFrame()
# lgb_res['id'] = test_x.index
# lgb_res['score'] = pred_lgb
# lgb_res.to_csv('../data/output/result_new_1.csv', index=None, header=None, sep='\t')