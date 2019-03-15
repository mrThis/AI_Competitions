# 删除相关性高的0.9
# 跑完数据后的相关性
# 相关性绝对扑街
# TODO：这个绝对有用

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
import lightgbm as lgb

pd.options.mode.use_inf_as_na = True


def read_p(name):
    return pd.read_pickle('../data/input/' + name + '.p')


def save_p(var, name):
    var.to_pickle('../data/input/' + name + '.p')


def auc_cv(model, train_data, label):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train_data)
    r_auc = cross_val_score(model, train_data, label, scoring="roc_auc", cv=kf, verbose=5)  # auc评价
    return (r_auc)


# 反序列化
train_x = read_p('train_x')
test_x = read_p('test_x')
train_y = read_p('train_y')
train_x.drop(['APPLYNO', 'CREATETIME', 'FTR51mode'], axis=1, inplace=True)
test_x.drop(['APPLYNO', 'CREATETIME', 'FTR51mode'], axis=1, inplace=True)
# TODO:添加次数新特征
train_x_imp = read_p('train_x_imp')
test_x_imp = read_p('test_x_imp')
train_x = train_x.join(train_x_imp)
test_x = test_x.join(test_x_imp)

all_x = train_x.append(test_x)
#
#
def select_feature(corr):
    sf = []
    d = []
    for x in corr:
        if x in d or corr[x].dropna().empty:
            continue
        sf = sf + [x]
        #0.9 28
        #0.95 29
        #0.99 311
        #0.999 （861）
        de = list(corr[corr[x] > 0.999].index)
        print('round' + str(x) + '-------------------------------------')
        print(de)
        d = d+de
    return sf
#
#
sf = select_feature(all_x.corr())
print(len(sf))
train_x = train_x[sf]
test_x = test_x[sf]

model_lgb = lgb.LGBMClassifier(
    boosting_type='gbdt', scale_pos_weight=1423 / 77, num_leaves=64,
    reg_alpha=0.0, reg_lambda=10,
    max_depth=-1, n_estimators=750, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
)

lgb_auc = auc_cv(model_lgb, train_x, train_y)
print('model_lgb auc : ', lgb_auc)
print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
