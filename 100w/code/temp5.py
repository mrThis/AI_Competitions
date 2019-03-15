# 特征融合，只做相关性高的 差和商


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
train_i = train_x.index
test_i = test_x.index
all_x = train_x.append(test_x)


#
# 高相关度特征融合
def comb_feature(data):
    corr = data.corr()
    d = []
    for x in corr:
        if x in d or corr[x].dropna().empty:
            continue
        # 相关性大的特征
        # 0.9的时候会有，sum-1980 .9298
        # 0.95 sum-1720 .9285
        de = list(corr[corr[x] > 0.9].index)
        # 特征列除了自己的1还有其它的
        if (len(de) > 1):
            # 其它的与自己进行特征融合
            for dde in list(set(de) - set([x])):
                print(dde)
                print(x)
                # 融合特征
                c1 = data[dde] / data[x]
                c1.name = dde + '/' + x
                c2 = data[dde] - data[x]
                c2.name = dde + '-' + x
                data = data.join(c1).join(c2)
        # 将融合过的特征加入不再融合列
        print('round' + str(x) + '-------------------------------------')
        print(de)
        d = d + de
    print(len(d))
    return data


#
#

all_x = comb_feature(all_x)
save_p(all_x,'temp1')
train_x = all_x.loc[train_i, :].sort_index()
test_x = all_x.loc[test_i, :].sort_index()
print(all_x.shape)
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
