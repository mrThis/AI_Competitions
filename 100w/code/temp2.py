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



#标准化对结果影响不大
train_x = (train_x-train_x.mean())/(train_x.max()-train_x.min())
test_x = (test_x-test_x.mean())/(test_x.max()-test_x.min())

#解决inf值问题
train_x=train_x.fillna(0)
test_x=test_x.fillna(0)

#合并结果做为fit依据
all_x = train_x.append(test_x)
all_x = (all_x-all_x.mean())/(all_x.max()-all_x.min())
#解决inf值问题
all_x=all_x.fillna(0)

#将pca后的数据作为补充集
pca = PCA(n_components=100)
pca.fit(all_x)
#可视化结果
if False:
    plt.figure(1, figsize=(14, 13))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_, linewidth=2)
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')
    plt.show()

#创造线性融合集
train_x_imp2=pca.transform(train_x)
test_x_imp2=pca.transform(test_x)
train_x_imp2 = pd.DataFrame(train_x_imp2,index=train_x.index)
train_x_imp2.rename(columns=lambda x: 'PCA'+str(x) , inplace=True)
test_x_imp2 = pd.DataFrame(test_x_imp2,index=test_x.index,columns=train_x_imp2.columns)


#补充trainx，和testx
train_x = train_x.join(train_x_imp2)
test_x = test_x.join(test_x_imp2)



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


