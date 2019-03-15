# out第一弹
import pandas as pd
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
# FTR51train = read_p('FTR51train')
# FTR51test = read_p('FTR51test')

train_x = train_x.join(FTR33train)
test_x = test_x.join(FTR33test)
# train_x = train_x.join(FTR51train)
# test_x = test_x.join(FTR51test)
#不骗钱的用户千篇一律,那我删掉也无所谓
#3,0.6  mean= 0.931775,std= 0.007116
#4,0.6  mean= 0.931998,std= 0.007281
#5,0.6  mean= 0.932935,std= 0.008079  #bingo 选择5
#6,0.6  mean= 0.932438,std= 0.007130
#9,0.6  mean= 0.931421,std= 0.007230

# 使用新的抽样方法的算法很稳定，所以trick2应该是可以work的
#5,0.5 mean= 0.931048,std= 0.007802
#5 0.7 mean= 0.932521,std= 0.007443
#5 0.8 mean= 0.933058,std= 0.007032
#5 0.9 mean= 0.933410,std= 0.007546
#5 1 mean= 0.932637,std= 0.007685


#是用modle1_3,聚类系列最后一次

# modeler = CURmodel(3)

#7-26 baseline mean= 0.933948,std= 0.008717
# 不加入freq特征 mean= 0.934793,std= 0.009037
# 加入freq的特征 mean= 0.934999,std= 0.009950
modeler=lgb.LGBMClassifier(
        boosting_type='gbdt', is_unbalance=True, num_leaves=64,
        reg_alpha=0.0, reg_lambda=10,
        max_depth=-1, n_estimators=750, objective='binary',
        subsample=0.8, colsample_bytree=0.6,
        learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1)
#重要性


if False:
    def getip(m, x, y):  # 计算重要性并输出
        ip = pd.DataFrame(m.fit(x, y).feature_importances_, index=x.columns)
        return ip
    getip(modeler, train_x, train_y).to_csv('../data/output/importance.csv')


if True:
    # 评估
    lgb_auc = auc_cv(modeler, train_x.values, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))
    # BASE 0.9319 0.01


#输出结果

modeler.fit(train_x.values, train_y)
pred_lgb = modeler.predict_proba(test_x.values)[:, 1]
lgb_res = pd.DataFrame()
lgb_res['id'] = test_x.index
lgb_res['score'] = pred_lgb
lgb_res.to_csv('../data/output/result5.csv', index=None, header=None, sep='\t')
#result 1是代表最好结果，线下 0.9339 0.00837
#result 3_1是加了我频数特征后的结果，线下 mean= 0.933608,std= 0.009698
#result 3_2是删去一部分特征（cui）后的结果，mean= 0.933665,std= 0.009675 # 删特征是有帮助的，但如果明天结果没超过清华，也没有意义
#result 3_3是我自选的一部分特征频数后的结果，mean= 0.933399,std= 0.009471
#result 2是代表一般结果（我的CURBOOST）结果降低，两个千分位.
#result 2-2代表改进（5,0.9）以后的结果
#result 3理论最强,加上了该加的 mean= 0.935943,std= 0.008829
#result 4理论应该更强，但是新加特征可能降分,感觉线上表现会优于result3
#result 5增加了玄学筛特征的方法，如果提升，则进一步探索.线下提升1个千分位，线上降低






