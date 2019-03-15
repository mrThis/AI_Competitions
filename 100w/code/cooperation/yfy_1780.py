import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
import lightgbm as lgb

def read_p(name):
    return pd.read_pickle(r'C:\Users\jary_\Desktop\cui3\\' + name + '.p')
def auc_cv(model, train_data, label):
    skf = StratifiedKFold(5, shuffle=True, random_state=42).get_n_splits(train_data)
    r_auc = cross_val_score(model, train_data, label, scoring="roc_auc", cv=skf, verbose=5)  # auc评价
    return (r_auc)

train = pd.read_csv('C:/Users/jary_/Desktop/yfy/train.csv')
test = pd.read_csv('C:/Users/jary_/Desktop/yfy/test.csv')
train.set_index('PERSONID', inplace=True)
test.set_index('PERSONID', inplace=True)
train_y = train['LABEL']
train_x = train.drop(['id', 'LABEL'], axis=1)
print(train_x.shape)

modeler = lgb.LGBMClassifier(
    boosting_type='gbdt', is_unbalance=True, num_leaves=80,
    reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=900, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=8222223388, n_jobs=-1,
    verbose=-1,
)
# 选取重要特征 420
if False:
    t = pd.read_csv(r'..\model_selection\yfy_imp.csv')
    feature = list(t.feature)
    train_x = train_x[feature]
    print(train_x.shape)
if True:
    t = pd.read_csv(r'..\model_selection\imp.csv')
    feature = list(t.name)[:75]
    train_x = train_x[feature]
    print(train_x.shape)

# baseline all 0.9340222321599694(第一折)
# baseline 75 mean= 0.939449,std= 0.003725
# 筛特征 420 mean= 0.936914,std= 0.003309  [ 0.93878855  0.94003203  0.93320997  0.9325985   0.93994077]
if False:
    # 评估
    lgb_auc = auc_cv(modeler, train_x.values, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))

# 输出结果
# modeler.fit(train.values, train_y)
# pred_lgb = modeler.predict_proba(test.values)[:, 1]
# lgb_res = pd.DataFrame()
# lgb_res['id'] = test.index
# lgb_res['score'] = pred_lgb
# lgb_res.to_csv('result_cui_3.csv', index=None, header=None, sep='\t')
