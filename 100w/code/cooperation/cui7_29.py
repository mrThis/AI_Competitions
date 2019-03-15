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


train1 = read_p('train_x').join(read_p('FTR33train'))  # 最初的版本
test1 = read_p('test_x').join(read_p('FTR33test'))

train2 = read_p('train_x2').join(read_p('train_x').iloc[:, -18:]).join(read_p('Ctrain'))  # 重构特征工程后的结果
test2 = read_p('test_x2').join(read_p('test_x').iloc[:, -18:]).join(read_p('Ctest'))

train3 = read_p('ltrain75')  # YFY筛选75个的结果
test3 = read_p('ltest75')

train_y = read_p('train_y')

modeler = lgb.LGBMClassifier(
    boosting_type='gbdt', is_unbalance=True, num_leaves=80,
    reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=900, objective='binary',
    subsample=0.8, colsample_bytree=0.6,
    learning_rate=0.01, min_child_weight=10, random_state=8222223388, n_jobs=-1,
    verbose=-1,
)


# print(yfytest_imp.shape)
#
# 选取特征重要性前排
print(train2.shape)
t = pd.read_csv(r'C:\Users\jary_\Desktop\cui3\t.csv')
feature = list(t.feature)
train = train2[feature]
test = test2[feature]

if True:
    # 加入强特
    yfytrain_imp = read_p('yfytrain_imp')
    yfytest_imp = read_p('yfytest_imp')
    train = train.join(yfytrain_imp)
    test = test.join(yfytest_imp)
# 加入我的新特征
if True:
    FTR51_new2 = read_p('FTR51_new2')
    train = train.join(FTR51_new2)


# 今天实验这一版，试一下它有没有提升
if True:
    # 丢弃与cui 相同的
    repeat = ['FTR33__agg_linear_trend__f_agg_"max"__chunk_len_50__attr_"intercept"',
              'FTR33__approximate_entropy__m_2__r_0.1',
              'FTR33__ar_coefficient__k_10__coeff_3',
              'FTR33__energy_ratio_by_chunks__num_segments_10__segment_focus_7',
              'FTR33__energy_ratio_by_chunks__num_segments_10__segment_focus_8',
              'FTR33__fft_coefficient__coeff_3__attr_"imag"',
              'FTR33__fft_coefficient__coeff_4__attr_"imag"',
              'FTR33__last_location_of_maximum', 'FTR33__quantile__q_0.6',
              'FTR33__quantile__q_0.9', 'FTR33__spkt_welch_density__coeff_2',
              'FTR33__spkt_welch_density__coeff_5']
    train75 = read_p('yfy75train')
    test75 = read_p('yfy75test')
    train75.drop(repeat, axis=1, inplace=True)
    test75.drop(repeat, axis=1, inplace=True)
    train = train.join(train75)
    test = test.join(test75)
print(train.shape)
# baseline mean= 0.941393,std= 0.008973 [ 0.95137856  0.94113178  0.94964909  0.92636281  0.93844402]
# 加yfy说的强特 mean= 0.941758,std= 0.009592   [ 0.95363965  0.94089677  0.94912431  0.92568061  0.93944794]
# 筛特征 mean= 0.941665,std= 0.008378   [ 0.95237335  0.94147174  0.94726935  0.92748537  0.9397263 ]
# 加我的新特征 mean= 0.942414,std= 0.009934 [ 0.95142191  0.94314645  0.95167973  0.92438921  0.94143295]
# 添加yfy的新特征75 mean= 0.944110,std= 0.007281 [ 0.95207674  0.94510637  0.95041571  0.93177027  0.94117969]
## 融合方案: cui筛选特征 + yfy新特征+我的新特征 mean= 0.945127,std= 0.009233 [ 0.95287987  0.94634757  0.95446788  0.92856458  0.9433769 ]
## 融合方案：yfy的新特征+我的新特征+加yfy说的强特（重复,增长小）mean= 0.945127,std= 0.009233
if True:
    # 评估
    lgb_auc = auc_cv(modeler, train.values, train_y)
    print('model_lgb auc : ', lgb_auc)
    print('mean= %f,std= %f' % (lgb_auc.mean(), lgb_auc.std()))

# 输出结果
# modeler.fit(train.values, train_y)
# pred_lgb = modeler.predict_proba(test.values)[:, 1]
# lgb_res = pd.DataFrame()
# lgb_res['id'] = test.index
# lgb_res['score'] = pred_lgb
# lgb_res.to_csv('result_cui_3.csv', index=None, header=None, sep='\t')
