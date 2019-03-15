# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p



all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/cm1.csv')


# describe more info like level of attributs
def describe_more(df):
    # 注意作者在这里用的同行加分号的形式，也是可以的，不符合pep8规范
    var = []
    l = []
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels


# data info
# print(all.shape)
# all.info()
# print(describe_more(all))

# split data as x y
#y_all = all.Defective.values * 1  # values can make data_typy turn to ndarry,trick is boolean*1=binary
y_all=all.Defective.map(dict(Y=1, N=0))
rate=(y_all==1).sum()/y_all.shape[0]
print('错误率为：',rate)
x_all = all.drop(['Defective'], axis=1)
print('feature 个数是,模块数目是:,错误类个数是',x_all.shape[1],x_all.shape[0],(y_all==1).sum())




# print(y_all)
# smote=SMOTE(random_state=42)
# x_all_a,y_all=smote.fit_sample(x_all,y_all)
# x_all=pd.DataFrame(x_all_a,columns=x_all.columns)
# print(x_all.head())
# preprocessing数据预处理
##不处理,标准化，或者正态化,或者box-cox
from sklearn.preprocessing import scale,MinMaxScaler
columns=x_all.columns
min_max_scaler = MinMaxScaler()
x_all=min_max_scaler.fit_transform(x_all)
x_all=pd.DataFrame(x_all,columns=columns)
print(x_all.values)
##接下来把属性数据的峰度偏度用box-cox降低一下.
# Check the skew of all numerical features
# skewed_feats = x_all.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# print("\nSkew in numerical features: \n")
# skewness = pd.DataFrame({'Skew': skewed_feats})
# skewness = skewness[abs(skewness) > 0.75]
# print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))
# print(skewness)
# skewed_features = skewness.index
# lam = 0.15
# for feat in skewed_features:
#     # all_data[feat] += 1
#     x_all[feat] = boxcox1p(x_all[feat], lam)

# x_all.to_csv()
# 特征工程

# 特征选取，特征重要性
# 特征降维度


###验证模块 for evaluation
from sklearn.model_selection import KFold, cross_val_score, train_test_split
# 自建balance
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


# 10折交叉验证
n_folds = 10


# evaluate modle
##accuracy,recall,f-measure,auc,pd,pf,balance
def rmsle_cv(model):
    # 这个rmse,crosevalidation
    scoring = ['precision', 'recall', 'f1', 'roc_auc', ]
    # recall=2 * (precision * recall) / (precision + recall)
    # precision=tp/(tp+fp)
    # recall=tp / (tp + fn)
    # roc_auc=Area under the ROC curve
    # fpr=fp/fp+tn
    ##the main difference between precision and recall is Denominator.一个分母是所有预测为真(可能含假)的值，一个分母是所有为真(是真实数据的)的值.
    ###precision可能预测出来的是有缺陷的很少，但 是都对了.recall可能预测出来的有缺陷很多，但是错的也很多
    # 可以返回每一次折的分数
    e_rmse = []
    # 这里的random_state好像不起作用
    for i in range(20):
        kf = KFold(n_folds, random_state=i*10).get_n_splits()
        rmse = cross_val_score(model, x_all.values, y_all, scoring='roc_auc', cv=kf)
        # print(rmse)
        # 数组拼接用这个方法append会直接附加成二维的list
        e_rmse.extend(rmse)
        # print(e_rmse)
    return (np.array(e_rmse))


# 模型
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

# NB#假设了分布的正态性
# model = GaussianNB()
# 随机森林信息熵增益划分的忽略细节
model = RandomForestClassifier(n_estimators=10)
# 提升树
# model = GradientBoostingClassifier()
# model=LogisticRegression()
# adaboost
# model = AdaBoostClassifier(base_estimator=GaussianNB())
score = rmsle_cv(model)
# print(score)
X_train, X_test, y_train, y_test = train_test_split(x_all.values, y_all, test_size=0.4, random_state=42)
# smote=SMOTE(random_state=42)
# X_train,y_train=smote.fit_sample(X_train,y_train)
# # x_all=pd.DataFrame(x_all_a,columns=x_all.columns)
y_pred = model.fit(X_train, y_train).predict(X_test)

from sklearn.metrics import auc, roc_curve, recall_score

print(y_pred)
print(list(y_test))
print("Number of mislabeled points out of a total %d points : %d" % (y_train.data.shape[0], (y_test != y_pred).sum()))
# print(recall_score(y_test, y_pred))
print("model score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
