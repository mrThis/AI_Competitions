# -*- coding: utf-8 -*-
# 3.4 测试证明分割test train部分是正确的
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score

from sklearn.feature_selection import SelectKBest, chi2

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest

all = pd.read_csv('./data/pc1.csv')
all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
rate2 = (all['Defective'] == 0).sum() / (all['Defective'] == 1).sum()



# get balance score
def bal_pf_score(test, pred):
    # labelArr[i] is actual value,predictArr[i] is predict value
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    for i in range(len(test)):
        if test[i] == 1 and pred[i] == 1:
            TP += 1.
        if test[i] == 1 and pred[i] == 0:
            FN += 1.
        if test[i] == 0 and pred[i] == 1:
            FP += 1.
        if test[i] == 0 and pred[i] == 0:
            TN += 1.
    pd = TP / (TP + FN)  # Sensitivity = TP/P  and P = TP + FN
    pf = FP / (FP + TN)  # Specificity = TN/N  and N = TN + FP
    bal = 1 - pow(pow((1 - pd), 2) + pow((0 - pf), 2), 0.5) / pow(2, 0.5)
    return bal, pf


def get_train(list, i, k_folds):
    train = pd.DataFrame()
    for k in range(k_folds):
        if k == i:
            continue
        train = train.append(list[k])
    return train.reset_index(drop=True)


def minmaxto0_1(df):
    columns = df.columns
    min_max_scaler = MinMaxScaler((0, 10))
    df = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    return df


# 正态化数据
from scipy.special import boxcox1p


# from scipy.stats import boxcox

def log(df):
    lam = 0
    for feat in df.columns:
        # all_data[feat] += 1
        df[feat] = boxcox1p(df[feat], lam)
    return df


# M*N=10*5
def evaluat_model(model, k_folds):
    sum_M = 0
    sum_auc_M = 0
    sum_bal_M = 0
    sum_pf_M = 0
    for m in range(10):
        # Shuffle data rows to achieve random sampling
        random_all = all.sample(frac=1).reset_index(drop=True)
        y = random_all['Defective']
        X = random_all.drop(['Defective'], axis=1)
        # split the data into n part
        # all_cross_list = np.array_split(random_all, k_folds)  # 是否应该保证划分比例的相同
        sum_N = 0
        sum_auc_N = 0
        sum_bal_N = 0
        sum_pf_N = 0
        i=0
        skf = StratifiedKFold(n_splits=k_folds)
        # kf = KFold(n_splits=k_folds)
        for train_index, test_index in skf.split(X,y):
            i+=1
            y_train=y[train_index]
            x_train=X.iloc[train_index]
            y_test=y[test_index]
            x_test=X.iloc[test_index]
            x_train=x_train.reset_index(drop=True)
            x_test=x_test.reset_index(drop=True)
            y_train=y_train.reset_index(drop=True)
            y_test=y_test.reset_index(drop=True)
        # for i in range(k_folds):
        #     # Get each round of test set and training set
        #     test = all_cross_list[i].reset_index(drop=True)
        #     train = get_train(all_cross_list, i, k_folds)
        #     y_train = train['Defective']
        #     x_train = train.drop(['Defective'], axis=1)
        #     y_test = test['Defective']
        #     x_test = test.drop(['Defective'], axis=1)
            # preprocess
            # Put the data in the range 0 to 1
            # x_train = minmaxto0_1(x_train)
            # x_test = minmaxto0_1(x_test)
            # x_train = log(x_train)
            # x_test =  log(x_test)
            # skb = SelectKBest(chi2, 7)
            # skbor = skb.fit(x_train, y_train)
            # x_train = skbor.transform(x_train)
            # x_test = skbor.transform(x_test)
            ##beacause of small amount of data but with many features,we should reduce dimensions
            ###in order to obtain better results ,we choose to use wrapper
            # modeling and prediction
            #添加新特征

            #
            # k = IsolationForest(max_features=3)
            # model_t = k.fit(x_train,y_train)
            # newtrain =pd.Series(model_t.predict(x_train)).rename('iso')
            # newtest = pd.Series(model_t.predict(x_test)).rename('iso')
            # x_train = x_train.join(newtrain)
            # x_test = x_test.join(newtest)



            modle_fit = model.fit(x_train, y_train)
            y_pred = modle_fit.predict(x_test)
            y_prob = modle_fit.predict_proba(x_test)[:, 1]
            # get score1
            bal, pf = bal_pf_score(y_test, y_pred)
            score = recall_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            sum_N += score
            sum_auc_N += auc
            sum_bal_N += bal
            sum_pf_N += pf
            # Output the wrong number of predictions
            print('-----------------group %d---------round %d-------------- ' % (m + 1, i + 1))
            print(
                "Number of mislabeled points out of a total %d points : %d and recall score %f and auc=%f and balance=%f and pf=%f" % (
                    x_test.shape[0], (y_test != y_pred).sum(), score, auc, bal, pf))
        sum_M += sum_N
        sum_auc_M += sum_auc_N
        sum_bal_M += sum_bal_N
        sum_pf_M += sum_pf_N
    print('the final recall score using model random forest is %f and auc=%f and bal=%f and pf=%f' % (
        sum_M / (10 * k_folds), sum_auc_M / (10 * k_folds), sum_bal_M / (10 * k_folds), sum_pf_M / (10 * k_folds)))

    # split_space=all.shape[0]//k_folds#ROUND DOWN
    # for i in range(k_folds):
    # all_cross=all.ix[i*split_space:(i+1)*split_space]


# model=GradientBoostingClassifier()
# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
#                          algorithm="SAMME",
#                          n_estimators=200, learning_rate=0.8)
# model = RandomForestClassifier(n_estimators=100)
# model = GaussianNB()  # 高斯模型下的贝叶斯
# model=NaiveBayes(bins=10)
# model=LogisticRegression()
# model=GaussianProcessClassifier()#new tech，时间太长
# model=DecisionTreeClassifier()#cart we should use C4.5
# model=SVC(probability=True)
# model = MLPClassifier()
# model=KNeighborsClassifier(30)#knn 特点，可以自己调节想要的召回率，和错误率这个很棒.


# xgboost
model = XGBClassifier(scale_pos_weight=rate2)

#lightgbm
# model = LGBMClassifier(is_unbalance = True)
from rgf import RGFClassifier
#RGF
# model =RGFClassifier()

evaluat_model(model, 10)

# evaluation result
# def evaluat_model():
# for i in range(10):
