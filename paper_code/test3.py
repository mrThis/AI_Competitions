#Add attribute selection(wrapper) on the basis of test2
#beacause of the special of NB

import pandas as pd
import numpy as np
from skfeature.function.statistical_based import CFS


from sklearn.preprocessing import scale, MinMaxScaler

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import GaussianNB

from ibmdbpy.learn.naive_bayes import NaiveBayes
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score

from sklearn.feature_selection import SelectKBest, chi2

# read file
all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/pc1.csv')
all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
y = all['Defective']
X = all.drop(['Defective'], axis=1)
rate = (all['Defective'] == 1).sum() / all['Defective'].shape[0]
print('错误率为：', rate)


# get balance and pf score
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


#split dataframe by instance index
def getEachLeveldata(train_index, test_index):
    y_train = y[train_index]
    x_train = X.iloc[train_index]
    y_test = y[test_index]
    x_test = X.iloc[test_index]
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return x_train, y_train, x_test, y_test


def attri_select(x_train, y_train, x_test):
    # Chi-Square#先用这个.
    skb = SelectKBest(chi2, 7)
    skbor = skb.fit(x_train, y_train)
    x_train_a = skbor.transform(x_train)
    x_test_a = skbor.transform(x_test)
    x_train = pd.DataFrame(x_train_a)
    x_test = pd.DataFrame(x_test_a)
    return x_train, y_train, x_test



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

def data_preprocess(x_train, x_test):
    x_train = minmaxto0_1(x_train)
    x_test = minmaxto0_1(x_test)
    x_train = log(x_train)
    x_test = log(x_test)
    return x_train, x_test


def class_balance(x_train, y_train):
    smote = SMOTE(random_state=42)
    x_train_a, y_train = smote.fit_sample(x_train, y_train)
    x_train = pd.DataFrame(x_train_a, columns=x_train.columns)
    return x_train,y_train



def evaluat_model(model, k_folds):
    sum_M = 0  # recall
    sum_auc_M = 0  # auc
    sum_bal_M = 0  # bal
    sum_pf_M = 0  # pf
    for m in range(10):
        i=0
        sum_N = 0
        sum_auc_N = 0
        sum_bal_N = 0
        sum_pf_N = 0  # Same as above
        # k-folds evaluation the model
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        for train_index, test_index in skf.split(X, y):
            i=i+1
            x_train, y_train, x_test, y_test = getEachLeveldata(train_index, test_index)
            # feature eng.
            x_train, y_train, x_test = attri_select(x_train, y_train, x_test)
            # data preprocessing
            x_train, x_test = data_preprocess(x_train, x_test)
            #sampling for class balance
            x_train, y_train=class_balance(x_train, y_train)

            #model evaluation
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
            print('-----------------group %d---------round %d-------------- ' % (m + 1, i))
            print(
                "Number of mislabeled points out of a total %d points : %d and recall score %f and auc=%f and balance=%f and pf=%f" % (
                    x_test.shape[0], (y_test != y_pred).sum(), score, auc, bal, pf))
        sum_M += sum_N
        sum_auc_M += sum_auc_N
        sum_bal_M += sum_bal_N
        sum_pf_M += sum_pf_N
    print('the final recall score using model random forest is %f and auc=%f and bal=%f and pf=%f' % (
        sum_M / (10 * k_folds), sum_auc_M / (10 * k_folds), sum_bal_M / (10 * k_folds), sum_pf_M / (10 * k_folds)))



# model=GradientBoostingClassifier()
# model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
#                          algorithm="SAMME",
#                          n_estimators=200, learning_rate=0.8)
# model = RandomForestClassifier(n_estimators=100)
model = GaussianNB()  # 高斯模型下的贝叶斯
# model=NaiveBayes(bins=10)
# model=LogisticRegression()
# model=GaussianProcessClassifier()#new tech，时间太长
# model=DecisionTreeClassifier()#cart we should use C4.5
# model=SVC(probability=True)
# model = MLPClassifier()
# model=KNeighborsClassifier(30)#knn 特点，可以自己调节想要的召回率，和错误率这个很棒.
evaluat_model(model, 10)


