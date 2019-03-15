# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from skfeature.function.statistical_based import CFS
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale, MinMaxScaler

from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.naive_bayes import GaussianNB
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

from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score, precision_score

from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression

# start jvm
import weka.core.jvm as jvm

jvm.start(class_path=['D:\\python_code\\venv\Lib\\site-packages\\weka\\lib\\python-weka-wrapper.jar',
                      'C:\\Program Files\\Weka-3-8\\weka.jar'])

# read file
all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/cm1.csv')
all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
y = all['Defective']
X = all.drop(['Defective'], axis=1)
rate = (all['Defective'] == 1).sum() / float(all['Defective'].shape[0])

print('the ratio of the wrong：%f' % (rate))
record = pd.DataFrame(columns=['fpr', 'tpr'])


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
    # precision=TP/(TP+FP) #precision
    # 由于f-measure是由pre/pd出来的，所以它的这个大小，我们也挺doubt的.
    bal = 1 - pow(pow((1 - pd), 2) + pow((0 - pf), 2), 0.5) / pow(2, 0.5)
    return bal, pf


# split dataframe by instance index
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
    skb = SelectKBest(f_regression, 7)
    skbor = skb.fit(x_train, y_train)
    x_train_a = skbor.transform(x_train)
    x_test_a = skbor.transform(x_test)
    x_train = pd.DataFrame(x_train_a)
    x_test = pd.DataFrame(x_test_a)
    return x_train, y_train, x_test


from util import pandas2arff
from weka.core.converters import Loader
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection


def attri_select2(x_train, y_train, x_test):
    # CFS Correlation-based Feature Selection
    # combine x_train and y_train
    temp = x_train
    temp['Defective'] = y_train
    # convert to attr
    pandas2arff(temp, 'temp.arff')
    # load data
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("temp.arff")

    # cfs alg.
    search = ASSearch(classname="weka.attributeSelection.GreedyStepwise")
    evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
    attsel = AttributeSelection()
    attsel.search(search)
    attsel.evaluator(evaluator)
    attsel.select_attributes(data)
    attri_index = (attsel.selected_attributes - 1)[:-1]
    # handle x
    x_train = x_train.iloc[:, attri_index]
    x_test = x_test.iloc[:, attri_index]
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
    return x_train, y_train


def evaluat_model(model):
    record = pd.DataFrame(columns=['bal', 'Gmean2', 'auc', 'recall', 'pf'])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    # data preprocessing
    x_train, x_test = data_preprocess(x_train, x_test)
    # feature eng.
    x_train, y_train, x_test = attri_select(x_train, y_train, x_test)
    # sampling for class balance
    x_train, y_train = class_balance(x_train, y_train)
    # model evaluation
    modle_fit = model.fit(x_train, y_train)
    y_pred = modle_fit.predict(x_test)
    y_prob = modle_fit.predict_proba(x_test)[:, 1]
    # get score1
    bal, pf = bal_pf_score(y_test.values, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    gmean2 = pow(recall * (1 - pf), 0.5)
    auc=roc_auc_score(y_test, y_prob)
    fpr, tpr, threshold = roc_curve(y_test, y_prob)
    rec = [bal, gmean2, auc,recall, pf]
    print(fpr)
    print(tpr)
    print(rec)

# model = KNeighborsClassifier(100)
# model = DecisionTreeClassifier()
model = GaussianNB()
# model = RandomForestClassifier(n_estimators=100)
# model=SVC(probability=True)
evaluat_model(model)
jvm.stop()
