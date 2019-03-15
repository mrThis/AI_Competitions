import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif as mic
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif as mic
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import pickle
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import cross_validation
from sklearn.base import BaseEstimator
import random
from lightgbm.sklearn import LGBMClassifier
import copy
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

#Stacking的python实现。
class OnegoStackingClassifier(BaseEstimator):
    def __init__(self, base_classifiers, combiner, n=3):
        self.base_classifiers = base_classifiers
        self.combiner = combiner
        self.n = n
    def fit(self, X, y):
        print('.',end='')
        stacking_train = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            cv = cross_validation.KFold(len(X), n_folds=self.n)
            for j, (traincv, testcv) in enumerate(cv):
                self.base_classifiers[model_no].fit(X[traincv, ], y[traincv])
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(X[testcv,])[:, 1]
                stacking_train[testcv, model_no] = predicted_y_proba

            self.base_classifiers[model_no].fit(X, y)
        self.combiner.fit(stacking_train, y)

    def predict_proba(self, X):
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict_proba(X)[:, 1]
        return self.combiner.predict_proba(stacking_predict_data)
    
#把使用不同特征处理方法的几个模型stacking。
class Idea4s(BaseEstimator):
    def __init__(self, base_classifiers, combiner, split_points=( (0,160),(160,160+470)),n=5):
        self.base_classifiers = base_classifiers
        self.combiner = combiner
        self.n = n
        self.split_points=split_points
    def fit(self, X, y):
        #print('.',end='')
        stacking_train = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for (start, stop),model_no in zip(self.split_points,range(len(self.base_classifiers))):
            cv = cross_validation.KFold(len(X[:,start:stop]), n_folds=self.n)
            for j, (traincv, testcv) in enumerate(cv):
                self.base_classifiers[model_no].fit(X[:,start:stop][traincv, ], y[traincv])
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(X[:,start:stop][testcv,])[:, 1]
                stacking_train[testcv, model_no] = predicted_y_proba

            self.base_classifiers[model_no].fit(X[:,start:stop], y)
        self.combiner.fit(stacking_train, y)

    def predict_proba(self, X):
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for (start, stop),model_no in zip(self.split_points,range(len(self.base_classifiers))):
            stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict_proba(X[:,start:stop])[:, 1]
        return self.combiner.predict_proba(stacking_predict_data)



