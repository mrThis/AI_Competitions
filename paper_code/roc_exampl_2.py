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


fpr1=[0.,0.05179283,0.05179283,0.05577689,0.05577689,0.06374502
,0.06374502,0.07171315,0.07171315,0.09561753,0.09561753,0.11553785
,0.11553785,0.11952191,0.11952191,0.15139442,0.15139442,0.15936255
,0.15936255,0.16334661,0.16334661,0.17131474,0.17131474,0.1752988
,0.1752988,0.187251,0.187251,0.21513944,0.21513944,0.2310757
,0.2310757,0.23904382,0.23904382,0.24302789,0.24302789,0.25896414
,0.25896414,0.29083665,0.29083665,0.30677291,0.30677291,0.33067729
,0.33067729,0.34262948,0.34262948,0.3625498,0.3625498,0.39043825
,0.39043825,0.42629482,0.42629482,0.45816733,0.45816733,0.49800797
,0.49800797,0.6374502,0.64940239,0.76494024,0.76494024,0.92031873
,0.92828685,0.93227092,0.94023904,0.95219124,0.96414343,0.98007968
,0.99203187,1.]


tpr1=[0.,0.17073171,0.2195122,0.2195122,0.24390244,0.24390244
,0.29268293,0.29268293,0.36585366,0.36585366,0.3902439,0.3902439
,0.41463415,0.41463415,0.43902439,0.43902439,0.46341463,0.46341463
,0.51219512,0.51219512,0.53658537,0.53658537,0.56097561,0.56097561
,0.58536585,0.58536585,0.6097561,0.6097561,0.63414634,0.63414634
,0.65853659,0.65853659,0.68292683,0.68292683,0.73170732,0.73170732
,0.75609756,0.75609756,0.7804878,0.7804878,0.80487805,0.80487805
,0.82926829,0.82926829,0.85365854,0.85365854,0.87804878,0.87804878
,0.90243902,0.90243902,0.92682927,0.92682927,0.95121951,0.95121951
,0.97560976,0.97560976,0.97560976,0.97560976,1.,1.
,1.,1.,1.,1.,1.,1.
,1.,1.]

fpr2=[0.,0.,0.01158301,0.01158301,0.01930502,0.01930502
,0.03088803,0.03088803,0.05791506,0.05791506,0.06177606,0.07335907
,0.1042471,0.1042471,0.10810811,0.10810811,0.12741313,0.12741313
,0.13513514,0.13513514,0.16988417,0.16988417,0.18918919,0.18918919
,0.19305019,0.19305019,0.20849421,0.20849421,0.21621622,0.21621622
,0.28185328,0.28185328,0.28957529,0.28957529,0.31660232,0.31660232
,0.32432432,0.32432432,0.35135135,0.35135135,0.37065637,0.37837838
,0.47490347,0.47490347,0.48262548,0.48262548,0.49420849,0.50965251
,0.6023166,0.61003861,0.62934363,0.63706564,0.71042471,0.72586873
,0.72972973,0.74131274,0.74903475,0.76833977,0.77220077,0.78378378
,0.78764479,0.81081081,0.86100386,0.86872587,1.]

tpr2=[0.03030303,0.09090909,0.09090909,0.12121212,0.12121212,0.18181818
,0.18181818,0.36363636,0.36363636,0.39393939,0.39393939,0.39393939
,0.39393939,0.42424242,0.42424242,0.48484848,0.48484848,0.54545455
,0.54545455,0.57575758,0.57575758,0.60606061,0.60606061,0.66666667
,0.66666667,0.6969697,0.6969697,0.75757576,0.75757576,0.78787879
,0.78787879,0.81818182,0.81818182,0.84848485,0.84848485,0.87878788
,0.87878788,0.90909091,0.90909091,0.93939394,0.93939394,0.93939394
,0.93939394,0.96969697,0.96969697,1.,1.,1.
,1.,1.,1.,1.,1.,1.
,1.,1.,1.,1.,1.,1.
,1.,1.,1.,1.,1.]

# plot ROC
lw = 2
_, ax = plt.subplots(figsize=(7, 7))
#设置字体大小
# 设置刻度字体大小
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# 设置坐标标签字体大小
ax.set_xlabel('PF',fontsize=20)
ax.set_ylabel('recall',fontsize=20)
# 设置图例字体大小
# line, = plt.plot(fpr, tpr, color='darkorange',
#                  lw=lw, label='ROC curve (AUC=%.2f)'%auc1)  ###假正率为横坐标，真正率为纵坐标做曲线
line, = plt.plot(fpr1, tpr1, color='darkorange',
                     lw=lw, label='CFS')
line.set_antialiased(False)
line2 = plt.plot(fpr2, tpr2, color='black',
                     lw=lw, label='chi-square',linestyle='--')
# line2.set_antialiased(False)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
# plt.xlabel('PF')
# plt.ylabel('recall')
plt.title('PC4_ROC',fontsize=20)
plt.legend(loc="lower right",fontsize=20)
# plt.savefig('figure.eps')
plt.show()


#pc5
# chisquare:[0.7121891946059147,,0.7579842962989832,,0.9603133706312713,,0.5943396226415094,,0.03331332533013205]
# CFS:[0.,,0.8180512216745801,,0.9400398111089935,,0.696969696969697,,0.039832285115303984]7838820050387473