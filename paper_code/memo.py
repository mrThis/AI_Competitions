# 3.4 测试证明分割test train部分是正确的

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, MinMaxScaler,StandardScaler


from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.metrics import auc, roc_curve, roc_auc_score, recall_score, f1_score

all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/mc1.csv')
all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
rate = (all['Defective'] == 1).sum() / all['Defective'].shape[0]
print('错误率为：', rate)
# print('有无缺失值',np.isnan(all).any())



def get_train(list, i, k_folds):
    train = pd.DataFrame()
    for k in range(k_folds):
        if k == i:
            continue
        train = train.append(list[k])
    return train.reset_index(drop=True)


def minmaxto0_1(df):
    columns = df.columns
    min_max_scaler = MinMaxScaler((0,10))
    df = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    return df

def scaledf(df1,df2):
    columns = df1.columns
    scaler =StandardScaler()
    model=scaler.fit(df1)
    df1=model.transform(df1)
    df2=model.transform(df2)
    df1 = pd.DataFrame(df1, columns=columns)
    df2 = pd.DataFrame(df2, columns=columns)
    return df1,df2

# 正态化数据
from scipy.special import boxcox1p


def log(df):
    lam = 0
    for feat in df.columns:
        # all_data[feat] += 1
        df[feat] = boxcox1p(df[feat], lam)
    return df


# M*N=10*5
def evaluat_model(model, k_folds):
    sum_M = 0
    for m in range(10):
        # Shuffle data rows to achieve random sampling
        random_all = all.sample(frac=1).reset_index(drop=True)
        # split the data into n part
        all_cross_list = np.array_split(random_all, k_folds)
        sum_N = 0
        for i in range(k_folds):
            # Get each round of test set and training set
            test = all_cross_list[i].reset_index(drop=True)
            train = get_train(all_cross_list, i, k_folds)
            y_train = train['Defective']
            x_train = train.drop(['Defective'], axis=1)
            y_test = test['Defective']
            x_test = test.drop(['Defective'], axis=1)
            # preprocess
            # Put the data in the range 0 to 1
            x_train = minmaxto0_1(x_train)
            x_test = minmaxto0_1(x_test)
            # x_train,x_test=scaledf(x_train,x_test)
            x_train.to_csv('../data_set/NASA_PROMISE_DATASET/x_train.csv')
            x_train = log(x_train)
            x_test = log(x_test)
            # SMOTE One of the sampling methods
            smote = SMOTE(random_state=42)
            x_train_a, y_train = smote.fit_sample(x_train, y_train)
            x_train = pd.DataFrame(x_train_a, columns=x_train.columns)
            print('ratio,0:1=%d:%d' % ((y_train == 0).sum(), (y_train == 1).sum()))
            # attribure selection
            # pca = PCA(n_components=40)
            # pca.fit(x_train)#作用很小，原因显然是因为特征选择目的本身是为了去掉对于预测没帮助的，而PCA去掉的是相关性小的
            attriSelect=SelectKBest(chi2, k=10)
            attriSelector=attriSelect.fit(x_train,y_train)
            attriSelector.transform(x_train)
            attriSelector.transform(x_test)
            ##beacause of small amount of data but with many features,we should reduce dimensions
            ###in order to obtain better results ,we choose to use wrapper
            # modeling and prediction
            modle_fit = model.fit(x_train, y_train)
            y_pred = modle_fit.predict(x_test)
            # get score
            score = recall_score(y_test, y_pred)
            sum_N += score
            # Output the wrong number of predictions
            print('-----------------group %d---------round %d-------------- ' % (m + 1, i + 1))
            print("Number of mislabeled points out of a total %d points : %d and recall score %f" % (
                y_train.data.shape[0], (y_test != y_pred).sum(), score))
        sum_M += sum_N
    print('the final recall score using model random forest is %f' % (
    sum_M / (10 * k_folds)))

    # split_space=all.shape[0]//k_folds#ROUND DOWN
    # for i in range(k_folds):
    # all_cross=all.ix[i*split_space:(i+1)*split_space]

# model = AdaBoostClassifier(base_estimator=GaussianNB())
# model = RandomForestClassifier(n_estimators=5)
model = GaussianNB()
# model=LDA()
evaluat_model(model, 10)

# evaluation result
# def evaluat_model():
# for i in range(10):
