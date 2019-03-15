# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.preprocessing import scale, MinMaxScaler


from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.ensemble import GradientBoostingClassifier


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

results = pd.DataFrame(columns=['dataset', 'class_model', 'metrics', 'result'])

# metrics('precsion','recall','pf','G-mean2','bal','auc')
# class_model(smapling(SMOTE,ADASYN,RandomOverSampler)+fe(chi-square,mutual information,cfs)+classifier(gussian_byais,RF))
# dataset('mc2','jm1','kc3','cm1','pc3','pc4','mw1','pc1','pc5','mc1','pc2')

# read file

temp_result=pd.DataFrame(columns=['result_temp'])
tt=0
t = 0
for str in ['mc2', 'jm1', 'kc3', 'cm1', 'mw1', 'pc1', 'pc2']:
    print ('-----------------------------------------------------------' + 'run' + str + '---------------------')
    all = pd.read_csv('./data/' + str + '.csv')
    all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
    y = all['Defective']
    X = all.drop(['Defective'], axis=1)
    rate = (all['Defective'] == 1).sum() / float(all['Defective'].shape[0])
    #rate2 是正类负类的比例
    rate2 = (all['Defective'] == 0).sum()/(all['Defective'] == 1).sum()
    # print('the ratio of the wrong：%f' % (rate))


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
        #
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


    def log(df):
        lam = 0
        for feat in df.columns:
            # all_data[feat] += 1
            df[feat] = boxcox1p(df[feat], lam)
        return df


    def data_preprocess(x_train, x_test):
        x_train = minmaxto0_1(x_train)
        x_test = minmaxto0_1(x_test)
        # x_train = log(x_train)
        # x_test = log(x_test)
        return x_train, x_test



    def record_recall(recall, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('recall')
        list_temp.append(recall)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record_precision(precision, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('precision')
        list_temp.append(precision)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record_bal(bal, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('bal')
        list_temp.append(bal)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record_gmean2(gmean2, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('gmean2')
        list_temp.append(gmean2)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record_auc(auc, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('auc')
        list_temp.append(auc)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record_pf(pf, t):
        list_temp = []
        list_temp.append(str)
        list_temp.append('__CFS_RF')
        t = t + 1
        list_temp.append('pf')
        list_temp.append(pf)
        print(list_temp)
        results.loc[t] = list_temp
        return t


    def record(recall, auc, bal, pf, precision, Gmean2, t):

        t = record_precision(precision, t)
        t = record_recall(recall, t)
        t = record_pf(pf, t)
        t = record_bal(bal, t)
        t = record_gmean2(Gmean2, t)
        t = record_auc(auc, t)
        return t


    def evaluat_model(model, k_folds, t=t, tt=tt):
        # for i in range(10):
        sum_M = 0  # recall
        sum_auc_M = 0  # auc
        sum_bal_M = 0  # bal
        sum_pf_M = 0  # pf
        sum_f1_M = 0  # f1-score
        sum_precision_M = 0  # precision
        sum_Gmean2_M = 0  # Gmean2
        for m in range(10):
            i = 0
            sum_N = 0
            sum_auc_N = 0
            sum_bal_N = 0
            sum_pf_N = 0  # Same as above
            sum_f1_N = 0
            sum_precision_N = 0
            sum_Gmean2_N = 0
            # k-folds evaluation the model
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
            for train_index, test_index in skf.split(X, y):
                i = i + 1
                x_train, y_train, x_test, y_test = getEachLeveldata(train_index, test_index)
                # data preprocessing
                x_train, x_test = data_preprocess(x_train, x_test)
                # feature eng.
                x_train, y_train, x_test = attri_select(x_train, y_train, x_test)
                # sampling for class balance
                # model evaluation
                modle_fit = model.fit(x_train, y_train)
                y_pred = modle_fit.predict(x_test)
                y_prob = modle_fit.predict_proba(x_test)[:, 1]
                # get score1
                bal, pf = bal_pf_score(y_test, y_pred)
                score = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                f1 = f1_score(y_test, y_pred)
                Gmean2 = pow(score * (1 - pf), 0.5)
                sum_N += score
                sum_auc_N += auc
                sum_bal_N += bal
                sum_pf_N += pf
                sum_f1_N += f1
                sum_precision_N += precision
                sum_Gmean2_N += Gmean2
                # Output the wrong number of predictions
                print('-----------------group %d---------round %d-------------- ' % (m + 1, i))
                print(
                        "Number of mislabeled points out of a total %d points : %d and recall score=%f and auc=%f and balance=%f and pf=%f and f1_score=%f and precision=%f and G-mean2=%f" % (
                    x_test.shape[0], (y_test != y_pred).sum(), score, auc, bal, pf, precision, f1,Gmean2))
            sum_M += sum_N
            sum_auc_M += sum_auc_N
            sum_bal_M += sum_bal_N
            sum_pf_M += sum_pf_N
            sum_f1_M += sum_f1_N
            sum_precision_M += sum_precision_N
            sum_Gmean2_M += sum_Gmean2_N
        # print('recall=%f and auc=%f and bal=%f and pf=%f and f1=%f and precision=%f and Gmean2=%f' % (
        #     sum_M / (10 * k_folds), sum_auc_M / (10 * k_folds), sum_bal_M / (10 * k_folds), sum_pf_M / (10 * k_folds),
        #     sum_f1_M / (10 * k_folds), sum_precision_M / (10 * k_folds), sum_Gmean2_M / (10 * k_folds)))
        r1=sum_M / (10 * k_folds)
        r2=sum_precision_M / (10 * k_folds)
        r3=sum_pf_M / (10 * k_folds)


        tt = tt + 1
        tt_list=[r1]
        temp_result.loc[tt]=tt_list
        tt=tt+1
        tt_list=[r2]
        temp_result.loc[tt]=tt_list
        tt = tt + 1
        tt_list = [r3]
        temp_result.loc[tt] = tt_list
        tt_list = [r1]
        temp_result.loc[tt] = tt_list
        tt = tt + 1
        tt_list = [r2]
        temp_result.loc[tt] = tt_list
        tt = tt + 1
        tt_list = [r3]
        temp_result.loc[tt] = tt_list


        print('recall=%f and  precision=%f and pf=%f' % (
            sum_M / (10 * k_folds), sum_precision_M / (10 * k_folds),
            sum_pf_M / (10 * k_folds)))
        print('auc=%f and  bal=%f and Gmean2=%f' % (
            sum_auc_M / (10 * k_folds), sum_bal_M / (10 * k_folds),
            sum_Gmean2_M / (10 * k_folds)))

        return tt


    # NB
    # model = GaussianNB()  # 高斯模型下的贝叶斯

    # DT

    # model = DecisionTreeClassifier()  # cart we should use C4.5

    # RF

    # model = RandomForestClassifier()

    # SVM

    # model=SVC(probability=True)

    # NN

    # model = MLPClassifier()

    # KNN

    # model=KNeighborsClassifier(30)

    # regression

    # model=LogisticRegression()

    # gdbt
    # model = GradientBoostingClassifier()
    # xgboost
    # model = XGBClassifier(scale_pos_weight=rate2)

    #lightgbm
    model = LGBMClassifier(is_unbalance = True)

    tt = evaluat_model(model, 10, tt)

temp_result.to_csv('temp_r_w.csv',index=False)
# shut down jvm
# jvm.stop()
