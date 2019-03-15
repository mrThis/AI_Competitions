#cusboost
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import interp
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.model_selection import StratifiedKFold

dataset = 'pima.txt'

print("dataset : ", dataset)
df = pd.read_csv(dataset, header=None)
df['label'] = df[df.shape[1] - 1]
#
df.drop([df.shape[1] - 2], axis=1, inplace=True)
labelencoder = LabelEncoder()
df['label'] = labelencoder.fit_transform(df['label'])
#
X = np.array(df.drop(['label'], axis=1))
y = np.array(df['label'])

normalization_object = Normalizer()
X = normalization_object.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True)

top_auc = 0
#生成误判率的点
#聚类的大小
number_of_clusters = 23
#从每个聚类中选点的个数
percentage_to_choose_from_each_cluster = 0.5




for depth in range(2, 20, 10):
    for estimators in range(20, 50, 10):
        current_param_auc = []
        current_param_aupr = []
        tprs = []
        #将数据集分为5折
        for train_index, test_index in skf.split(X, y):
            X_train = X[train_index]
            X_test = X[test_index]

            y_train = y[train_index]
            y_test = y[test_index]

            idx_min = np.where(y_train == 1)[0]
            idx_maj = np.where(y_train == 0)[0]

            #找到多数类出现的行和标签
            majority_class_instances = X_train[idx_maj]
            majority_class_labels = y_train[idx_maj]

            #对多数类进行聚类，即用户分层
            kmeans = KMeans(n_clusters=number_of_clusters)
            kmeans.fit(majority_class_instances)


            X_maj = []
            y_maj = []

            #得到每个聚类（得到每个用户层）
            points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

            #对每个用户层进行操作，用户层中选取多少点
            for key in points_under_each_cluster.keys():
                points_under_this_cluster = np.array(points_under_each_cluster[key])
                number_of_points_to_choose_from_this_cluster = math.ceil(
                    len(points_under_this_cluster) * percentage_to_choose_from_each_cluster)
                selected_points = np.random.choice(points_under_this_cluster,
                                                   size=number_of_points_to_choose_from_this_cluster)
                X_maj.extend(majority_class_instances[selected_points])
                y_maj.extend(majority_class_labels[selected_points])

            #得到最终分用户层聚类后得到的点
            X_sampled = np.concatenate((X_train[idx_min], np.array(X_maj)))
            y_sampled = np.concatenate((y_train[idx_min], np.array(y_maj)))

            # 使用分类器进行分类
            classifier = AdaBoostClassifier(
                DecisionTreeClassifier(max_depth=depth),
                n_estimators=estimators,
                learning_rate=1, algorithm='SAMME')

            classifier.fit(X_sampled, y_sampled)
            predictions = classifier.predict_proba(X_test)
            auc = roc_auc_score(y_test, predictions[:, 1])
            current_param_auc.append(auc)

        current_mean_auc = np.mean(np.array(current_param_auc))

        if top_auc < current_mean_auc:
            top_auc = current_mean_auc

            best_depth = depth
            best_estimators = estimators
            best_auc = top_auc


