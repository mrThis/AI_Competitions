# cusboost
import numpy as np
from scipy import interp
import pandas as pd
from sklearn.cluster import KMeans
import lightgbm as lgb
import math
from sklearn.base import BaseEstimator

class CURmodel(BaseEstimator):

    # 传入用户层级
    # 传入抽样比例
    # 先抽0.5，多数类数量减半
    # 0.8
    #
    def __init__(self, number_of_clusters=12, percentage_to_choose_from_each_cluster=0.8, model=lgb.LGBMClassifier(
        boosting_type='gbdt', is_unbalance=True, num_leaves=64,
        reg_alpha=0.0, reg_lambda=10,
        max_depth=-1, n_estimators=750, objective='binary',
        subsample=0.8, colsample_bytree=0.6,
        learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
    )):
        self.number_of_clusters = number_of_clusters
        self.percentage_to_choose_from_each_cluster = percentage_to_choose_from_each_cluster
        self.model = model
        self.moedlor = None

    def fit(self, X, y):

        idx_min = np.where(y == 1)[0]
        idx_maj = np.where(y == 0)[0]

        # 找到多数类出现的行和标签
        majority_class_instances = X[idx_maj]
        majority_class_labels = y[idx_maj]

        # 对多数类进行聚类，即用户分层
        kmeans = KMeans(n_clusters=self.number_of_clusters)
        kmeans.fit(majority_class_instances)

        X_maj = []
        y_maj = []

        # 得到每个聚类（得到每个用户层）
        points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        # 对每个用户层进行操作，用户层中选取多少点
        for key in points_under_each_cluster.keys():
            points_under_this_cluster = np.array(points_under_each_cluster[key])
            number_of_points_to_choose_from_this_cluster = math.ceil(
                len(points_under_this_cluster) * self.percentage_to_choose_from_each_cluster)
            #这里的random_choice 简直
            selected_points = np.random.choice(points_under_this_cluster,
                                               size=number_of_points_to_choose_from_this_cluster,replace=False)
            X_maj.extend(majority_class_instances[selected_points])
            y_maj.extend(majority_class_labels[selected_points])

        # 得到最终分用户层聚类后得到的点
        X_sampled = np.concatenate((X[idx_min], np.array(X_maj)))
        y_sampled = np.concatenate((y[idx_min], np.array(y_maj)))
        self.moedlor = self.model.fit(X_sampled, y_sampled)
        # 使用分类器进行分类

    def predict(self, X):
        return self.moedlor.predict(X)

    def predict_proba(self, X):
        return self.moedlor.predict_proba(X)