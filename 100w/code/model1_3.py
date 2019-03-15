# 骗保用户差异多，不骗保用户差异小（所以导致我的分类没效果，并且因为删数据导致了分数降低2个千分位）
# 直接给违规用户做画像（聚类）
# 分为2个（默认）类别后分别建模，得到model1，model2...modeln(理论上只设置为2)
# （相应的test集传入后也先聚类，再做判断）
##其实这两类最好不要都删去，而是设置重要性参数，太复杂，我的实现肯定效果很差

# 数据会更加不平衡
# 数据不平衡依靠lgb默认参数解决


# 理论上，最起码不降低auc?，但是删除了一部分错误点出的训练集，更加的，学不到少数类了？，（我们先假设lgb可以解决这个问题）

# jary_model1
import numpy as np
from scipy import interp
import pandas as pd
from sklearn.cluster import KMeans
import lightgbm as lgb
import math
from sklearn.base import BaseEstimator


class CURmodel(BaseEstimator):

    # 传入用户层级
    # 传入训练模型
    def __init__(self, number_of_clusters=2, model=lgb.LGBMClassifier(
        boosting_type='gbdt', is_unbalance=True, num_leaves=64,
        reg_alpha=0.0, reg_lambda=10,
        max_depth=-1, n_estimators=750, objective='binary',
        subsample=0.8, colsample_bytree=0.6,
        learning_rate=0.01, min_child_weight=10, random_state=888, n_jobs=-1
    )):
        self.number_of_clusters = number_of_clusters
        self.model = model
        self.cluster = None
        self.moedlor = []

    def fit(self, X, y):
        idx_min = np.where(y == 1)[0]
        idx_maj = np.where(y == 0)[0]

        # 找到骗保用户出现的行和标签
        minority_class_instances = X[idx_min]
        minority_class_labels = y[idx_min]

        # 对多数类进行聚类，即骗保用户分层
        kmeans = KMeans(n_clusters=self.number_of_clusters)
        kmeans.fit(minority_class_instances)
        # 将聚类方法沿用到test集将test根据少数类分法分为2部分（这个刁钻的方法）
        self.cluster = kmeans
        # 得到每个聚类（得到每个用户层）
        points_under_each_cluster = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

        for key in points_under_each_cluster.keys():
            points_under_this_cluster = np.array(points_under_each_cluster[key])
            print(points_under_this_cluster)
            number_of_points_to_choose_from_this_cluster = math.ceil(
                len(points_under_this_cluster) * 1)
            selected_points = np.random.choice(points_under_this_cluster,
                                               size=number_of_points_to_choose_from_this_cluster)
            X_sampled = np.concatenate((X[idx_maj], np.array(minority_class_instances[selected_points])))
            y_sampled = np.concatenate((y[idx_maj], np.array(minority_class_labels[selected_points])))
            sub_moedlor = self.model.fit(X_sampled, y_sampled)
            self.moedlor = self.moedlor + [sub_moedlor]
        # 对每个用户层进行操作
        # for key in points_under_each_cluster.keys():
        #     # 该用户层下的点
        #     points_under_this_cluster = np.array(points_under_each_cluster[key])
        #     print(points_under_this_cluster)
        #     # 该用户层下的点，拼接正常用户，得到训练集
        #     X_sampled = np.concatenate((X[idx_maj], np.array(minority_class_instances[points_under_this_cluster])))
        #     y_sampled = np.concatenate((y[idx_maj], np.array(minority_class_labels[points_under_this_cluster])))
        #     sub_moedlor = self.model.fit(X_sampled, y_sampled)
        #     # 将模型加入
        #     self.moedlor = self.moedlor + [sub_moedlor]

    def predict_proba(self, X):
        prob = np.zeros((X.shape[0], 2))
        for sub_modelor in self.moedlor:
            prob = prob + sub_modelor.predict_proba(X)
            print(prob)
        pd.DataFrame(prob).to_csv('temp.csv',index=False)
        return prob/self.number_of_clusters
