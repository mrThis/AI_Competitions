import numpy as np
from sklearn import cross_validation
from sklearn.base import BaseEstimator

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