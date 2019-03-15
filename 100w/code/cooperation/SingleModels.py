from sklearn.base import BaseEstimator

from lightgbm import LGBMClassifier

# m1 = LGBMClassifier(boosting_type='dart', is_unbalance=True, objective='xentropy',
#                     seed=1239329, n_estimators=200, colsample_bytree=0.9)
bestargs={'colsample_bytree': 0.69, 'drop_rate': 0.062, 'learning_rate': 0.117, 'max_drop': 51, 'n_estimators': 280, 'num_leaves': 40, 'reg_alpha': 0.905, 'reg_lambda': 1.48, 'skip_drop': 0.42, 'subsample': 0.9, 'subsample_freq': 1}
m1 = LGBMClassifier(boosting_type='dart', is_unbalance=True, objective='xentropy',seed=20180718,
                    **bestargs)

from sklearn.ensemble import ExtraTreesClassifier

m2 = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35,
                          min_samples_leaf=14, min_samples_split=15,
                          n_estimators=200, class_weight='balanced', n_jobs=4)

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier

m3 = GradientBoostingClassifier(max_features=0.35,
                                min_samples_leaf=14, min_samples_split=15,
                                n_estimators=200)

m4=AdaBoostClassifier(   n_estimators=200)

# m5=RandomForestClassifier(max_features=0.35,
#                           min_samples_leaf=14, min_samples_split=15,
#                           n_estimators=200, class_weight='balanced')
m5 = RandomForestClassifier(bootstrap=False,
                            criterion='entropy', max_features=0.4,
                            min_samples_leaf=16, min_samples_split=14, n_estimators=200)
#rf奇慢无比

from xgboost import XGBClassifier

m6 = XGBClassifier(n_estimators=200, max_depth=4, n_jobs=4, scale_pos_weight=0.05)

if True:
    from keras.datasets import mnist
    from keras.utils import np_utils
    from keras.models import Sequential
    from keras.models import Model
    from keras.optimizers import RMSprop
    from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from keras.layers import Dense, Dropout, Input, Embedding, Flatten, concatenate, Activation
    from keras.layers.advanced_activations import PReLU, LeakyReLU
    from keras import regularizers
    from keras import initializers
    # from keras import optimizers
    from keras.layers.normalization import BatchNormalization


    def make_model1():
        model = Sequential([
            Dense(200, input_dim=483, activation='relu'),
            Dropout(0.85),
            Dense(200, activation='relu'),
            Dense(150, activation='relu'),
            Dense(70, activation='relu'),
            BatchNormalization(),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid'),
        ])
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def make_model2():
        model = Sequential([
            Dense(200, input_dim=483, activation='relu'),
            Dropout(0.85),
            Dense(200, activation='relu'),
            Dense(70, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid'),
        ])
        rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model


    mk = KerasClassifier(make_model1)

    m7 = Pipeline([('a', StandardScaler()), ('keras', mk)])





import catboost as cb
m8 = cb.CatBoostClassifier()#在xbest上表现不错

from mymodels import OnegoStackingClassifier
from sklearn.linear_model import LogisticRegression
m9 = OnegoStackingClassifier([m1, m2, m3, m5, m6,m8], LogisticRegression(), n=3)

from sklearn.feature_selection import SelectKBest,SelectFromModel,SelectFdr,SelectFpr,SelectFwe,RFECV
from sklearn.base import clone
m10=Pipeline([('select',SelectFromModel(clone(m1),'mean',False)),('predict',m1)])


from rgf.sklearn import RGFClassifier
m11=RGFClassifier(max_leaf=1000,
                    algorithm="RGF_Sib",
                    test_interval=100,
                  learning_rate=0.1,
                    verbose=True)
#0.902

from GCForest import gcForest
m12=gcForest(shape_1X=[1,483],window=[483] ,tolerance=0.0)
#0.866

from sklearn.neural_network import MLPClassifier
m12 = Pipeline([('a', StandardScaler()), ('MLP', MLPClassifier())])
#0.8205

from sklearn.semi_supervised import LabelPropagation
class Semi(BaseEstimator):
    def fit(self,train_x,train_y):
        self.train_x=train_x
        self.train_y=train_y
        return self
    def predict_proba(self,test_x):
        total_x=self.train_x.append(test_x)
        total_y=self.train_y.reindex_like(total_x)
        total_y[test_x.index]=-1

        return LabelPropagation().fit(total_x,total_y).predict_proba(test_x)

m13=Semi()

from tools import SelectandPredict
m14=SelectandPredict(mb,60)