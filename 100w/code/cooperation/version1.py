import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.base import clone
import numpy as np

from tools import *

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings('ignore')
    # 数据的预处理
    if False:

        # 先读取数据，得到DataFrame形式的数据
        test_x_raw = pd.read_csv('./data/test_A.tsv', sep='\t')
        train_y_raw = pd.read_csv('./data/train_id.tsv', sep='\t')
        train_x_raw = pd.read_csv('./data/train.tsv', sep='\t')

        train_y = train_y_raw.set_index('PERSONID').sort_index().LABEL

        train_i = train_x_raw.PERSONID.unique()
        test_i = test_x_raw.PERSONID.unique()
        all_x_raw = train_x_raw.append(test_x_raw)
        all_x = agg(all_x_raw)
        # train_x=agg(train_x_raw)
        # test_x=agg(test_x_raw)

        if True:
            # 试验项目：
            mj_all = ajax_agg(all_x_raw)  # TODO:用一个函数直接处理all_x_raw
            all_x = all_x.join(mj_all)
            train_x = all_x.loc[train_i, :].sort_index()
            test_x = all_x.loc[test_i, :].sort_index()

            # mj_train = ajax_agg(train_x_raw)
            # mj_test = ajax_agg(test_x_raw)

            # dm = docvec()
            # myvec_train = dm.fit_transform(train_x_raw)
            # myvec_test = dm.transform(test_x_raw)

            # train_x = train_x.join(mj_train)  # .join(myvec_train)
            # test_x = test_x.join(mj_test)  # .join(myvec_test)

        save_p(train_x, 'train_x')
        save_p(test_x, 'test_x')
        save_p(train_y, 'train_y')

    else:
        test_x_raw = pd.read_csv('./data/test_A.tsv', sep='\t')
        train_y_raw = pd.read_csv('./data/train_id.tsv', sep='\t')
        train_x_raw = pd.read_csv('./data/train.tsv', sep='\t')

        train_x = read_p('train_x')
        test_x = read_p('test_x')
        train_y = read_p('train_y')

    # 一些在加载后定义的工具库
    if True:
        def qeva(df, model, seed=950307, ):  # 快速eva
            if type(df) is pd.Series:
                df = df.to_frame()

            return evaa(df, train_y, model)

        def output(model, train_x, test_x, name='result6.csv'):
            p = model.fit(train_x.values, train_y.values).predict_proba(test_x.values)[:, 1]
            r = pd.DataFrame(p, index=test_x.index)
            r.to_csv('./data/' + name, header=False, sep='\t')

    # 待使用的模型
    from lightgbm import LGBMClassifier

    m1 = LGBMClassifier(boosting_type='dart', is_unbalance=True, objective='xentropy',
                        seed=1239329, n_estimators=200, colsample_bytree=0.9)

    from sklearn.ensemble import ExtraTreesClassifier

    m2 = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35,
                              min_samples_leaf=14, min_samples_split=15,
                              n_estimators=200, class_weight='balanced', n_jobs=4)
    from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier

    m3 = GradientBoostingClassifier(max_features=0.35,
                                    min_samples_leaf=14, min_samples_split=15,
                                    n_estimators=200)
    # m4=AdaBoostClassifier(   n_estimators=200)
    # m5=RandomForestClassifier(max_features=0.35,
    #                           min_samples_leaf=14, min_samples_split=15,
    #                           n_estimators=200, class_weight='balanced')
    m5 = RandomForestClassifier(bootstrap=False,
                                criterion='entropy', max_features=0.4,
                                min_samples_leaf=16, min_samples_split=14, n_estimators=200)

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
                Dense(200, input_dim=44, activation='relu'),
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


        mk = KerasClassifier(make_model1)

        m7 = Pipeline([('a', StandardScaler()), ('keras', mk)])

    # nn=neww()
    # new=nn.fit_transform(train_x_raw)
    # 统计各种药的出现频率

    # best=['max3', 'm_sum', 'FTR9maxd', 'FTR9kurt', 'FTR8max', 'FTR5sum', 'FTR5kurt', 'FTR50avg', 'FTR48std',
    #       'FTR48avg', 'FTR47skew', 'FTR44sum', 'FTR43sum', 'FTR42sum', 'FTR41sum', 'FTR41avgd', 'FTR38sum',
    #       'FTR36sum', 'FTR36stdd', 'FTR35skewd', 'FTR34sum', 'FTR34skewd', 'FTR34skew', 'FTR34maxd',
    #       'FTR34avg', 'FTR33sum', 'FTR33skewd', 'FTR33kurtd', 'FTR33avgd', 'FTR32sum', 'FTR32kurtd',
    #       'FTR30sum', 'FTR29stdd', 'FTR28sum', 'FTR28skewd', 'FTR18sum', 'FTR16kurt', 'FTR14avg', 'FTR12sum',
    #       'FTR10avgd', 'FTR0sum', 'counts', 'countd', 'm_cate_sum']

    # x=train_x[best]
    # import catboost as cb
    # m8 = cb.CatBoostClassifier()#在xbest上表现不错
    # m9 = OnegoStackingClassifier([m1, m2, m3, m5, m6,m8], LogisticRegression(), n=3)

    # from sklearn.feature_selection import SelectKBest,SelectFromModel,SelectFdr,SelectFpr,SelectFwe,RFECV
    # m10=Pipeline([('select',SelectFromModel(clone(m1),'mean',False)),('predict',m1)])
    # s=pd.Series(index=list(range(5,100,5)))
    # for i in s.index:
    #     m11=SelectandPredict(m1,i)
    #     score=qeva(train_x,m11)
    #     print(i,score)
    #     s[i]=score

    # import auto_ml

    # column_descriptions={'LABEL':'output'}
    # m10=auto_ml.Predictor('classifier', column_descriptions=column_descriptions)
    # m11=m10.train(x.join(train_y),model_names=['DeepLearningClassifier'])
    # for m in [m1,m2,m3,m5,m6]:
    #    print(evaa(x.fillna(-99),train_y,m))

    # new2=train_x_raw.FTR0.apply(lambda x: 0 if round(x, 2) == 0.05 else 1).to_frame().join(train_x_raw.PERSONID)\
    #     .groupby( 'PERSONID').mean() #对FTR0的重新处理
    # s=pd.Series(index=train_x.columns)
    # seed=123
    # import random
    # for ftr in s.index:
    #     seed+=1
    #     random.seed(seed)
    #     sr=evaa(train_x.drop(ftr,axis=1).fillna(-99),train_y,m2)
    #     print(ftr,'   ',sr)
    #     s[ftr]=sr

    # 先聚类，再分类
    if False:
        k = KMeans()
        c = pd.DataFrame(k.fit_transform(train_x.fillna(-99)), train_x.index)
        # print(qeva(train_x.join(c),m1))

    if True:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation


        def list2str(l):
            s = ''
            for sec in l:
                s = s + sec + ','
            return s


        uni_word = train_x_raw.groupby('PERSONID').apply(lambda ff: list2str(list(ff.FTR51)))
        cui = CountVectorizer()
        bag = cui.fit_transform(uni_word)
        lda = LatentDirichletAllocation(3)
        theme = pd.DataFrame(lda.fit_transform(bag), index=train_x.index, )

        #
        # a2=pd.DataFrame(a1.toarray(),index=train_x.index,columns=cui.get_feature_names())
        # l=(a2!=0).sum().sort_values()[1000:]
        # keyword=a2.apply(lambda x:x.argmax(),axis=1)
        #
        # a3=keyword.apply(lambda x: x if x in l else 'wtf')
        # from gensim.models import LdaModel,LsiModel,TfidfModel
        #
        # from gensim.corpora import Dictionary
        #
        # d=Dictionary([list(k)])
        # corpus = [d.doc2bow([text]) for text in k]
        # tfidf = TfidfModel(corpus)
        # corpus_tfidf = tfidf[corpus]
        # lda = LdaModel(corpus_tfidf, id2word=d, num_topics=10)
        #
        # class mycorpus():
        #     def __init__(self,a):
        #         self.a=a
        #     def __iter__(self):
        #         for i in self.a.shape[0]:
        #             yield dictionary.doc2bow(line.lower().split())
        #
        #
        # c = mycorpus(a1)
        # lda = LsiModel(corpus=c, num_topics=20)

        #
        # from sklearn.preprocessing import LabelEncoder
        # le = LabelEncoder()
        # a4 = pd.Series(le.fit_transform(a3))
        # k=train_x_raw.groupby('PERSONID').apply(lambda ff: list(ff.FTR51.str.split(',', expand=True).unstack().unique()))
        # vectorizer = CountVectorizer()
        # transformer = TfidfTransformer()
        # a1=vectorizer.fit_transform(train_x_raw.FTR51)
        # p = pd.Series(index=list(range(a1.shape[1])))
        #
        # for i in range(a1.shape[1]):
        #     ai = a1[:, i]
        #     #c = a1.
        #     p[i] = ai.nnz
        # a2=pd.DataFrame(data=a1,index=train_x_raw.PERSONID).groupby('PERSONID').sum()
        # tfidf = transformer.fit_transform(a2)
        #
        # a = train_x_raw[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',',expand=True)
        # aa=a.unstack()
        # from sklearn.preprocessing import LabelEncoder
        # b = aa.dropna().reset_index().drop(
        #     'level_0', axis=1).rename(columns={0: 'FTR51'})
        # b.loc[-99,'FTR51'] = 'wtf'
        # le = LabelEncoder().fit(b.FTR51)
        # c=a.fillna('wtf').apply(le.transform)
        # f33 = train_x_raw.set_index('APPLYNO').FTR33.sort_index()
        # model = Sequential()
        # model.add(Embedding(len(le.classes_), 10, input_length=69))
        # model.add(Flatten())
        # model.add(Dense(200, activation='relu'))
        # model.add(Dense(50, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # model.fit(c.sort_index(),f33,10000)

    # Keras模型试验
    if False:
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
                Dense(200, input_dim=44, activation='relu'),
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


        mk = KerasClassifier(make_model1)

        m7 = Pipeline([('a', StandardScaler()), ('keras', mk)])
        # print(qeva(x.fillna(-99), m3), )


        def make_model2():
            model = Sequential([
                Dense(200, input_dim=428, activation='tanh'),
                Dropout(0.5),
                Dense(100, activation='sigmoid'),
                Dropout(0.5),
                Dense(1, activation='sigmoid'),
            ])
            model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model


        def make_model3():
            model = Sequential([
                Dense(44, input_dim=44, kernel_initializer=initializers.he_normal(),
                      kernel_regularizer=regularizers.l2(0.001),
                      bias_regularizer=regularizers.l2(0.001)),
                PReLU(),
                BatchNormalization(),
                Dropout(0.75),

                Dense(16, kernel_initializer=initializers.he_normal(), kernel_regularizer=regularizers.l2(0.001),
                      bias_regularizer=regularizers.l2(0.001)),
                PReLU(),
                BatchNormalization(),
                Dropout(0.5),

                Dense(1, kernel_initializer=initializers.he_normal(), activation='sigmoid')])

            # sgd = optimizers.sgd(lr=0.001, decay=0.995)
            model.compile(loss='binary_crossentropy', optimizer='adam')
            return model


        mk = KerasClassifier(make_model1)

        m3 = Pipeline([('a', StandardScaler()), ('keras', mk)])
        r = evaa(train_x.fillna(-99), train_y, m3)
        print(r)

    # 模型压缩
    # if True:
    #     input = Input(shape=(428,))
    #     encoded = Dense(200, activation='relu')(input)
    #     encoded = Dense(64, activation='relu')(encoded)
    #     encoded = Dense(20, activation='relu')(encoded)
    #     encoder_output = Dense(2)(encoded)
    #
    #     decoded = Dense(20, activation='relu')(encoder_output)
    #     decoded = Dense(64, activation='relu')(decoded)
    #     decoded = Dense(128, activation='relu')(decoded)
    #     decoded = Dense(428, activation='tanh')(decoded)
    #     encoder = Model(input=input, output=encoder_output)
    #     autoencoder = Model(input=input, output=decoded)
    #     autoencoder.compile(optimizer='adam', loss='mse')
    #     x_std=StandardScaler().fit_transform( train_x.fillna(-99))
    #     autoencoder.fit(x_std,x_std)
    #     x_compassed=pd.DataFrame( encoder.predict(x_std),index=train_x.index)
    #     #得到压缩后的特征

    # gplearn符号回归
    # from gplearn.genetic import SymbolicRegressor,SymbolicTransformer

    # from gplearn.fitness import make_fitness
    # m=SymbolicRegressor()
    # def my_auc(y,y_pred,w):
    #     if len(y)==len(np.array([1, 1])) :
    #         return 0.5
    #     return eva()
    # #m = SymbolicRegressor(metric=make_fitness(my_auc,True))
    # from sklearn.pipeline import Pipeline
    # m=Pipeline([('gpt',SymbolicTransformer()),
    #             ('lgbm',m1)])
    #
    # m.fit(train_x.fillna(-99),train_y)

    # topt寻优
    # from tpot import TPOTClassifier
    # from sklearn.datasets import load_digits
    # from sklearn.model_selection import train_test_split
    #
    # tpot = TPOTClassifier(generations=5, population_size=60, verbosity=3, scoring='roc_auc')
    # tpot.fit(train_x.fillna(-99), train_y)

    # from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
    # import hyperopt
    # space={'n_estimators':hp.quniform('n_estimators',50,300,20),
    #         'learning_rate':hp.uniform('learning_rate',0.05,0.5),
    #         'num_leaves':hp.quniform('num_leaves',20,50,5),
    #         'colsample_bytree':hp.uniform('colsample_bytree',0.3,1),
    #         'subsample':hp.uniform('subsample',0.3,1),
    #         'subsample_freq':hp.quniform('subsample_freq',1,3,1),
    #         'reg_alpha':hp.uniform('reg_alpha',0.5,2),
    #         'reg_lambda':hp.uniform('reg_lambda',0.5,2),
    #         'drop_rate':hp.uniform('drop_rate',0.05,0.15),
    #         'skip_drop':hp.uniform('skip_drop',0.4,0.6),
    #         'max_drop':hp.quniform('max_drop',40,60,1)}
    # x=train_x.loc[:,SelectandPredict(m1,40).fit(train_x,train_y).best]
    # def my(args):
    #     for a in ['n_estimators','num_leaves','subsample_freq','max_drop']:
    #         args[a]=int(args[a])
    #     print(args)
    #
    #     s=qeva(x,LGBMClassifier(boosting_type='dart', is_unbalance=True, objective='xentropy',
    #                     seed=1239329,**args))
    #     print(s)
    #     print('-----------')
    #     return -s
    # best=fmin(my,space,algo=hyperopt.tpe.suggest,max_evals=1000)

bestargs = {'colsample_bytree': 0.69, 'drop_rate': 0.062, 'learning_rate': 0.117, 'max_drop': 51, 'n_estimators': 280,
            'num_leaves': 40, 'reg_alpha': 0.905, 'reg_lambda': 1.48, 'skip_drop': 0.42, 'subsample': 0.9,
            'subsample_freq': 1}
