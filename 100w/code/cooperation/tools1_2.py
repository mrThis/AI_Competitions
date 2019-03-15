import pandas as pd
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import f1_score,roc_auc_score
from sklearn.base import clone
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
def agg(x):
    dff=pd.DataFrame(index=x.PERSONID.sort_values().unique(),columns=x.columns)
    columns=x.columns
    l=[]
    print('Step1：去除无用列')
    #去除缺失率极高的几个特征
    x=x.drop(['FTR19','FTR11','FTR6','FTR3','FTR1','FTR13','FTR22'],axis=1)

    #spe=(x[['FTR24','FTR40','FTR26','FTR49']]==0).join(x.PERSONID).groupby('PERSONID').sum()

    #去除需要特别处理的几个特征
    x = x.drop(['FTR24','FTR40','FTR26','FTR49'],axis=1)

    x.FTR46=x.FTR37-x.FTR46
    x.FTR47=x.FTR16-x.FTR47
    x.FTR4=x.FTR7-x.FTR4
    x.FTR9=x.FTR43-x.FTR9
    x.FTR38=x.FTR12-x.FTR38
    #去除重复特征
    x = x.drop(['FTR20'],axis=1)

    nuni = x.groupby('PERSONID').apply(lambda k: k.nunique()).drop(['PERSONID','APPLYNO'],axis=1)
    l+=['nuni']

    #FTR51暂时取其长度
    x['FTR51'] = x.FTR51.apply(lambda k:k.count(',')+1)



    #x=x.replace(0,np.nan)  #：有可能0代表缺失值

    print('Step2：直接计算统计特征')
    q=pd.to_datetime(x.CREATETIME).apply(lambda x:1 if x.month in [3,4,5,6,7,8] else 0.25)
    countadj=x.join(q.rename('adj')).groupby('PERSONID')['adj'].sum()

    def compare(s):
        gb=s.groupby('season')
        if gb.ngroups==1:
            return pd.Series(index=s.columns)
        else:
            sum=gb.sum()
            div=sum.loc[1.00]/(sum.loc[0.25])
            div=div.replace(np.inf,np.nan).replace(-np.inf,np.nan)
            return div

    gdp=x.drop(['CREATETIME', 'APPLYNO'], axis=1).groupby('PERSONID')
    import time

    incresing=x.join(q.rename('season')).groupby('PERSONID').apply(compare).unstack().drop(['APPLYNO','CREATETIME','PERSONID','season'],axis=1)

    #x=x[~q]
    print('debug-start', time.time())
    t1=time.time()
    counts = gdp.count().iloc[:,0]
    #lasts = gdp.last()
    avg=gdp.mean()
    max=gdp.max()
    std = gdp.std()
    skew = gdp.apply(lambda x:x.skew())
    kurt=gdp.apply(lambda x:x.kurt())
    sum = gdp.sum()
    mode = gdp.apply(lambda s: s.apply(lambda o:o.value_counts().iloc[0]))
    print('debug-stop', time.time()-t1)
    l+=['countadj','incresing','counts','avg','max','std','skew','kurt','sum','mode']

    print('Step3：按天加总，再计算统计特征')
    lastdm=(pd.to_datetime('2016-03-01')-pd.to_datetime( x.groupby('PERSONID').CREATETIME.last()))\
        .apply(lambda x:x.days).rename('lastdm')
    #countd = gdp.apply(lambda k: k.CREATETIME.nunique()).rename('countdate')


    #按天加总再统计
    gdpd=x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum()).groupby('PERSONID')

    avgd = gdpd.apply(lambda k:k.mean())
    maxd = gdpd.apply(lambda k:k.max())
    stdd = gdpd.apply(lambda k:k.std())
    skewd = gdpd.apply(lambda k:k.skew())
    kurtd = gdpd.apply(lambda k:k.kurt())


    l+=['lastdm','avgd','maxd','stdd','skewd','kurtd']

    print('Step4：新想法')
    weekmean=pd.to_datetime(x.set_index('PERSONID').CREATETIME).groupby('PERSONID').apply(lambda t:(t.dt.week + 1).mean())
    year2015 = pd.to_datetime(x.set_index('PERSONID').CREATETIME).apply(lambda x: 1 if x.year == 2015 else 0).groupby(
        'PERSONID').mean().rename('year2015')
    weekday = pd.get_dummies(pd.to_datetime(x.set_index('PERSONID').CREATETIME).apply(lambda x: x.weekday())).groupby(
        'PERSONID').mean().rename(columns={i:'weekday'+str(i) for i in range(7)})
    l+=['weekmean','year2015','weekday']

    # x1=x.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum()).reset_index().set_index('PERSONID')
    # m = pd.to_datetime(x1.CREATETIME).apply(lambda o: o.month).rename('month')
    # x2 = x1.reset_index().join(m.reset_index(drop=True))
    # x3 = x2.groupby("PERSONID").apply(lambda x: x.groupby('month').mean())

    # #月均值的统计特征
    # countm_av = x3.groupby('PERSONID').count().iloc[:,0]
    # avgm_av =  x3.groupby('PERSONID').mean()
    # maxm_av =  x3.groupby('PERSONID').max()
    # stdm_av =  x3.groupby('PERSONID').std()
    # skewm_av =  x3.groupby('PERSONID').skew()
    # kurtm_av =  x3.groupby('PERSONID').apply(lambda x:x.kurt())
    #
    # l+=['countm_av','avgm_av','maxm_av','stdm_av','skewm_av','kurtm_av']

    #调整月总和的统计特征


    for colname in l:
        col=eval(colname)
        if type(col) is pd.Series:
            dff=dff.join(col.rename(colname))
        else:
            dff=dff.join(col,rsuffix=colname)
    dff=dff.drop(columns,axis=1)
    return dff


def ajax_agg(train):
    def quarter(x):
        if x.month in [3, 4,5,6,7,8]:
            return 1
        else:
            return 0

    def cate(arr):
        return len(arr.unique())

    def mean(arr):
        return pd.value_counts(arr).mean()

    def max(arr):
        return pd.value_counts(arr).max()

    def min(arr):
        return pd.value_counts(arr).min()

    print('Step5：FTR51精细处理')
    a = train[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',',
                                                                         expand=True).unstack().dropna().reset_index().drop(
        'level_0', axis=1).rename(columns={0: 'FTR51'})
    b = train[['PERSONID', 'APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
    c = train[['APPLYNO', 'CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()
    x = pd.merge(a, b, on='APPLYNO')
    x = pd.merge(x, c, on='APPLYNO')

    A=x.FTR51.apply(lambda s: int(s.split('B')[0][1:])).to_frame().join(x.PERSONID)

    #xx = x.groupby('PERSONID').FTR51.apply(lambda y: y.drop_duplicates()).reset_index()
    # oftenmed = list(aa.FTR51.value_counts()[:800].index)
    # bb = aa.FTR51.apply((lambda k: 1 if k in oftenmed else 0))
    # cc = bb.to_frame().join(aa.PERSONID).groupby('PERSONID').sum().FTR51.rename('raremed')

    # m_sum :已被取代
    # m_sum = x.groupby('PERSONID')['FTR51'].count().rename('m_sum')
    # m_cate_sum
    m_cate_sum = x.groupby('PERSONID')['FTR51'].agg(cate).rename('m_cate_sum')
    # m_freq1,max1,min1
    mj1 = x.groupby('PERSONID')['APPLYNO'].agg({'m_freq1': mean, 'max1': max})
    # m_freq2,max2,min2
    mj2 = x.groupby('PERSONID')['CREATETIME'].agg({'m_freq2': mean, 'max2': max})
    # m_freq3,max3,min3
    x['MONTH'] = pd.to_datetime(x.CREATETIME).apply(lambda x: x.month)
    x[lambda k: k['MONTH'] in (1, 2, 9, 10, 11, 12)]=x[lambda k:k['MONTH'] in (1,2,9,10,11,12)]/4
    mj3 = x.groupby('PERSONID')['MONTH'].agg({'m_freq3': mean, 'max3': max })
    #add4=A.groupby('PERSONID')
    # # m_freq4,max4,max4
    # x['Quarter'] = pd.to_datetime(x.CREATETIME).apply(quarter)
    # mj4 = x.groupby('PERSONID')['Quarter'].agg({'m_freq4': mean, 'max4': max, 'min4': min})

    mj = mj1.join(mj2).join(mj3).join(m_cate_sum)#.join(mj4)
    return mj


def evaa(train_x,train_y,m):
    skf=StratifiedKFold(5,True,20180718)
    s=cross_val_score(m, train_x, train_y, cv=skf, scoring='roc_auc')
    print(s)
    return s.mean()

def getip(m,x,y): #计算重要性到剪贴板
    ip = pd.Series(m.fit(x,y).feature_importances_, index=x.columns).to_clipboard()

work_path  = r"./"
pickle_path= "./pickles/"
#将变量以指定的名字pickle下来
def read_p(name):
    return pd.read_pickle(pickle_path+name+'.p')

def save_p(var,name):
    var.to_pickle(pickle_path+name+'.p')

def real_level(var):
    return var.nunique()+var.isnull().any()

class cui:
    def __init__(self, lis):
        self.lis = lis

    def __iter__(self):
        for idx, doc in enumerate(self.lis):
            yield TaggedDocument(words=doc, tags=[idx])
def my(gss):
    ls = []
    for ll in gss:
        ls += ll
    return ls

class docvec():
    def __init__(self):
        self.d2vm=None
    def fit(self,x):
        a = x[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',')
        b = a.to_frame().join(x.reset_index()[['CREATETIME', 'PERSONID', 'APPLYNO']].set_index('APPLYNO'))
        c = b.groupby('PERSONID').FTR51.apply(my)
        self.d2vm = Doc2Vec(cui(c), vector_size=10)
        return self
    def transform(self,x):
        a = x[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',')
        b = a.to_frame().join(x.reset_index()[['CREATETIME', 'PERSONID', 'APPLYNO']].set_index('APPLYNO'))
        c = b.groupby('PERSONID').FTR51.apply(my)
        d = c.apply(lambda doc: pd.Series(self.d2vm.infer_vector(doc)))
        return d
    def fit_transform(self,x):
        a = x[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',')
        b = a.to_frame().join(x.reset_index()[['CREATETIME', 'PERSONID', 'APPLYNO']].set_index('APPLYNO'))
        c = b.groupby('PERSONID').FTR51.apply(my)
        self.d2vm = Doc2Vec(cui(c), vector_size=10)
        d = c.apply(lambda doc: pd.Series(self.d2vm.infer_vector(doc)))
        return d


from sklearn.decomposition import LatentDirichletAllocation

class neww():
    def fit_transform(self, train):

        a = train[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',',
                                                                             expand=True).unstack().dropna().reset_index().drop(
            'level_0', axis=1).rename(columns={0: 'FTR51'})
        b = train[['PERSONID', 'APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
        c = train[['APPLYNO', 'CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()
        x = pd.merge(a, b, on='APPLYNO')
        x = pd.merge(x, c, on='APPLYNO')

        self.often = x.FTR51.value_counts().index[:100]
        self.lda = LatentDirichletAllocation(8)
        sn = pd.Series(index=x.index)
        for i, med in enumerate(self.often):
            sn[x.FTR51 == med] = i
        snn = sn.fillna(20)
        new = x.PERSONID.to_frame().join(pd.get_dummies(snn)).groupby('PERSONID').sum()


        return pd.DataFrame( self.lda.fit_transform(new),index=new.index)

    def transform(self, train):
        a = train[['APPLYNO', 'FTR51']].set_index('APPLYNO').FTR51.str.split(',',
                                                                             expand=True).unstack().dropna().reset_index().drop(
            'level_0', axis=1).rename(columns={0: 'FTR51'})
        b = train[['PERSONID', 'APPLYNO']].set_index('APPLYNO').PERSONID.reset_index()
        c = train[['APPLYNO', 'CREATETIME']].set_index('APPLYNO').CREATETIME.reset_index()
        x = pd.merge(a, b, on='APPLYNO')
        x = pd.merge(x, c, on='APPLYNO')
        sn = pd.Series(index=x.index)
        for i, med in enumerate(self.often):
            sn[x.FTR51 == med] = i
        snn = sn.fillna(20)
        new = x.PERSONID.to_frame().join(pd.get_dummies(snn)).groupby('PERSONID').sum()
        return pd.DataFrame(self.lda.transform(new),index=new.index)


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
                print('fitting',self.base_classifiers[model_no])
                self.base_classifiers[model_no].fit(X[traincv, ], y[traincv])
                predicted_y_proba = self.base_classifiers[model_no].predict_proba(X[testcv,])[:, 1]
                stacking_train[testcv, model_no] = predicted_y_proba

            self.base_classifiers[model_no].fit(X, y)
        self.combiner.fit(stacking_train, y)
        return self
    def predict_proba(self, X):
        stacking_predict_data = np.full(
            (np.shape(X)[0], len(self.base_classifiers)),
            np.nan
        )
        for model_no in range(len(self.base_classifiers)):
            stacking_predict_data[:, model_no] = self.base_classifiers[model_no].predict_proba(X)[:, 1]
        return self.combiner.predict_proba(stacking_predict_data)

from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest

class MyCC(BaseEstimator):
    def __init__(self,total_x,n):
        self.n=n
        self.total_x=total_x
        self.cluster=IsolationForest().fit(total_x)
        self.lb=pd.Series( self.cluster.predict(total_x),index=total_x.index)
        self.lb[self.lb==1]=0
        self.lb[self.lb==-1]=1
        m = LGBMClassifier(boosting_type='dart', is_unbalance=True, objective='xentropy',
                            seed=1239329, n_estimators=200, colsample_bytree=0.9)
        self.estimators=[clone(m),clone(m)]
    def fit(self,train_x,train_y):
        train_lb=self.lb[train_x.index]
        for i in [0,1]:
            self.estimators[i].fit(train_x[train_lb == i], train_y[train_lb == i])

        return self
    def predict_proba(self,test_x):
        yp=np.zeros((test_x.shape[0],2))
        test_lb=self.lb[test_x.index]
        for i in [0,1]:
            yp[test_lb == i,:]=self.estimators[i].predict_proba(test_x[test_lb == i])
        return yp

# class SelectandPredict(BaseEstimator):
#     def __init__(self,m,n):
#         self.m=m
#         self.n=n
#         self.best=[]
#     def fit(self,train_x,train_y):
#         self.best=getbest()
#         self.m.fit(train_x[self.best],train_y)
#         return self
#     def predict_proba(self,test_x):
#         return self.m.predict_proba(test_x[self.best])

# def getbest(m,n,train_x,train_y):
#     ip = pd.Series(clone(m).fit(train_x, train_y).feature_importances_, index=train_x.columns)
#     return ip.sort_values(ascending=False)[:n]

def eva_for_gcf(gcf,x,y,**args):
    x=x.values
    y = y.values
    sum_auc_N = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(x, y):
        x_train, y_train, x_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]
        modell_fit =gcf(**args)
        modell_fit.cascade_forest(x_train, y_train)
        y_prob = modell_fit.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        sum_auc_N.append(auc)
    return np.mean(sum_auc_N)
