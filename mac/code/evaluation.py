from utils import timer
from sklearn.model_selection import StratifiedKFold,cross_val_score
def read_dict(name):
    with open('../dicts/'+name,'r',encoding='utf-8') as f:
        ct=f.readlines()

    return [j.strip() for j in ct]


import jieba

jieba.load_userdict('../dicts/all_words.txt')
cuter1=lambda j:list(jieba.cut(j))

import thulac
with timer('加载thulac词库'):
    t=thulac.thulac(T2S=True,seg_only=True,model_path='../models/'
                    ,user_dict='../dicts/all_words.txt')

cuter2=lambda string:[j[0] for j in t.cut(string)]

from pyhanlp import HanLP,JClass,SafeJClass


#NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
CustomDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CustomDictionary")
aw=read_dict('all_words.txt')
for w in aw:
    CustomDictionary.add(w)
cuter3=lambda string: [j.word for j in HanLP.segment(string)]
#cuter3_2=lambda string: [j.word for j in NLPTokenizer.segment(string)]

from pyltp import Segmentor

segmentor = Segmentor()

segmentor.load_with_lexicon('../ltp_data/cws.model', '../dicts/all_words.txt')
cuter4=segmentor.segment

from settings import use_cuter3_NLP
if use_cuter3_NLP:
    NLPTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    cuter3_2 = lambda string: [j.word for j in NLPTokenizer.segment(string)]
    cs=(cuter1,cuter2,cuter3_2,cuter4) #TODO:3——2
else:
    cs=(cuter1,cuter2,cuter3,cuter4)

def my_cuter(string):
    l=[]
    for cuter in cs:
        l+=list(cuter(string))
    return l

def release():
    segmentor.release()


#一个分词器能切出的所有词
def get_words(cuter,all_text):
    l=Counter()
    for id,s in all_text.iterrows():
        c=Counter( clean( cuter( s['context']+s['title'] )))
        l+=c
    return l


if __name__=='__main__':
    import pandas as pd
    from tqdm import tqdm
    from utils import load

    tqdm.pandas(ncols=75)

    train_text=load('train_text')
    all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')

    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')

    train_set=load('train_set')
    train_x = load('train_x')
    train_y = load('train_y')




    # 加载stopwords词典
    with open('../stopwords.txt', encoding='utf-8') as f:
        stop_words = set(f.read().split('\n'))

    #分词器用的，清洗切出来的词
    def clean(l):
        return [w for w in l if not (len(w.strip()) < 2 or w.lower() in stop_words)]


    from collections import Counter

    #评估训练集中每个词被分出来的次数
    def hows_word():
        ll=Counter()
        for cuter in cs:
            l = Counter()
            for id, s in train_text.iterrows():
                c = Counter(clean(cuter(s['context'])))
                l += c
            ll+=Counter(list(l))
        return ll

    #对分词器评分
    def eva_cuter(cuter):

        k = train.keyword.str.split(',')
        r=pd.Series(0,index=k.index)
        for id, s in train_text.iterrows():
            c = set( cuter( s['context'] )).union(set(cuter( s['title'] ) ))
            for j in k.loc[id]:#对于每一关键词
                if j not in c:
                    r[id]+=1
        return r.sum()

    #my_cuter-->300--->69
    #cuter1:491--->164
    #cuter2:500--->197
    #cuter3:972--->154
    #cuter3_2:724-->727
    #cuter4:560-->319


    #检查一下切不出来的词
    def what_the_word(cuter):
        l=[]
        k = train.keyword.str.split(',')
        r=pd.Series(0,index=k.index)
        for id, s in train_text.iterrows():
            c = set(cuter(s['context'])).union(set(cuter(s['title'])))
            for j in k.loc[id]: #对于每一关键词
                if j not in c:
                    l.append((id,j))
        return l

    #用来定位一个奇葩的词是在哪里出现的
    def search_word(word):
        l=[]
        for id, s in train_text.iterrows():
            if word in s['context'] or word in s['title']:
                l.append( ( s,train.keyword.loc[id] ))
        for id, s in all_text.iterrows():
            if word in s['context'] or word in s['title']:
                l.append(s)
        return l



    from sklearn.externals.joblib import load as load_m

    m1 = load_m('../pickles/model.p')
    ks=[]
    # for kk in train.keyword.str.split(','):
    #     ks += kk
    # ck = Counter(ks)
    # dump(ck, '../pickles/train_freq.p')

    def get_label(dfp):
        k=dfp.word[-2:]
        return pd.Series({'label1':k[0],'label2':k[1]})

    def score(tx):

        p = pd.DataFrame(m1.fit(tx, train_y).predict_proba(tx)[:, 1], columns=['proba'], index=tx.index)
        p['id'] = train_set.id
        p['word'] = train_set.word

        rr = p.sort_values(by=['id', 'proba']).groupby('id').apply(get_label)

        return rr.join(train).apply(
            lambda j: (j['label1'] in j['keyword']) + (j['label2'] in j['keyword'])
            , 1).apply(lambda j: j / 2).sum()
    import numpy as np
    #评价模型的二分类能力
    def qeva(x,model, verbose=True):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        scores = cross_val_score(model, x, train_y, scoring='roc_auc', verbose=verbose, cv=skf, )
        return np.mean(scores), scores

    print('分类器的性能是：',round( qeva(train_x,m1)[0] ,5 ))
    # print('分词器不能成功发现的词数为：',eva_cuter(my_cuter))
    # print('在训练集上的评分是：',score(train_x))


    #is_book=train_set.progress_apply(lambda k:k.word in bid ,axis=1)
    # s=train_set
    #r=pd.merge(s.groupby('id').mean(), s, left_index=True, right_on='id')
    #r=s.groupby('id')['td_title'].apply(lambda j:j-j.mean())
    #ss=s.groupby('id')['td_title'].apply(lambda j:pd.Series(j.mean(),index=j.index))
    # l=[]
    # for name in [ 'td_title', 'idf']:
    #     l.append(s.groupby('id')[name].apply(lambda j:pd.Series(j.mean(),index=j.index)))
    # for name in [ 'td_title', 'idf']:
    #     l.append(s.groupby('id')[name].apply(lambda j:j.rank()))
    #ss=s.groupby('id')['td_title'].apply(lambda j:j.rank())

    #sss = pd.concat(l, 1)
    # q=qeva(train_x.join(sss.reindex(index=train_y.index), rsuffix='d'), m1)
    # print(q)
    x = train_x

    #import re
    #have_letter=train_set.word.map(lambda w: bool(re.search('[A-Za-z]', w) )).rename('letter')
    #have_digit = train_set.word.map(lambda w: bool(re.search('[0-9]', w))).rename('digit')
    #print(qeva(x.join(have_digit).join(have_letter),m1,verbose=False))
    # for name in x:
    #     print(name,qeva(x.drop(name,1),m1,verbose=False))
    from lightgbm.sklearn import LGBMClassifier,LGBMRegressor
    #m2=LGBMClassifier(n_estimators=100,reg_lambda=1,num_leaves=14)
    #qeva(x,LGBMClassifier(n_estimators=100,reg_lambda=3,num_leaves=14))
    raise

    r=pd.read_csv('../output/final_op12.csv')
    pdw=r.label1.append(r.label2).reset_index(drop=True)
    tpdw=pd.Series(ks)
    tpdw.to_frame().join(tpdw.apply(len).rename('len')).sort_values('len')

    j = SafeJClass('com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger')()
    d=train_text.loc['D103531', 'context']
    from collections import Counter
    l=my_cuter(d)
    poss=j.tag(l)
    c=Counter(zip(l, poss))

    c=sorted(c.items(),key=lambda kk:(kk[0][0],kk[1]),reverse=True)
    def search_pos(word,c,default='WTF'):
        return next((x[0][1] for x in c if x[0][0] == word), default)


    #def get_from_c(word,c):
        #for w,pos in

    from mj_usefulmodel import *

    m5 = MyKeras(deepth=4, dp_rate=0.5, size=256, bn=True, activation='prelu',
                   lr=0.002, l2=0)
    xx=x.copy()
    from sklearn.preprocessing import LabelEncoder
    xx.pos=LabelEncoder().fit_transform(xx.pos)
    xx.new_pos = LabelEncoder().fit_transform(xx.new_pos)



    raise
    from gensim.models import Word2Vec,Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    docs = [cuter1(ser.title + '。' + ser.context) for k, ser in train_text.iterrows()]
    all_text = all_text.fillna('')

    class alldocs():
        def __iter__(self):
            for k,ser in tqdm(all_text.iterrows()):
                yield TaggedDocument(words=clean(cuter1(ser.title + '。' + ser.context)), tags=[k])
    class allwdls():
        def __iter__(self):
            for k,ser in tqdm(all_text.iterrows()):
                words=cuter1(ser.title + '。' + ser.context)
                yield clean(words)



    model_w2v=Word2Vec(allwdls(),min_count=2,iter=10,sg=1, hs=1,size=20, window=10)
    model_d2v=Doc2Vec(alldocs(),size=20, window=10, min_count=1,iter=10,train_words=False)
    model_w2v.delete_temporary_training_data()
    model_d2v.delete_temporary_training_data()
    model_w2v.save('../pickles/model_w2v')
    model_d2v.save('../pickles/model_d2v')


    l = clean(cuter1(train_text.loc['D103531', 'context']))
    l = [k for k in l if k in model_d2v.wv]
    model_d2v.most_similar(l)
    docvecs=train_text.index.to_series().apply(lambda k:pd.Series(model_d2v.docvecs[k]))
    from sklearn.cluster import KMeans

    lei=pd.Series(KMeans().fit_predict(docvecs),index=docvecs.index)
    p=[]
    for w in docs[0]:
        try:
            p.append(model.wv[w])
        except:
            pass

    dv=np.mean(np.array(p), axis=0)

    j=SafeJClass('com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger')()
    j.tag(cuter1(l))
    from functools import lru_cache
    #用word2vec计算互信息，得到关键词权重
    @lru_cache(maxsize=1000)
    def predict_proba(oword, iword,mm):
        iword_vec = mm[iword]
        oword = mm.wv.vocab[oword]
        oword_l = mm.syn1[oword.point].T
        dot = np.dot(iword_vec, oword_l)
        lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
        return lprob

    def keywords(s,mm):
        s = [w for w in s if w in mm] #所有词汇
        ws = {w: sum([predict_proba(u, w,mm) for u in s]) for w in s} #某个词汇与其他所有词汇的相似度之和
        return Counter(ws)


    # train_list = train.keyword.str.split(',')
    # t=train_text.join(train_list)
    # def search(ser):
    #     for w in ser.keyword:
    #
    #         c=ser.title+ser.context
    #         if w=='':
    #             continue
    #         if c.find(w)<0:
    #             print("'{}':('{}',''),".format(ser.name,w))
    # eee=t.apply(search,1)



    """
    分类器的性能是： 0.90159
    分词器不能成功发现的词数为： 88
    在训练集上的评分是： 818.0
    """

    """
    分类器的性能是： 0.89911
[Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.5s finished
分词器不能成功发现的词数为： 96
在训练集上的评分是： 809.5
PyDev console: starting.
    """