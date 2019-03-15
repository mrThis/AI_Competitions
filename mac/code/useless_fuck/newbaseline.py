from utils import timer
from segmentor import read_dict
from sklearn.model_selection import StratifiedKFold,cross_val_score

import jieba
import jieba.posseg
jieba.load_userdict('../dicts/all_words.txt')
cuter1=lambda j:list(jieba.cut(j))

import thulac
with timer('加载thulac词库'):
    t=thulac.thulac(T2S=True,seg_only=True,model_path='../models/'
                    ,user_dict='../dicts/all_words.txt')

cuter2=lambda string:[j[0] for j in t.cut(string)]

from pyhanlp import HanLP,JClass


#NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
aw=read_dict('all_words.txt')
for w in aw:
    CustomDictionary.add(w)
cuter3=lambda string: [j.word for j in HanLP.segment(string)]
#cuter3_2=lambda string: [j.word for j in NLPTokenizer.segment(string)]

from pyltp import Segmentor

segmentor = Segmentor()

segmentor.load_with_lexicon('../ltp_data/cws.model', '../dicts/all_words.txt')
cuter4=segmentor.segment

cs=(cuter1,cuter2,cuter3,cuter4) #TODO:3——2

def my_cuter(string):
    l=[]
    for cuter in cs:
        l+=list(cuter(string))
    return l

import jieba
import jieba.analyse
import jieba.posseg
import pandas as pd
import numpy as np
from mj_utils import timer
from operator import itemgetter
from jieba.analyse.tfidf import IDFLoader, DEFAULT_IDF
from collections import Counter
from pyhanlp import HanLP
from mj_utils import *



# 加载idf词典
idf_loader = IDFLoader('../newidf.txt')
idf_freq, median_idf = idf_loader.get_idf()

# 加载stopwords词典
with open('../stopwords.txt', encoding='utf-8') as f:
    stop_words = set(f.read().split('\n'))

from tqdm import tqdm

tqdm.pandas(ncols=75)

 #全角转半角
def d2c(ustring):
    if type(ustring) is not str:
        ustring=''
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 0x3000:
            inside_code = 0x0020
        else:
            inside_code -= 0xfee0
        if not (0x0021 <= inside_code and inside_code <= 0x7e):
            rstring += uchar
            continue
        rstring += chr(inside_code)
    l=['◆','&amp','▼','\u200b','●','☟','�','\u3000','▼','❖','―End―','|']
    for ss in l:
        rstring = rstring.replace(ss, ' ')
    l1=['〖','「','【','[','<','{']
    for ss in l1:
        rstring = rstring.replace(ss, '《')
    l2=['〗','」','】',']','>','}']
    for ss in l2:
        rstring = rstring.replace(ss, '》')
    rstring=rstring.replace('!','！')
    return rstring

def clean(l):
    return [w for w in l if not (len(w.strip()) < 2 or w.lower() in stop_words)]

def gen_feature(id,text):
    title_words = clean( my_cuter(d2c(text.title )) )
    context_words =clean( my_cuter(d2c( text.context)) )
    postag = dict(jieba.posseg.cut(d2c(text.title+'  。'+ text.context)))

    td_title = Counter(title_words)
    td_context = Counter(context_words)
    freq = td_title + td_context
    total = sum(freq.values())  # 总词数

    # 计算tdidf并排序
    tdidf = {}
    for k in freq:
        tdidf[k] = freq[k] * idf_freq.get(k, median_idf) / total
    tdidf_tuple = sorted(tdidf, key=tdidf.__getitem__, reverse=True)

    # 得到候选词（tdidf前五加上title里面的所有word）
    r = pd.DataFrame(index=set([j for j in tdidf_tuple[:5]] + title_words))

    # tdidf作为特征
    for w in r.index:
        r.loc[w, 'tdidf'] = tdidf[w]
        r.loc[w, 'td_title'] = td_title[w]
        r.loc[w, 'td_context'] = td_context[w]
        r.loc[w, 'rank'] = tdidf_tuple.index(w)
        r.loc[w, 'pos'] = postag.get(w, 'NULL')
        r.loc[w, 'position'] = text['context'].find(w)
        # r.loc[w, 'pos_pct']=s['context'].find(w)/len(s['context'])
        r.loc[w, 'length'] = len(w)
        r.loc[w,'idf']=idf_freq.get(w, median_idf)
        # 在不在标题里作为特征
        if w in title_words:
            r.loc[w, 'in_title'] = 1
        else:
            r.loc[w, 'in_title'] = 0
    r['wid'] = id + '__' + r.index.to_series()
    r['id'] = id
    r = r.reset_index().set_index('wid').rename({'index': 'word'}, axis=1)
    return r

if __name__=='__main__':
    all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')

    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')

    train_text = all_text.loc[train.index, :]
    test_text = all_text.drop(train.index)
    train_list = train.keyword.str.split(',')
    tl=[]
    for id,text in tqdm(train_text.iterrows(),total=train_text.shape[0],ncols=75):
        tl.append(gen_feature(id,text))
    train_set=pd.concat(tl)
    train_set.pos = pd.Categorical(train_set.pos)

    def get_label(dfp):
        k=dfp.word[-2:]
        return pd.Series({'label1':k[0],'label2':k[1]})

    from lightgbm.sklearn import LGBMClassifier

    m1 = LGBMClassifier(n_estimators=65,reg_alpha=3)

    train_x = train_set.drop(['word', 'id'], 1)

    train_y = train_set.apply(lambda j:1 if j.word in train_list.loc[j.id] else 0 , axis=1)
    def score(tx):

        p = pd.DataFrame(m1.fit(tx, y).predict_proba(tx)[:, 1], columns=['proba'], index=tx.index)
        p['id'] = train_set.id
        p['word'] = train_set.word

        rr = p.sort_values(by=['id', 'proba']).groupby('id').apply(get_label)

        return rr.join(train).apply(
            lambda j: (j['label1'] in j['keyword']) + (j['label2'] in j['keyword'])
            , 1).apply(lambda j: j / 2).sum()

    #评价模型的二分类能力
    def qeva(x,model, verbose=True):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        scores = cross_val_score(model, x, train_y, scoring='roc_auc', verbose=verbose, cv=skf, )
        return np.mean(scores), scores
    m1.fit(train_x,train_y)
    kl=[]
    tt=test_text[:1000]
    for id, text in tqdm(tt.iterrows(), total=tt.shape[0], ncols=75):
        test_set=gen_feature(id, text)
        test_set.pos = pd.Categorical(test_set.pos)
        test_x=test_set.drop(['word', 'id'], 1)

        p = pd.DataFrame(m1.predict_proba(test_x)[:, 1], columns=['proba'], index=test_x.index)
        keywords=pd.Series(list(test_set.join(p).sort_values('proba').word[-2:]),index=['label1','label2'])
        kl.append(keywords)