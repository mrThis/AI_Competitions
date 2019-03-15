from utils import timer,read_dict

import logging

logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)
#
# import jieba
# import jieba.posseg
# jieba.load_userdict('../dicts/all_words.txt')
# cuter1=lambda j:list(jieba.cut(j))
#
# import thulac
# t=thulac.thulac(T2S=True,seg_only=True,model_path='../models/'
#                 ,user_dict='../dicts/all_words.txt')
#
# cuter2=lambda string:[j[0] for j in t.cut(string)]
#
# from pyhanlp import HanLP,JClass
# from pyhanlp import JClass as SafeJClass
#
#
#
# CustomDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CustomDictionary")
# #jtag = SafeJClass('com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger')()
# #NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
# aw=read_dict('all_words.txt')
# for w in aw:
#     CustomDictionary.add(w)
# cuter3=lambda string: [j.word for j in HanLP.segment(string)]
# #cuter3_2=lambda string: [j.word for j in NLPTokenizer.segment(string)]
#
# from pyltp import Segmentor
#
# segmentor = Segmentor()
#
# segmentor.load_with_lexicon('../ltp_data/cws.model', '../dicts/all_words.txt')
# cuter4=segmentor.segment
#
# from settings import use_cuter3_NLP
# if use_cuter3_NLP:
#     NLPTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
#     cuter3_2 = lambda string: [j.word for j in NLPTokenizer.segment(string)]
#     cs=(cuter1,cuter2,cuter3_2,cuter4) #TODO:3——2
# else:
#     cs=(cuter1,cuter2,cuter3,cuter4)
#
# def my_cuter(string):
#     l=[]
#     for cuter in cs:
#         l+=list(cuter(string))
#     return l
import pymongo
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db=client['cuters']
cuter1ed=db['cuter1ed']
cuter2ed=db['cuter2ed']
cuter3ed=db['cuter3ed']
cuter4ed=db['cuter4ed']
cuter5ed=db['cuter5ed']
cuter6ed=db['cuter6ed']
#@profile
def my_cuter(id):
    titles=[]
    contexts=[]
    for c in (cuter1ed,cuter2ed,cuter3ed,cuter4ed):
        d=c.find_one({'id':id})
        titles+=d['title']
        contexts+=d['context']
    return titles,contexts

import pandas as pd
from jieba.analyse.tfidf import IDFLoader, DEFAULT_IDF
from collections import Counter

# 加载idf词典
idf_loader = IDFLoader('../newidf.txt')
idf_freq, median_idf = idf_loader.get_idf()
#median_idf=10.2063
local_idf,useless=IDFLoader('../localnewidf.txt').get_idf()

# 加载stopwords词典
with open('../stopwords.txt', encoding='utf-8') as f:
    stop_words = set(f.read().split('\n'))

from settings import num_of_word_extract


def have_comma(s):

    for j in [',','.',';','#','、','《','》','。',' ']:
        if s.find(j)!=-1:
            return True
    return False

def clean(r):
    #去掉：过短的、全是数字的、标点开头的、有逗号的
    rr=[j for j in r if (len(j.split())==1) and j.isalnum() and not j.strip().isdigit()
        and not have_comma(j) and (1 < len(j) < 20) and not (j.lower() in stop_words)]
    return rr


# from gensim.models import Word2Vec,Doc2Vec
# model_w2v=Word2Vec.load('../pickles/model_w2v')
# model_d2v=Doc2Vec.load('../pickles/model_d2v')

from sklearn.externals.joblib import load
ck=load('../pickles/train_freq.p')


def search_pos(word, c, default='WTF'):
    return next((x[0][1] for x in c if x[0][0] == word), default)
#
# from functools import lru_cache
# import numpy as np
# from collections import Counter
# #用word2vec计算互信息，得到关键词权重
# @lru_cache(1000) #缓存结果，避免重复计算
# def predict_proba(oword, iword):
#     iword_vec = model_w2v[iword]
#     oword = model_w2v.wv.vocab[oword]
#     oword_l = model_w2v.syn1[oword.point].T
#     dot = np.dot(iword_vec, oword_l)
#     lprob = -sum(np.logaddexp(0, -dot) + oword.code * dot)
#     return lprob
#
# def keywords(s):
#     s = [w for w in s if w in model_w2v] #所有词汇
#     ws = {w: sum([predict_proba(u, w) for u in s]) for w in s} #某个词汇与其他所有词汇的相似度之和
#     return Counter(ws)

def gen_feature(id,text):
    # a = my_cuter(d2c(text.title))
    # b = my_cuter(d2c(text.context))
    # title_words = clean(a)
    # context_words = clean(b)
    # postag = dict(jieba.posseg.cut(d2c(text.title + '  。' + text.context)))

    a,b=my_cuter(id)
    title_words = clean( a )
    context_words =clean( b )
    postag = dict(cuter5ed.find_one({'id':id})['postag'])
    c=cuter6ed.find_one({'id':id},)['pos']
    # l=a+b
    # poss=jtag.tag(l)
    # c=sorted(Counter(zip(l, poss)).items(),key=lambda kk:(kk[0][0],kk[1]),reverse=True)
    #预排序用来寻找出现次数最多的词性

    td_title = Counter(title_words)
    td_context = Counter(context_words)
    freq = td_title + td_context
    total = sum(freq.values())  # 总词数

    # 计算tdidf并排序
    tdidf = {}
    for k in freq:
        tdidf[k] = freq[k] * idf_freq.get(k, median_idf) / total
    tdidf_tuple = sorted(tdidf, key=tdidf.__getitem__, reverse=True)


    candidates=set([j for j in tdidf_tuple[:num_of_word_extract]] + title_words)
    #print(id)
    # tdidf作为特征
    rrl={}
    for w in candidates:
        pos_ = postag.get(w, 'WTF')
        newpos = pos_[0]
        
        tdidf_r=tdidf[w]
        td_title_r= td_title[w]
        td_context_r = td_context[w]
        rank_r = tdidf_tuple.index(w)
        pos_r = pos_
        new_pos_r = newpos if newpos in ['n','v','x','W'] else 'H'
        hanlp_pos_r = search_pos(w,c)
        title_position_r = text['title'].find(w)
        title_position_r_r =len(text['title'])- text['title'].rfind(w)
        title_position_rr_r =  text['title'].rfind(w)
        position_r = text['context'].find(w)
        position_r_r = len(text['context']) - text['context'].rfind(w)
        position_rr_r = text['context'].rfind(w)
        length_r = len(w)
        idf_r=idf_freq.get(w, median_idf)
        local_idf_r=local_idf.get(w,10.2063)
        train_freq_r=ck.get(w,1)-1 #此处可能导致严重的过拟合，所以需要relu一下
        in_title_r= 1 if w in title_words else 0

        s=dict(zip(['tdidf', 'td_title', 'td_context', 'rank', 'pos','new_pos','hanlp_pos' ,'title_position',
        'title_position_r','title_position_rr', 'position', 'position_r', 'position_rr', 'length', 'idf','local_idf'
        'train_freq', 'in_title'],
        [tdidf_r,td_title_r,td_context_r,rank_r,pos_r,new_pos_r,hanlp_pos_r,title_position_r,title_position_r_r,
                   title_position_rr_r,position_r,position_r_r,position_rr_r,length_r,idf_r,local_idf_r,train_freq_r,
                   in_title_r]))
        rrl[w]=s
    r=pd.DataFrame.from_dict(rrl,orient='index')

    #r[ ['d2vec_{}'.format(k) for k in range(20)] ]=model_d2v.docvecs[id]

    r['td_title_gm'] = r['td_title'].mean()
    r['idf'] = r['idf'].mean()
    r['doclen'] = len(text.context)
    r['td_title_rk']=r['td_title'].rank()
    r['idf'] = r['idf'].rank()



    r['id'] = id
    r['wid'] = r.index.map(lambda j:j+id)
    r.reset_index(inplace=True,drop=False)
    r.set_index('wid',inplace=True)
    r.rename({'index': 'word'}, axis=1,inplace=True)



    return r
if __name__=='__main__':
    all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')
    r=gen_feature(all_text.loc['D014419'].name,all_text.loc['D014419',:])