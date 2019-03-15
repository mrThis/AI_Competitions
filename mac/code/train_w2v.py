from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import timer,read_dict,mynext
from collections import Counter

import pymongo
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db=client['cuters']
cuter1ed=db['cuter1ed']
cuter2ed=db['cuter2ed']
cuter3ed=db['cuter3ed']
cuter4ed=db['cuter4ed']
cuterall=db['cuterall']

all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')
#raise
all_text = all_text.fillna('')
# 加载stopwords词典
with open('../stopwords.txt', encoding='utf-8') as f:
    stop_words = set(f.read().split('\n'))

def have_comma(s):

    for j in [',','.',';','#','、','《','》','。',' ']:
        if s.find(j)!=-1:
            return True
    return False

def clean(r):
    #去掉：过短的、全是数字的、标点开头的、有逗号的
    rr=[j for j in r if (len(j.split())==1) and j.isalnum() and not j.strip().isdigit()
        and not have_comma(j) and (1 < len(j) < 20) and not (j.lower() in stop_words)]
    return set(rr)
if False:  #合并分词结果
    def combine_cuter(id):
        titles=[]
        contexts=[]
        for c in (cuter1ed,cuter2ed,cuter3ed,cuter4ed):
            d=c.find_one({'id':id})
            titles+=d['title']
            contexts+=d['context']
        #k=titles+contexts
        cuterall.insert_one({'id':id,'title':titles,'context':contexts})
        return titles+contexts

    cuterall.drop()
    cuterall.create_index([('id', pymongo.ASCENDING)],
                                   unique=True)
    for k,v in tqdm(all_text.iterrows(),ncols=65,total=all_text.shape[0]):
        combine_cuter(k)

else:
    def my_cuter(id):
        d=cuterall.find_one({'id':id})
        k=d['title']+d['context']
        return k

    class alldocs():
        # def __init__(self):
        #     self.db=cuterall.find(no_cursor_timeout=True)
        def __iter__(self):
            # for d in tqdm(self.db,total=all_text.shape[0],ncols=65):
            #     k = d['title'] + d['context']
            #     yield TaggedDocument(words=clean(k), tags=[d['id']])
            for k,ser in tqdm(all_text.iterrows(),ncols=65,total=all_text.shape[0]):
                yield TaggedDocument(words=clean(my_cuter(k)), tags=[k])

    class allwdls():
        # def __init__(self):
        #     self.db=cuterall.find(no_cursor_timeout=True)
        def __iter__(self):
            # for d in tqdm(self.db,total=all_text.shape[0],ncols=65):
            #     k = d['title'] + d['context']
            #     yield clean(k)
            for k,ser in tqdm(all_text.iterrows(),ncols=65,total=all_text.shape[0]):
                words=clean(my_cuter(k))
                yield clean(words)


    #cuterall.drop()
    # cuterall.create_index([('id', pymongo.ASCENDING)],
    #                                unique=True)

    model_w2v=Word2Vec(allwdls()
                       ,min_count=5,iter=10,sg=1, hs=1,size=30, window=10,workers=8)
    model_w2v.save('../pickles/model_w2v')
    raise
    model_d2v=Doc2Vec(alldocs(),vector_size=30, window=10, min_count=1,epochs=10,workers=8)
    #model_w2v.delete_temporary_training_data()
    #model_d2v.delete_temporary_training_data()

    model_d2v.save('../pickles/model_d2v')