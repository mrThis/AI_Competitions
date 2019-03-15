from utils import timer,read_dict
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import logging
tqdm.pandas()
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)

import sys
use_which_cuter=int(sys.argv[1])
#use_which_cuter=5
if use_which_cuter==1:
    import jieba
    import jieba.posseg
    jieba.load_userdict('../dicts/all_words.txt')
    cuter=lambda j:list(jieba.cut(j))
elif use_which_cuter==2:
    import thulac
    t = thulac.thulac(T2S=True, seg_only=True, model_path='../models/'
                      , user_dict='../dicts/all_words.txt')
    cuter = lambda string: [j[0] for j in t.cut(string)]
elif use_which_cuter==3:
    from pyhanlp import HanLP, JClass,SafeJClass
    #from pyhanlp import JClass as SafeJClass
    CustomDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    aw = read_dict('all_words.txt')
    for w in aw:
        CustomDictionary.add(w)
    NLPTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    cuter = lambda string: [j.word for j in NLPTokenizer.segment(string)]

elif use_which_cuter==4:

    from pyltp import Segmentor
    segmentor = Segmentor()
    segmentor.load_with_lexicon('../ltp_data/cws.model', '../dicts/all_words.txt')
    cuter = lambda j: list(segmentor.segment(j))
elif use_which_cuter==5:
    import jieba
    import jieba.analyse
    import jieba.posseg
    jieba.load_userdict('../dicts/all_words.txt')


all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')
all_text=all_text.fillna('')
train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                    names=['id', 'keyword'], index_col='id')

train_text = all_text.loc[train.index, :]
test_text = all_text.drop(train.index)
from settings import modify_kw, modify_title

for id, (old, new) in modify_kw.items():
    train.loc[id, 'keyword'] = train.loc[id, 'keyword'].replace(old, new)
for id, (old, new) in modify_title.items():
    train_text.loc[id, 'title'] = train_text.loc[id, 'title'].replace(old, new)


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


import pymongo
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db=client['cuters']
cuterXed=db['cuter{}ed'.format(use_which_cuter)]
cuterXed.drop()
cuterXed.create_index([('id', pymongo.ASCENDING)],
                               unique=True)

def cut_and_store(text):
    id=text.name
    if use_which_cuter!=5:
        cut_title=cuter(d2c(text.title))
        cut_context=cuter(d2c(text.context))
        cuterXed.insert_one({'id':id,'title':cut_title,'context':cut_context})
    else:
        postag = [tuple(j) for j in jieba.posseg.cut(d2c(text.title + '  。' + text.context))]
        cuterXed.insert_one({'id': id,'postag':postag})


for id,text in tqdm(all_text.iterrows(),total=all_text.shape[0],ncols=75,desc='cuter{}'.format(use_which_cuter)):
    cut_and_store(text)