

from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import timer,read_dict
from collections import Counter
from pyhanlp import HanLP,JClass,SafeJClass
CustomDictionary = SafeJClass("com.hankcs.hanlp.dictionary.CustomDictionary")
aw = read_dict('all_words.txt')
for w in aw:
    CustomDictionary.add(w)
jtag = SafeJClass('com.hankcs.hanlp.model.perceptron.PerceptronPOSTagger')()

import pymongo
from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db=client['cuters']
cuter1ed=db['cuter1ed']
cuter2ed=db['cuter2ed']
cuter3ed=db['cuter3ed']
cuter4ed=db['cuter4ed']
cuter6ed=db['cuter6ed']

all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')
#raise
all_text = all_text.fillna('')
# 加载stopwords词典
with open('../stopwords.txt', encoding='utf-8') as f:
    stop_words = set(f.read().split('\n'))

def have_comma(s):

    for j in [',','.',';','#','、','《','》','。']:
        if s.find(j)!=-1:
            return True
    return False

def clean(dic):
    #去掉：过短的、全是数字的、标点开头的、有逗号的
    rr={j:v for j,v in dic.items() if (len(j[0].split())==1) and j[0].isalnum() and not j[0].strip().isdigit()
        and not have_comma(j[0]) and (1 < len(j[0]) < 20) and not (j[0].lower() in stop_words)}
    return rr

def my_cuter(id):
    titles=[]
    contexts=[]
    for c in (cuter1ed,cuter2ed,cuter3ed,cuter4ed):
        d=c.find_one({'id':id})
        titles+=d['title']
        contexts+=d['context']
    return titles,contexts

cuter6ed.drop()
cuter6ed.create_index([('id', pymongo.ASCENDING)],
                               unique=True)

for id,text in tqdm(all_text.iterrows(),total=all_text.shape[0],ncols=75,
                    desc='cuter{}'.format('HANLP词性标注')):
    a,b=my_cuter(id)
    poss = jtag.tag(a+b)
    c = sorted(clean(Counter(zip(a+b, poss))).items(), key=lambda kk: (kk[0][0], kk[1]), reverse=True)
    cuter6ed.insert_one({'id':id,'pos':c})