from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import timer,read_dict,mynext
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

def my_cuter(id):
    titles=[]
    contexts=[]
    for c in (cuter1ed,cuter2ed,cuter3ed,cuter4ed):
        d=c.find_one({'id':id})
        titles+=d['title']
        contexts+=d['context']
    return titles,contexts

c=Counter()
for id,text in tqdm(all_text.iterrows(),total=all_text.shape[0],ncols=75,
                    desc='计算idf'):
    a, b = my_cuter(id)
    words=clean(set(a+b))
    c.update(words)

df=pd.DataFrame.from_dict(c,'index')[0]
allnum=all_text.shape[0]
df=df[df>2]
s=allnum/(df+1)
idf=s.apply(np.log)
idf.to_frame().to_csv('../newidf.txt',header=None,sep=' ')