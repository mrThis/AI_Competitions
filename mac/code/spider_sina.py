from bs4 import BeautifulSoup
import requests
import json
import time
import tqdm
import pandas as pd
import numpy as np
import re

def unpack(l):
    return [k for i in l for k in i]
def fuckit(l):
    return [j for j in l if len(j)>1]
def dropit(l):
    return list(set(l))

def handleit(l):
    return dropit(fuckit(unpack(l)))


#找到相关联的关键字
def get_connect_word(w,errorlist):
    try:
        url="http://api.search.sina.com.cn/"
        d={'q':w,'c':'news','range':'keywords','num':50}
        r=requests.get(url,d,allow_redirects=False)
        c=json.loads(r.content)
        rl=c['result']['list']
        kws=list(set(unpack( [j['kl'].split(' ') for j in rl] )))
        titles=[j['title'] for j in rl]
    except:
        print(w)
        errorlist.append(w)
        return [],[]
    return list(set(kws)),titles


if __name__=='__main__':
    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                            names=['id', 'keyword'], index_col='id')

    ks = []
    for kk in train.keyword.str.split(','):
        ks += kk
    #ks:所有训练集中的关键词

    cw=[]
    err_l=[]
    for w in tqdm.tqdm(ks):
        cw+=get_connect_word(w,err_l)

    idxs=np.arange(0,4574,2)

    new_kw=[cw[i] for i in idxs]
    new_titles=[cw[i+1] for i in idxs]

    new_kw=fuckit(dropit(unpack(new_kw)))

    new_kw=[s[1:-1] if (s[0]=="《" and s[-1]=="》") else s for s in new_kw ]

    #从title里面提取信息
    new_books=handleit([re.findall('《(.*?)》',s) for s in unpack(new_titles)])

    new_words=dropit(new_kw+new_books)

    with open('./dicts/spiderkw1.txt','w',encoding='utf-8') as f:
        f.writelines([j+' \n' for j in new_words])
