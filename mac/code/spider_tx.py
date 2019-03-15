from bs4 import BeautifulSoup
import requests
import json
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from retry import retry
#解包list
def unpack(l):
    return [k for i in l for k in i]
#去掉长度为1的词
def fuckit(l):
    return [j for j in l if len(j)>1]
#去重
def dropit(l):
    return list(set(l))
#上面三个合起来
def handleit(l):
    return dropit(fuckit(unpack(l)))

if __name__=='__main__':
    path='../spider_tx/'

    import os
    if not os.path.exists(path):
        os.mkdir(path)
    from sklearn.externals.joblib import dump,load

    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')

    ks = []
    for kk in train.keyword.str.split(','):
        ks += kk
    # ks:所有训练集中的关键词

    # url="https://new.qq.com/tag/81149"
    # url="https://pacaio.match.qq.com/tags/tag2articles"
    # query={'id':81149,'num':60}
    # r=requests.get(url,query)
    # c=json.loads(r.content)['data']

    from collections import deque,OrderedDict
    #
    #105625  微软
    #81149  成龙
    #289444  拍拍贷
    #92183 馒头
    #82542 NBA
    #205041 穿搭
    #96900 汉字
    #114223 睡眠
    start_with=deque([105625,81149,289444,92183,82542,205041,96900,114223]) #从哪里开始
    already_seen=[] #已经看过的不再回头看
    to_see=[] #接下来看的

    intros=[]#查看过的所有intro
    titles=[]#查看过的所有title
    tags=OrderedDict()#查看过的所有关键词与其代码
    raise
    #start_with=load( path + 'start_with.p', )
    #already_seen=load( path + 'already_seen.p', )
    #to_see=load( path + 'to_see.p')
    #intros=load( path + 'intros.p')
    #titles=load( path + 'titles.p')
    #tags=load( path + 'tags.p')
    
    @retry(delay=2)
    def get_connected_word(num):
        url = "https://pacaio.match.qq.com/tags/tag2articles"
        query = {'id': num, 'num': 60}
        r = requests.get(url, query)
        c = json.loads(r.content)['data']
        connect_nums=[]
        for k in c:
            try:
                intros.append(k['intro'])
                titles.append(k['title'])
            except Exception as e:
                print(e)

            try:
                for word, num in k['tag_label']:

                    tags[word] = num
                    if not num in already_seen:
                        #print(word)
                        already_seen.append(num)
                        connect_nums.append(num)
            except Exception as e:
                print(e)

        return connect_nums

    for i in tqdm(range(100000),desc='腾讯新闻',ncols=75):
        #time.sleep(0)
        
        new_nums=get_connected_word(start_with.popleft())
        for num in new_nums:
            start_with.append(num)
        if i%1000==0 and i!=0:
            print('\nSaving。。',len(tags))
            w=list(tags.keys())[-1]
            print('Last added:',w)
            dump(start_with, path + 'start_with.p', )
            dump(already_seen, path + 'already_seen.p', )
            dump(to_see, path + 'to_see.p')
            dump(intros, path + 'intros.p')
            dump(titles, path + 'titles.p')
            dump(tags, path + 'tags.p')











    #c=BeautifulSoup(r.content,features="html5lib")


