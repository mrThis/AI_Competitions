
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from utils import load,save
tqdm.pandas(ncols=75)


if __name__=='__main__':
    all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')

    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')

    train_text = all_text.loc[train.index, :]
    test_text = all_text.drop(train.index)
    from settings import modify_kw, modify_title

    for id, (old, new) in modify_kw.items():
        train.loc[id, 'keyword'] = train.loc[id, 'keyword'].replace(old, new)
    for id, (old, new) in modify_title.items():
        train_text.loc[id, 'title'] = train_text.loc[id, 'title'].replace(old, new)

    from sklearn.externals.joblib import dump
    from collections import Counter
    train_list = train.keyword.str.split(',')
    ks=[]
    for kk in train.keyword.str.split(','):
        ks += kk
    ck = Counter(ks)
    dump(ck, '../pickles/train_freq.p')


    from featuretool import gen_feature
    tl=[]
    for id,text in tqdm(train_text.iterrows(),total=train_text.shape[0],ncols=75):
        tl.append(gen_feature(id,text))
    train_set=pd.concat(tl)
    save('temp', train_set)
    from for_set import handle_features
    train_set=handle_features(train_set)

    #TODO:groupby mean里面，tdtitle奇怪

    from lightgbm.sklearn import LGBMClassifier

    #m1 = LGBMClassifier(n_estimators=65,reg_lambda=1,num_leaves=7)
    #m1 = LGBMClassifier(n_estimators=135,reg_lambda=3,num_leaves=20)
    m1=LGBMClassifier(n_estimators=200, reg_lambda=10, num_leaves=31, learning_rate=0.05)
    train_x = train_set.drop(['word', 'id'], 1)

    train_y = train_set.apply(lambda j:1 if j.word in train_list.loc[j.id] else 0 , axis=1)

    m1.fit(train_x,train_y)


    save('train_x',train_x)
    save('train_y', train_y)
    save('train_set',train_set)
    save('train_text',train_text)

    from sklearn.externals.joblib import dump

    dump(m1,'../pickles/model.p')
