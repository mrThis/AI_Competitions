import warnings
from utils import timer,read_dict
warnings.filterwarnings('ignore')

from gensim.models import Word2Vec, Doc2Vec
import gc
import re
import pandas as pd
bid=read_dict('books_in_data.txt')
bid=set(bid)
a3 = read_dict('spiderkw1.txt')
a3 = list(map(lambda j: j.upper(), a3))
a3 = set(a3)  #注意：在set中查找比在list中查找要快600多倍
a4 = read_dict('tx_words.txt')
a4 = list(map(lambda j: j.upper(), a4))
a4 = set(a4)
#@profile
def handle_features(train_set):
    model_w2v = Word2Vec.load('../pickles/model_w2v')
    wv_name = ['w2vec_{}'.format(k) for k in range(30)]

    words=[k for k in set(train_set.word) if k in model_w2v]
    w2v_dict = dict(zip(words,model_w2v.wv[words]))
    w2v=pd.DataFrame.from_dict(w2v_dict, 'index')
    w2v.columns=wv_name
    train_set=train_set.join(w2v,on='word').fillna(0)
    del model_w2v,w2v
    gc.collect()

    dv_name = ['d2vec_{}'.format(k) for k in range(30)]
    model_d2v = Doc2Vec.load('../pickles/model_d2v')
    ids=set(train_set.id)
    d2v_dict=dict(zip(ids,[model_d2v.docvecs[j] for j in ids]))
    dv = pd.DataFrame.from_dict(d2v_dict, 'index')
    dv.columns = dv_name
    train_set = train_set.join(dv,on='id')
    del model_d2v,dv
    gc.collect()


    train_set.pos = pd.Categorical(train_set.pos)
    train_set.new_pos = pd.Categorical(train_set.new_pos)
    train_set.hanlp_pos = pd.Categorical(train_set.hanlp_pos)

    train_set['is_book'] = train_set.word.str.upper().apply(lambda k: k in bid)
    train_set['is_txkw'] = train_set.word.str.upper().apply((lambda k: k in a4))
    train_set['is_sinakw'] = train_set.word.str.upper().apply(lambda k: k in a3)
    letter =re.compile('[A-Za-z]')
    digit = re.compile('[0-9]')
    train_set['have_letter'] = train_set.word.map(lambda w: bool(letter.search( w)))
    train_set['have_digit'] = train_set.word.map(lambda w: bool(digit.search( w)))

    return train_set
if __name__=='__main__':
    from utils import load
    train_set = load('temp')
    handle_features(train_set)