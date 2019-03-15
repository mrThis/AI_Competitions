
import time
from contextlib import contextmanager


#计时器
@contextmanager
def timer(title):
    t0 = time.time()
    print("{} - start ".format(title))
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

_count=0

def countit(name=''):
    global _count
    _count+=1
    if _count%100==0:
        print(name+str(_count))
    return

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
    l1=['〖','「','【','[','<','{','“']
    for ss in l1:
        rstring = rstring.replace(ss, '《')
    l2=['〗','」','】',']','>','}','”']
    for ss in l2:
        rstring = rstring.replace(ss, '》')
    rstring=rstring.replace('!','！')
    return rstring

import numpy as np
import pandas as pd
from joblib import Parallel,delayed

def para_apply(data, func,cores=4,axis=1):
    from multiprocessing import cpu_count
    data_split = np.array_split(data, cpu_count(), axis=1-axis)
    my_apply=lambda j:j.apply(func,axis=axis)
    l=Parallel(n_jobs=cpu_count())(delayed(my_apply)(k) for k in data_split)
    r=pd.concat(l,axis=0)
    return r


def save(name, v):
    v.to_pickle('../pickles/' + name + '.p')


def load(name):
    return pd.read_pickle('../pickles/' + name + '.p')

from tqdm import tqdm
import gc
def split_save(df,name):
    sl = np.array_split(df, 16, axis=0)
    for i in tqdm(range(16),desc='分块储存',ncols=75):
        r=sl[i]
        save(name+'_'+str(i),r)
    del sl
    gc.collect()
    return

def read_dict(name):
    with open('../dicts/'+name,'r',encoding='utf-8') as f:
        ct=f.readlines()

    return [j.strip() for j in ct]

def mynext(iterable):
    try:
        return next(iterable)
    except TypeError:
        return next(iter(iterable))
