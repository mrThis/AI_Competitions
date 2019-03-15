from utils import *
from tqdm import tqdm
tqdm.pandas()
from collections import Counter
from jieba.analyse.tfidf import IDFLoader, DEFAULT_IDF
import gc
idf_loader = IDFLoader('../newidf.txt')
idf_freq, median_idf = idf_loader.get_idf()



def gen_features(u_name, is_train=False, train_list=None):
    import time
    tt = time.time()

    df=load(u_name)

    l = []
    for id in tqdm(list(df.index),ncols=75):
        #与组均值的差和商 TODO


        s = df.loc[id]

        title_words = s['title_words']
        context_words = s['context_words']
        postag = s['postag']
        # 计算tf
        tf_title = Counter(title_words)
        tf_context = Counter(context_words)
        freq = tf_title + tf_context
        total = sum(freq.values())  # 总词数

        # 计算tfidf并排序
        tfidf = {}
        for k in freq:
            tfidf[k] = freq[k] * idf_freq.get(k, median_idf) / total
        tfidf_tuple = sorted(tfidf, key=tfidf.__getitem__, reverse=True)

        # 得到候选词（tfidf前五加上title里面的所有word）
        r = pd.DataFrame(index=set([j for j in tfidf_tuple[:5]] + title_words))

        # tfidf作为特征
        for w in r.index:
            r.loc[w, 'tfidf'] = tfidf[w]
            r.loc[w, 'tf_title'] = tf_title[w]
            r.loc[w, 'tf_context'] = tf_context[w]
            r.loc[w, 'rank'] = tfidf_tuple.index(w)
            r.loc[w, 'pos'] = postag.get(w, 'NULL')
            p=s['context'].find(w)
            r.loc[w, 'position'] = p
            r.loc[w,'pos_pct']=-1
            if len(s['context'])!=0:
                r.loc[w, 'pos_pct']=p/len(s['context'])
            r.loc[w, 'length'] = len(w)

            # 在不在标题里作为特征
            if w in title_words:
                r.loc[w, 'in_title'] = 1
            else:
                r.loc[w, 'in_title'] = 0

            # 如果是训练集，顺手提取target
            if is_train:
                r.loc[w, 'target'] = 1 if w in train_list.loc[id] else 0


        r['wid'] = id + '__' + r.index.to_series()
        r['id'] = id
        r = r.reset_index().set_index('wid').rename({'index': 'word'}, axis=1)
        l.append(r)

    d = pd.concat(l)
    if is_train:
        print('平均每个id用时：' + str((time.time() - tt) / 1000))

    return d

if __name__=='__main__':
    #print(222)
    #test_u=load('test_u')

    train_list = load('train_list', )

    train_set = gen_features('train_u', is_train=True, train_list=train_list)

    from sklearn.externals.joblib import Parallel, delayed

    u_names=['test_u_'+str(j) for j in range(16)]

    from multiprocessing import cpu_count
    l = Parallel(n_jobs=cpu_count())(delayed(gen_features)(k) for k in u_names)

    test_set = pd.concat(l)

    train_set['idf'] = train_set.word.apply(lambda k: idf_freq[k] if k in idf_freq else median_idf)
    test_set['idf'] = test_set.word.apply(lambda k: idf_freq[k] if k in idf_freq else median_idf)

    save('train_set', train_set)
    save('test_set', test_set)


    train_x = train_set.drop(['word', 'target', 'id'], 1)
    train_y = train_set.target

    test_x = test_set.drop(['word', 'id'], 1)

    train_x.pos = pd.Categorical(train_x.pos)
    test_x.pos = pd.Categorical(test_x.pos)

    save('train_x', train_x)
    save('test_x', test_x)
    save('train_y', train_y)

