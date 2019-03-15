
from tqdm import tqdm
import pandas as pd
import numpy as np

if __name__=='__main__':
    import sys
    #sys.argv=['','0','27073','2']
    start=int(sys.argv[1])
    stop=int(sys.argv[2])
    core_i=int(sys.argv[3])

    tqdm.pandas(ncols=75,desc=sys.argv[1]+' ---> '+ sys.argv[2],disable=core_i!=0)

    all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
                           names=['id', 'title', 'context'], index_col='id')
    #raise
    all_text = all_text.fillna('')
    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')

    train_text = all_text.loc[train.index, :]
    test_text = all_text.drop(train.index)
    from sklearn.externals.joblib import load
    m1=load('../pickles/model.p')

    #print(start,stop)
    #print(test_text.shape)
    tt = test_text[start:stop]
    del test_text,all_text
    from settings import already_have_set
    if not already_have_set:
        from featuretool import gen_feature
        tl=[]
        for id, text in tqdm(tt.iterrows(), total=tt.shape[0], ncols=75,position=core_i,mininterval=2,miniters=500,
                             desc='线程{}'.format(core_i),leave=False):
            test_set = gen_feature(id, text)
            tl.append(test_set)

        all_set=pd.concat(tl)
        del tl
        from for_set import handle_features
        all_set=handle_features(all_set)

        from sklearn.externals.joblib import dump
        dump(all_set,'../output/test_set_'+str(start)+'_'+str(stop)+'.p')
    else:

        all_set=load('../output/test_set_'+str(start)+'_'+str(stop)+'.p')
    kl = {}
    probs=pd.Series(m1.predict_proba(all_set.drop(['word', 'id'], 1))[:, 1],name='proba',index=all_set.index)
    temp=all_set.join(probs)
    for id,test_prob in tqdm(temp.groupby('id'), total=tt.shape[0], ncols=75,desc='预测结果,线程{}'.format(core_i),
	position=core_i,mininterval=0.5, miniters=500,leave=False):
        keywords=dict(zip(['label1', 'label2'],test_prob.sort_values('proba').word[-2:]))
        kl[id]=keywords

    result=pd.DataFrame.from_dict(kl,'index')
    #result.index.name='id'
    #print('../output/result_'+str(start)+'_'+str(stop)+'.csv')
    result.to_csv('../output/result_'+str(start)+'_'+str(stop)+'.csv',index_label='id')
