from utils import *
from sklearn.model_selection import StratifiedKFold,cross_val_score


if __name__=='__main__':
    train_x=load('train_x',)
    test_x=load('test_x', )
    train_y=load('train_y', )

    train_set=load('train_set')
    test_set=load('test_set')
    train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
                        names=['id', 'keyword'], index_col='id')



    #raise
    from lightgbm.sklearn import LGBMClassifier

    m1 = LGBMClassifier(n_estimators=65, reg_alpha=3)

    from sklearn.linear_model import LogisticRegression

    m2 = LogisticRegression()

    def get_label(dfp):
        k=dfp.word[-2:]
        return pd.Series({'label1':k[0],'label2':k[1]})

    #评价模型的二分类能力
    def qeva(x,model, verbose=True):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        scores = cross_val_score(model, x, train_y, scoring='roc_auc', verbose=verbose, cv=skf, )
        return np.mean(scores), scores

    #在训练集上打分
    def score(tx):

        p = pd.DataFrame(m1.fit(tx, train_y).predict_proba(tx)[:, 1], columns=['proba'], index=tx.index)
        p['id'] = train_set.id
        p['word'] = train_set.word

        rr = p.sort_values(by=['id', 'proba']).groupby('id').apply(get_label)

        return rr.join(train).apply(
            lambda j:(j['label1'] in j['keyword'])+(j['label2'] in j['keyword'])
            ,1).apply(lambda j:j/2).sum()

    with timer('输出结果：'):
        p = pd.DataFrame(m1.fit(train_x, train_y).predict_proba(test_x)[:, 1], columns=['proba'], index=test_x.index)
        p['id'] = test_set.id
        p['word'] = test_set.word
        r = p.sort_values(by=['id', 'proba']).groupby('id').apply(get_label)
        r=r.applymap(lambda j:j if not ',' in j else '')
        r.to_csv('../output/useall4.csv')


