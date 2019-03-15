from segmentor import make_cuter
from utils import *
import tqdm
tqdm.tqdm.pandas(ncols=75)
import pandas as pd


def cut_word(df, cuter, cleaner, par=True):
    title_words = df.title.progress_apply(lambda j: cleaner(cuter(j)))

    global co
    co = 0

    def f(j):
        global co
        co += 1
        if co % 100 == 0:
            print('\r    大概进行到了',round(co*4*100/df.shape[0],2),'%',end='')
        try:
            return cleaner(cuter(j.context))
        except:
            return []

    if par:
        print('    多线程分词：')
        context_words = para_apply(df, f)
        print('    多线程分词完成。')
    else:
        context_words = df.context.progress_apply(lambda j: cleaner(cuter(j)))

    return pd.concat([title_words, context_words], axis=1)
import sys
if len(sys.argv)>1:#命令行启动
    n=int(sys.argv[1])
    par=int(sys.argv[2])
else:#自行设置
    n = 4
    par = False

#print(n,par)
if __name__=='__main__':



    cuter=make_cuter(n)

    #print(cuter)

    train_text=load('train_text' )
    test_text=load('test_text', )
    train_list=load('train_list', )


    # 加载stopwords词典
    with open('../stopwords.txt', encoding='utf-8') as f:
        stop_words = set(f.read().split('\n'))


    def clean(l):
        return [w for w in l if not (len(w.strip()) < 2 or w.lower() in stop_words or ',' in w )]

    train_word=cut_word(train_text,cuter,clean,par)
    train_word.columns = ['title', 'context']
    #raise
    test_word=cut_word(test_text,cuter,clean,par)
    test_word.columns = ['title', 'context']

    split_save(test_word,'test_word'+str(n))

    save('train_word'+ str(n), train_word)
    #save('test_word' + str(n), test_word)