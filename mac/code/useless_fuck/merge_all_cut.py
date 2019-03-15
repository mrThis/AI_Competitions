from utils import *
import gc
from tqdm import tqdm

def split_file(name):
    sl = np.array_split(load(name), 16, axis=0)
    for i in tqdm(range(16)):
        r = sl[i]
        save(name + '_' + str(i), r)
    del sl
    gc.collect()


train_text = load('train_text')
test_text = load('test_text', )

train_word1 = load('train_word1')
train_word1.columns=['title','context']
train_word2 = load('train_word2')
train_word2.columns = ['title', 'context']
train_word3 = load('train_word3')
train_word3.columns = ['title', 'context']
train_word4 = load('train_word4')
train_word4.columns = ['title', 'context']
train_pos = load('train_pos')
train_words = train_word1 + train_word2 + train_word3 + train_word4
train_u = pd.concat([train_text, train_words, train_pos], 1)
train_u.columns = ['title', 'context', 'title_words', 'context_words', 'postag']
save('train_u', train_u)

del train_word1, train_word2, train_word3, train_word4,train_pos,train_u
gc.collect()

tts = np.array_split(test_text, 16, axis=0)
for i in tqdm(range(16),ncols=75,desc='测试集分块:'):
    test_pos = load('test_pos'+'_'+str(i))
    test_word1 = load('test_word1'+'_'+str(i))
    test_word2 = load('test_word2'+'_'+str(i))
    test_word3 = load('test_word3'+'_'+str(i))
    test_word4 = load('test_word4'+'_'+str(i))
    for j in (test_word1, test_word2, test_word3, test_word4):
        j.columns=['title','context']
    test_words = test_word1 + test_word2 + test_word3 + test_word4

    test_u = pd.concat([tts[i], test_words, test_pos], 1)
    test_u.columns = ['title', 'context', 'title_words', 'context_words', 'postag']
    save('test_u'+'_'+str(i), test_u)
    del test_pos,test_word1,test_word2,test_word3,test_word4,test_words,test_u
    gc.collect()




