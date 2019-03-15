from utils import *
from segmentor import make_cuter

def pos_cut(df,cuter):
    global co
    co = 0

    def f(j):
        global co
        co += 1
        if co % 100 == 0:
            print('\r    大概进行到了', round(co * 4 * 100 / df.shape[0], 2), '%',end='')
        try:
            return dict(cuter(j.title+'  。'+j.context))
        except:
            return {}

    pos_words = para_apply(df, f)
    return pos_words

if __name__=='__main__':
    cuter = make_cuter(5)

    # print(cuter)

    train_text = load('train_text')
    test_text = load('test_text', )
    train_list = load('train_list', )

    train_pos=pos_cut(train_text,cuter)
    test_pos = pos_cut(test_text, cuter)

    save('train_pos',train_pos)
    split_save(test_pos,'test_pos')
