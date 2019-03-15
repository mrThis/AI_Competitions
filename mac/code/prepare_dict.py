
def have_comma(s):

    for j in [',','.',';','#','、','《','》','。',' ']:
        if s.find(j)!=-1:
            return True
    return False


def cleanit(r):
    #去掉：过短的、全是数字的、标点开头的、有逗号的
    rr=[j for j in r if j[0].isalnum() and not j.strip().isdigit()
        and not have_comma(j) and 2 < len(j) < 20]
    return rr



def write_dict(name,l,mode='w'):
    rr=list(set(cleanit(l)))

    with open('../dicts/'+name,mode,encoding='utf-8') as f:
        f.writelines([j+'\n' for j in rr])



#修正一个字典文件
def make_it_right(path):
    with open(path,'r',encoding='utf-8') as f:
        r=f.readlines()
    rr=[j for j in r if len(j.split())!=0 and j[0].isalnum() and not j.strip().isdigit()
        and not have_comma(j)]
    with open(path,'w',encoding='utf-8') as f:
        f.writelines(rr)


if __name__=='__main__':
    import pandas as pd
    from tqdm import tqdm
    from utils import timer,d2c,read_dict
    import re
    tqdm.pandas(ncols=75)

    with timer('清洗文本'):
        pass
        # all_text = pd.read_csv('../all_docs.txt', '\001', header=None,
        #                        names=['id', 'title', 'context'], index_col='id')
        #
        # all_text.context = all_text.context.progress_apply(d2c)
        # all_text.title = all_text.title.progress_apply(d2c)
        #
        # train = pd.read_csv('../train_docs_keywords.txt', '\t', header=None,
        #                     names=['id', 'keyword'], index_col='id')
        #
        # train_text = all_text.loc[train.index, :]

    def search_book():
        l=[]
        for id, s in all_text.iterrows():

            l1=re.findall('《(.*?)》', s.context)
            l2=re.findall('《(.*?)》', s.title)
            l+=l1+l2
        return [k for k in set(l) if len(k)<=20 and len(k)>1]

    #
    def clean_bookname(name):
        if "‘" in name or '。' in name :
            return
        elif name[-1]=='?' or name[-1]=="!" or (',' in name) or ('《' in name) or ('》' in name) :
            return
        else:
            return name

    def get_in_brace(name,l):
        if ('(' in name )and (')' in name):
            n=re.findall('\((.*?)\)', name)[0]
            l.append(n)
        return re.sub('\(.*?\)','',name)


    def dropit(l):
        return list(set(l))
    #
    # with timer('寻找文本中所有在书名号里的词，并将其切分'):
    #     books = search_book()
    #     books=[j for j in [clean_bookname(i) for i in books] if (j is not None )]
    #     l=[]
    #     books=[get_in_brace(j,l) for j in books]
    #     books+=l
    #     write_dict('books_in_data.txt',books)
    #
    # with timer('提取训练集中所有关键词'):
    #     ks = []
    #     for kk in train.keyword.str.split(','):
    #         ks += kk
    #     write_dict('kw_in_train.txt',ks)

    with timer('爬虫寻找关键词'):
        pass

    from sklearn.externals.joblib import load
    tx_words=list(load( '../spider_tx/' + 'tags.p').keys())

    def fix_human_name(list):
        nl=[]
        for name in list:
            if '\x1a' in name:
                name=name.replace('\x1a','·')
            nl.append(name)
        return nl

    # with timer('将上述词典合并为一个'):
    #
    #
    #     a1=read_dict('books_in_data.txt')
    #     a2=read_dict('kw_in_train.txt')
    #     a3=read_dict('spiderkw1.txt')
    #     a4=read_dict('tx_words.txt')
    #
    #
    #
    #     a=dropit(a1+a2+a3)
    #
    #     write_dict('all_words.txt',a,)