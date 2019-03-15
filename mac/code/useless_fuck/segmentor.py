from utils import timer
def read_dict(name):
    with open('../dicts/'+name,'r',encoding='utf-8') as f:
        ct=f.readlines()

    return [j.strip() for j in ct]

def make_cuter(n):
    if n==1:
        import jieba

        jieba.load_userdict('../dicts/all_words.txt')
        cuter1=lambda j:list(jieba.cut(j))
        return cuter1
    elif n==2:

        import thulac
        with timer('加载thulac词库'):
            t=thulac.thulac(T2S=True,seg_only=True,model_path='../models/'
                            ,user_dict='../dicts/all_words.txt')

        cuter2=lambda string:[j[0] for j in t.cut(string)]
        return cuter2
    elif n==3:
        from pyhanlp import HanLP,JClass


        #NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
        CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
        aw=read_dict('all_words.txt')
        for w in aw:
            CustomDictionary.add(w)
        cuter3=lambda string: [j.word for j in HanLP.segment(string)]
        #cuter3_2=lambda string: [j.word for j in NLPTokenizer.segment(string)]
        return cuter3
    elif n==4:

        from pyltp import Segmentor

        segmentor = Segmentor()

        segmentor.load_with_lexicon('../ltp_data/cws.model', '../dicts/all_words.txt')
        cuter4=segmentor.segment
        return cuter4
    elif n==5:
        import jieba
        import jieba.posseg
        jieba.load_userdict('../dicts/all_words.txt')
        return jieba.posseg.cut
