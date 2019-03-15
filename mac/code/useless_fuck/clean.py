from utils import *
from tqdm import tqdm
tqdm.pandas(ncols=75)
#with timer('读取文本'):
all_text =pd.read_csv('../all_docs.txt' ,'\001' ,header=None,
                     names=['id' ,'title' ,'context'] ,index_col='id')

train =pd.read_csv('../train_docs_keywords.txt' ,'\t' ,header=None,
                  names=['id' ,'keyword'] ,index_col='id')

#with timer('清洗文本'):
all_text.context = all_text.context.progress_apply(d2c ,)
all_text.title = all_text.title.progress_apply(d2c)


train_text =all_text.loc[train.index ,:]
test_text =all_text.drop(train.index)
train_list = train.keyword.str.split(',')

save('train_text',train_text)
save('test_text',test_text)
save('train_list',train_list)