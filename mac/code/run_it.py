import subprocess
import time
start_time=time.time()
from settings import *
print('当前设置：')
if use_cuter3_NLP:
    print('\t使用hanlp新词发现')
print('\t使用内核数：',cores)
print('\t输出文件名:',output_name)
print('\t提取词数：',num_of_word_extract)
from utils import timer
with timer('训练模型'):
   subprocess.call('python train_model.py')
# with timer('模型评估'):
#     subprocess.call('python evaluation.py')

with timer('多线程预测'):
    shape=108295
    #shape=1000
    from settings import cores

    pool=[]
    if cores==1:
        batch = shape
        subprocess.call('python predict_keyword.py 0 {} 0'.format(shape))
    else:

        batch=shape//cores
        i=0
        for i in range(cores-1):
            #print(i*batch,(i+1)*batch)
            s1=subprocess.Popen(['python','predict_keyword.py',str(i*batch),str((i+1)*batch),str(i)])
            pool.append(s1)
        s1=subprocess.Popen(['python','predict_keyword.py',str((i+1)*batch),str(shape),str(i+1)])
        pool.append(s1)

        for s in pool:
            s.wait()
with timer('合并预测结果'):
    import pandas as pd
    rl=[]
    i=0
    for i in  range(cores-1):
        rl.append(pd.read_csv('../output/result_'+str(i*batch)+'_'+str((i+1)*batch)+'.csv'))
    rl.append(pd.read_csv('../output/result_'+str((i+1)*batch)+'_'+str(shape)+'.csv'))

    from settings import output_name
    pd.concat(rl).set_index('id')[['label1','label2']].to_csv('../output/'+output_name)

print('运行完成，文件输出在',output_name)
m=(time.time()-start_time)
print('用时{}分钟{}秒。'.format(int(m//60),int(m%60)))