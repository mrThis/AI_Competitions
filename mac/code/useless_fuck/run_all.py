import sys
import os
from utils import timer
import time
t0 = time.time()
with timer('清洗文本'):
    os.system('python clean.py')
with timer('jieba分词'):
    os.system('python cut_word.py 1 1')
with timer('thulac分词'):
    os.system('python cut_word.py 2 1')
with timer('hanlp分词'):
    os.system('python cut_word.py 3 0')
with timer('pyltp分词'):
    os.system('python cut_word.py 4 0')
with timer('jieba标注词性'):
    os.system('python cut_pos_tag.py')
with timer('融合所有切词结果'):
    os.system('python merge_all_cut.py')
with timer('生成特征'):
    os.system('python gen_feature.py')
with timer('预测并输出结果'):
    os.system('python get_output.py')

print('全部结束，总用时'+str(round((time.time()-t0)/60,2))+'min')