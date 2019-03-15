import pandas as pd
import orangecontrib.associate.fpgrowth as oaf

# b 分析一笔报销id下所有药品与0,1的关系 单纯分析每笔没有太大意义
# c 分析一个病人id下所有药品与0,1的关系 具体到每个病人


test = pd.read_csv('../data/input/train_id.tsv', sep='\t', header=0)
implement = pd.read_csv('../data/output/implement.csv')
a = implement[['PERSONID', 'FTR51']]
a = pd.merge(a, test, on='PERSONID').drop('PERSONID', axis=1)