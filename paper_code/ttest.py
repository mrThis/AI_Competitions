# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
# all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/cm1.csv')
# print(all.shape[0])
# list=np.array_split(all,5)
# for i in range(5):
#     print(list[i].shape[0])
# train=pd.DataFrame()
# for k in range(5):
#     if k == 1:
#         continue
#     train = train.append(list[k])
#
#


# skf=StratifiedKFold(n_splits=5)
# np.random.seed(100)
# df = pd.DataFrame(np.random.random((10,5)), columns=list('ABCDE'))
# df.index = df.index * 10
# print (df)
# train = df.iloc[[0, 2, 3, 5, 6, 7, 8, 9]]
# print(train)



#测试一下使用weka包

# import weka.core.jvm as jvm
# from weka.core.converters import Loader, Saver
# from weka.classifiers import Classifier
#
# jvm.start(class_path=['D:\\python_code\\venv\Lib\\site-packages\\weka\\lib\\python-weka-wrapper.jar',
# 'C:\\Program Files\\Weka-3-8\\weka.jar'])
#
# cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.3"])
#
# loader = Loader(classname="weka.core.converters.ArffLoader")
# data = loader.load_file("all.arff")
#
# from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
# search = ASSearch(classname="weka.attributeSelection.GreedyStepwise")
# evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval")
# attsel = AttributeSelection()
# attsel.search(search)
# attsel.evaluator(evaluator)
# attsel.select_attributes(data)
# # print("# attributes: " + str(attsel.number_attributes_selected))
# # print("attributes: " + str(attsel.selected_attributes))
# # print("result string:\n" + attsel.results_string)
# print((attsel.selected_attributes-1)[:-1])
#
#
# jvm.stop()
#测试转换
# from util import pandas2arff
# from arff2pandas import a2p
#
#
# all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/cm1.csv')
# all.to_csv('compare.csv',index=False)
# pandas2arff(all,'all.arff')
#
# with open('all.arff') as f:
#     df = a2p.load(f)
#     df.to_csv('all.csv',index=False)

# all_raw,meta =loadarff('all.arff')
# all_new=pd.DataFrame(all_raw[0])
# all_new.to_csv('all.csv')



# t=0
# def count(t):
#     t=t+1
#     return t
# for i in range(10):
#     t = count(t)


# data=pd.DataFrame(columns=['s'])
# print(data)
str='kc3'
all = pd.read_csv('../data_set/NASA_PROMISE_DATASET/' + str + '.csv')
all['Defective'] = all['Defective'].map(dict(Y=1, N=0))
y = all['Defective']
X = all.drop(['Defective'], axis=1)
rate = (all['Defective'] == 1).sum() / float(all['Defective'].shape[0])
print('the ratio of the wrong：%f' % (rate))