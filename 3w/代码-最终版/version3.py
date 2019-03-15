###

#python文件载入部分

###

#加载验证数据集

import pandas as pd

test_consumer_A=pd.read_csv("./test/scene_A/test_consumer_A.csv")

test_consumer_B=pd.read_csv("./test/scene_B/test_consumer_B.csv")

test_behavior_A=pd.read_csv("./test/scene_A/test_behavior_A.csv")

test_behavior_B=pd.read_csv("./test/scene_B/test_behavior_B.csv")

test_ccx_A=pd.read_csv("./test/scene_A/test_ccx_A.csv")

# 自测代码时读取的验证集数据样本

# test_consumer_A=pd.read_csv("./testdemo/scene_A/test_consumer_A.csv")

# test_consumer_B=pd.read_csv("./testdemo/scene_B/test_consumer_B.csv")

# test_behavior_A=pd.read_csv("./testdemo/scene_A/test_behavior_A.csv")

# test_behavior_B=pd.read_csv("./testdemo/scene_B/test_behavior_B.csv")

# test_ccx_A=pd.read_csv("./testdemo/scene_A/test_ccx_A.csv")

###

###

#选手将自己的代码附于此处

# 必要的训练数据加载、数据处理、特征工程、数据建模、模型预测、结果输出代码

# 耗时的模型训练寻优代码不需要提交 30分钟以内的运行时间 超过30分钟没有结果的成绩为0 并消耗了一次评测机会



import warnings
warnings.filterwarnings('ignore')  # 忽略警告信息（来自于scikit-learn和pandas）

from Preprocess import *



print('数据读取完毕，开始数据清理...')
test_total_A=clean_and_aggregate(test_behavior_A,test_consumer_A,test_ccx_A)
test_total_B=clean_and_aggregate(test_behavior_B,test_consumer_B)

print('数据清理完毕，开始特征提取...')
FS_test_A = transform(test_total_A, 'FS', 'A')
FN_test_A = transform(test_total_A, 'FN', 'A')
FG_test_A = transform(test_total_A, 'FG', 'A')
#FA_test_A = transform(test_total_A, 'FA', 'A')
FW_test_A = transform(test_total_A, 'FW', 'A')
#FA_test_B = transform(test_total_B, 'FA', 'B')
FW_test_B = transform(test_total_B, 'FW', 'B')


print('特征提取完毕，开始载入模型...')
from sklearn.externals import joblib
modelA=joblib.load(r'./pickles/modelA_3.p')
modelB=joblib.load(r'./pickles/modelB_3.p')

print('模型A预测中...')
k=[]
for x_i in (FS_test_A,FN_test_A,FG_test_A,FW_test_A):
    k.append(x_i.values)
xA = np.hstack(k)  #将不同方案处理的特征一起输入stacking模型中
pa=modelA.predict_proba(xA)[:,1]
predict_result_A=test_behavior_A.index.to_frame()
predict_result_A['prob']=pa

print('模型B预测中...')
xB=FW_test_B
pb=modelB.predict_proba(xB.values)[:,1]
predict_result_B=test_behavior_B.index.to_frame()
predict_result_B['prob']=pb

print('完成。')

###

###

#python文件结束部分

###

# 保存预测的结果 predict_result_A predict_result_B为您构建的模型预测出的概率和唯一索引构成的DataFrame

predict_result_A.to_csv('./predict_result_A.csv',encoding='utf-8',index=False)

predict_result_B.to_csv('./predict_result_B.csv',encoding='utf-8',index=False)


