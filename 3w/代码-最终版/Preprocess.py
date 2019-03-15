import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif as mic
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
import pickle
import os

work_path  = r"C:/Users/hunzh/Desktop/jianmo/"
test_path = r"C:/Users/hunzh/Desktop/jianmo/test/"
dataA_path = work_path + r"train/train_scene_A/"
dataB_path = work_path+r"train/train_scene_B/"
pickle_path= work_path+"pickles/"
#将变量以指定的名字pickle下来
def read_p(name):
    return pd.read_pickle(pickle_path+name+'.p')

def save_p(var,name):
    var.to_pickle(pickle_path+name+'.p')

def load(name):
    with open(pickle_path +name+ ".p", 'rb') as f:
        file = pickle.load(f)
    return file
def save(var,name):
    with open(pickle_path + name + ".p", 'wb') as f:
        pickle.dump(var,f)


        #对Dataframe使用的特殊方法，因为直接apply时，dtype判断不准确。和wash结合使用
#用法：df_clean=apply(df,s_wash)
def apply(df,fun):
    ndf=pd.DataFrame(index=df.index)
    for var in df:
        r=fun(df[var])
        if r is not None:
            ndf[var]=r
    return ndf

def make_date(x):
    if pd.isna(x):
        return x
    else:
        try:
            return pd.to_numeric(str(x)[:4])
        except:
            return  np.nan

def p_lis_dum(var,columns):
    if var.isnull().any():

        rr= pd.get_dummies(var, prefix=var.name, dummy_na=True)
    else:
        rr= pd.get_dummies(var, prefix=var.name, dummy_na=False)

    return rr.reindex_axis(columns,axis=1)


def lis_dum(var):
    if var.isnull().any():
        # if real_level(var)==2:
        #     return var.isnull().replace([False,True],[0,1])
        return pd.get_dummies(var, prefix=var.name, dummy_na=True)
    else:
        return pd.get_dummies(var, prefix=var.name, dummy_na=False)


#连续--->哑变量
def con_dum(var, nn):
    if var.isnull().any():
        a,bins=pd.qcut(var, nn -1,retbins=True,duplicates='drop')
        cuted=pd.cut(var,bins)
        rr=pd.get_dummies(cuted,prefix=var.name, dummy_na=True)
    else:
        a,bins=pd.qcut(var, nn ,retbins=True,duplicates='drop')
        cuted = pd.cut(var, bins)
        rr=pd.get_dummies(cuted, prefix=var.name,dummy_na=False)
    return rr,bins

def p_con_dum(var, bin):  #用于已经寻好优之后的处理
    if var.isnull().any():
        rr=pd.get_dummies(pd.cut(var,bin ),prefix=var.name, dummy_na=True)
    else:
        rr=pd.get_dummies(pd.cut(var, bin ), prefix=var.name,dummy_na=False)
    return rr


# 清洗:处理日期，从字符串中提取数字
def s_wash(var):
    #print(var.name,var.dtype,end='--->')
    if var.isnull().all():
        #print('全空')
        return var
    if var.name == 'target' or var.name == 'ccx_id':
        #print('无需处理')
        return var
    level = len(pd.value_counts(var))
    if level is 0:  # 完全缺失的废字段，舍弃
        #print('舍弃')
        return var
    if var.name in ['var16', 'var17']: #特殊字段特殊处理
        #print('日期型')
        var=var.replace(0, np.nan).apply(make_date)

        return var
    if var.name in ['var19']:
        #print('日期型')
        var=var.isnull().replace([False,True],[0,1])
    if var.name in ['var_06']:
        #print('日期型')
        return pd.to_datetime(var)
        #return
    if var.name in ['V_7','V_11']:
        #print('日期型')
        var=var.replace('0000-00-00 00:00:00', np.nan)
        return pd.to_datetime(var)
    if pd.api.types.is_string_dtype(var) :
        #print('字符藏着数字型')
        var = var.map(cut_out_numbers)
    else:
        pass
        #print('数值型')  # 数值变量
    return var

# 去掉字母保留数字（并转换）
def cut_out_numbers(string):
    if pd.isnull(string):
        return string
    else:
        return pd.to_numeric(''.join(filter(str.isdigit, string)))

def clean_and_aggregate(behavior,consumer,ccx=None):
    if ccx is not None:
        handle_type='A'
    else:
        handle_type='B'
    behavior=behavior.set_index('ccx_id')
    behavior=behavior.sort_index()
    index=behavior.index
    behavior_new = apply(behavior, s_wash)
    consumer_clean = apply(consumer, s_wash)
    consumer_new = aggregate(consumer_clean, index,True).rename(columns={'size': 'consumer_size'})
    if ccx is not None:
        handle_type='A'
        ccx_clean = apply(ccx, s_wash).rename(columns={'size': 'ccx_size'})
        ccx_new = aggregate(ccx_clean, index, False).rename(columns={'size': 'ccx_size'})
        #ccx_new = aggregate(ccx_clean, index, False).join(self.index.to_frame())
        return behavior_new.join(consumer_new).join(ccx_new)
    return behavior_new.join(consumer_new)

def aggregate(n, index, is_consumer=True):  # TODO:index

    n_new = pd.DataFrame(index=index)

    # 日期序列单独处理
    if is_consumer:
        dts = n.loc[:, n.columns.isin(['V_7', 'V_11', 'ccx_id'])]
        time = consumertime(dts)
        #n_new = pd.DataFrame(index,index=index)
        #
    else:
        dts = n.loc[:, n.columns.isin(['var_06', 'ccx_id'])]
        time = ccxtime(dts)
        n_new = pd.DataFrame(index,index=index)
        #


    if not is_consumer:
        var_01 = n[['var_01', 'ccx_id']]
        avgvar_01 = var_01['var_01'].replace([2, 3], [0, 1]).to_frame().join(var_01.ccx_id). \
            groupby('ccx_id').mean()
        sum1var_01 = var_01['var_01'].replace([2, 3], [0, 1]).to_frame().join(var_01.ccx_id). \
            groupby('ccx_id').sum()
        sum0var_01 = var_01['var_01'].replace([2, 3], [1, 0]).to_frame().join(var_01.ccx_id). \
            groupby('ccx_id').sum()
        avgvar_01.columns = ['avgvar_01']
        sum1var_01.columns = ['sum1var_01']
        sum0var_01.columns = ['sum0var_01']
        sum1var_01 = pd.DataFrame(sum1var_01, n_new.index).fillna(0)
        sum0var_01 = pd.DataFrame(sum0var_01, n_new.index).fillna(0)
        n_new = n_new.join(avgvar_01).join(sum1var_01).join(sum0var_01)

    # 数值型变量的整理：
    # 计数项（0填缺失值）
    n = n.drop(['V_7', 'V_11', 'var_06', 'var_01'], axis=1, errors='ignore')
    gd = n.groupby('ccx_id')

    size = gd.size()
    size = size.to_frame('size')
    size = pd.DataFrame(size, n_new.index).fillna(0).astype('int64')
    n_new = n_new.join(size, )  # .fillna(0)

    # 均值项（不填均值）
    mean = gd.mean()
    new_name = []
    for x in mean:
        new_name.append('avg' + x)
    mean.columns = new_name
    n_new = n_new.join(mean, rsuffix='mean')  # .fillna(0)

    # 求和项（0填均值）#DOne
    s = gd.sum()
    new_name = []
    for x in s:
        new_name.append('sum' + x)
    s.columns = new_name
    s = pd.DataFrame(s, n_new.index).fillna(0)
    n_new = n_new.join(s, rsuffix='sum')

    # 最大值项（不填）
    s = gd.max()
    new_name = []
    for x in s:
        new_name.append('max' + x)
    s.columns = new_name
    n_new = n_new.join(s, rsuffix='max')  # .fillna(0)

    # 最小值项
    s = gd.min()
    new_name = []
    for x in s:
        new_name.append('min' + x)
    s.columns = new_name
    n_new = n_new.join(s, rsuffix='min')  # .fillna(0)
    # 平均值项

    return n_new.join(time)  # 最后把日期时间项整合回来

# consumer表专用的日期时间处理方法  dts：包含了V_7和V_11、ccx_id的DataFrame
def consumertime(dts):
    id=dts.ccx_id
    v7=dts.V_7 #v7无缺
    v11=dts.V_11 #v11有缺

    #计算经常购物的时间段，结果类似于： A：0.3 B:0.2 C:0.5 D:0 ，代表每个时间段的比例。
    hour=v7.apply(lambda p:p.hour)
    v7A=(hour>=23) | (hour<9)
    v7B= (hour>=9) & (hour<17)
    v7C = (hour>=17) & (hour<23)
    often=pd.DataFrame([v7A,v7B,v7C],['V_7A','V_7B','V_7C']).transpose()

    often=often.join(id).groupby('ccx_id').mean()
    weekday=pd.DataFrame(v7.apply(lambda x:x.weekday()>=5)).join(id).groupby('ccx_id').mean()
    weekday.columns = ['weekday']
    # delta=pd.cut( (v7-v11).apply(lambda t:t.seconds) ,
    #               bins=[0.0, 6.0, 9.0, 13.0, 27.0, 172.0, 86399.0],labels=[x for x in 'abcdef'])

    # 虽然不知道v7和v11的含义，但是二者的差值一定有含义。
    delta=pd.DataFrame((v7 - v11).apply(lambda t: t.seconds)).join(id).groupby('ccx_id').mean().iloc[:,0].rename('deltaV_7')
    gd=pd.DataFrame(v7).join(id).groupby('ccx_id')

    #第一次购物到最后一次购物
    days=(gd.max()-gd.min())['V_7'].apply(lambda x:x.days).rename('daysV_7')

    # 第一次和最后一次距离last的天数。
    last=pd.to_datetime("2017-06-01 00:00:01")
    later=(gd.max().apply(lambda x:last-x))['V_7'].apply(lambda x:x.days).rename('laterV_7')
    early=(gd.min().apply(lambda x:last-x))['V_7'].apply(lambda x:x.days).rename('earlyV_7')

    #汇总并返回
    return often.join(delta,lsuffix='o').join(days,lsuffix='d').join(later,lsuffix='l')\
        .join(early,lsuffix='e').join(weekday)

#跟上面的差不多
def ccxtime(dts):
    gd=dts.groupby('ccx_id')
    days = (gd.max() - gd.min())['var_06'].apply(lambda x: x.days).rename('daysvar_06')

    last = pd.to_datetime("2017-06-01 00:00:01")
    later = (gd.max().apply(lambda x: last - x))['var_06'].apply(lambda x: x.days).rename('latervar_06')
    early = (gd.min().apply(lambda x: last - x))['var_06'].apply(lambda x: x.days).rename('earlyvar_06')
    return pd.DataFrame( days).join(later,lsuffix='l').join(early,lsuffix='e')

#计算level 输入：Series
def level(var):
    return len(var.value_counts())

def real_level(var):
    return level(var)+var.isnull().any()


#填补缺失值（用指定的方法）  var：dataframe+isnull u：代表处理方法的flag
def fill(var,u=-1):  #用中位数填充的默认方法
    if var.isnull().any():
       #r = pd.DataFrame(var.isnull().replace([False, True], [0, 1]))
       if u==-1:
           return var.fillna(var.median())
       elif u==-2:
           return var.fillna(var.mode()[0])
       elif u==-3:
           return var.fillna(0)
    else:
        return var
    return

def iv(var,y):
    t = var.to_frame().join(y)
    yesi = t[var == 1].target.sum()
    yest = y.sum()
    noi = (t[var == 1].target == 0).sum()
    no_t = (y == 0).sum()
    if yesi==0: yesi=1
    if yest==0: yest=1
    if noi==0:noi=1
    if no_t==0:no_t=1
    return (yesi/yest - noi/no_t) *  np.log((yesi / yest) / (noi / no_t))

def cal_iv(df,y):
    return df.apply(lambda k:iv(k,y)).sum()

def transform(total,method='FS',t='A'):
    flags = load(method+'_flags_'+t)
    return proc(total,flags)

def proc(total,flags):
    dff = pd.DataFrame(index=total.index)
    for varname, (flag,info) in flags.items():
        r = how_deal(total[varname], flag,info)
        if r is None:
            continue
        dff = dff.join(r, lsuffix='ERROR')  # debug
    return dff

def how_deal(var,flag,info): #自动按flag处理单个变量
    #print('处理',var.name,end='')
    if flag > 0:
        #print('按照已有标准，连续变量离散化--->', end='')
        r= p_con_dum(var, info)
        #print(r.shape[1], '个')
        return r
    elif flag ==0:
        #print('被筛掉')
        return None
    elif flag == -1:
        #print('作为连续变量，中位数填')
        return fill(var, flag)
    elif flag == -2:
        #print('作为连续变量，众数填')
        return fill(var, flag)
    elif flag == -3:
        #print('作为连续变量，0填')
        return fill(var, flag)
    elif flag == -4:
        #print('作为离散变量')
        return p_lis_dum(var,info)
    elif flag == -5:
        #print('直接取na dummy')
        return var.isnull().replace([False,True],[0,1])
    elif flag == -6:
        #print('不作处理')
        return var