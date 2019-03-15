from Preprocess import *



def eva(x, yy,model=None):
    if model==None:
        model=LogisticRegression()
    if len(x.shape) == 1:
        x = x.values.reshape(-1, 1)
    else:
        x=x.values

    y=yy.values
    sum_auc_N = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    #l.append(y)
    #print(y.shape)
    for train_index, test_index in skf.split(x, y):
        #try:
        x_train, y_train, x_test, y_test = x[train_index],y[train_index], x[test_index], y[test_index]
        modell=clone(model)
        modell_fit = modell.fit(x_train, y_train)
        y_prob = modell_fit.predict_proba(x_test)[:, 1]
        #print(y_prob)
        auc = roc_auc_score(y_test, y_prob)
        if auc<0.5:
            return 0.5 # 不稳定的变量，可能产生过拟合
        sum_auc_N.append(auc)
        # except:
        #     pass
    return np.mean(sum_auc_N)



def find_best(total,y, method='FN'):
    if method == 'FS':
        flags = {varname: search_fs(total[varname],y) for varname in total}
        return flags
    elif method == 'FN':
        flags = {varname: search_fn(total[varname],y) for varname in total}
        return flags
    elif method == 'FG':
        flags = {varname: search_fg(total[varname],y) for varname in total}
        return flags
    elif method == 'FA':
        scores =find_fa(total,y)
        flags={varname:(-6,None) for n,varname in enumerate( scores.index) if n<160 }
        return flags
    elif method =='FW':
        flags = {varname: search_fw(total[varname],y) for varname in total}
        return flags

def search_fs(var,y):
    #print('Method FS:', var.name, end=': ')
    if var.count() <= 10:
        #print(0)
        return 0, None
    lv=real_level(var)
    if lv == 1:
        #print(0)
        return 0,None
    if lv==2 and var.isnull().any():
        score=eva(var.isnull().replace([False, True], [0, 1]).to_frame(),y)
        #print(-5,score)
        if score>0.525:
            return -5,None
        else:
            return 0,None
    if (lv>2) and (lv <= 4):  # 离散型
        lis=lis_dum(var)
        lis_name = list(lis.columns)
        score=eva(lis_dum(var),y)
        #print(-4,score)
        if score>0.525:
            return -4,lis_name
        return 0,None
    else:
        r = pd.Series(index=[n for n in range(4, 10)])
        for nn in range(4,10):
            boxed,bins=con_dum(var, nn)
            if boxed.shape[1]<nn and nn!=4:  #如果数据不够分这么多箱
                break
            else:
                r[nn]=eva(boxed,y)
        r[-1] = eva(fill(var, u=-1),y)  # -1：中位数填充
        r[-2] = eva(fill(var, u=-2),y)  # -2：众数填充
        r[-3] = eva(fill(var, u=-3),y)  # -3:0填充
        r[-5] = eva(var.isnull().replace([False, True], [0, 1]).to_frame(),y)  # -5:只用na矩阵
        #print(r.idxmax(), round(r.max(), 4))  # 注意一下nn的问题
        flag,score=r.idxmax(), r.max()
        print(flag,score)
        if score>0.525:
            if flag<=0:
                return flag,None
            else:
                return flag,con_dum(var, flag)[1]
        return 0,None

def search_fn(var,y):
    if var.count() <= 10:
        return 0, None
    lv=real_level(var)
    if lv==1:
        return 0, None
    return -6, None

def search_fg(var,y):
    if var.count() <= 10:
        return 0, None
    lv=real_level(var)
    if lv == 1:
        return 0,None
    if lv==2 and var.isnull().any():
        return -5,None
    if (lv>2) and (lv <= 4):  # 离散型
        lis=lis_dum(var)
        lis_name = list(lis.columns)
        return -4,lis_name
    r = pd.Series(index=[-1,-2,-3,-4,-5])
    r[-1] = f_classif(fill(var, u=-1).values.reshape(-1,1),y)[0]  # -1：中位数填充
    r[-2] = f_classif(fill(var, u=-2).values.reshape(-1,1),y)[0]  # -2：众数填充
    r[-3] = f_classif(fill(var, u=-3).values.reshape(-1,1),y)[0]  # -3:0填充
    r[-5] = f_classif(var.isnull().replace([False, True], [0, 1]).to_frame().values.reshape(-1,1),y)[0]  # -5:只用na矩阵
    flag, score = r.idxmax(), r.max()
    return flag,None

def find_fa(total,y):
    l = []
    x = total.values
    for seed in range(50):  # 用不同随机种子跑10次
        np.random.seed(seed)
        skf = StratifiedKFold(5, True)
        for train_index, test_index in skf.split(x, y.values):
            xt, yt = x[train_index], y.values[train_index]
            print(seed, '.')
            l.append(
                XGBClassifier(eval_metric='auc', scale_pos_weight=0.5, subsample=0.7, colsample_bytree=0.7)
                    .fit(xt, yt).feature_importances_)
            break
    ip10 = pd.DataFrame(l, columns=total.columns).transpose()
    return ip10.mean(axis=1).sort_values(ascending=False)

def search_fw(var,y):
    # 计算IV寻优&筛选
    if var.count() <= 10:
        return 0, None
    lv=real_level(var)
    if lv == 1:
        return 0,None
    if lv == 2 and var.isnull().any():
        isna=var.isnull().replace([False, True], [0, 1]).to_frame()
        if cal_iv(isna,y)>=0.01:
            return -5,None
        return 0,None
    if (lv>2) and (lv <= 4):  # 离散型
        lis=lis_dum(var)
        lis_name = list(lis.columns)
        if cal_iv(lis,y)>=0.01:
            return -4,lis_name
        return 0,None
    else:  # 数值型
        r = pd.Series(index=[n for n in range(4, 10)])
        for nn in range(4, 10):
            boxed, bins = con_dum(var, nn)
            if boxed.shape[1] < nn and nn!=4:  # 如果数据不够分这么多箱
                break
            else:
                r[nn] = cal_iv(boxed,y)
        flag, score = r.idxmax(), r.max()
        if score>0.01:
            return flag,con_dum(var, flag)[1]
        else:
            return 0,None


if __name__=='__main__':
    work_path  = r"C:/Users/hunzh/Desktop/jianmo/"
    dataA_path = work_path + r"train/train_scene_A/"
    dataB_path = work_path+r"train/train_scene_B/"
    pickle_path= work_path+"pickles/"

    train_consumer_A=pd.read_csv(dataA_path + r"train_consumer_A.csv")

    train_behavior_A=pd.read_csv(dataA_path + r"train_behavior_A.csv")

    train_ccx_A=pd.read_csv(dataA_path + r"train_ccx_A.csv")

    train_target_A=pd.read_csv(dataA_path + r"train_target_A.csv")

    total_A     =clean_and_aggregate(train_behavior_A,train_consumer_A,train_ccx_A)
    total_A_forB=clean_and_aggregate(train_behavior_A,train_consumer_A).drop('var10',axis=1)

    train_y=train_target_A.set_index('ccx_id').sort_index().target
    y = train_y

    FS_flags_A = find_best(total_A, y, 'FS')
    FN_flags_A = find_best(total_A, y, 'FN')
    FG_flags_A = find_best(total_A, y, 'FG')
    FA_flags_A = find_best(total_A, y, 'FA')
    FW_flags_A = find_best(total_A, y, 'FW')

    save(FS_flags_A, 'FS_flags_A')
    save(FN_flags_A, 'FN_flags_A')
    save(FG_flags_A, 'FG_flags_A')
    save(FA_flags_A, 'FA_flags_A')
    save(FW_flags_A, 'FW_flags_A')

    FS_flags_A_forB = find_best(total_A_forB, y, 'FS')
    FN_flags_A_forB = find_best(total_A_forB, y, 'FN')
    FG_flags_A_forB = find_best(total_A_forB, y, 'FG')
    FA_flags_A_forB = find_best(total_A_forB, y, 'FA')
    FW_flags_A_forB = find_best(total_A_forB, y, 'FW')

    save(FS_flags_A_forB, 'FS_flags_B')
    save(FN_flags_A_forB, 'FN_flags_B')
    save(FG_flags_A_forB, 'FG_flags_B')
    save(FA_flags_A_forB, 'FA_flags_B')
    save(FW_flags_A_forB, 'FW_flags_B')
