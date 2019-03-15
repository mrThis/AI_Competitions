from Preprocess import *
from imblearn.over_sampling import SMOTE

from mymodels import *
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    skf = StratifiedKFold(2, True)
    y = read_p('train_y')

    #将A集的算法在本地拟合，并保存下来
    if True:
        l=[]
        for name in ('FS_train_A','FN_train_A','FG_train_A','FA_train_A','FW_train_A'):
            l.append(read_p(name).values)
        n=0
        splits=[]
        for x in l:
            long=x.shape[1]
            splits.append((n,n+long))
            n=n+long
        xt = np.hstack(l)
        y = y.values.ravel()
        xgb1=XGBClassifier(booster='dart',eval_metric='auc', scale_pos_weight=0.5, subsample=0.9, colsample_bytree=0.5,
                    random_state=111321,gamma=0,reg_lambda=1, reg_alpha=1)
        lgbm1=LGBMClassifier(boosting_type='dart',subsample=0.9, colsample_bytree=0.5,class_weight='balanced',
                            metric='auc',scale_pos_weight=0.5,reg_lambda=1, reg_alpha=1)
        for_fs =OnegoStackingClassifier([xgb1,lgbm1],LogisticRegression(class_weight='balanced'))
        for_fn=clone(for_fs)

        for_fg=XGBClassifier(booster='dart',eval_metric='auc', scale_pos_weight=0.5, subsample=0.9, colsample_bytree=0.5,
                    random_state=131211,gamma=0,reg_lambda=1, reg_alpha=1)
        for_fa=XGBClassifier(booster='dart',eval_metric='auc', scale_pos_weight=0.5, subsample=0.9, colsample_bytree=0.5,
                    random_state=133161,gamma=0)
        for_fw=XGBClassifier(booster='dart',eval_metric='auc', scale_pos_weight=0.5, subsample=0.9, colsample_bytree=0.5,
                              random_state=223442,reg_lambda=2, reg_alpha=1)
        models=[for_fs,for_fn,for_fg,for_fa,for_fw ]
        myA = Idea4s(base_classifiers=models,
                    combiner=LogisticRegression(class_weight='balanced'),
                    split_points=splits,
                    n=3)
        #skf = StratifiedKFold(2, True)
        # s = cross_val_score(my, xt, y, scoring='roc_auc', cv=skf, n_jobs=2)
        #上面这两行代码用来测试模型的分数
        myA.fit(xt,y)
        joblib.dump(myA,pickle_path+'modelA_3.p')

    #将B集的算法在本地拟合并保存下来
    if True:
        from Predict_B_3 import trans_scene
        xa=read_p('FW_train_A_forB').values
        ya=y.values.ravel()
        myB=trans_scene(xa, ya)
        myB.fit(read_p('FW_train_B').values)
        joblib.dump(myB, pickle_path + 'modelB_3.p')

