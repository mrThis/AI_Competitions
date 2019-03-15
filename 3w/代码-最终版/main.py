from Preprocess import *


if __name__ == "__main__":

    if True:

        #
        test_consumer_A=pd.read_csv(test_path + r"test_consumer_A.csv")
        test_consumer_B=pd.read_csv(test_path+r"test_consumer_B.csv")
        test_behavior_A=pd.read_csv(test_path + r"test_behavior_A.csv")
        test_behavior_B=pd.read_csv(test_path+r"test_behavior_B.csv")
        test_ccx_A=pd.read_csv(test_path + r"test_ccx_A.csv")

        train_consumer_A=pd.read_csv(dataA_path + r"train_consumer_A.csv")
        train_consumer_B=pd.read_csv(dataB_path+r"train_consumer_B.csv")
        train_behavior_A=pd.read_csv(dataA_path + r"train_behavior_A.csv")
        train_behavior_B=pd.read_csv(dataB_path+r"train_behavior_B.csv")
        train_ccx_A=pd.read_csv(dataA_path + r"train_ccx_A.csv")

        train_target_A=pd.read_csv(dataA_path + r"train_target_A.csv")

    train_total_A=clean_and_aggregate(train_behavior_A,train_consumer_A,train_ccx_A)
    train_total_B=clean_and_aggregate(train_behavior_B,train_consumer_B)
    train_total_A_forB=clean_and_aggregate(train_behavior_A,train_consumer_A).drop('var10',axis=1)
    train_y=train_target_A.set_index('ccx_id').sort_index().target

    test_total_A=clean_and_aggregate(test_behavior_A,test_consumer_A,test_ccx_A)
    test_total_B=clean_and_aggregate(test_behavior_B,test_consumer_B)

    print('A训练集')
    FS_train_A=transform(train_total_A,'FS','A')
    FN_train_A=transform(train_total_A,'FN','A')
    FG_train_A=transform(train_total_A,'FG','A')
    FA_train_A=transform(train_total_A,'FA','A')
    FW_train_A=transform(train_total_A,'FW','A')
    #
    print('A训练集的子集（用来训练B）')
    FS_train_A_forB = transform(train_total_A_forB, 'FS', 'B')
    FN_train_A_forB = transform(train_total_A_forB, 'FN', 'B')
    FG_train_A_forB = transform(train_total_A_forB, 'FG', 'B')
    FA_train_A_forB = transform(train_total_A_forB, 'FA', 'B')
    FW_train_A_forB = transform(train_total_A_forB, 'FW', 'B')
    #
    print('B训练集')
    FS_train_B=transform(train_total_B,'FS','B')
    FN_train_B=transform(train_total_B,'FN','B')
    FG_train_B=transform(train_total_B,'FG','B')
    FA_train_B=transform(train_total_B,'FA','B')
    FW_train_B=transform(train_total_B,'FW','B')
    #
    print('A测试集')
    FS_test_A = transform(test_total_A, 'FS', 'A')
    FN_test_A = transform(test_total_A, 'FN', 'A')
    FG_test_A = transform(test_total_A, 'FG', 'A')
    FA_test_A = transform(test_total_A, 'FA', 'A')
    FW_test_A = transform(test_total_A, 'FW', 'A')
    #
    FS_test_B = transform(test_total_B, 'FS', 'B')
    FN_test_B = transform(test_total_B, 'FN', 'B')
    FG_test_B = transform(test_total_B, 'FG', 'B')
    FA_test_B = transform(test_total_B, 'FA', 'B')
    FW_test_B = transform(test_total_B, 'FW', 'B')

    for name in ('FS_train_A','FN_train_A','FG_train_A','FA_train_A','FW_train_A','FA_train_A_forB',
                 'FW_train_A_forB','FA_train_B','FW_train_B','FS_test_A','FN_test_A','FG_test_A','FA_test_A',
                 'FW_test_A','FS_test_B','FN_test_B','FG_test_B','FA_test_B','FW_test_B'):
        save_p(eval(name),name)

    for name in ('FS_train_A', 'FN_train_A', 'FG_train_A', 'FA_train_A', 'FW_train_A', 'FA_train_A_forB',
                 'FW_train_A_forB', 'FA_train_B', 'FW_train_B', 'FS_test_A', 'FN_test_A', 'FG_test_A', 'FA_test_A',
                 'FW_test_A', 'FS_test_B', 'FN_test_B', 'FG_test_B', 'FA_test_B', 'FW_test_B'):
        print(name,read_p(name).shape)

    save_p(train_y,'train_y')


