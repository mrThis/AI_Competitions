import pandas as pd


# 根据特别项目出现频率做特征
# 先测支持度大的
# 统计该用户中出现该种药的次数（对置信度大的），与占比（对支持度大的）
# TODO：同时考虑有些药坏人不会用它来诈骗


# 还有支持度更低的，暂时不考虑
# 传入数据集，与得到的columns
def ftr51(data, columns):
    def count(ser, col):
        if col not in ser.values:
            return 0
        return pd.value_counts(ser)[[col]][0]

    def freq(ser, col):
        print(len(ser))
        return count(ser, col) / len(ser)

    df = pd.DataFrame(index=data.index.unique())
    gpb = data.groupby('PERSONID')
    for col in columns:
        # 得到series
        # 得到医疗项目计数count
        ts_c = gpb['FTR51'].agg(count, col=col)
        ts_c.rename('FTR51' + col + '_count', inplace=True)
        # 得到医疗项目频率freq
        ts_f = gpb.FTR51.agg(freq, col=col)
        ts_f.rename('FTR51' + col + '_freq', inplace=True)
        # 直接拼接在df上
        df = df.join(ts_c).join(ts_f)
    return df


# 得到新特征
train = pd.read_csv('../data/output/implement.csv')
test = pd.read_csv('../data/output/implement_test.csv')
train = train[['PERSONID', 'FTR51']].set_index('PERSONID')
test = test[['PERSONID', 'FTR51']].set_index('PERSONID')
# 新增特征#用了标签信息，所以线下可能会升高
# 支持度>1e-4，置信度>0.9
columns1 = ['A2B107C705E0D0', 'A153B193C704E0D0', 'A153B193C6551E0D0', 'A2B309C6070E0D0']
# 支持度>1e-4，置信度>0.6 #最科学
columns2 = ['A2B107C705E0D0', 'A3B233C350E29D15', 'A2B172C320E0D0', 'A0B3C75E0D0', 'A0B3C6E13D0', 'A2B241C361E0D0',
            'A2B387C1055E0D0', 'A2B662C362E0D0', 'A153B193C704E0D0', 'A0B3C26E0D0', 'A2B107C9069E0D0', 'A2B113C756E0D0',
            'A2B171C25E0D0', 'A2B111C1009E0D0', 'A0B34C51E0D0']
columns3 = ['A0B0C0E0D0', 'A0B0C6E0D0', 'A0B0C14E0D0', 'A0B0C10E5D3', 'A0B0C10E0D0', 'A0B0C2E4D1', 'A0B0C2E1D1',
            'A7B25C0E0D0', 'A0B0C2E0D0', 'A0B0C50E0D0', 'A0B0C1E0D0']

columns4 = ['A0B0C0E0D0', 'A0B0C6E0D0', 'A0B0C14E0D0', 'A0B0C10E5D3', 'A0B0C10E0D0', 'A0B0C2E4D1', 'A0B0C2E1D1',
            'A7B25C0E0D0', 'A0B0C2E0D0', 'A0B0C50E0D0', 'A0B0C1E0D0', 'A0B0C41E5D3', 'A0B0C9E0D0', 'A0B0C11E0D0',
            'A0B0C67E0D0', 'A0B0C7E0D0', 'A0B0C3E0D0', 'A7B25C10E5D3', 'A0B4C14E0D0', 'A0B0C4E0D0']

columns5 = ['A0B0C0E0D0', 'A0B0C6E0D0', 'A0B0C14E0D0', 'A0B0C10E5D3', 'A0B0C10E0D0', 'A0B0C2E4D1', 'A0B0C2E1D1',
            'A7B25C0E0D0', 'A0B0C2E0D0', 'A0B0C50E0D0', 'A0B0C1E0D0', 'A0B0C41E5D3', 'A0B0C9E0D0', 'A0B0C11E0D0',
            'A0B0C67E0D0', 'A0B0C7E0D0', 'A0B0C3E0D0', 'A7B25C10E5D3', 'A0B4C14E0D0', 'A0B0C4E0D0', 'A0B0C26E0D0',
            'A0B0C65E0D0', 'A13B3C10E0D0', 'A1B65C62E0D0', 'A2B48C73E0D0', 'A0B0C5E0D0', 'A0B0C8E0D0', 'A0B0C36E0D0',
            'A0B126C2E0D0', 'A7B25C2E4D1', 'A0B0C70E0D0', 'A7B25C41E5D3', 'A2B32C72E0D0', 'A2B94C73E0D0', 'A0B0C63E0D0',
            'A18B52C235E0D0', 'A8B76C47E0D0', 'A0B0C64E0D0', 'A1B81C159E0D0', 'A0B0C76E0D0', 'A2B113C184E0D0',
            'A2B232C72E0D0', 'A0B4C2E0D0', 'A0B0C10E7D0', 'A0B0C63E5D3', 'A0B0C81E0D0', 'A0B126C14E0D0', 'A2B49C74E0D0',
            'A0B0C2E0D1', 'A0B0C77E0D0', 'A2B9C19E0D0', 'A2B259C189E0D0', 'A2B124C184E0D0', 'A2B128C74E0D0',
            'A3B67C48E0D0', 'A0B74C279E0D0', 'A0B0C2E594D1', 'A0B0C2E16D1', 'A6B145C210E0D0', 'A0B74C70E0D0',
            'A0B0C322E0D0', 'A0B0C2E3D1', 'A0B126C65E0D0', 'A3B156C60E0D0', 'A3B14C93E0D0', 'A3B137C60E0D0',
            'A0B0C0E9D0', 'A1B208C182E0D0', 'A0B0C78E0D0', 'A2B143C177E0D0', 'A0B0C0E3D0', 'A0B0C75E0D0', 'A0B0C41E0D0',
            'A0B0C94E0D0', 'A0B74C213E0D0', 'A2B133C118E0D0', 'A0B0C2E2D1', 'A2B79C143E0D0', 'A3B32C48E0D0',
            'A0B0C174E0D0', 'A1B65C187E0D0', 'A7B25C10E0D0', 'A0B74C225E0D0', 'A12B74C279E0D0', 'A2B70C118E0D0',
            'A0B74C128E0D0']

# ###专注1置信度
# # 得到新的train补充集
# train_imp = ftr51(train, columns1)
# print(train_imp.head(5))
# # 得到新的test补充集
# test_imp = ftr51(test, columns1)
# train_imp.to_pickle('../data/input/FTR51_2train.p')
# test_imp.to_pickle('../data/input/FTR51_2test.p')
#
# ###专注1支持度
# # 得到新的train补充集
# train_imp = ftr51(train, columns2)
# # 得到新的test补充集
# test_imp = ftr51(test, columns2)
# train_imp.to_pickle('../data/input/FTR51_3train.p')
# test_imp.to_pickle('../data/input/FTR51_3test.p')


###只放data支持度高的 0.01
# train_imp = ftr51(train, columns3)
# print(train_imp.head(5))
# # 得到新的test补充集
# test_imp = ftr51(test, columns3)
# train_imp.to_pickle('../data/input/FTR51_5train.p')
# test_imp.to_pickle('../data/input/FTR51_5test.p')

# ###pindata>0.005
# train_imp = ftr51(train, columns4)
# print(train_imp.head(5))
# # 得到新的test补充集
# test_imp = ftr51(test, columns4)
# train_imp.to_pickle('../data/input/FTR51_6train.p')
# test_imp.to_pickle('../data/input/FTR51_6test.p')

###pindata>0.001
# train_imp = ftr51(train, columns5)
# print(train_imp.head(5))
# # 得到新的test补充集
# test_imp = ftr51(test, columns5)
# train_imp.to_pickle('../data/input/FTR51_7train.p')
# test_imp.to_pickle('../data/input/FTR51_7test.p')
