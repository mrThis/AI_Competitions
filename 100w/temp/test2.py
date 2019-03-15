import pandas as pd

def describe_more(df):
    tem = len(df)
    var = []
    l = []
    r = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        r.append(df[df[x] == 0].count()[x] / tem)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'ratio': r})
    levels.sort_values(by='Levels', inplace=True)
    return levels
tt = pd.read_pickle('../data/input/FTR51train.p')
print(tt.head(5))
print(describe_more(tt))

te = tt.index.to_frame('feature')
print(te.head(5))
te.to_csv('del.csv',index=False)
