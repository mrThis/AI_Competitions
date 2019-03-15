# test auc,g-mean2,bal
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
resultss = pd.read_csv('D:\\python_code\\data_set\\paper_result\\TDPM3_VS_AUC.csv')

groubyed = resultss.groupby(['classifier', 'metrics'], as_index=False).mean()


def fun(x):
    for index, row in groubyed.iterrows():
        if x.classifier == row.classifier and x.metrics == row.metrics:
            x.results = x.results - row.results
    return x


new_resultss = resultss.apply(fun, axis=1)
new_resultss.to_csv('D:\\python_code\\data_set\\paper_result\\new_resultss.csv', index=False)
# new=groubyed.apply()



# ax = sns.pointplot(x="metrics", y="results", hue="classifier", data=resultss, join=False,
#                     dodge=0.8,scale=1,markers=["o", "x", 'v', 's', '+', '*','1','2','3','4','8','h'])
# markers=["o", "x", 'v', 's', '+', '*'],

# ax1 = sns.pointplot(x="metrics", y="results", hue="classifier", data=resultss, join=False,markers=["1", "2", '3', '4', '+', '*','x','X','v','^','<','>']
#                     ,dodge=0.8,scale=1.5,ci='sd')
#
# ax1 = sns.pointplot(x="metrics", y="results", hue="classifier", data=new_resultss, join=False,markers=["1", "2", '3', '4', '+', '*','x','X','v','^','<','>']
#                     ,dodge=0.8,scale=1.5,ci='sd')

# ax1 = sns.pointplot(x="metrics", y="results", data=new_resultss, join=False,
#                     dodge=0.5,scale=0.8,ci='sd')

# ax = sns.pointplot(x="metrics", y="results",hue="classifier", data=resultss, join=False,
#                    dodge=True,ci='sd')

ax1 = sns.pointplot(x="metrics", y="results", hue="classifier", data=new_resultss, join=False,markers=["1", "2", '3', '4', '+', '*','x']
                    ,dodge=0.8,scale=1.5,ci='sd')



def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
handles, labels = ax1.get_legend_handles_labels()
plt.legend(flip(handles, 4), flip(labels, 4),loc='upper left',ncol=4,fancybox=True,shadow=True)
# plt.legend(flip(handles, 4), flip(labels, 4),loc='upper center',bbox_to_anchor=(0.6,1),ncol=4,fancybox=True,shadow=True)

#bbox_to_anchor=(0.5, 0.5)
plt.show()
# markers=["o", "x",'v']
# tips = sns.load_dataset("tips")
# ax = sns.barplot(x="day", y="total_bill",hue='sex', data=tips)
# plt.show()
