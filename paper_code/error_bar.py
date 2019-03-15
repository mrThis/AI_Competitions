# test
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
results = pd.read_csv('D:\\python_code\\data_set\\paper_result\\model_r_1.csv')

groubyed = results.groupby(['class_model', 'metrics'], as_index=False).mean()


def fun(x):
    for index, row in groubyed.iterrows():
        if x.class_model == row.class_model and x.metrics == row.metrics:
            x.result = x.result - row.result
    return x


new_results = results.apply(fun, axis=1)
new_results.to_csv('D:\\python_code\\data_set\\paper_result\\new_results.csv', index=False)
# new=groubyed.apply()



# ax = sns.pointplot(x="metrics", y="result", hue="class_model", data=results, join=False,
#                     dodge=0.8,scale=1,markers=["o", "x", 'v', 's', '+', '*','1','2','3','4','8','h'])
# markers=["o", "x", 'v', 's', '+', '*'],

# ax1 = sns.pointplot(x="metrics", y="result", hue="class_model", data=results, join=False,markers=["1", "2", '3', '4', '+', '*','x','X','v','^','<','>']
#                     ,dodge=0.8,scale=1.5,ci='sd')
#
ax1 = sns.pointplot(x="metrics", y="result", hue="class_model", data=new_results, join=False,markers=["1", "2", '3', '4', '+', '*','x','X','v','^','<','>']
                    ,dodge=0.8,scale=1.5,ci='sd')

# ax1 = sns.pointplot(x="metrics", y="result", data=new_results, join=False,
#                     dodge=0.5,scale=0.8,ci='sd')

# ax = sns.pointplot(x="metrics", y="result",hue="class_model", data=results, join=False,
#                    dodge=True,ci='sd')

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
handles, labels = ax1.get_legend_handles_labels()
plt.legend(flip(handles, 4), flip(labels, 4),loc='upper left',ncol=4,fancybox=True,shadow=True)
# plt.legend(flip(handles, 4), flip(labels, 4),loc='upper center',bbox_to_anchor=(0.6,1),ncol=4,fancybox=True,shadow=True)
plt.figure(figsize=(16,8))
plt.savefig('figure.png',dpi=300)
plt.show()
# markers=["o", "x",'v']
# tips = sns.load_dataset("tips")
# ax = sns.barplot(x="day", y="total_bill",hue='sex', data=tips)
# plt.show()
