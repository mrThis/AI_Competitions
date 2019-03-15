import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv('../data_set/paper_result/model_r2_new3.csv')

grid = sns.FacetGrid(df, col="classification_model", hue="metrics", col_wrap=4, size=3,hue_kws={"marker": ["^", "x",'o']})



grid.map(plt.plot, "dataset", "result", marker="o",ms=4)



grid.fig.tight_layout(w_pad=1)

# grid.ax.invert_xaxis()
grid.add_legend()
plt.savefig('figure.png')
plt.show()
