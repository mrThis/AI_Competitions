import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv('../paper_result/lstm.csv')

grid = sns.FacetGrid(df, col="classifier", hue="metrics", col_wrap=1,hue_kws={"marker": ["^", "x",'o']})



grid.map(plt.plot, "dataset", "result", marker="o",ms=4)



grid.fig.tight_layout(w_pad=1)

# grid.ax.invert_xaxis()
grid.add_legend()
plt.savefig('figure.svg',dpi=300)
plt.show()
