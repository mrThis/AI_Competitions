import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



df=pd.read_csv('../paper_result/validation.csv')

grid = sns.FacetGrid(df, col="dataset", hue="validation", col_wrap=4, size=3,hue_kws={"marker": ["^", "x",'o']})



grid.map(plt.plot, "classifier", "results", marker="o",ms=4)



grid.fig.tight_layout(w_pad=1)

# grid.ax.invert_xaxis()
grid.add_legend()
plt.savefig('figure.png',dpi=300)
plt.show()
