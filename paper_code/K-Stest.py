from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
df = pd.read_csv('../paper_result/validation.csv')
#
# for str in ['mc2', 'jm1', 'kc3', 'cm1', 'pc3', 'pc4', 'mw1', 'pc1', 'pc5', 'mc1', 'pc2']:
#     df1=df[df['dataset'] == str]
#     s1 = df1[df1['validation'] == 'sampling']['results'].values
#     s2 = df1[df1['validation'] == 'non-sampling']['results'].values
#     # out = ks_2samp(s1,s2)
#     # print("{:.3f}".format(out.pvalue))
#     # print str
#     print("{:.3f}".format(np.std(s2)))
#     # print np.std(s2)


s3 = [0.640246528,
0.552152598,
0.633696181,
0.658363608,
0.716389971,
0.869338363,
0.621750659,
0.724347413,
0.871910735,
0.703041323,
0.787932839]

print 'cat'
print("{:.3f}".format(np.mean(s3)))
print("{:.3f}".format(np.std(s3)))

s4 = [0.665274306,
0.590731701,
0.662378472,
0.676179803,
0.71980744,
0.85171058,
0.665996377,
0.738781679,
0.861151415,
0.675181655,
0.713717656]
print 'lgb'
print("{:.3f}".format(np.mean(s4)))
print("{:.3f}".format(np.std(s4)))

s5 = [0.632652778,
0.60296235,
0.575791667,
0.573045874,
0.65774785,
0.567412301,
0.617831028,
0.68651634,
0.796483117,
0.589860957,
0.599328387
]
print 'xgb'
print("{:.3f}".format(np.mean(s5)))
print("{:.3f}".format(np.std(s5)))

s6 = [0.596718774,
0.685956597,
0.611452894,
0.639643897,
0.647392897,
0.70523386,
0.671169357,
0.940865397,
0.750078536,
0.733768075]
print 'NB'
print("{:.3f}".format(np.mean(s6)))
print("{:.3f}".format(np.std(s6)))

s7 = [0.640246528,
0.552152598,
0.633696181,
0.658363608,
0.716389971,
0.869338363,
0.621750659,
0.724347413,
0.871910735,
0.703041323,
0.787932839]

print 'RF'
print("{:.3f}".format(np.mean(s7)))
print("{:.3f}".format(np.std(s7)))

