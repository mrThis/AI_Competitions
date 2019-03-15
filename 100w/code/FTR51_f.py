# FTR51 时序特征提取

import pandas as pd

from tsfresh import extract_features
dic = {'abs_energy': None,
           'absolute_sum_of_changes': None,
           'agg_autocorrelation': [{'f_agg': 'mean'}, {'f_agg': 'median'}, {'f_agg': 'var'}],
           'agg_linear_trend': [
               {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'},
               {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'min'},
               {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'},
               {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'},

               {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'},
               {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'},
               {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'},
               {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'},

               {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'},
               {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'min'},
               {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'},
               {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'},

               {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'max'},
               {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'},
               {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'},
               {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'var'}],
           'approximate_entropy': [{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.3}, {'m': 2, 'r': 0.5}, {'m': 2, 'r': 0.7},
                                   {'m': 2, 'r': 0.9}],
           'ar_coefficient': [{'coeff': 2, 'k': 10}, {'coeff': 3, 'k': 10}, {'coeff': 4, 'k': 10}],
           'augmented_dickey_fuller': [{'attr': 'teststat'}, {'attr': 'pvalue'}, {'attr': 'usedlag'}],
           'c3': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
           'change_quantiles': [{'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'},
                                {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'},
                                {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'},
                                {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}],
           'cid_ce': [{'normalize': True}, {'normalize': False}],
           'count_above_mean': None,
           'count_below_mean': None,
           'cwt_coefficients': [{'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 2},
                                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 5},
                                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 10},
                                {'widths': (2, 5, 10, 20), 'coeff': 0, 'w': 20}, ],
           'energy_ratio_by_chunks': [{'num_segments': 10, 'segment_focus': 0},
                                      {'num_segments': 10, 'segment_focus': 1},
                                      {'num_segments': 10, 'segment_focus': 2},
                                      {'num_segments': 10, 'segment_focus': 3},
                                      {'num_segments': 10, 'segment_focus': 4},
                                      {'num_segments': 10, 'segment_focus': 5},
                                      {'num_segments': 10, 'segment_focus': 6},
                                      {'num_segments': 10, 'segment_focus': 7},
                                      {'num_segments': 10, 'segment_focus': 8},
                                      {'num_segments': 10, 'segment_focus': 9}],
           'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'},
                              {'aggtype': 'kurtosis'}],
           'fft_coefficient': [{'coeff': 0, 'attr': 'real'}, {'coeff': 1, 'attr': 'real'}, {'coeff': 2, 'attr': 'real'},
                               {'coeff': 3, 'attr': 'real'}, {'coeff': 4, 'attr': 'real'}, {'coeff': 5, 'attr': 'real'},
                               {'coeff': 6, 'attr': 'real'}, {'coeff': 7, 'attr': 'real'}, {'coeff': 8, 'attr': 'real'},
                               {'coeff': 9, 'attr': 'real'}, {'coeff': 10, 'attr': 'real'},
                               {'coeff': 15, 'attr': 'real'}, {'coeff': 20, 'attr': 'real'},
                               {'coeff': 25, 'attr': 'real'},
                               {'coeff': 30, 'attr': 'real'},
                               {'coeff': 0, 'attr': 'imag'}, {'coeff': 1, 'attr': 'imag'},
                               {'coeff': 2, 'attr': 'imag'}, {'coeff': 3, 'attr': 'imag'}, {'coeff': 4, 'attr': 'imag'},
                               {'coeff': 5, 'attr': 'imag'}, {'coeff': 6, 'attr': 'imag'}, {'coeff': 7, 'attr': 'imag'},
                               {'coeff': 8, 'attr': 'imag'}, {'coeff': 9, 'attr': 'imag'},
                               {'coeff': 10, 'attr': 'imag'},
                               {'coeff': 15, 'attr': 'imag'},
                               {'coeff': 20, 'attr': 'imag'}, {'coeff': 25, 'attr': 'imag'},
                               {'coeff': 30, 'attr': 'imag'},
                               {'coeff': 0, 'attr': 'abs'},
                               {'coeff': 1, 'attr': 'abs'}, {'coeff': 2, 'attr': 'abs'}, {'coeff': 3, 'attr': 'abs'},
                               {'coeff': 4, 'attr': 'abs'}, {'coeff': 5, 'attr': 'abs'}, {'coeff': 6, 'attr': 'abs'},
                               {'coeff': 7, 'attr': 'abs'}, {'coeff': 8, 'attr': 'abs'}, {'coeff': 9, 'attr': 'abs'},
                               {'coeff': 10, 'attr': 'abs'}, {'coeff': 15, 'attr': 'abs'}, {'coeff': 20, 'attr': 'abs'},
                               {'coeff': 25, 'attr': 'abs'}, {'coeff': 30, 'attr': 'abs'},
                               ],
           'last_location_of_maximum': None,
           'first_location_of_maximum': None,
           'last_location_of_minimum': None,
           'first_location_of_minimum': None,
           'friedrich_coefficients': [{'coeff': 0, 'm': 3, 'r': 30}, {'coeff': 1, 'm': 3, 'r': 30},
                                      {'coeff': 2, 'm': 3, 'r': 30}, {'coeff': 3, 'm': 3, 'r': 30}],
           'index_mass_quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.3}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}, {'q': 0.8},
                                   {'q': 0.9}],
           'large_standard_deviation': [{'r': 0.2}, {'r': 0.3}, ],
           'linear_trend': [{'attr': 'intercept'}, {'attr': 'stderr'}],
           'longest_strike_below_mean': None,
           'longest_strike_above_mean': None,
           'mean_abs_change': None,
           'number_cwt_peaks': [{'n': 1}, {'n': 5}],
           'number_peaks': [{'n': 1}, {'n': 3}, {'n': 5}, {'n': 10}, {'n': 50}],
           'partial_autocorrelation': [{'lag': 8}, ],
           'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
           'percentage_of_reoccurring_values_to_all_values': None,
           'quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.3}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}, {'q': 0.8}, {'q': 0.9}],
           'range_count': [{'min': -1, 'max': 1}],
           'ratio_beyond_r_sigma': [{'r': 1}, {'r': 2}, {'r': 3}],
           'ratio_value_number_to_time_series_length': None,
           'sample_entropy': None,
           'spkt_welch_density': [{'coeff': 2}, {'coeff': 5}, {'coeff': 8}],
           'sum_of_reoccurring_values': None,
           'sum_of_reoccurring_data_points': None,
           'time_reversal_asymmetry_statistic': [{'lag': 1}, {'lag': 2}, {'lag': 3}],
           }
##没有办法具体到药品变换，只能具体到药品数量变换

# 将笔数作为序列，应该没有意义

# 将天数作为序列，它们之间的变化应该有意义

# 将月作为序列，它们之间的变化应该也有意义

#
def handle_applyno(data):
    data['FTR51'] = 1
    x = data.groupby('PERSONID').apply(lambda k: k.groupby('APPLYNO').sum())
    df = extract_features(x.FTR51.reset_index(), column_id="PERSONID",default_fc_parameters=dic)
    return df

#
def handle_createtime(data):
    data.drop('APPLYNO',inplace=True,axis=1)
    data['FTR51'] = 1
    x = data.groupby('PERSONID').apply(lambda k: k.groupby('CREATETIME').sum())
    print(x.head(5))
    df = extract_features(x.FTR51.reset_index(), column_id="PERSONID",column_sort="CREATETIME",default_fc_parameters=dic)
    return df

#
def handle_createmonth(data):
    pass

# train_imp = handle_applyno(train)
# print(train_imp.head(5))
# print(train_imp.shape)
# test_imp = handle_applyno(test)
# print(test_imp.shape)
# train_imp.to_pickle('../data/input/FTR51_new1.p')
# test_imp.to_pickle('../data/input/FTR51_new1.p')

if __name__ == '__main__':
    #处理前
    train = pd.read_csv('../data/output/implement.csv')
    test = pd.read_csv('../data/output/implement_test.csv')

    #保留index信息
    train_i = train.PERSONID.unique()
    test_i = test.PERSONID.unique()
    all = train.append(test)

    #处理后，返回PERSONID_index
    all = handle_createtime(all)

    #处理后切割
    train_imp = all.loc[train_i, :].sort_index()
    test_imp = all.loc[test_i, :].sort_index()

    print(train_imp.shape)
    print(test_imp.shape)

    #FTR51_new2 根据时间的 #
    #FTR51 new3 根据每笔的 #
    #FTR51 new4 根据每月的 #
    train_imp.to_pickle('../data/input/FTR51_new2train.p')
    test_imp.to_pickle('../data/input/FTR51_new2test.p')

