#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from utils.precision_recall_f1 import precision_recall_f1
from scipy.fft import fft, fftfreq
from scipy import stats, signal
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
import statsmodels.api as sm
import ordpy
from sklearn import preprocessing
from utils.mergeNearestIndex import nearstIndex_v1, nearstIndex_v2
from PyEMD import EMD
import warnings
warnings.filterwarnings("ignore")

# load data
data_path = '../EMD-PERIOD/data/A3Benchmark'
file_name_list = []
file_list = os.listdir(data_path)
for i in file_list:
    if os.path.splitext(i)[1] == '.csv':
        if i != 'A3Benchmark_all.csv':
            file_name_list.append(i)
file_name_list.sort(key=lambda x: int(x.split('-TS')[1].split('.')[0]))
fileLength = len(file_name_list)

# decompositon
YahooA3_EMD = dict()
for i in tqdm(range(fileLength)):
    file_name = file_name_list[i]
    df = pd.read_csv(data_path + '/' + file_name, header=0)
    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='s')
    y = df[['value']].values.reshape(-1, )
    y_len = len(y)
    y_standard = preprocessing.scale(y)
    scaleFactor= 1
    Ts = 1 * scaleFactor
    fs = 1.0 / Ts
    y_emd = y_standard
    emd = EMD()
    IMF = emd(y_emd)
    YahooA3_EMD[file_name.split('.')[0]] = np.concatenate((y_emd.reshape(1, -1), IMF), axis=0)

# identify
IMF_period_dict = {}
IMF_index_period_dict = {}
IMF_LBtest_dict = {}
IMF_aftest_dict = {}
IMF_pearsonCorr_dict = {}
IMF_permutationEntropy_dict = {}
IMF_statisticalComplexity_dict = {}
for key, value in YahooA3_EMD.items():
    print('--', key, '--')
    y = value[0, :]
    IMF = value[1::, :]
    IMF_LBtest = []
    for n, imf in enumerate(IMF):
        LBtest = sm.stats.acorr_ljungbox(imf, return_df=False)
        if all(LBtest[1] > 0.05):
            IMF_LBtest.append(0)
        else:
            IMF_LBtest.append(1)
    IMF_aftest = []
    IMF_pptest = []
    IMF_kpsstest = []
    for n, imf in enumerate(IMF):
        dftest = adfuller(imf)
        if dftest[0] < dftest[4]['1%']:
            IMF_aftest.append(1)
        else:
            IMF_aftest.append(0)
    IMF_permutationEntropy = []
    IMF_statisticalComplexity = []
    for n, imf in enumerate(IMF):
        p, s = ordpy.complexity_entropy(imf)
        IMF_permutationEntropy.append(p)
        IMF_statisticalComplexity.append(s)
    y_len = len(y)
    Ts = 1 * 60 * 60
    fs = 1.0 / Ts
    freqs = fftfreq(y_len, 1 / fs)[:y_len // 2]
    IMF_period_fft = []
    IMF_index_period = []
    for i, imf in enumerate(IMF):
        if IMF_LBtest[i] == 0 or IMF_permutationEntropy[i] > 0.8:
            period_fft = -2
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])

        elif IMF_aftest[i] == 0:
            period_fft = -1
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])
        else:
            numPks = 1
            fft_imf = fft(imf)
            abs_fft_imf = np.abs(fft_imf)[:y_len // 2]
            pks = signal.argrelextrema(abs_fft_imf, np.greater)[0]
            top_k = np.argsort(abs_fft_imf[pks])[::-1][:numPks]
            top_k = pks[top_k]
            if len(top_k) > 0:
                period_fft = (1.0 / freqs[top_k] / 60 / 60)[0]
            else:
                period_fft = 0
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])
    indexMergedRecommended = nearstIndex_v2(np.array(IMF_index_period))
    IMFmerged = []
    for i in range(len(indexMergedRecommended)):
        if isinstance(indexMergedRecommended[i], list) and len(indexMergedRecommended[i]) > 1:

            IMFtemp = np.sum(IMF[np.array(indexMergedRecommended[i]).astype(int), :], axis=0)
        else:
            IMFtemp = IMF[int(indexMergedRecommended[i])]

        IMFmerged.append(IMFtemp)
    IMF = np.array(IMFmerged)
    IMF_LBtest = []
    for n, imf in enumerate(IMF):
        LBtest = sm.stats.acorr_ljungbox(imf, return_df=False)
        if all(LBtest[1] > 0.05):
            IMF_LBtest.append(0)
        else:
            IMF_LBtest.append(1)
    IMF_LBtest = np.array(IMF_LBtest)
    IMF_LBtest_dict[key] = IMF_LBtest
    IMF_aftest = []
    IMF_pptest = []
    IMF_kpsstest = []
    for n, imf in enumerate(IMF):
        dftest = adfuller(imf)
        if dftest[0] < dftest[4]['1%']:
            IMF_aftest.append(1)
        else:
            IMF_aftest.append(0)
    IMF_aftest = np.array(IMF_aftest)
    IMF_aftest_dict[key] = IMF_aftest
    IMF_season = IMF[np.where(np.array(IMF_aftest) > 0)[0], :]
    IMF_trend = IMF[np.where(np.array(IMF_aftest) < 1)[0], :]
    y_emd_detrend = np.sum(IMF_season, axis=0)
    data_temp = np.concatenate((y_emd_detrend.reshape(1, -1), IMF), axis=0)
    pearson_corr = np.corrcoef(data_temp)
    pearson_corr = pearson_corr[1::, 0]
    IMF_pearsonCorr_dict[key] = pearson_corr
    IMF_permutationEntropy = []
    IMF_statisticalComplexity = []
    for n, imf in enumerate(IMF):
        p, s = ordpy.complexity_entropy(imf)
        IMF_permutationEntropy.append(p)
        IMF_statisticalComplexity.append(s)

    IMF_permutationEntropy_dict[key] = np.array(IMF_permutationEntropy)
    IMF_statisticalComplexity_dict[key] = np.array(IMF_statisticalComplexity)


    y_len = len(y)
    Ts = 1 * 60 * 60
    fs = 1.0 / Ts
    freqs = fftfreq(y_len, 1 / fs)[:y_len // 2]
    IMF_period_fft = []
    IMF_index_period = []

    for i, imf in enumerate(IMF):

        if IMF_LBtest[i] == 0 or IMF_permutationEntropy[i] > 0.8:
            period_fft = -2
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])
        elif IMF_aftest[i] == 0:
            period_fft = -1
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])
        else:
            numPks = 1
            fft_imf = fft(imf)
            abs_fft_imf = np.abs(fft_imf)[:y_len // 2]
            pks = signal.argrelextrema(abs_fft_imf, np.greater)[0]
            top_k = np.argsort(abs_fft_imf[pks])[::-1][:numPks]
            top_k = pks[top_k]
            if len(top_k) > 0:
                period_fft = (1.0 / freqs[top_k] / 60 / 60)[0]
            else:
                period_fft = 0
            IMF_period_fft.append(period_fft)
            IMF_index_period.append([i, period_fft])
    IMF_index_period_dict[key] = np.array(IMF_index_period)
    IMF_period_dict[key] = np.array(IMF_period_fft)

# evaluation
IMF_period_detected_dict = {}
period_expected = np.array([12, 24, 168])
precision_list = []
recall_list = []
f1score_list = []
for key, value in IMF_period_dict.items():
    bool_aftest = IMF_aftest_dict[key] > 0
    bool_pearsonCorr = IMF_pearsonCorr_dict[key] > 0.15
    bool_permutationEntropy = IMF_permutationEntropy_dict[key] < 0.8
    bool_syn = np.logical_and(np.logical_and(bool_aftest, bool_pearsonCorr), bool_permutationEntropy)
    IMF_period_detected_dict[key] = IMF_period_dict[key][bool_syn]
    true_bkps = period_expected
    my_bkps = IMF_period_detected_dict[key]
    precision, recall, f1score = precision_recall_f1(true_bkps, my_bkps, margin_percent=0)
    precision_list.append(precision)
    recall_list.append(recall)
    f1score_list.append(f1score)
f1score_result = np.mean(np.array(f1score_list))
print('f1score_result: ', f1score_result)

import xlwt
file = xlwt.Workbook('encoding = utf-8')
sheet1=file.add_sheet('sheet1',cell_overwrite_ok=True)
sheet1.write(0,0,"precision")
sheet1.write(0,1,"recall")
sheet1.write(0,2,"f1score")
for i in range(len(f1score_list)):
    sheet1.write(i+1,0,precision_list[i])
    sheet1.write(i+1,1,recall_list[i])
    sheet1.write(i+1,2,f1score_list[i])
file.save('EMD_YahooA3_result.xls')
