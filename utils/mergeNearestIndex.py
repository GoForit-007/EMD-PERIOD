#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/29 17:32
import numpy as np
import pandas as pd


def nearstIndex_v1(period, threshold=0.8, thresholdType='soft'):
    assert len(period.shape) == 1, 'Array shape is not (N,)'
    indexMergedRecommended = []
    N = len(period)
    if N == 1:
        indexMergedRecommended.append(0)
        return indexMergedRecommended
    period[np.isnan(period)] = 10000 * np.nanmax(period)
    i = 0
    while (i < N):
        j = i + 1
        if j == N:
            return indexMergedRecommended
        if thresholdType == 'soft':
            thresholdValue = period[i] * threshold
        if thresholdType == 'hard':
            thresholdValue = threshold
        diffTemp = period[j] - period[i]
        if diffTemp >= thresholdValue:
            indexMergedRecommended.append(int(i))
            i = i + 1
            if i == N - 1:
                indexMergedRecommended.append(int(i))
            continue
        if diffTemp < thresholdValue:
            candidate = [int(i), int(j)]
            if j == N - 1:
                indexMergedRecommended.append(candidate)
                return indexMergedRecommended
            elif j < N - 1:
                k = j
                k = k + 1
                while (k <= N - 1):

                    diffTemp = period[k] - period[i]
                    if diffTemp <= thresholdValue:
                        candidate.append(int(k))
                        if k == N - 1:
                            indexMergedRecommended.append(candidate)
                            return indexMergedRecommended
                        k = k + 1
                        continue
                    elif diffTemp > thresholdValue:
                        indexMergedRecommended.append(candidate)
                        if k == N - 1:
                            indexMergedRecommended.append(k)
                        i = k
                        break
    return indexMergedRecommended

def nearstIndex_v2(IMF_index_period, threshold=0.8, thresholdType='soft'):
    assert len(IMF_index_period.shape) == 2, 'Array shape is not (N,2)'
    indexMergedRecommended = []
    N = len(IMF_index_period)
    if N == 1:
        indexMergedRecommended.append(int(IMF_index_period[0, 0]))
        return indexMergedRecommended
    df = pd.DataFrame(IMF_index_period)
    temp = df[df.iloc[:, 1] < 0].iloc[:, 0].values.tolist()
    if len(temp) > 0:
        for x in temp:
            indexMergedRecommended.append(int(x))
    df_period = df.drop(df[df.iloc[:, 1] < 0].index)
    df_period = df_period.sort_values(by=1)
    df_period=df_period.reset_index(drop=True)
    if df_period.empty:
        return IMF_index_period[:,0].tolist()
    period=df_period.iloc[:,1].values
    mergedIndextemp=nearstIndex_v1(period, threshold=threshold, thresholdType=thresholdType)
    if len(mergedIndextemp)>0:
        for x in mergedIndextemp:
            temp=df_period.iloc[x,0].tolist()
            indexMergedRecommended.append(temp)
    return indexMergedRecommended




