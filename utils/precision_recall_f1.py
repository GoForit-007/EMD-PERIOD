#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/10 15:30

import numpy as np
from itertools import product

def pr2_f1score(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def precision_recall_f1(true_bkps, my_bkps, margin_percent=5):

    assert margin_percent >= 0, "margin_percent of error must be non-negative (magin_percent={})".format(margin_percent)
    assert len(true_bkps) > 0, "currently onlg assume at least one element in true_bkps"

    if len(my_bkps) == 0:
        return 0.0, 0.0, 0.0

    used=set()
    true_pos=set(
        true_b for true_b, my_b in product(true_bkps,my_bkps)
        if true_b*(1-margin_percent/100.0)<=my_b<=true_b*
        (1+margin_percent/100.0) and not (my_b in used or used.add(my_b))
    )

    tp_=len(true_pos)
    precision=tp_/len(my_bkps)
    recall=tp_/len(true_bkps)
    f1score=pr2_f1score(precision,recall)
    return precision,recall,f1score


