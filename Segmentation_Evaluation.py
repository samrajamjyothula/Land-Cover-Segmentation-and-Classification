import numpy as np
import math


def Segmentation_Evaluation(sp, act):
    Tp = np.zeros((len(act), 1))
    Fp = np.zeros((len(act), 1))
    Tn = np.zeros((len(act), 1))
    Fn = np.zeros((len(act), 1))
    for i in range(len(act)):
        p = sp[i]
        a = act[i]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for j in range(len(p)):
            if a[j] == 1 and p[j] == 1:
                tp = tp + 1
            elif a[j] == 0 and p[j] == 0:
                tn = tn + 1
            elif a[j] == 0 and p[j] == 1:
                fp = fp + 1
            elif a[j] == 1 and p[j] == 0:
                fn = fn + 1
        Tp[i] = tp
        Fp[i] = fp
        Tn[i] = tn
        Fn[i] = fn

    tp = np.squeeze(sum(Tp))
    fp = np.squeeze(sum(Fp))
    tn = np.squeeze(sum(Tn))
    fn = np.squeeze(sum(Fn))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    Dice = (2*tp)/((2*tp) + fp + fn)
    Jaccard = tp/(tp + fp + fn)
    EVAL = [accuracy,Dice, Jaccard]
    return EVAL



