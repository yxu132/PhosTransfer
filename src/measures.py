from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
import numpy as np
import sys

fpr_cutoff = 0.09


def tpr_fpr(gold, predicted, cutoff):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in xrange(len(gold)):
        if gold[i] == 1:
            if predicted[i] > cutoff:
                tp += 1
            else:
                fn += 1
        else:
            if predicted[i] > cutoff:
                fp += 1
            else:
                tn += 1
    fpr = fp / float(fp + tn)
    tpr = tp / float(tp + fn)
    return tpr, fpr

def count(gold, predicted, cutoff):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in xrange(len(gold)):
        if gold[i] == 1:
            if predicted[i] > cutoff:
                tp += 1
            else:
                fn += 1
        else:
            if predicted[i] > cutoff:
                fp += 1
            else:
                tn += 1
    sp = tn / float(tn+fp+sys.float_info.epsilon)
    pr = tp / float(tp+fp+sys.float_info.epsilon)
    se = tp / float(tp+fn+sys.float_info.epsilon)
    return pr, se, sp

def mcc(gold, predict, cutoff):
    predict_b = []
    for p in predict:
        if p > cutoff:
            predict_b.append(1)
        else:
            predict_b.append(-1)
    return metrics.matthews_corrcoef(gold, predict_b)


def auc(gold, predict):
    fpr, tpr, thresholds = metrics.roc_curve(np.array(gold), np.array(predict), pos_label=1)
    ind = -1
    for indj, o in enumerate(fpr):
        if o > fpr_cutoff:
            ind = indj
            break
    cutoff = thresholds[ind]
    auc = metrics.auc(fpr, tpr)
    # fpr, tpr = tpr_fpr(gold, predict)
    return fpr, tpr, thresholds, auc, cutoff

def evaluate(golds, predicts):
    pr, sp, se = count(golds, predicts, 0.5)
    mcc_score = mcc(golds, predicts, 0.5)
    _, _, _, auc_score = auc(golds, predicts)
    print auc_score, se, sp, mcc_score

