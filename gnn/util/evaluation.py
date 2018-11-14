# coding=utf-8
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate(preds, labels, masks):
    masked_node_indices = np.nonzero(masks)
    masked_preds = preds[masked_node_indices]
    masked_labels = labels[masked_node_indices]

    accuracy = accuracy_score(masked_labels, masked_preds)
    macro_f1 = f1_score(masked_labels, masked_preds, pos_label=None, average='macro')
    micro_f1 = f1_score(masked_labels, masked_preds, pos_label=None, average='micro')

    return accuracy, macro_f1, micro_f1

    # corrects = (masked_preds == masked_labels).astype(np.float32)
    # accuracy = corrects.mean()
    # return accuracy


