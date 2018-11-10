# coding=utf-8
import numpy as np


def evaluate_accuracy(preds, labels, masks):
    masked_node_indices = np.nonzero(masks)
    masked_preds = preds[masked_node_indices]
    masked_labels = labels[masked_node_indices]
    corrects = (masked_preds == masked_labels).astype(np.float32)
    accuracy = corrects.mean()
    return accuracy


