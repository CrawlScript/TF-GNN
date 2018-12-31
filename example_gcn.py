# coding=utf-8
from gnn.data.dataset import GraphDataset, WhiteSpaceTokenizer
from gnn.data.example import load_M10, load_cora, load_dblp
from gnn.model.gcn import GCN, GCNTrainer
import tensorflow as tf

# eager mode must be enabled
from tensorflow.contrib.eager.python import tfe

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tfe.enable_eager_execution()

# read graph dataset: data/M10 data/dblp
# dataset = GraphDataset("data/dblp", ignore_featureless_node=True)
dataset = load_M10("data/M10", ignore_featureless_node=True)

adj = dataset.adj_matrix(sparse=True)
feature_matrix, feature_masks = dataset.feature_matrix(bag_of_words=True, sparse=True)
labels, label_masks = dataset.label_list_or_matrix(one_hot=False)

train_node_indices, test_node_indices, train_masks, test_masks = dataset.split_train_and_test(training_rate=0.3)

gcn_model = GCN([16, dataset.num_classes()])
gcn_trainer = GCNTrainer(gcn_model)
gcn_trainer.train(adj, feature_matrix, labels, train_masks, test_masks, learning_rate=1e-3, l2_coe=1e-3)
