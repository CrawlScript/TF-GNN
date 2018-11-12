# coding=utf-8

from tensorflow.contrib.eager.python import tfe

from gnn.data.dataset import GraphDataset
from gnn.data.meta_network import NODE_TYPE_NODE
from gnn.model.gcn import GCN, GCNTrainer

tfe.enable_eager_execution()

m10 = GraphDataset("data/M10")
adj = m10.adj_matrix(sparse=True)
feature_matrix = m10.feature_matrix(bag_of_words=True)
labels = m10.label_list_or_matrix(one_hot=False)

train_node_indices, test_node_indices, train_masks, test_masks = m10.split_train_and_test(training_rate=0.3)

gcn_model = GCN([100, m10.num_classes()])
gcn_trainer = GCNTrainer(gcn_model)

gcn_trainer.train(adj, feature_matrix, labels, train_masks)


