# coding=utf-8

from gnn.data.dataset import GraphDataset
from gnn.model.gcn import GCN, GCNTrainer

# eager mode must be enabled
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

# read graph dataset: data/M10 data/dblp
dataset = GraphDataset("data/M10")

adj = dataset.adj_matrix(sparse=True)
feature_matrix = dataset.feature_matrix(bag_of_words=True, sparse=True)
labels = dataset.label_list_or_matrix(one_hot=False)

train_node_indices, test_node_indices, train_masks, test_masks = dataset.split_train_and_test(training_rate=0.3)

gcn_model = GCN([100, dataset.num_classes()])
gcn_trainer = GCNTrainer(gcn_model)
gcn_trainer.train(adj, feature_matrix, labels, train_masks, learning_rate=1e-3, l2_coe=1e-4)


