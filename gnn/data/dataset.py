# coding=utf-8

import os
import numpy as np
import re

from scipy.sparse import coo_matrix, csr_matrix


class Tokenizer(object):
    def __init__(self):
        self.token_id_index_dict = {}
        self.token_index_id_dict = {}

    # get token_index if token_id exists
    # otherwise create token_index for token_id
    def get_or_create_token_index(self, token_id):
        if token_id in self.token_id_index_dict:
            return self.token_id_index_dict[token_id]
        else:
            token_index = self.num_tokens()
            self.token_index_id_dict[token_index] = token_id
            self.token_id_index_dict[token_id] = token_index
            return token_index

    def get_token_index(self, token_id):
        return self.token_id_index_dict[token_id]

    def num_tokens(self):
        return len(self.token_id_index_dict)

    # sentence:str => words:list[str]
    def tokenize(self, s):
        raise NotImplementedError()

    def tokenize_to_indices(self, s, create_token_index=True):
        token_ids = self.tokenize(s)
        if create_token_index:
            token_indices = [self.get_or_create_token_index(token_id) for token_id in token_ids]
        else:
            token_indices = [self.get_token_index(token_id) for token_id in token_ids]
        return token_indices


# default tokenizer, splitting by white spaces
class WhiteSpaceTokenizer(Tokenizer):
    def tokenize(self, s):
        return s.split()


class EnglishWordTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
        self.punc_re = re.compile("[^a-zA-Z]")

    def tokenize(self, s):
        s = s.lower()
        s = self.punc_re.sub(" ", s)
        return s.split()


# construct dataset from a data directory
# the data directory should contain:
#   - adjedges.txt or edgelist.txt
#   - docs.txt
#   - labels.txt
class GraphDataset(object):
    FORMAT_ADJEDGES = "adjedges"
    FORMAT_EDGELIST = "edgelist"

    # read data from data_dir
    # if ignore_featureless_node is True, nodes without content or features will be ignored
    def __init__(self, data_dir, data_format=FORMAT_ADJEDGES, ignore_featureless_node=True, tokenizer=None):
        self.data_dir = data_dir
        self.data_format = data_format
        self.ignore_featureless_node = True
        # node_id:str => node_index:int
        self.node_id_index_dict = {}
        # node_index:int => node_id:str
        self.node_index_id_dict = {}

        self.node_index_label_dict = {}

        # key0: node_index0 => key1 => node_index1 => value: weight
        self.adj_dict = {}

        # type of label is str
        self.label_id_index_dict = {}
        self.label_index_id_dict = {}

        self.node_index_feature_dict = {}

        if tokenizer is None:
            self.tokenizer = EnglishWordTokenizer()
        else:
            self.tokenizer = tokenizer


        # read document first such that nodes with content will have small index
        self._read_docs()
        self._read_labels()
        self._read_structure()

    def add_edge(self, node_index0, node_index1, weight):
        if node_index0 in self.adj_dict:
            weight_dict = self.adj_dict[node_index0]
        else:
            weight_dict = {}
            self.adj_dict[node_index0] = weight_dict

        weight_dict[node_index1] = weight

    def add_edges(self, node_index0, node_index1, weight):
        self.add_edge(node_index0, node_index1, weight)
        self.add_edge(node_index1, node_index0, weight)

    # get node_index if node_id exists
    # otherwise create node_index for node_id
    def get_or_create_node_index(self, node_id):
        if node_id in self.node_id_index_dict:
            return self.node_id_index_dict[node_id]
        else:
            node_index = self.num_nodes()
            self.node_index_id_dict[node_index] = node_id
            self.node_id_index_dict[node_id] = node_index
            return node_index

    # will raise exception when node_id does not exist
    def get_node_index(self, node_id):
        return self.node_id_index_dict[node_id]

    def get_node_id(self, node_index):
        return self.node_index_id_dict[node_index]

    def has_node_id(self, node_id):
        return node_id in self.node_id_index_dict

    # get label_index if label_id exists
    # otherwise create label_index for label_id
    def get_or_create_label_index(self, label_id):
        if label_id in self.label_id_index_dict:
            return self.label_id_index_dict[label_id]
        else:
            label_index = self.num_labels()
            self.label_index_id_dict[label_index] = label_id
            self.label_id_index_dict[label_id] = label_index
            return label_index

    def get_label_index(self, label_id):
        return self.label_id_index_dict[label_id]

    def get_label_id(self, label_index):
        return self.label_index_id_dict[label_index]

    def num_nodes(self):
        return len(self.node_id_index_dict)

    def num_labels(self):
        return len(self.label_id_index_dict)

    # if sparse, return a csr_matrix
    def adj_matrix(self, sparse=False):
        data = []
        row = []
        col = []
        for node_index0 in self.adj_dict:
            weight_dict = self.adj_dict[node_index0]
            for node_index1 in weight_dict:
                data.append(weight_dict[node_index1])
                row.append(node_index0)
                col.append(node_index1)
        adj = csr_matrix((data, (row, col)), shape=(self.num_nodes(), self.num_nodes()))
        if sparse:
            return adj
        else:
            return adj.todense()
        # if sparse:
        #     adj = csr_matrix((self.num_nodes(), self.num_nodes()), dtype=np.float32)
        # else:
        #     adj = np.zeros((self.num_nodes(), self.num_nodes()), dtype=np.float32)
        #
        # for node_index0 in self.adj_dict:
        #     weight_dict = self.adj_dict[node_index0]
        #     for node_index1 in weight_dict:
        #         adj[node_index0, node_index1] = weight_dict[node_index1]
        # return adj

    def _read_structure(self):
        if self.data_format == GraphDataset.FORMAT_ADJEDGES:
            self._read_adjedges()
        else:
            raise Exception("not support yet")

    def _read_adjedges(self):
        adjedges_fpath = os.path.join(self.data_dir, "adjedges.txt")
        with open(adjedges_fpath, "r", encoding="utf-8") as f:
            for line in f:
                node_ids = line.split()
                node_id0 = node_ids[0]
                if self.ignore_featureless_node and not self.has_node_id(node_id0):
                    continue
                node_index0 = self.get_or_create_node_index(node_id0)

                for node_id1 in node_ids[1:]:
                    if self.ignore_featureless_node and not self.has_node_id(node_id1):
                        continue
                    node_index1 = self.get_or_create_node_index(node_id1)
                    self.add_edges(node_index0, node_index1, 1.0)

    def _read_docs(self):
        docs_fpath = os.path.join(self.data_dir, "docs.txt")
        with open(docs_fpath, "r", encoding="utf-8") as f:
            for line in f:
                node_id, sentence = re.split(r"\s+", line, 1)
                node_index = self.get_or_create_node_index(node_id)
                token_indices = self.tokenizer.tokenize_to_indices(sentence)
                self.node_index_feature_dict[node_index] = token_indices

    def _read_labels(self):
        labels_fpath = os.path.join(self.data_dir, "labels.txt")
        with open(labels_fpath, "r", encoding="utf-8") as f:
            for line in f:
                node_id, label_id = line.split()
                if self.ignore_featureless_node:
                    node_index = self.get_node_index(node_id)
                else:
                    node_index = self.get_or_create_node_index(node_id)
                label_index = self.get_or_create_label_index(label_id)
                self.node_index_label_dict[node_index] = label_index

    def get_feature(self, node_index):
        return self.node_index_feature_dict[node_index]

    def feature_matrix(self, bag_of_words=False):
        if bag_of_words:
            feature_dim = self.tokenizer.num_tokens()
            feature_matrix = np.zeros((self.num_nodes(), feature_dim), dtype=np.float32)

            for node_index in range(self.num_nodes()):
                token_indices = self.node_index_feature_dict[node_index]
                feature_matrix[node_index][token_indices] = 1.0
            return feature_matrix

        else:
            raise NotImplementedError()

    def num_classes(self):
        return len(self.label_id_index_dict)

    def label_list_or_matrix(self, one_hot=False):
        label_indices = [self.node_index_label_dict[node_index] for node_index in range(self.num_nodes())]
        if one_hot:
            label_matrix = np.zeros((self.num_nodes(), self.num_classes()), dtype=np.int32)
            label_matrix[np.arange(self.num_nodes()), label_indices] = 1
            return label_matrix
        else:
            return np.array(label_indices)

    def split_train_and_test(self, training_rate):
        random_node_indices = np.random.permutation(self.num_nodes())
        training_size = int(self.num_nodes() * training_rate)
        train_node_indices = random_node_indices[:training_size]
        test_node_indices = random_node_indices[training_size:]
        train_masks = np.zeros_like(random_node_indices, dtype=np.int32)
        train_masks[train_node_indices] = 1
        test_masks = np.zeros_like(random_node_indices, dtype=np.int32)
        test_masks[test_node_indices] = 1
        return train_node_indices, test_node_indices, train_masks, test_masks
