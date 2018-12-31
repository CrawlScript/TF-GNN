# coding=utf-8
from scipy.sparse import csr_matrix, _sparsetools
import numpy as np
import random
from multiprocessing.dummy import Pool as ThreadPool

# N_TYPE denotes node type
N_TYPE_NODE = "N_NODE"
N_TYPE_LABEL = "N_LABEL"


def dict_get_or_create_value(dict_object, key, default_value):
    if key in dict_object:
        return dict_object[key]
    else:
        dict_object[key] = default_value
        return default_value


class IdIndexer(object):
    def __init__(self):
        self.id_index_dict = {}
        self.index_id_dict = {}

    def get_index(self, id, create=False):
        if not create or id in self.id_index_dict:
            return self.id_index_dict[id]
        index = len(self.id_index_dict)
        self.id_index_dict[id] = index
        self.index_id_dict[index] = id
        return index

    def get_indices(self, ids):
        return [self.get_index(id) for id in ids]

    def get_id(self, index):
        return self.index_id_dict[index]

    def get_ids(self, indices):
        return [self.index_id_dict[index] for index in indices]

    def has_id(self, id):
        return id in self.id_index_dict

    def list_ids(self):
        return list(self.id_index_dict)

    def list_indices(self):
        return list(self.index_id_dict)

    def __len__(self):
        return len(self.id_index_dict)




class Adj(object):
    def __init__(self):
        self.data = []
        self.row = []
        self.col = []
        self.cached_csr = None
        self.cached_sample_list = None

    def build_cache(self, shape=None):
        self.cached_csr = self.to_csr(shape=shape)
        self.cached_sample_list = self.get_sample_list()

    def to_csr(self, shape):
        return csr_matrix((self.data, (self.row, self.col)), shape=shape)

    def add_edge(self, node_index0, node_index1, weight=1.0):
        self.data.append(weight)
        self.row.append(node_index0)
        self.col.append(node_index1)

    def add_edges(self, node_index0, node_index1, weight=1.0):
        self.add_edge(node_index0, node_index1, weight)
        self.add_edge(node_index1, node_index0, weight)

    def get_neighbor_dict(self, node_index):
        return {col_index: self.cached_csr[node_index, col_index] for col_index in self.get_neighbors(node_index)}

    def get_neighbors(self, node_index):
        return [col_index for col_index in self.cached_csr.getrow(node_index).indices]

    def get_sample_list(self):
        return list(set(self.row))

    def sample_node(self, excluded_node_indices=None):
        while True:
            node_index = self.cached_sample_list[random.randrange(0, len(self.cached_sample_list))]
            if excluded_node_indices is None or node_index not in excluded_node_indices:
                return node_index

    def sample_neighbor(self, node_index):
        neighbors = self.get_neighbors()
        if len(neighbors) == 0:
            return None
        return neighbors[random.randrange(0, len(neighbors))]

    def sample_triple(self, node_a=None):
        if node_a is None:
            node_a = self.sample_node()
        neighbors = self.get_neighbors(node_a)
        node_b = self.sample_neighbor(node_a)
        excluded = set([node_a] + neighbors)
        node_neg = self.sample_node(excluded_node_indices=excluded)
        return node_a, node_b, node_neg



# Heterogeneous Network
# edges are based on meta-paths
class MetaNetwork(object):

    # read data from data_dir
    # if ignore_featureless_node is True, nodes without content or features will be ignored
    def __init__(self):
        self.node_type_indexer_dict = {}
        self.meta_adj_dict = {}
        # node_type:str => node_index:int => node_attrdict: dict
        self.node_type_index_attrdict_dict = {}
        # node_type0:str => node_type1:str => node_index:int => neighbor_node_indices:list
        self.meta_neighbors_dict = {}
        # key0: node_type0 => key1: node_type1 => key2: node_index0 => key3 => node_index1 => value: weight
        self.meta_adj_dict = {}
        # key0: node_type0 => key1: node_type1 => sample_list: list[int]
        self.meta_sample_list_dict = {}

    def get_indexer(self, node_type, create=False):
        if not create or node_type in self.node_type_indexer_dict:
            return self.node_type_indexer_dict[node_type]
        indexer = IdIndexer()
        self.node_type_indexer_dict[node_type] = indexer
        return indexer

    def get_adj(self, meta, create=False):
        if not create or meta in self.meta_adj_dict:
            return self.meta_adj_dict[meta]
        adj = Adj()
        self.meta_adj_dict[meta] = adj
        return adj

    def list_node_ids(self, meta):
        return self.get_indexer(meta).list_ids()

    def list_node_indices(self, meta):
        return self.get_indexer(meta).list_indices()

    def get_node_attrdict(self, node_type, node_index, create=False):
        if not create:
            return self.node_type_index_attrdict_dict[node_type][node_index]
        node_index_attrdict = dict_get_or_create_value(self.node_type_index_attrdict_dict, node_type, {})
        attrdict = dict_get_or_create_value(node_index_attrdict, node_index, {})
        return attrdict

    def get_node_attr(self, node_type, node_index, attr_name, return_none_if_not_exist=False):
        try:
            return self.get_node_attrdict(node_type, node_index, create=False)[attr_name]
        except Exception as e:
            if return_none_if_not_exist:
                return None
            else:
                raise e

    def has_node_attr(self, node_type, node_index, attr_name):
        return self.get_node_attr(node_type, node_index, attr_name, return_none_if_not_exist=True) is not None

    def get_node_attrs(self, node_type, node_indices, attr_name, return_none_if_not_exist=False):
        return [self.get_node_attr(node_type, node_index, attr_name, return_none_if_not_exist) for node_index in node_indices]

    def set_node_attr(self, node_type, node_index, attr_name, attr_value):
        attrdict = self.get_node_attrdict(node_type, node_index, create=True)
        attrdict[attr_name] = attr_value

    # get node_index if node_id exists
    # otherwise create node_index for node_id
    def get_node_index(self, node_type, node_id, create=False):
        return self.get_indexer(node_type, create).get_index(node_id, create)

    def get_node_indices(self, node_type, node_ids):
        return self.get_indexer(node_type, create=False).get_ids(node_ids)

    def get_node_id(self, node_type, node_index, create=False):
        return self.get_indexer(node_type, create).get_id(node_index)

    def get_node_ids(self, node_type, node_indices):
        return self.get_indexer(node_type, create=False).get_ids(node_indices)

    def has_node_id(self, node_type, node_id):
        return self.get_indexer(node_type, create=False).has_id(node_id)

    def add_edge(self, meta, node_index0, node_index1, weight=1.0):
        self.get_adj(meta, create=True).add_edge(node_index0, node_index1, weight=weight)

    def add_edges(self, meta, node_index0, node_index1, weight=1.0):
        self.get_adj(meta, create=True).add_edges(node_index0, node_index1, weight=weight)

    def num_nodes(self, node_type):
        return len(self.get_indexer(node_type, create=False))

    # if sparse, return a csr_matrix
    def adj_matrix(self, meta, sparse=False):
        csr = self.get_adj(meta, create=False).cached_csr
        return csr if sparse else csr.todense()

    def split_train_and_test(self, node_type, training_rate, func_should_mask=None):
        masked_node_indices = []
        num_nodes = self.num_nodes(node_type)
        if func_should_mask is not None:
            for node_index in range(num_nodes):
                if func_should_mask(node_index):
                    masked_node_indices.append(node_index)
            random_node_indices = np.random.permutation(masked_node_indices)
        else:
            random_node_indices = np.random.permutation(num_nodes)

        training_size = int(len(random_node_indices) * training_rate)
        train_node_indices = random_node_indices[:training_size]
        test_node_indices = random_node_indices[training_size:]

        train_masks = np.zeros([num_nodes], dtype=np.int32)
        train_masks[train_node_indices] = 1
        test_masks = np.zeros([num_nodes], dtype=np.int32)
        test_masks[test_node_indices] = 1
        return train_node_indices, test_node_indices, train_masks, test_masks

    def sample_node(self, node_type, excluded_node_indices=None):
        indexer = self.get_indexer(node_type, create=False)
        while True:
            node_index = random.randrange(0, len(indexer))
            if excluded_node_indices is None or node_index not in excluded_node_indices:
                return node_index

    def sample_meta_node(self, meta, excluded_node_indices=None):
        return self.get_adj(meta, create=False).sample_node(excluded_node_indices)

    def sample_meta_neighbor(self, meta, node_index):
        return self.get_adj(meta, create=False).sample_neighbor(node_index)

    def random_walk(self, node_types, start_node_index=None, padding=True):
        if start_node_index is None:
            start_node_index = self.sample_meta_node(tuple(node_types[:2]))
        path = [start_node_index]
        for i, node_type in enumerate(node_types[:-1]):
            meta = tuple(node_types[i:i+2])
            node_index0 = path[-1]
            node_index1 = self.sample_meta_neighbor(meta, node_index0)
            if node_index1 is None:
                break
            path.append(node_index1)

        while len(path) < len(node_types):
            node_type = node_types[len(path)]
            random_node_index = self.sample_node(node_type, excluded_node_indices=path)
            path.append(random_node_index)
        return path

    def multi_random_walk(self, node_types, start_node_indices=None, num_paths=None, num_threads=None):
        if (start_node_indices is None) == (num_paths is None):
            print("please specify either 'start_node_indices' or 'num_paths'")
        if start_node_indices is None:
            start_node_indices = [None] * num_paths
        if num_threads is None:
            num_paths = num_paths

        def random_walk_func(start_node_index):
            return self.random_walk(node_types, start_node_index)

        pool = ThreadPool(4)
        paths = pool.map(random_walk_func, start_node_indices)

        return paths

    def get_adj_shape(self, meta):
        return [self.num_nodes(meta[0]), self.num_nodes(meta[1])]


    def build_cache(self):
        for meta, adj in self.meta_adj_dict.items():
            adj.build_cache(shape=self.get_adj_shape(meta))

    def sample_triple(self, meta, node_a=None):
        return self.get_adj(meta, create=False).sample_triple(node_a)

    def sample_triples(self, meta, num):
        samples = []
        for i in range(num):
            samples.append(self.sample_triple(meta))
        return [list(t) for t in list(zip(*samples))]

