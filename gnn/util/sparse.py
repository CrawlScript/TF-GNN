# coding=utf-8
import numpy as np
import tensorflow as tf
import scipy.sparse as sp


def dense_to_sparse_tensor(dense_matrix, dtype=np.float32):
    coo_matrix = sp.coo_matrix(dense_matrix, dtype=dtype)
    return coo_to_sparse_tensor(coo_matrix)


def csr_to_sparse_tensor(csr_matrix, dtype=np.float32):
    coo_matrix = csr_matrix.tocoo().astype(dtype)
    return coo_to_sparse_tensor(coo_matrix)


def coo_to_sparse_tensor(coo_matrix):
    t = tf.SparseTensor(indices=np.stack((coo_matrix.row, coo_matrix.col), axis=1), values=coo_matrix.data, dense_shape=coo_matrix.shape)
    return t