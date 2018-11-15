# coding=utf-8

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from scipy.sparse.base import spmatrix

from gnn.util.evaluation import evaluate


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, activation=tf.nn.relu,
                 trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)

        self.num_units = num_units
        self.activation = activation
        self.W = None
        self.b = None

    def build(self, input_shape):
        super().build(input_shape)
        input_dim = int(input_shape[1][1])
        self.W = self.add_weight("W", shape=[input_dim, self.num_units],initializer=tf.glorot_uniform_initializer)
        self.b = self.add_weight("b", shape=[self.num_units], initializer=tf.zeros_initializer)

    def l2_loss(self):
        return tf.nn.l2_loss(self.W)

    def call(self, inputs, **kwargs):
        A, H = inputs
        A_is_sparse = isinstance(A, tf.SparseTensor)
        H_is_sparse = isinstance(H, tf.SparseTensor)

        if H_is_sparse:
            HW = tf.sparse_tensor_dense_matmul(H, self.W) + self.b
        else:
            HW = tf.matmul(H, self.W) + self.b

        if A_is_sparse:
            AHW = tf.sparse_tensor_dense_matmul(A, HW)
        else:
            AHW = tf.matmul(A, HW)

        if self.activation is not None:
            AHW = self.activation(AHW)
        return AHW


class GCN(tf.keras.Model):
    def __init__(self, num_units_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_units_list = num_units_list
        self.gcn_funcs = []
        for i, num_units in enumerate(num_units_list):
            activation = tf.nn.relu if i < num_units - 1 else None
            gcn_func = GCNLayer(num_units, activation)
            setattr(self, "gcn_func{}".format(i), gcn_func)
            self.gcn_funcs.append(gcn_func)

    def l2_loss(self):
        return tf.add_n([gcn_func.l2_loss() for gcn_func in self.gcn_funcs])

    def call(self, inputs, drop_rate=None, training=None, mask=None):
        A, H = inputs
        for i, gcn_func in enumerate(self.gcn_funcs):
            H = gcn_func([A, H])
            if drop_rate is not None and i == len(self.gcn_funcs) - 2:
                H = tf.layers.dropout(H, rate=drop_rate)
        return H

    @classmethod
    def gcn_kernal(cls, adj):
        inv_D = np.array(adj.sum(axis=1)).flatten()
        inv_D = np.power(inv_D, -0.5)
        inv_D[np.isinf(inv_D)] = 0.0
        inv_D = sp.diags(inv_D)
        return inv_D.dot(adj).dot(inv_D) + sp.eye(inv_D.shape[0])

    @classmethod
    def gcn_kernal_tensor(cls, adj, sparse=True):
        adj = GCN.gcn_kernal(adj)
        if sparse:
            A = adj.tocoo().astype(np.float32)
            A = tf.SparseTensor(indices=np.stack((A.row, A.col), axis=1), values=A.data, dense_shape=A.shape)
        else:
            A = tf.get_variable("A", initializer=adj.todense().astype(np.float32), trainable=False)
        return A


class GCNTrainer(object):
    def __init__(self, gcn_model):
        self.model = gcn_model

    def train(self,
              adj,
              feature_matrix,
              labels,
              train_masks,
              test_masks,
              steps=1000,
              learning_rate=1e-3,
              l2_coe=1e-3,
              drop_rate=1e-3,
              show_interval=20,
              eval_interval=20):

        if test_masks is None:
            test_masks = 1 - np.array(train_masks)

        A = GCN.gcn_kernal_tensor(adj, sparse=True)
        num_classes = self.model.num_units_list[-1]
        one_hot_labels = tf.one_hot(labels, num_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate)

        if feature_matrix is None:
            feature_matrix = sp.diags(range(adj.shape[0]))

        if isinstance(feature_matrix, spmatrix):
            coo_feature_matrix = feature_matrix.tocoo().astype(np.float32)
            x = tf.SparseTensor(indices=np.stack((coo_feature_matrix.row, coo_feature_matrix.col), axis=1),
                                values=coo_feature_matrix.data, dense_shape=coo_feature_matrix.shape)
        else:
            x = tf.get_variable("x", initializer=feature_matrix, trainable=False)

        num_masked = tf.cast(tf.reduce_sum(train_masks), tf.float32)
        for step in range(steps):
            with tf.GradientTape() as tape:
                logits = self.model([A, x], drop_rate=drop_rate)
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=one_hot_labels
                )
                losses *= train_masks
                mean_loss = tf.reduce_sum(losses) / num_masked
                loss = mean_loss + self.model.l2_loss() * l2_coe

            watched_vars = tape.watched_variables()
            grads = tape.gradient(loss, watched_vars)
            optimizer.apply_gradients(zip(grads, watched_vars))

            if step % show_interval == 0:
                print("step = {}\tloss = {}".format(step, loss))

            if step % eval_interval == 0:
                preds = self.model([A, x])
                preds = tf.argmax(preds, axis=-1).numpy()
                accuracy, macro_f1, micro_f1 = evaluate(preds, labels, test_masks)
                print("step = {}\taccuracy = {}\tmacro_f1 = {}\tmicro_f1 = {}".format(step, accuracy, macro_f1, micro_f1))
