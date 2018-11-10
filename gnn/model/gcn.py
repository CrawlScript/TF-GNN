# coding=utf-8

import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from gnn.util.evaluation import evaluate_accuracy


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, activation=tf.nn.relu, l2_coe=0.0,
                 trainable=True, name=None, dtype=None, **kwargs):
        super().__init__(trainable, name, dtype, **kwargs)

        self.activation = activation
        self.dense_func = tf.layers.Dense(num_units, activation=activation,
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_coe))

    def call(self, inputs, **kwargs):
        A, H = inputs
        sparse = isinstance(A, tf.SparseTensorValue)
        if sparse:
            AH = tf.sparse_tensor_dense_matmul(A, H)
        else:
            AH = tf.matmul(A, H)
        AHW = self.dense_func(AH)
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

    def call(self, inputs, training=None, mask=None):
        A, H = inputs
        for gcn_func in self.gcn_funcs:
            H = gcn_func([A, H])
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
            A = tf.SparseTensorValue(indices=np.stack((A.row, A.col), axis=1), values=A.data, dense_shape=A.shape)
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
              test_masks=None,
              steps=1000,
              learning_rate=1e-3,
              show_interval=20,
              eval_interval=20):

        if test_masks is None:
            test_masks = 1 - np.array(train_masks)

        A = GCN.gcn_kernal_tensor(adj, True)
        num_classes = self.model.num_units_list[-1]
        one_hot_labels = tf.one_hot(labels, num_classes)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        x = tf.get_variable("x", initializer=feature_matrix, trainable=False)
        for step in range(steps):
            with tf.GradientTape() as tape:
                logits = self.model([A, x])
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits,
                    labels=one_hot_labels
                )
                losses *= train_masks
            watched_vars = tape.watched_variables()
            grads = tape.gradient(losses, watched_vars)
            optimizer.apply_gradients(zip(grads, watched_vars))

            if step % show_interval == 0:
                print("step = {}\tloss = {}".format(step, tf.reduce_mean(losses)))

            if step % eval_interval == 0:
                preds = self.model([A, x])
                preds = tf.argmax(preds, axis=-1).numpy()
                accuracy = evaluate_accuracy(preds, labels, test_masks)
                print("step = {}\taccuracy = {}".format(step, accuracy))
