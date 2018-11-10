# coding=utf-8
import numpy as np
import tensorflow as tf
from gat.model.module import GCN
from gat.util.evaluation import evaluate_accuracy


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
