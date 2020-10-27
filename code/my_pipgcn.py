""" Implementation of graph convolution operation, modified slightly from the implementation used in
    Protein Interface Prediction using Graph Convolutional Networks
    https://github.com/fouticus/pipgcn
    available under the MIT License, Copyright 2020 Alex Fout """
import numpy as np
import tensorflow as tf


def node_average_gc(inputs, adj_mtx, activation, filters=None, trainable=True):
    # node_average_gc_dist_thresh

    vertices = inputs  # shape: (batch_size, number_of_vertices, encoding_len)
    v_shape = vertices.get_shape()

    # create new weights # (v_dims, filters)
    center_weights = tf.Variable(initializer("he", (v_shape[-1].value, filters)), name="Wc", trainable=trainable)
    neighbor_weights = tf.Variable(initializer("he", (v_shape[-1].value, filters)), name="Wn", trainable=trainable)
    bias = tf.Variable(initializer("zero", (filters,)), name="b", trainable=trainable)

    # center signals are simply the center node value times the weight
    # shape: (batch_size, number_of_vertices, num_filters)
    center_signals = tf.reshape(tf.matmul(tf.reshape(vertices, (-1, v_shape[-1])),
                                          center_weights),
                                (-1, v_shape[1], filters))

    # apply neighbor weight to each neighbor
    # shape: (batch_size, number_of_vertices, num_filters)
    neighbor_signals_sep = tf.reshape(tf.matmul(tf.reshape(vertices, (-1, v_shape[-1])), neighbor_weights),
                                      (-1, v_shape[1], filters))

    # compute full neighbor signals
    neighbor_signals = tf.divide(tf.matmul(tf.tile(adj_mtx[None], (tf.shape(vertices)[0], 1, 1)),
                                           neighbor_signals_sep),
                                 tf.reshape(tf.maximum(tf.constant(1, dtype=tf.float32),
                                                       tf.reduce_sum(adj_mtx, axis=1)), (-1, 1)))

    # final output signal
    output_signal = activation(center_signals + neighbor_signals + bias)

    return output_signal


def initializer(init, shape):
    if init == "zero":
        return tf.zeros(shape)

    elif init == "he":
        fan_in = np.prod(shape)
        std = 1 / np.sqrt(fan_in)
        return tf.random_uniform(shape, minval=-std, maxval=std, dtype=tf.float32)


def main():
    pass


if __name__ == "__main__":
    main()
