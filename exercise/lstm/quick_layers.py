import numpy as np
import tensorflow as tf
import random
import cv2
import os
import datetime
from tensorflow.python import debug as tf_debug

from tensorflow.contrib.layers.python.layers import batch_norm as batch_nm
from functools import reduce
slim = tf.contrib.slim
flags = tf.app.flags


def conv2d1(inputs, filters, name, keep_prod=0.5, is_train=True):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_initializer=tf.random_normal_initializer(), bias_initializer=tf.random_normal_initializer(),kernel_size=3, padding='same', activation=tf.nn.relu)
        # conv = tf.layers.batch_normalization(conv, training=is_train)
        conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        return tf.nn.dropout(conv, keep_prod)


def conv2d2(inputs, filters, name, prob=0.5, times=0 ,is_train=True):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=is_train)
        conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        return tf.nn.dropout(conv, (prob+times)/ (prob+times+1))


def conv2d3(inputs, filters, name, prob=0.5, is_train=True):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_initializer=tf.random_normal_initializer(mean=0.3, stddev=0.4), bias_initializer=tf.random_normal_initializer(mean=0.3, stddev=0.4), kernel_size=3, padding='same', activation=None)
        conv = tf.nn.dropout(conv, prob)
        conv = tf.layers.batch_normalization(conv, training=is_train,scale=False)
        conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        return tf.nn.relu(conv)


def conv2d0(inputs, filters, name, prob=0.5):
    conv = inputs
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        # weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.1)
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        padding="same",
                        # weights_regularizer=slim.l2_regularizer(0.0005)):
                        ):
        with tf.name_scope(name):
            conv = slim.repeat(conv, 2, slim.conv2d, filters, [3, 3])
            conv = slim.max_pool2d(inputs=conv, kernel_size=[2,2],padding="same")
            return tf.nn.dropout(conv, prob)


def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size,
                                             activation_fn=tf.nn.relu,
                                             scope=scope)

def dense_batch_relu(x, phase, scope, is_train=True):
    with tf.name_scope(scope):
        h1 = tf.layers.dense(inputs=x, units=512, use_bias=False, activation= None)
        # h2 = tf.contrib.layers.batch_norm(h1,
        #                                   center=True, scale=True,
        #                                   is_training=phase,
        #                                   scope='bn')
        h2 = tf.layers.batch_normalization(h1, training=is_train)
        return tf.nn.relu(h2, 'relu')