#coding:utf-8

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

codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']


# 图像大小
IMG_HEIGHT = 60
IMG_WIDTH = 160
LABEL_LEN = len(codeList)

MAX_LABEL = 4
#print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
train_path = "/Volumes/d/t1/"
valid_path = "/Volumes/d/t2/"



####################################################################
# 申请占位符 按照图片
X = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH],name="X")
Y = tf.placeholder(tf.float32, [None, MAX_LABEL*LABEL_LEN], name="Y")
is_train = tf.placeholder(tf.bool, None, name="is_train")
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout

#w 权重,b 偏置量, keep_prob, stddev=学习率
vex = [[0.01,0.1,0.75,0.01],
       [1e-2, 1e-2, 0.5, 1e-2],
       [1e-2, 1e-1, 0.5, 1e-1],
       [1e-4, 1e-3, 0.75, 1e-1],
       [1e-2, 1e-1, 0.5, 1e-3],
       [1e-2, 0.2, 0.75, 1e-1],
       [1, 1, 0.75, 1e-1]
       ]



def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name):
    #w = tf.Variable(tf.zeros([5, 5, size_in, size_out]), name="W")
    #b = tf.Variable(tf.zeros([size_out]), name="B")
    w = tf.Variable(tf.truncated_normal([4, 4, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    act = tf.nn.relu(conv + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# Add fully connected layer
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)

    return act


def cnn1():
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    conv1 = conv_layer(x, 1, 32, "conv1")
    conv2 = conv_layer(conv1, 32, 64, "conv2")
    conv3 = conv_layer(conv2, 64, 64, "conv3")

def vgg16(inputs, feature):
  batch_norm_params = {
      "is_training": is_train,
      'zero_debias_moving_mean': True,
      # 'decay': batch_norm_decay,
      'updates_collections': None,
  }

  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      # weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params):
    net = inputs
    for i in range(3)[1:]:
        with tf.name_scope("conv{0}".format(i)):
            # net = slim.repeat(net, 2, slim.conv2d, 32*i, [3, 3])
            net = slim.conv2d(net , 32 * i, [3, 3])
            net = slim.batch_norm(net)
            net = slim.max_pool2d(net, [2, 2])
            net = slim.dropout(net, i/(i+1))

    net = slim.flatten(net, scope='flatten')

    net = slim.fully_connected(net, 256, scope='fc6')
    #print(net)
    net = slim.dropout(net, scope='dropout6')
    net = slim.fully_connected(net, feature, activation_fn=None, scope='fc8')
  return net

def train_vgg16():
    inputs = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    feature = 104
    vgg_out = vgg16(inputs, feature)

    #print("out",vgg_out)
    #w = tf.Variable(initial_value=tf.truncated_normal(shape=[feature, LABEL_LEN * MAX_LABEL], stddev=0.1,  dtype=tf.float32))
    # w = slim.variable('weights',
    #                         shape=[feature, LABEL_LEN * MAX_LABEL],
    #                         initializer=tf.truncated_normal_initializer(stddev=0.1),
    #                         # regularizer=slim.l2_regularizer(0.05),
    #                         device='/CPU:0')
    # #print("w", w)
    # b = slim.variable('bias',
    #                   shape=[LABEL_LEN * MAX_LABEL],
    #                   initializer=tf.truncated_normal_initializer(stddev=0.01),
    #                   # regularizer=slim.l2_regularizer(0.05),
    #                   device='/CPU:0')
    # # b = tf.Variable(initial_value=tf.truncated_normal(shape=[LABEL_LEN * MAX_LABEL], stddev=0.1))
    # #print("b", b)
    # #tf.summary.histogram("")
    # fc = tf.matmul(vgg_out, w)
    # fc = tf.nn.bias_add(fc, b)
    # tf.summary.histogram("weight", w)
    # tf.summary.histogram("bias", b)

    #dense = tf.reshape(vgg_out, [-1, MAX_LABEL * LABEL_LEN])
    #bias_init_val = tf.constant(0.0, shape=[MAX_LABEL * LABEL_LEN], dtype=tf.float32)
    #biases = tf.Variable(bias_init_val, trainable=True, name='b')
    #z = tf.nn.bias_add(dense, biases)
    #activation = tf.nn.relu(z, name="out")
    return vgg_out


def conv2d1(inputs, filters, name, prob=0.5):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_initializer=tf.random_normal_initializer(mean=0.3, stddev=0.4), bias_initializer=tf.random_normal_initializer(mean=0.3, stddev=0.4),kernel_size=3, padding='same', activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=is_train)
        conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        return tf.nn.dropout(conv, prob)

def conv2d2(inputs, filters, name, prob=0.5,times=0):
    with tf.name_scope(name):
        conv = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu)
        conv = tf.layers.batch_normalization(conv, training=is_train)
        conv = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=[2, 2], padding="same")
        return tf.nn.dropout(conv, (prob+times)/ (prob+times+1))

def conv2d3(inputs, filters, name, prob=0.5):
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

def dense_batch_relu(x, phase, scope):
    with tf.name_scope(scope):
        h1 = tf.layers.dense(inputs=x, units=512, use_bias=False, activation= None)
        # h2 = tf.contrib.layers.batch_norm(h1,
        #                                   center=True, scale=True,
        #                                   is_training=phase,
        #                                   scope='bn')
        h2 = tf.layers.batch_normalization(h1, training=is_train)
        return tf.nn.relu(h2, 'relu')


def crack_captcha_cnn0(w_alpha=0.01, b_alpha=0.1, keep=0.5, stddev = 0.005):
    print("use crack_captcha_cnn0")
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    total_out = []
    net = x
    # 输入60*160
    for i in range(3):
        # with tf.name_scope("conv{0}".format(i)):
        net = conv2d0(net, 32*(i+1), "conv{0}".format(i), keep_prob)
    # slim.flatten
    net = slim.flatten(net, scope='flatten')
    # net = tf.reshape(net, [-1, reduce(lambda a, b: a * b, net.shape.as_list()[1:])], name="flatten")
    #dense = tf.reshape(conv3, [-1, 8*20*64])
    #dense = tf.layers.dense(inputs=dense, units=1024, use_bias=False, activation= tf.nn.relu)
    #dense = tf.nn.dropout(dense, keep)
    # net = dense(net, 1024, "dense1")

    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.relu, scope='fc1')
    # fc = dense_batch_relu(fc, is_train, "fc2")
    net = tf.nn.dropout(net, keep_prob)

    # out = tf.layers.dense(inputs=net, units=MAX_LABEL*LABEL_LEN, name="dense2")

    out = slim.fully_connected(net, MAX_LABEL*LABEL_LEN, activation_fn=tf.nn.relu, scope='fc2')

    return out

# 定义CNN卷积 批处理
def crack_captcha_cnn1(w_alpha=0.01, b_alpha=0.1, keep=0.5, stddev = 0.005):
    print("use crack_captcha_cnn1")
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    total_out = []
    net = x


    for i in range(3):
        # with tf.name_scope("conv{0}".format(i)):
        net = conv2d1(net, 32*(i+1), "conv{0}".format(i), keep_prob)
    #conv3.shape.as_list()
    # Fully connected layer
    net = tf.reshape(net, [-1, reduce(lambda a, b: a * b, net.shape.as_list()[1:])])
    #dense = tf.reshape(conv3, [-1, 8*20*64])
    #dense = tf.layers.dense(inputs=dense, units=1024, use_bias=False, activation= tf.nn.relu)
    #dense = tf.nn.dropout(dense, keep)
    fc = dense(net, 2*10*64, "fc1")
    # fc = dense_batch_relu(fc, is_train, "fc2")


    out = tf.layers.dense(inputs=fc, units=MAX_LABEL*LABEL_LEN, use_bias=False)
    return out


def crack_captcha_cnn2(w_alpha=0.01, b_alpha=0.1, keep=0.5, stddev = 0.005):
    print("use crack_captcha_cnn2")
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    total_out = []

    # 输入60*160
    conv1 = conv2d2(x, 32, "conv1", keep_prob,0)

    # 输入30*80
    conv2 = conv2d2(conv1, 64, "conv2", keep_prob,1)

    # 输入15*40
    conv3 = conv2d2(conv2, 64, "conv3", keep_prob,2)
    conv3.shape.as_list()
    # Fully connected layer
    dense = tf.reshape(conv3, [-1, reduce(lambda a, b: a * b, conv3.shape.as_list()[1:])])
    # dense = tf.reshape(conv3, [-1, 8*20*64])
    dense = tf.layers.dense(inputs=dense, units=512,  activation=tf.nn.relu)
    # dense = tf.layers.batch_normalization(dense, training=is_train)
    dense = tf.nn.dropout(dense, keep_prob)

    out = tf.layers.dense(inputs=dense, units=MAX_LABEL * LABEL_LEN)
    return out


def crack_captcha_cnn3(w_alpha=0.01, b_alpha=0.1, keep=0.5, stddev = 0.005):
    print("use crack_captcha_cnn3")
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    total_out = []

    # 输入60*160
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32])) # 从正太分布输出随机值,乘以一个固定值，使每个参数值结果在[0,1]之间
    b_c1 = tf.constant(0, dtype=tf.float32, shape=[32])#tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    # conv1 = tf.nn.relu(conv1)
    #print(conv1)
    # stddev 标准差=方差开方, mean = 均值
    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=stddev),is_train, scope='bn_1')
    conv1 = tf.layers.batch_normalization(conv1, training=is_train)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        #print(conv1)
        #print(tf.expand_dims(tf.transpose(tf.reshape(conv1,[30, 80, 32]), [2,0,1]),3))
    #tf.summary.image("img1", tf.expand_dims(tf.transpose(tf.reshape(conv1,[30, 80, 32]), [2,0,1]),3), 32)
    conv1 = tf.nn.dropout(conv1, keep_prob)

#    tf.summary.image('conv1', conv1, max_outputs=1)

    # 输入30*80
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.constant(0, dtype=tf.float32, shape=[64])#tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=stddev),is_train, scope='bn_2')

    # conv2 = tf.nn.relu(conv2)
    conv2 = tf.layers.batch_normalization(conv2, training=is_train)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #tf.summary.image("img2", tf.expand_dims(tf.transpose(tf.reshape(conv2,[15, 40, 64]), [2,0,1]),3), 64)
    conv2 = tf.nn.dropout(conv2, keep_prob)

#    tf.summary.image('conv2', conv2, max_outputs=1)

    # 输入15*40
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.constant(0, dtype=tf.float32, shape=[64])#tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)

    # conv3 = tf.nn.relu(conv3)
    conv3 = tf.layers.batch_normalization(conv3, training=is_train)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=stddev),is_train, scope='bn_3')
    #tf.summary.image("img3", tf.expand_dims(tf.transpose(tf.reshape(conv3,[8, 20, 64]), [2,0,1]),3), 64)
    conv3 = tf.nn.dropout(conv3, keep_prob)


    # Fully connected layer
    dense = tf.reshape(conv3, [-1, 8*20*64])
    # dense = tf.layers.dense(inputs=dense, units=1024, use_bias=False, activation= tf.nn.relu)
    w_d = tf.Variable(w_alpha*tf.random_normal([8*20*64, 1024]))
    b_d = tf.Variable(b_alpha*tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024, MAX_LABEL*LABEL_LEN]))
    b_out = tf.Variable(b_alpha*tf.random_normal([MAX_LABEL*LABEL_LEN]))

    tf.summary.histogram("weight", w_out)
    tf.summary.histogram("bias", b_out)

    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.layers.dense(inputs=dense, units=MAX_LABEL*LABEL_LEN, use_bias=False)
    return out

#待验证，一定要添加此项
def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
	with tf.variable_scope(scope):
		#beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
		#gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
        #对输入的[0,1,2]纬度求均值，标准差，并在后续不断更新这两个值
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        #滑动平均，
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)
        #当phase_train 为True时 返回mean_var_with_update, Flase时，返回lambda表达式内容
		mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
	return normed


trainList = get_image_and_tensor(train_path)
verfifyList = get_image_and_tensor(valid_path)
# 训练


def getTime(starttime):
    return datetime.datetime.now() - starttime

def train_crack_captcha_cnn():

    # output = crack_captcha_cnn(*vex[4])
    # output = crack_captcha_cnn3(*vex[0])
    output = crack_captcha_cnn0(*vex[4])
    # output = train_vgg16()

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    #optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
    optimizer = tf.train.AdamOptimizer(1e-2)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=1e-2,
                                               global_step=global_step,
                                               decay_steps=10,
                                               decay_rate=0.95,
                                               staircase=True,
                                               # If `True` decay the learning rate at discrete intervals
                                               # staircase = False,change learning rate at every step
                                               )


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        #train_op = optimizer.minimize(loss)
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    predict = tf.reshape(output, [-1, MAX_LABEL, LABEL_LEN],name="predict")
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_LABEL, LABEL_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)


    tf.add_to_collection("predict", predict)
    tf.add_to_collection("loss",loss)

    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        step = 0
        writer = tf.summary.FileWriter("./logs", sess.graph)
        starttime = datetime.datetime.now()



        with tf.device('/cpu:0'):
            while True:
                batch_x, batch_y = get_next_batch(trainList,100)
                _, loss_ , summary, acc = sess.run([train_op, loss, merged,accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5, is_train: True})
                #print(oo.flatten())
                writer.add_summary(summary, step)
                if step % 10 == 0 :

                    #slim.model_analyzer.analyze_ops(tf.get_default_graph(), print_info=True)
                    variables = tf.model_variables()
                    #slim.model_analyzer.analyze_vars(variables, print_info=False)
                    batch_x_test, batch_y_test = get_next_batch(verfifyList,50)
                    test_accuracy = accuracy.eval(feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0, is_train: False})
                    # acc, = sess.run([accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., is_train: False})
                    print("第{0}次，准确率{1}, 损失率{2}, 用时{3}, 训练准确率{4}".format(step, test_accuracy, loss_, getTime(starttime), acc))
                    tf.summary.scalar('accuracy_run', acc)
                    tf.summary.scalar('test_accuracy', test_accuracy)
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.95 or step >= 3200:
                        #saver = tf.train.Saver()
                        var_list = tf.trainable_variables()
                        g_list = tf.global_variables()
                        for i in var_list:
                            print("name", i.name)
                        for i in g_list:
                            print("name", i.name)
                        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
                        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
                        var_list += bn_moving_vars
                        # saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
                        saver = tf.train.Saver()
                        saver.save(sess, "/Users/konglinghong/tensor/50/crack_capcha.model", global_step=step)
                        break
                step += 1

train_crack_captcha_cnn()
