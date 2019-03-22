
# import sys
# sys.path.append(r'/Volumes/d/code/codeverifica/exercise/lstm')
import numpy as np
import tensorflow as tf
import random
import cv2
import os
import datetime
from tensorflow.python import debug as tf_debug
from sklearn import preprocessing
import math
from imageBatch import *
from quick_layers import *
IMG_HEIGHT, IMG_WIDTH = (60, 160)

MAX_TIMES = 160

HIDDEN_SIZE = 256

slim = tf.contrib.slim
flags = tf.app.flags

codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
NCLASSES = len(codeList)+2

# 图像大小
IMG_HEIGHT = 60
IMG_WIDTH = 160
LABEL_LEN = len(codeList)

MAX_LABEL = 4
#print("验证码文本最长字符数", MAX_CAPTCHA)   # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐

# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
train_path = "/Volumes/d/t1/"
valid_path = "/Volumes/d/t2/"


def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0

    if len(original_list) != len(detected_list):
        # print(original_list, decoded_list)
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

def single_layer_dynamic_bi_lstm(input_x, n_steps, n_hidden):
    '''
    返回单层动态双向LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 正向
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden//2, forget_bias=1.0)
    # 反向
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden//2, forget_bias=1.0)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
    outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=input_x, dtype=tf.float32)

    print('hiddens:\n', type(outputs), len(outputs), outputs[0].shape, outputs[1].shape)
    # <class 'tuple'> 2 (?, 28, 128) (?, 28, 128)
    # 按axis=2合并 (?,28,128) (?,28,128)按最后一维合并(?,28,256)
    outputs = tf.concat(outputs, axis=2)
    # 使用WX+B ,转成结果
    lstm_out = tf.reshape(outputs, [-1, n_hidden])
    # init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.001, mode='FAN_AVG', uniform=False)
    # init_weights = tf.contrib.layers.xavier_initializer()
    init_weights = tf.truncated_normal_initializer(stddev=0.1)
    init_biases = tf.constant_initializer(0.0)

    # print(input_x.shape)
    shape = tf.shape(input_x)
    # batch_size = input_x.shape[0]
    print(shape)
    W = tf.get_variable("weights", [n_hidden, NCLASSES], initializer=init_weights, trainable=True)

        # self.make_var('weights', [num_hids, cfg.NCLASSES], init_weights, trainable, \
        #               regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
    b = tf.get_variable("biases", [NCLASSES], initializer=init_biases, trainable=True)
    # b = self.make_var('biases', [cfg.NCLASSES], init_biases, trainable)
    logits = tf.matmul(lstm_out, W) + b
    logits = tf.reshape(logits, [shape[0], -1, NCLASSES])
    logits = tf.transpose(logits, (1, 0, 2))

    print("[时序，批次，特征量]",logits.shape)
    # 注意这里输出需要转置  转换为时序优先的
    # outputs = tf.transpose(outputs, [1, 0, 2])

    return logits, state

# 定义CNN卷积 批处理
def crack_captcha_cnn(inputs, keep_prob, num):
    print("use crack_captcha_cnn1")
    # 将占位符 转换为 按照图片给的新样式
    #x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    net = tf.reshape(inputs, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])


    for i in range(num):
        # with tf.name_scope("conv{0}".format(i)):
        net = conv2d1(net, 32*(i+1), "conv{0}".format(i), keep_prob)
    # Fully connected layer
    net = tf.reshape(net, [-1, reduce(lambda a, b: a * b, net.shape.as_list()[1:])])
    dense_num = math.ceil(IMG_HEIGHT / 8) * math.ceil(IMG_WIDTH / 8)
    #only for forget
    net = dense(net, dense_num,"fc1")
    #dense = tf.reshape(conv3, [-1, 8*20*64])
    #dense = tf.layers.dense(inputs=dense, units=1024, use_bias=False, activation= tf.nn.relu)
    #dense = tf.nn.dropout(dense, keep)
    #fc = dense(net, 2*10*64, "fc1")
    # fc = dense_batch_relu(fc, is_train, "fc2")

    print("input",inputs.shape)
    #out = tf.layers.dense(inputs=fc, units=MAX_LABEL*LABEL_LEN, use_bias=False)
    return net, dense_num



X = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH],name="X")
Y = tf.placeholder(tf.float32, [None, MAX_LABEL*LABEL_LEN], name="Y")
is_train = tf.placeholder(tf.bool, None, name="is_train")
keep_prob = tf.placeholder(tf.float32, name="keep_prob") # dropout


def train_lstm():
    img_obj = ImageTensorBuilder(train_path, codeList, (IMG_WIDTH, IMG_HEIGHT))
    valid_obj = ImageTensorBuilder(valid_path, codeList, (IMG_WIDTH, IMG_HEIGHT))

    batch_size = 100
    hidden_num = 256
    #batch 信息


    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32, name="targets")
    # 定义seq_len
    sequence_length = tf.placeholder(tf.int32, [None])




    max_time = math.ceil(IMG_WIDTH / 8)
    num_feature = math.ceil(IMG_HEIGHT / 8)
    #cnn
    cnn_out, dense_num = crack_captcha_cnn(X,0.5,3)
    print("fdsafadsf",cnn_out.shape[0])
    #TODO , 全连接转 【batch_size, image_height, image_width】=> [batch_size, max_time, -1]
    # num_feature, max_time = math.ceil(IMG_HEIGHT / 8) , math.ceil(IMG_WIDTH / 8)
    cnn_out = tf.reshape(cnn_out, shape=[-1, num_feature , max_time])
    cnn_out = tf.transpose(cnn_out, [0, 2, 1])
    #  全连接转 【batch_size, max_time, -1】
    # cnn_out = tf.reshape(cnn_out, shape=[batch_size, max_time, num_feature])
    lstm_outputs, bw_state = single_layer_dynamic_bi_lstm(cnn_out, max_time, hidden_num)

    #loss
    loss = tf.nn.ctc_loss(labels=targets, inputs=lstm_outputs, sequence_length=sequence_length)
    cost = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(lstm_outputs, sequence_length, merge_repeated=True)

    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))


    #output = tf.contrib.layers.fully_connected(inputs=lstm_outputs[-1], num_outputs=n_classes, activation_fn=tf.nn.sigmoid)
    init = tf.global_variables_initializer()
    #loss = tf.nn.ctc_loss(logits, labels, seq_len)



    def do_report(batch_size):
        test_inputs, test_labels, test_seq_len = img_obj.next_sparse_batch(batch_size)#get_next_batch(BATCH_SIZE)
        test_inputs = np.reshape(test_inputs, (batch_size, IMG_HEIGHT*IMG_WIDTH))
        test_seq_len = np.ones((batch_size)) * 20
        test_feed = {X: test_inputs,
                     targets: test_labels,
                     sequence_length: test_seq_len}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        report_accuracy(dd, test_labels)

    with tf.Session() as session:
        session.run(init)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for i in range(1000):
            inputs, labels, seq_len = img_obj.next_sparse_batch(batch_size)
            seq_len = np.ones((batch_size)) * 20
            _out, _state, _loss, _acc, _cost = session.run([lstm_outputs,bw_state, loss, acc, cost],
                                                    feed_dict={X: np.reshape(inputs, (batch_size, IMG_HEIGHT*IMG_WIDTH)),
                                                               sequence_length:seq_len,
                                                               targets:labels})
            print("step={}, loss={}, acc={}".format(i, _cost, _acc))
            if i % 10 == 0:
                # print("-----------------")
                do_report(100)

train_lstm()