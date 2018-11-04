#coding:utf-8

import numpy as np
import tensorflow as tf
import random
import cv2
import os
import datetime
from tensorflow.python import debug as tf_debug


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
train_path = "D:\\t1\\"
valid_path = "D:\\t2\\"


# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len > MAX_LABEL:
        raise ValueError('验证码最长{0}个字符'.format(MAX_LABEL))
    vector = np.zeros(MAX_LABEL * LABEL_LEN)

    for i, c in enumerate(text):
        idx = i * LABEL_LEN + codeList.index(c)
        vector[idx] = 1
    return vector

# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        text.append(codeList[c-i*len(codeList)])
    return ''.join(text)

def get_next_batch(imageList=None,batch_size=256):
    batch_x = np.zeros([batch_size, IMG_HEIGHT * IMG_WIDTH])
    batch_y = np.zeros([batch_size, MAX_LABEL * LABEL_LEN])

    randomList = random.sample(range(0, len(imageList)), batch_size)
    #if batch_size == 1:
    #    randomList = [0]
    for i, e in enumerate(randomList):
        imagePath, text = imageList[e]
        batch_x[i, :] = cv2.imread(imagePath, 0).flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text

    return batch_x, batch_y

def get_image_and_tensor(imgFilePath):
    for root, dirs, files in os.walk(imgFilePath):
        res = []
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                res.append((root + file, text2vec(file.split(".")[0])))
        return res

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
       [1e-2, 1e-1, 0.75, 1e-1],
       [1e-2, 0.2, 0.75, 1e-1],
       [1, 1, 0.75, 1e-1]
       ]

# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1, keep=0.75, stddev = 0.01):
    # 将占位符 转换为 按照图片给的新样式
    x = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    total_out = []

    # 输入60*160
    # 3 conv layer
    w_c1 = tf.Variable(w_alpha*tf.random_normal([3, 3, 1, 32])) # 从正太分布输出随机值
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)
    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=stddev),is_train, scope='bn_1')
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

#    tf.summary.image('conv1', conv1, max_outputs=1)


    # 输入30*80
    w_c2 = tf.Variable(w_alpha*tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)
    conv2 = batch_norm(conv2, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=stddev),is_train, scope='bn_2')
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

#    tf.summary.image('conv2', conv2, max_outputs=1)

    # 输入15*40
    w_c3 = tf.Variable(w_alpha*tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)
    conv3 = batch_norm(conv3, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=stddev),is_train, scope='bn_3')
    conv3 = tf.nn.relu(conv3)
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

#    tf.summary.image('conv3', conv3, max_outputs=1)

    # w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    # b_c4 = tf.Variable(b_alpha * tf.random_normal([64]))
    # conv4 = tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4)
    # conv4 = batch_norm(conv4, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02),is_train, scope='bn_4')
    # conv4 = tf.nn.relu(conv4)
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # conv4 = tf.nn.dropout(conv4, keep_prob)

    # Fully connected layer
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
#out = tf.nn.softmax(out)
    return out


def vgg16(inputs):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    #print(net)
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    net = slim.max_pool2d(net, [2, 2], scope='pool5')

    net = slim.flatten(net, scope='flatten')

    net = slim.fully_connected(net, 4096, scope='fc6')
    #print(net)
    net = slim.dropout(net, scope='dropout6')
    net = slim.fully_connected(net, 4096, scope='fc7')
    net = slim.dropout(net, scope='dropout7', is_training=is_train)
    net = slim.fully_connected(net, 1000, activation_fn=None, scope='fc8')
    # net = tf.reshape(net, [-1, MAX_LABEL*LABEL_LEN])
    #net = slim.fully_connected(net, MAX_LABEL * LABEL_LEN, scope='fc8')
    #net = slim.dropout(net, 0.5, scope='dropout8')
    #net = tf.reshape(net, [-1, MAX_LABEL * LABEL_LEN])
    #slim.ba

  return net


#待验证，一定要添加此项
def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
	with tf.variable_scope(scope):
		#beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
		#gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=decay)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
	return normed


def train_vgg16():
    inputs = tf.reshape(X, shape=[-1, IMG_HEIGHT, IMG_WIDTH, 1])
    vgg_out = vgg16(inputs)

    #print("out",vgg_out)
    w = tf.Variable(initial_value=tf.random_normal(shape=[1000, LABEL_LEN * MAX_LABEL], dtype=tf.float32))
    #print("w", w)
    b = tf.Variable(initial_value=tf.zeros(shape=[LABEL_LEN * MAX_LABEL]))
    #print("b", b)
    #tf.summary.histogram("")
    fc = tf.matmul(vgg_out, w)
    fc = tf.nn.bias_add(fc, b)
    tf.summary.histogram("weight", w)
    tf.summary.histogram("bias", b)

    #dense = tf.reshape(vgg_out, [-1, MAX_LABEL * LABEL_LEN])
    #bias_init_val = tf.constant(0.0, shape=[MAX_LABEL * LABEL_LEN], dtype=tf.float32)
    #biases = tf.Variable(bias_init_val, trainable=True, name='b')
    #z = tf.nn.bias_add(dense, biases)
    #activation = tf.nn.relu(z, name="out")
    return fc



def getacc(out):
    y1 = tf.nn.softmax(out)
    y2 = tf.nn.softmax(out)
    y3 = tf.nn.softmax(out)
    y4 = tf.nn.softmax(out)



    # y1_1 = tf.clip_by_value(y1, 1e-4, tf.reduce_max(y1))
    # y2_1 = tf.clip_by_value(y2, 1e-4, tf.reduce_max(y2))
    # y3_1 = tf.clip_by_value(y3, 1e-4, tf.reduce_max(y3))
    # y4_1 = tf.clip_by_value(y4, 1e-4, tf.reduce_max(y4))

    label = tf.reshape(Y, [-1, MAX_LABEL, LABEL_LEN])

    y1_ = label[:,0,:]
    y2_ = label[:,1,:]
    y3_ = label[:,2,:]
    y4_ = label[:,3,:]

    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y1, labels=y1_))
    loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2, labels=y2_))
    loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y3, labels=y3_))
    loss4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y4, labels=y4_))


    # 定义5个损失
    # loss1 = tf.reduce_mean(-tf.reduce_sum(y1_ * tf.log(y1_1)))
    # loss2 = tf.reduce_mean(-tf.reduce_sum(y2_ * tf.log(y2_1)))
    # loss3 = tf.reduce_mean(-tf.reduce_sum(y3_ * tf.log(y3_1)))
    # loss4 = tf.reduce_mean(-tf.reduce_sum(y4_ * tf.log(y4_1)))
    # 取个平均损失
    loss = (loss1 + loss2 + loss3 + loss4 ) / 4

    train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # 定义各自的精确度
    correct_predict1 = tf.equal(tf.argmax(y1_, 1), tf.argmax(y1, 1))
    correct_predict2 = tf.equal(tf.argmax(y2_, 1), tf.argmax(y2, 1))
    correct_predict3 = tf.equal(tf.argmax(y3_, 1), tf.argmax(y3, 1))
    correct_predict4 = tf.equal(tf.argmax(y4_, 1), tf.argmax(y4, 1))


    auc1 = tf.reduce_mean(tf.cast(correct_predict1, dtype=tf.float32))
    auc2 = tf.reduce_mean(tf.cast(correct_predict2, dtype=tf.float32))
    auc3 = tf.reduce_mean(tf.cast(correct_predict3, dtype=tf.float32))
    auc4 = tf.reduce_mean(tf.cast(correct_predict4, dtype=tf.float32))
    # 取个平均精度
    auc = (auc1 + auc2 + auc3 + auc4 ) / 4
    return loss, auc, train


trainList = get_image_and_tensor(train_path)
verfifyList = get_image_and_tensor(valid_path)
# 训练


def getTime(starttime):
    return datetime.datetime.now() - starttime

def train_crack_captcha_cnn():
    #output = crack_captcha_cnn(1e-3, 1e-3)
    # total_loss = []
    # with tf.variable_scope(tf.get_variable_scope()):
    #     for i in range(0,2):
    #         with tf.device('/gpu:%d' % i):
    #             with tf.name_scope('%s_%s' % ('tower', i)):
    #                 output = crack_captcha_cnn(*vex[2])
    #                 with tf.variable_scope("loss"):
    #                     total_loss.append(output)
    #                     tf.get_variable_scope().reuse_variables()
    output = crack_captcha_cnn(*vex[6])
#    tf.summary.image('out', output, max_outputs=1)
    # loss = tf.concat(axis=0, values=total_loss)
    #output = train_vgg16()
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # loss = -tf.reduce_sum(Y * tf.log(output))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    #optimizer = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_LABEL, LABEL_LEN],name="predict")
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_LABEL, LABEL_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    #loss, accuracy, optimizer = getacc(output)

    tf.summary.scalar('loss', loss)
    # tf.summary.histogram('optimizer', optimizer)
    tf.summary.scalar('accuracy', accuracy)


    tf.add_to_collection("predict", predict)
    tf.add_to_collection("loss",loss)



    #交叉商，分别计算

    #train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    #correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        step = 0
        writer = tf.summary.FileWriter(".", sess.graph)
        starttime = datetime.datetime.now()


        with tf.device('/cpu:0'):
            while True:
                batch_x, batch_y = get_next_batch(trainList,200)
                #output = sess.run([ output], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5, is_train:True})
                #print("out", step, output)
                _, loss_ , summary = sess.run([optimizer, loss, merged], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75, is_train: True})
                #print("第{0}次，损失率{1}".format(step, loss_))

                writer.add_summary(summary, step)
                # 每100 step计算一次准确率
                # print(step % 200)
                if step % 10 == 0 :
                    batch_x_test, batch_y_test = get_next_batch(verfifyList,100)
                    acc, = sess.run([accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., is_train: False})
                    print("第{0}次，准确率{1}, 损失率{2}, 用时{3}".format(step, acc, loss_, getTime(starttime)))
                    tf.summary.scalar('accuracy', acc)
                    # 如果准确率大于50%,保存模型,完成训练
                    if acc > 0.95 or step > 2000:
                        saver.save(sess, "./crack_capcha.model", global_step=step)
                        break
                step += 1

train_crack_captcha_cnn()
