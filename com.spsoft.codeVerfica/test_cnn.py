# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time

#coding:utf-8

import numpy as np
import tensorflow as tf
import random
import cv2
import os
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
train_path = "/Volumes/d/t1/"
valid_path = "/Volumes/d/t2/"


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
X = tf.placeholder(tf.float32, [None, IMG_HEIGHT*IMG_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_LABEL*LABEL_LEN])
T = tf.placeholder(tf.bool, None)
keep_prob = tf.placeholder(tf.float32) # dropout






trainList = get_image_and_tensor(train_path)
verfifyList = get_image_and_tensor(valid_path)
# 训练



def predict_captcha(captcha_image):

    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('/Users/konglinghong/tensor/50/crack_capcha.model-3010.meta')
    r_t = 0
    r_f = 0
    with tf.Session() as sess:
        #saver.restore(sess, tf.train.latest_checkpoint('../c/'+str(MAX_CAPTCHA)+'/',))
        #new_saver = tf.train.import_meta_graph('/Users/konglinghong/tensor/50/crack_capcha.model-60.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/Users/konglinghong/tensor/50', ))

        graph = tf.get_default_graph()

        X = graph.get_tensor_by_name("X:0")
        # Y = graph.graph.get_tensor_by_name("Y:0")

        is_train = graph.get_tensor_by_name("is_train:0")
        keep_prob = graph.get_tensor_by_name("keep_prob:0")  # dropout

        predict = graph.get_tensor_by_name("predict:0")

        max_idx_p = tf.argmax(predict, 2)


        images , lables = get_next_batch(verfifyList,200)
        for index, e in enumerate(images):
            #print(e)
            predict_tensor = sess.run(max_idx_p, feed_dict={X: [e], keep_prob: 1, is_train: False})

            #max_idx_p = tf.argmax(predict_tensor, 2)
            v = np.reshape(predict_tensor, [MAX_LABEL])
            ze = np.zeros([MAX_LABEL, LABEL_LEN])
            for i, a in enumerate(v):
                ze[i][a] = 1
            # vector = np.zeros(MAX_LABEL * LABEL_LEN)
            result = vec2text(np.reshape(ze,[MAX_LABEL * LABEL_LEN]))
            captcha_text = vec2text(lables[index])
            #print(result, captcha_text)
            if captcha_text == result:
                r_t += 1
            else :
                r_f +=1
            print("预测值:{0}, 正确值:{1}, {2}, 正确率{3}".format(captcha_text, result, captcha_text==result, r_t/(r_t+r_f)))
            #print(str(captcha_text == result)+"  i: "+str(index)+" captcha_text: "+captcha_text+"  result: "+result+"   % :  "+str(r_t/(r_t+r_f)))
        return "end"


predict_text = predict_captcha("")
