# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
import sys
import logging
import pylab
from matplotlib import pyplot as plt
import random
from PIL import Image

# verifica code
codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

IMG_WIDTH = 160
IMG_HIGHT = 60

LABEL_LEN = len(codeList)

MAX_LABEL = 4


x = tf.placeholder("float", shape=[None, IMG_WIDTH * IMG_HIGHT])
y_ = tf.placeholder("float", shape=[None, len(codeList)])


#img1 = cv2.imread("/Volumes/d/t1/AAAB1.jpg", cv2.IMREAD_GRAYSCALE)

src = cv2.imread("/Volumes/d/t1/AAAB.jpg")

img1 = cv2.imread("/Volumes/d/t1/AAAB.jpg", cv2.IMREAD_GRAYSCALE)
ax = plt.subplot(111)

# convert all to grayscale


# Denoise 3rd frame considering all the 5 frames
#dst = cv2.fastNlMeansDenoising(img1, h=10, templateWindowSize=10, searchWindowSize = 10)

kernel = np.ones((3, 3), np.float32) / 10
print(kernel)
# 卷积操作，-1表示通道数与原图相同
dst = cv2.filter2D(img1, -1, kernel)





plt.subplot(311),plt.imshow(src)
plt.subplot(312),plt.imshow(img1,'gray')
plt.subplot(313),plt.imshow(dst,'gray')

#cv2.imwrite('/Volumes/d/t1/AAAB1.jpg', img1)
#pim = np.array(dst)

pylab.show()


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