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

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# 读取图片和标签
def gen_captcha_text_and_image(imageFilePath, image_filename_list,imageAmount):
    num = random.randint(0, imageAmount - 1)
    img = cv2.imread(os.path.join(imageFilePath, image_filename_list[num]), 0)
    img = np.float32(img)
    text = image_filename_list[num].split('_')[0]
    return text, img