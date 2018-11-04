import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import time

input_data = tf.Variable(np.random.rand(10, 15, 40, 1), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(3, 3, 1, 32), dtype=np.float32)
y = tf.nn.convolution(input_data, filter_data, strides=[1, 1], padding='SAME')

CNN1 = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

init_op = tf.global_variables_initializer()
print(1e-3)

for i in range(2):
    print(i)

print('1. tf.nn.convolution : ', y)

print('1. tf.nn.max_pool : ', CNN1)

st = time.clock()
time.sleep(3)
def getTime(starttime):
    return time.clock() - starttime

print(getTime(st))
#help(CNN1)




with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(CNN1)
    #print(result[0])


ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

print(len(ALPHABET))
print(1e-4)

s1 = np.array([[[1,2,3],[4,5,6],[7,8,9],[21,22,23]],
               [[10,11,12],[13,14,15],[16,17,18],[31,32,33]]])
print(s1)
print(s1.shape)
s2 = s1[:,0,:]
print(s2)
print(s2.shape)


vex = [[0.01,0.1,0.75,0.01],
       [1e-2, 1e-2, 0.5, 1e-2],
       [1e-2, 1e-1, 0.5, 1e-1]]
c1, c2,c3,c4 = vex[0]
print(c1, c2,c3,c4)