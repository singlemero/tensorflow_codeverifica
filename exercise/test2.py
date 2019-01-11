import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
import time
import matplotlib.pyplot as plt
import cv2
input_data = tf.Variable(np.random.rand(10, 15, 40, 1), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(3, 3, 1, 32), dtype=np.float32)
y = tf.nn.convolution(input_data, filter_data, strides=[1, 1], padding='SAME')

CNN1 = tf.nn.max_pool(y, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#print(1e-3)

for i in range(2):
    pass
    #print(i)

print('1. tf.nn.convolution : ', y)

print('1. tf.nn.max_pool : ', CNN1)

st = time.clock()
time.sleep(3)
def getTime(starttime):
    return time.clock() - starttime

#print(getTime(st))
#help(CNN1)

tf.random_normal([3, 3, 1, 32])


#w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.05, mean=1))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    result = sess.run(CNN1)
    print(w1)
    #print("随机正太分布:", sess.run(w1))
    print(len(result.shape)-1)


ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']

#print(len(ALPHABET))
#print(1e-4)


# 切片
s1 = np.array([[[1,2,3],[4,5,6],[7,8,9],[21,22,23]],
               [[10,11,12],[13,14,15],[16,17,18],[31,32,33]]])
#print(s1) 3,2,4
print(s1.shape)
s2 = s1[:,:,0]

# 手动切片
st = np.zeros([3,2,4])

st[0] = s1[:,:,0]
st[1] = s1[:,:,1]
st[2] = s1[:,:,2]
print(st)
print(st.shape)
# 使用tensorflow切片
t = tf.constant(s1)
#transopose,要转化成的纬度目标纬度数量= 原始数量[纬度下标1， 维度下标2....]
#例如(3,2,4)维度需要转换成(4,3,2)维度, tf.transpose(t,[2,1,0]) 2,1,0对应（3，2，4）的下标，即转成(4,3,2)
#对于tensorflow卷积查看图片有用，卷积结果是(batchsize, height, weight, feature), feature数量决定一张图转出多少个特征
#但是对于tensorboard显示图片时需要的tensor为(feature, height,weight, channel), 此处的feature为卷积的结果,所以使用transopose转换
z1 = tf.transpose(t,[2,1,0])
with tf.Session() as sess:
    print(t.shape)
    a = sess.run(z1)
    print(a.shape)
    print(a)


#print(np.reshape(s1,[3,2,4]))
vex = [[0.01,0.1,0.75,0.01],
       [1e-2, 1e-2, 0.5, 1e-2],
       [1e-2, 1e-1, 0.5, 1e-1]]
c1, c2,c3,c4 = vex[0]
#print(c1, c2,c3,c4)


t = tf.constant([[[1, 1, 1, 1], [2, 2, 2, 2], [7, 7, 7, 7]],
                 [[3, 3, 3, 3], [4, 4, 4, 4], [8, 8, 8, 8]]])



z1 = tf.strided_slice(t, [0,0,0], [2,3,4])
z2 = tf.strided_slice(t, [1, 0], [-1, 1], [1, 1])
z3 = tf.strided_slice(t, [1, 0, 1], [-1, 2, 3], [1, 1, 1])

z1 = tf.transpose(t,[2,1,0])


reader = tf.WholeFileReader()

key, value = reader.read(tf.train.string_input_producer(["/Volumes/d/t2/SJIS.jpg"]))

image0 = tf.image.decode_jpeg(value)
gc = tf.image.rgb_to_grayscale(image0)
with tf.Session() as sess:
    #image_raw_data_jpg = tf.gfile.FastGFile("/Volumes/d/t2/SJIS.jpg", 'r').read()
    #image_raw_data_jpg = cv2.imread("/Volumes/d/t2/SJIS.jpg")
    #image_data = tf.image.decode_jpeg(image_raw_data_jpg)
    #image_data = sess.run(tf.image.rgb_to_grayscale(image_data))
    #print(image_data.shape)
    #plt.imshow(image_data[:, :, 0], cmap='gray')
    #plt.show()
    #print(sess.run(value))
    # print(sess.run(gc))
    plt.imshow(sess.run(gc))
    plt.show()
    # print(sess.run(z3))
    # print(tf.transpose())


