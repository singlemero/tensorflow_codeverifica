import tensorflow as tf
import numpy as np

const = tf.constant(2.0, name='const')
#python /Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages/tensorboard/main.py --logdir
# 创建TensorFlow变量b和c
#b = tf.Variable(2.0, name='b')

b = tf.placeholder(tf.float32, [None, 1], name="b")
c = tf.Variable(1.0, dtype=tf.float32, name='c')

d = tf.add(b, c, name='d')
e = tf.add(c, const, name='e')
a = tf.multiply(d, e, name='a')

init_op = tf.global_variables_initializer()

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())


#print(np.array([123,456,789]))


b = np.array([1,2,3,4])

x = np.ones([2,5])
y = x.reshape([10])
print(x.shape)
print(y.shape)

print(b[1:4])
#print(np.random.rand(10,10))

#print(np.random.normal(1.75, 0.1, (2, 3)))
#print(b.shape)
#print(b.ndim)
#print(b.dtype)

arr = np.random.normal(1.75, 0.1, (4, 5))
print(arr)
# 截取第1至2行的第2至3列(从第0行算起)
after_arr = arr[1:3, 2:4]
print(after_arr)

a1 = np.array([[1,2,3],[4,5,6]])
a2 = np.array([[1,4],[2,5],[3,6]])
print(a1.dot(a2))


with tf.Session() as sess:
    pass
    #2. 运行init operation
    #sess.run(init_op)
    # 计算
    #print(np.arange(0, 10)[:, np.newaxis])
    #a_out = sess.run(a, feed_dict={b: np.arange(0, 10)[:, np.newaxis]})
    #print("Variable a is {}".format(a_out))

test = np.array([[[1, 2, 3],[4,5,6]], [[2, 3, 4],[7,8,9]], [[5, 4, 3],[10,11,12]], [[8, 7, 2],[12,14,16]]])
print(test)
print(test.shape)
print(np.argmax(test, 0))
print(np.argmax(test, 1))
print(np.argmax(test, 2))
print("#######")
with tf.Session() as sess:
    b = tf.argmax(test, 0)
    c = tf.argmax(test, 1)
    d = tf.argmax(test, 2)

    tf.truncated_normal([1], stddev=0.1)
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))


