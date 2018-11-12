from keras.models import *
from keras.layers import *
from keras.optimizers import *
import os
import random
import cv2

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
# input_tensor = Input((height, width, 3))
# x = input_tensor
# for i in range(4):
#     x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
#     x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#
# x = Flatten()(x)
# x = Dropout(0.25)(x)
# x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
# model = Model(input=input_tensor, output=x)
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='adadelta',
#               metrics=['accuracy'])


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

def get_image_and_tensor(imgFilePath):
    for root, dirs, files in os.walk(imgFilePath):
        res = []
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                res.append((root + file, text2vec(file.split(".")[0])))
        return res

def get_next_batch(imageList=None,batch_size=256):
    #batch_x = np.zeros([batch_size, IMG_HEIGHT * IMG_WIDTH])
    #batch_y = np.zeros([batch_size, MAX_LABEL * LABEL_LEN])
    batch_x = np.zeros([batch_size, IMG_HEIGHT * IMG_WIDTH])
    batch_y = np.zeros([batch_size, MAX_LABEL * LABEL_LEN])

    randomList = random.sample(range(0, len(imageList)), batch_size)
    #if batch_size == 1:
    #    randomList = [0]
    for i, e in enumerate(randomList):
        imagePath, text = imageList[e]
        batch_x[i, :] = cv2.imread(imagePath, 0).flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text
    return np.reshape(batch_x,[batch_size, IMG_HEIGHT, IMG_WIDTH, 1]), batch_y


def generate_next_batch(imageList=None,batch_size=256):

    while True:
        batch_x = np.zeros([batch_size, IMG_HEIGHT * IMG_WIDTH])
        batch_y = np.zeros([batch_size, MAX_LABEL * LABEL_LEN])
        randomList = random.sample(range(0, len(imageList)), batch_size)
        for i, e in enumerate(randomList):
            imagePath, text = imageList[e]
            batch_x[i, :] = cv2.imread(imagePath, 0).flatten() / 255  # (image.flatten()-128)/128  mean为0
            batch_y[i, :] = text
            yield np.reshape(batch_x, [batch_size, IMG_HEIGHT, IMG_WIDTH, 1]), batch_y

trainList = get_image_and_tensor(train_path)
verfifyList = get_image_and_tensor(valid_path)


X_train, Y_train = get_next_batch(trainList,1)

X_test, Y_test = get_next_batch(trainList,1000)


batch_axis=-1

drop=0.5
model = Sequential()

# First convolutional layer with max pooling,取消激活
model.add(Conv2D(26, (3, 3), padding="valid", input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), activation=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#添加batch normal, 激活函数
model.add(Dropout(drop))
#model.add(BatchNormalization(axis=batch_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(BatchNormalization(axis=batch_axis))
model.add(Activation('relu'))

#model.add(Dropout(0.25))

# Second convolutional layer with max pooling
model.add(Conv2D(52, (3, 3), padding="same", activation=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(drop))
#model.add(BatchNormalization(axis=batch_axis, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(BatchNormalization(axis=batch_axis))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

model.add(Conv2D(104, (3, 3), padding="same", activation=None))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(drop))
model.add(BatchNormalization(axis=batch_axis))
model.add(Activation('relu'))
#model.add(Dropout(0.25))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(256, activation="relu", activity_regularizer=regularizers.l2(0.01)))

# Output layer with 32 nodes (one for each possible letter/number we predict)
model.add(Dense(104, activation="sigmoid", activity_regularizer=regularizers.l2(0.01)))


def mean_pred(y_true, y_pred):
    max_idx_p = K.argmax(K.reshape(y_pred, [-1, MAX_LABEL, LABEL_LEN]), 2)
    max_idx_l = K.argmax(K.reshape(y_true, [-1, MAX_LABEL, LABEL_LEN]), 2)
    return K.mean(K.cast(K.equal(max_idx_l, max_idx_p),dtype="float32" ))

# Ask Keras to build the TensorFlow model behind the scenes

def binary_crossentropy(y_true, y_pred):
    t = K.reshape(y_true, [-1, MAX_LABEL, LABEL_LEN])
    p = K.reshape(y_pred, [-1, MAX_LABEL, LABEL_LEN])
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=1)

RMS = RMSprop(lr=0.001, rho=0.9, epsilon=None)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=["accuracy", mean_pred])
#model.compile(loss=binary_crossentropy, optimizer='rmsprop', metrics=["accuracy", mean_pred])



# Train the neural network
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=200, epochs=5, verbose=1)

model.fit_generator(generate_next_batch(trainList,200),steps_per_epoch=200, epochs=20000,max_queue_size=1,validation_data=(X_test, Y_test),workers=1)


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
