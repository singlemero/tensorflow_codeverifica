from imageBatch import *
from keras.callbacks import *

"""使用keras识别定长验证码，先卷积后使用GRU"""


OUTPUT_DIR = 'image_ocr'

#y_pred[批次量, 最大时序, 特征量]中，最大时序不可小于最大分类数量, 内部实际调用ctc.ctc_loss，传入函数的输入是y_pred,在LSTM时序中输入的max_times不可小于结果分类数量，so...
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def evaluate(model, batch_num=10):
    batch_acc = 0
    generator = train_gen.gen_batch(n_len, batch_num)

    # model.summary()
    for i in range(batch_num):
        __mydata = next(generator)[0]
        X_test, y_test = (__mydata["the_input"], __mydata["the_labels"])
        # [X_test, y_test, _, _], _ = next(generator)
        input_length = tf.to_int32(tf.squeeze(__mydata["input_length"], axis=1))
        y_pred = model.predict(__mydata)
        # print("predict:0", y_pred[0])
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=input_length)[0][0]
        out = K.get_value(ctc_decode)[:, :4]
        if out.shape[1] == 4:
            cur_acc = (y_test[:,:4] == out).sum(axis=1) == 4
            batch_acc += cur_acc.mean()
            for bol in cur_acc:
                if bol:
                    print("labels", y_test[:, :4])
                    print("pred", K.get_value(ctc_decode))

            # if (y_test[:,:4] == out).sum(axis=1) == 4:

    # if batch_acc / batch_num
    return batch_acc / batch_num

class Evaluate(Callback):

    def __init__(self, validation_func=None, inputs=None, outputs=None):
        self.accs = []
        # self.model = Model(inputs=inputs, outputs=outputs)
        self.outputs = outputs
        self.inputs = inputs
        self.validation_func = validation_func
        self.output_dir = os.path.join(OUTPUT_DIR, "keras_gru_first_four")
        # print(outputs)

    def on_epoch_end(self, epoch, logs=None):
        # print(self.outputs)
        if self.validation_func is not None:
            pass
        else:
            acc = evaluate(Model(inputs=self.inputs, outputs=self.outputs), batch_num=20) * 100
            self.accs.append(acc)
            if acc > 0.8:
                self.model.save_weights(
                    os.path.join(self.output_dir, 'epoch_%02d_%.2f.h5' % (epoch,acc)))
        print('acc: %f%%' % acc)






codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', ' ']
train_path = "/Volumes/d/t1/"
valid_path = "/Volumes/d/t2/"
width, height, n_len, n_class = 160, 60, 16, len(codeList)
rnn_size = 512
train_gen = ImageTensorBuilder(train_path, codeList, (height, width))
image_gen = ImageSequence(train_path,codeList, 200, 160, 60, 2)
base_model=None

dense_size = len(codeList) + 1

def train():

    input_tensor = Input(name='the_input', shape=(width, height, 1), dtype='float32')#Input((width, height, 1))
    x = input_tensor
    for i in range(2):
        x = Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(x)
        # x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
    conv_shape = x.get_shape()

    # conv_to_rnn_dims = (width // (2 ** 3),
    #                     (height // (2 ** 3)) * 32)

    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
    x = Dense(dense_size, activation='relu')(x)


    # (batch_size, 20, 8 )
    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = Add()([gru_1, gru_1b])#sum

    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                 kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = Concatenate()([gru_2, gru_2b])#concat
    x = Dropout(0.25)(x)
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)

    # Model(inputs=input_tensor, outputs=x).summary()
    base_model = Model(inputs=input_tensor, outputs=x)

    evaluator_func = K.function([input_tensor], [x])

    base_model.summary()
    evaluator = Evaluate(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])



    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])#.summary()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    # print(base_model == model)

    # model.fit_generator(train_gen.gen_batch(n_len, 200), steps_per_epoch=100, epochs=100, max_queue_size=1, workers=1, callbacks=[evaluator])
    model.fit_generator(image_gen, steps_per_epoch=100, epochs=100, callbacks=[evaluator],use_multiprocessing=True, workers=2)
    # model.fit_generator(image_gen, steps_per_epoch=2, epochs=100, callbacks=[evaluator])
train()