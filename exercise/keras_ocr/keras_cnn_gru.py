from imageBatch1 import *
from keras.callbacks import *
import itertools
"""使用keras识别定长验证码，先卷积后使用GRU"""
from keras.layers.core import K

OUTPUT_DIR = 'image_ocr'

SAVE_THRESHOLD = 10
MODEL_NAME = "keras_cnn_gru"

#y_pred[批次量, 最大时序, 特征量]中，最大时序不可小于最大分类数量, 内部实际调用ctc.ctc_loss，传入函数的输入是y_pred,在LSTM时序中输入的max_times不可小于结果分类数量，so...
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)



class Evaluate(Callback):

    def __init__(self, validation_func=None, val_seq=None, name = MODEL_NAME):
        self.validation_func = validation_func
        self.val_seq = val_seq
        self.output_dir = os.path.join(OUTPUT_DIR, name)
        # print(outputs)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs=None):
        # print(self.outputs)
        # K.set_learning_phase(0)
        if epoch >= 50:
            pass
        acc = self.evaluate(batch_num=10) * 100
        # K.set_learning_phase(1)
        # self.accs.append(acc)

        #if max(acc) > 0.8 or epoch >= SAVE_THRESHOLD:
        # self.model.summary()
        self.model.save_weights(os.path.join(self.output_dir, 'epoch_%02d.h5' % (epoch)))
        print(' acc: %f%%, ctc_acc: %f%%' % (acc[0], acc[1]))

    def evaluate(self, batch_num):
        batch_acc = 0
        o_acc = 0
        generator = self.val_seq.generator()
        #关闭学习率
        for i in range(batch_num):
            inputs = next(generator)
            x_test, y_test, source_str = (inputs["the_input"], inputs["the_labels"], inputs["source_str"])
            out = self.validation_func([x_test, 0])[0]
            current_acc = np.zeros([out.shape[0]])

            c_acc = np.zeros([out.shape[0]])
            #example one
            # ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
            ctc_decode  = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1], greedy=True)[0][0])
            # print(ctc_decode)
            for j in range(ctc_decode.shape[0]):
                print("ctc_decode",ctc_decode[j], y_test[j][:4])
                # out_best = list(np.argmax(decode_out[j, 2:], 1))
                out_best = list(ctc_decode[j])
                out_best = [k for k, g in itertools.groupby(out_best)]
                if self.val_seq.equals_after_trim(y_test[j], np.asarray(out_best)):
                    c_acc[j] = 1
                    print(source_str[j], y_test[j], out_best)
            o_acc += c_acc.mean()
            # print(" ctc_acc: %f%%" % (o_acc))

            for j in range(out.shape[0]):
                # 该层的输出结果是使用 max ,此处推断出最有可能的结果, 对每一列
                out_best = list(np.argmax(out[j, 2:], 1))
                out_best = [k for k, g in itertools.groupby(out_best)]
                if self.val_seq.equals_after_trim(y_test[j], np.asarray(out_best)):
                    current_acc[j] = 1
                    print(source_str[j], y_test[j], out_best)
            batch_acc += current_acc.mean()
        return batch_acc / batch_num, o_acc / batch_num


codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', ' ']
train_path = "/Volumes/d/t1/"
valid_path = "/Volumes/d/t2/"
width, height, n_len, n_class = 160, 60, 16, len(codeList)+1
rnn_size = 512
# train_gen = ImageTensorBuilder(train_path, codeList, (height, width))
image_gen = ImageSequence(train_path,codeList, 200, 160, 60, 3)
val_obj = ImageSequence(valid_path,codeList, 200, 160, 60, 3)
# base_model=None

dense_size = len(codeList) + 1

def train(epoch_num=None, name=MODEL_NAME):

    input_tensor = Input(name='the_input', shape=(width, height, 3), dtype='float32')#Input((width, height, 1))
    x = input_tensor
    for i in range(2):
        # x = Conv2D(filters=2 ** (3+i), kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(filters=16 * (i+1), kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(x)
        x = Conv2D(filters=16 * (i+1), kernel_size=(3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(x)
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
    gru_2 = TimeDistributed(BatchNormalization())(gru_2)
    gru_2b = TimeDistributed(BatchNormalization())(gru_2b)
    x = Concatenate()([gru_2, gru_2b])#concat

    # x = Dropout(0.25)(x)
    """
    最后结果是[batch_size, 最大时间序列, 分类总数+1位空白符+1位CTC校验位]，使用softmax函数，将所有结果的概率分布在（0，1）之间，激活用在每一帧时间序列上，求最大概率的分类，得出该帧的预测结果。
    因此，此处dense层设置 分类总数的数量为结果，并采用softmax多分类激活函数
    """
    x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)

    # Model(inputs=input_tensor, outputs=x).summary()
    # base_model = Model(inputs=input_tensor, outputs=x)
    # 评估回调函数
    evaluator_func = K.function([input_tensor, K.learning_phase()], [x])
    # evaluator_func.
    # base_model.summary()
    evaluator = Evaluate(validation_func=evaluator_func,val_seq=val_obj,name="keras_cnn_gru_add_batch")

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])#.summary()
    model.summary()
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    if epoch_num is not None:
        weight_file = os.path.join(
            OUTPUT_DIR,
            os.path.join(name, 'epoch_%02d.h5' % (epoch_num)))
        model.load_weights(weight_file)
    # print(base_model == model)

    # model.fit_generator(train_gen.gen_batch(n_len, 200), steps_per_epoch=100, epochs=100, max_queue_size=1, workers=1, callbacks=[evaluator])
    # model.fit_generator(image_gen.next_val(), steps_per_epoch=1, epochs=100, max_queue_size=1, workers=1, callbacks=[evaluator]) #单线程,易调试
    model.fit_generator(image_gen, steps_per_epoch=200, epochs=100, callbacks=[evaluator],use_multiprocessing=True, workers=2) #多线程
train()