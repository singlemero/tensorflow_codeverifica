from keras.models import *
from keras.layers import *
import os
import random
import cv2
import tensorflow as tf
from keras.utils import Sequence
import math

class ImageSequence(Sequence):
    #
    def __init__(self, dir_path, codes, batch_size,
                 img_w, img_h, downsample_factor,
                 absolute_max_string_len=16):
        """
        :param dir_path: 文件路径
        :param codes:    结果编码集
        :param batch_size: 批次大小
        :param img_w: 图片宽度
        :param img_h: 图片高度
        :param downsample_factor: 卷积次数
        :param absolute_max_string_len: label长度
        """

        self.codes = codes
        self.res = self.__get_image_path_and_name(dir_path)
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.downsample_factor = downsample_factor
        self.absolute_max_string_len = absolute_max_string_len

        # 添加空白位
        if codes[-1] != " ":
            self.codes.append(" ")
        self.blank = len(self.codes)
        pass


    # 名字转向量
    def __text2index(self, file_name):
        file_length = len(file_name)
        #长度是文件字符长度
        vector = np.zeros(file_length)
        #返回每个字符对应的下标
        for i, c in enumerate(file_name):
            vector[i] = self.codes.index(c)
        return vector

    # 获取当前目录下的文件的文件名
    def __get_image_path_and_name(self, img_dir_path):
        for root, dirs, files in os.walk(img_dir_path):
            file_paths = []
            file_names = []
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    # 目录，文件名
                    file_paths.append(root + file)
                    file_names.append(file.split(".")[0])
            return file_paths, file_names

    def __len__(self):
        print("but")
        return int(np.ceil(len(self.res[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_files = self.res[0]
        # batch_x = np.zeros([self.batch_size, self.img_w, self.img_h, 1])
        batch_x = np.zeros([self.batch_size, self.img_w, self.img_h, 3])
        batch_y = np.ones([self.batch_size, self.absolute_max_string_len]) * -1
        input_length = np.ones([self.batch_size, 1]) * math.ceil(self.img_w / 2**self.downsample_factor - 2)
        label_length = np.zeros([self.batch_size, 1])
        source_str = []

        random_list = random.sample(range(0, len(image_files)), self.batch_size)

        for i, e in enumerate(random_list):
            path = image_files[e]
            text = self.res[1][e]
            text_len = len(text)
            #添加一维度通道
            # batch_x[i, :] = np.expand_dims(np.transpose(cv2.imread(path, 0) / 255),axis=2)
            batch_x[i, :] = np.transpose(cv2.imread(path) / 255, [1,0,2])
            batch_y[i, :text_len] = self.__text2index(text)
            label_length[i,:] = text_len
            source_str.append(text)
            # input_length[:]
        inputs = {'the_input': batch_x,
                  'the_labels': batch_y,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        #不做任何处理
        outputs = {'ctc': np.zeros([self.batch_size])}
        return inputs, outputs

class ImageTensorBuilder:

    # 获取当前目录下的文件的文件名
    def __get_image_path_and_name(self, imgFilePath):
        for root, dirs, files in os.walk(imgFilePath):
            # res = []
            file_paths = []
            file_names = []
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    # 目录，文件名
                    file_paths.append(root + file)
                    file_names.append(file.split(".")[0])
                    # res.append((root + file, file.split(".")[0]))
            return file_paths, file_names
    #名字转向量
    def __text2vector(self, file_name):
        file_length = len(file_name)
        vector = np.zeros(file_length * len(self.labels))
        for i, c in enumerate(file_name):
            idx = i * file_length + self.labels.index(c)
            vector[idx] = 1
        return vector

    def __text2index(self, file_name):
        file_length = len(file_name)
        vector = np.zeros(file_length)
        for i, c in enumerate(file_name):
            # idx = i * file_length + self.labels.index(c)
            vector[i] = self.labels.index(c)
        return vector

    def vector2text(self, vector):
        char_pos = vector.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            text.append(self.labels[c - i * len(self.labels)])
        return ''.join(text)

    def __text2sparse_tensor(self, sample_list):
        """
        文本转换成稀松矩阵
        :param sample_list: 随机列表
        :return:  稀松矩阵，矩阵中的值对应的labels下标列表，矩阵实际的shape [batch_size ,max_feature_num]
        """
        indices = []
        values = []
        batch_size = len(sample_list)
        for row, num in enumerate(sample_list):
            text = self.res[1][num]
            for col, char in enumerate(text):
                indices.append((row, col))
                values.append(self.labels.index(char))
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([batch_size, indices.max(0)[1]+1], dtype=np.int64)
        return indices, values, shape

    def sparse_tensor2text(self, indices, values):

        pass

    def next_batch(self, feature_num, batch_size = 200):
        """ 用于生成固定结果长度的图片张量和labels向量数据
        :param feature_num: 最大结果长度
        :param batch_size: 批次数量
        :return: 4-D tensor[batch_size, image_hight, image_width, 1] , 1-D tensor labels
        """
        image_files = self.res[0]
        batch_x = np.zeros([batch_size, self.shape[0] * self.shape[1]])
        batch_y = np.zeros([batch_size, feature_num * len(self.labels)])

        randomList = random.sample(range(0, len(image_files)), batch_size)
        for i, e in enumerate(randomList):
            path = image_files[e]
            text = self.res[1][e]
            batch_x[i, :] = cv2.imread(path, 0).flatten() / 255  # (image.flatten()-128)/128  mean为0
            batch_y[i, :] = self.__text2vector(text)
        print(batch_x.shape)
        return np.reshape(batch_x, [batch_size, self.shape[0], self.shape[1], 1]), batch_y

    def gen_batch(self, feature_num, batch_size = 200):
        """

        :param feature_num:
        :param batch_size:
        :return:
        """
        image_files = self.res[0]
        # batch_x = np.zeros([batch_size, self.shape[1], self.shape[0], 1])
        batch_x = np.zeros([batch_size, self.shape[1], self.shape[0], 3])
        batch_y = np.ones([batch_size, feature_num]) * -1
        # input_len = np.zeros_like([])
        while True:
            random_list = random.sample(range(0, len(image_files)), batch_size)
            for i, e in enumerate(random_list):
                path = image_files[e]
                text = self.res[1][e]
                # batch_x[i, :] = np.expand_dims(np.transpose(cv2.imread(path, 0)),axis=2)
                batch_x[i, :] = np.transpose(cv2.imread(path)) / 255
                batch_y[i, :4] = self.__text2index(text)
            #print(batch_x.shape)
            inputs = {'the_input': batch_x,
                      'the_labels': batch_y,
                      'input_length': np.expand_dims(np.ones([batch_size]) * (self.shape[1] / 4) - 2, axis=1), #(self.shape[1] / 8),#除以卷积次数
                      'label_length': np.expand_dims(np.ones([batch_size]) * 4, axis=1)
                      # 'source_str': source_str  # used for visualization only
                      }
            outputs = {'ctc': np.zeros([batch_size])}
            #return np.reshape(batch_x, [batch_size, self.shape[0], self.shape[1], 1]), batch_y
            yield (inputs, outputs)


    def next_sparse_batch(self, batch_size = 200):
        """
        生成稀松矩阵
        :param batch_size:
        :return:
            A 3-D tensor[batch_size, image_width=max_time_step, image_hight = num_featrue],
            tf.spareTensor
            seq_len :对于每张图片，num_feature的数量,由于是batch_size张图片， 返回1-D张量，每个元素值对应num_feature TODO:对于经过多重卷积池化后，会出现矩阵小于不定长结果的最大长度
        """
        image_files = self.res[0]
        batch_x = np.zeros([batch_size, self.shape[0] * self.shape[1]])
        random_list = random.sample(range(0, len(image_files)), batch_size)
        feature_indices, feature_values, feature_shape = self.__text2sparse_tensor(random_list)
        for i, e in enumerate(random_list):
            path = image_files[e]
            #TODO 调试
            # text = self.res[1][e]
            # print(text)
            batch_x[i, :] = cv2.imread(path, 0).flatten() / 255  # (image.flatten()-128)/128  mean为0

        #print(batch_x.shape)
        #return np.reshape(batch_x, [batch_size, self.shape[1], self.shape[0]]), tf.SparseTensor(values=feature_values, indices=feature_indices, dense_shape=feature_shape), np.ones((batch_size)) * self.shape[1]
        return np.reshape(batch_x, [batch_size, self.shape[1], self.shape[0]]), tf.SparseTensorValue(values=feature_values, indices=feature_indices, dense_shape=feature_shape), np.ones((batch_size)) * (len(self.labels))

    def seq_gen(self, batch_size):
        return ImageSequence(self.res,self.labels, batch_size, self.shape)

    def __init__(self, path, labels, shape):
        """

        :param path:  图片目录
        :param labels: 结果列表
        :param shape: 图片shape
        """
        self.res = self.__get_image_path_and_name(path)
        self.labels = labels
        self.shape = shape


codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
train_path = "/Volumes/d/t1/"
# imageTensor = ImageTensorBuilder(train_path, codeList, (60, 160))
# dd = imageTensor.gen_batch(16, 20)
# for i in dd:
#     print(i[0])
    # print(i[0]["the_input"].shape)
    # break
# for i in range(10):
#     print(imageTensor.next_sparse_batch(2))
#

#DIGITS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def decode_sparse_tensor(sparse_tensor):

    # print("sparse_tensor = ", sparse_tensor)
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    # print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        # print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
        print(result)
    return result


def decode_a_seq(indexes, spars_tensor):
    decoded = []
    DIGITS = "".join(codeList)
    for m in indexes:
        if spars_tensor[1][m] >= len(DIGITS):
            print(spars_tensor[1][m], len(DIGITS))
            continue
        str = DIGITS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    # str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    # str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    print(decoded)
    return decoded

# imageTensor = ImageTensorBuilder()
# sparseTensor = imageTensor.next_sparse_batch(10)[1]
# sparseTensor
#ge = imageTensor.gen_batch(4, 2)
#next(ge)
#print(sparseTensor)
#print("#################")
#decode_sparse_tensor(sparseTensor)

# print("0",imageTensor.next_sparse_batch(4)[1][0])
# print("1",imageTensor.next_sparse_batch(4)[1][1])
# print("2",imageTensor.next_sparse_batch(4)[1][2])


def create_sparse(batch_size, dtype=np.int32):
    '''
    创建稀疏张量,ctc_loss中labels要求是稀疏张量,随机生成序列长度在150～180之间的labels
    '''
    indices = []
    values = []
    for i in range(batch_size):
        length = random.randint(1,10)
        for j in range(length):
            indices.append((i,j))
            value = random.randint(0,779)
            values.append(value)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    print("max", indices.shape)
    print("max1",indices.max(0)[1]+1)
    print(np.asarray(indices).max(0)[1] + 1)
    shape = np.asarray([batch_size, np.asarray(indices).max(0)[1] + 1], dtype=np.int64) #[64,180]
    return
    #return [indices, values, shape]



#print(create_sparse(2))