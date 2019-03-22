from keras.models import *
from keras.layers import *
import os
import random
import cv2
from keras.utils import Sequence
import math
import itertools

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
        return int(np.ceil(len(self.res[0]) / float(self.batch_size)))

    def __getitem__(self, idx):
        image_files = self.res[0]
        batch_x = np.zeros([self.batch_size, self.img_w, self.img_h, 3])
        # batch_x = np.zeros([self.batch_size, self.img_w, self.img_h, 1])
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

    def generator(self):
        while True:
            yield self.__getitem__(0)[0]

    #TODO __getitem__
    def next_val(self):
        while True:
            yield self.__getitem__(0)

    def tensor_to_text(self, target):
        ret = []
        for c in target:
            if c == len(self.codes):  # CTC Blank
                ret.append("")
            else:
                ret.append(self.codes[c])
        return "".join(ret)

    def __clean_spot(self, d):
        return np.delete(d, np.where((d) < 0 | (d == self.blank)))

    def equals_after_trim(self, origin, dest):
        _origin = self.__clean_spot(origin)
        _dest = self.__clean_spot(dest)
        if _origin.shape == _dest.shape:
            if (_origin == _dest).sum() == _origin.shape[0]:
                return True
        return False


codeList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
train_path = "/Volumes/d/t1/"
# imageTensor = ImageSequence(train_path,codeList,10,160,60,3)
# dd = imageTensor.generator()

# inp = next(dd)["the_labels"]
# print(inp.shape[0])

# print(np.delete(inp, np.where(inp == -1)))

# for j in range(inp.shape[0]):
#     print(np.delete(inp[j], np.where(inp[j] == -1)))
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





#print(create_sparse(2))