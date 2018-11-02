# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
from PIL import Image
import random
import cv2
import os
import math
 
# 验证码中的字符
#number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
 
# alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
#             'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
             'V', 'W', 'X', 'Y', 'Z']
 
# 验证码长度为4个字符
def random_captcha_text(char_set=ALPHABET, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text
 
 
# 生成字符对应的验证码
def gen_captcha_text_and_image(text=ALPHABET):

    DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
    DEFAULT_FONTS = [os.path.join("/Users/konglinghong/Downloads/", 'BarkingCatDEMO.ttf')]
    image = ImageCaptcha(fonts=DEFAULT_FONTS)



    captcha_text = random_captcha_text()
    #captcha_text = ''.join(captcha_text)
 
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image
 
 
if __name__ == '__main__':
    #保存路径
    path = ['/Volumes/d/t1', '/Volumes/d/t2']
    #path = '/Volumes/d/t2'
    #path = './validImage'
    for index, p in enumerate(path):
        ran = 10000/math.pow(10, index)
        for i in range(int(ran)):
            text, image = gen_captcha_text_and_image(i)
            fullPath = os.path.join(p, "".join(text) + ".jpg")
            #print(fullPath)
            cv2.imwrite(fullPath, image)
            print("{0}/{1}".format(i, math.pow(10, index )))
    print("/nDone!")

