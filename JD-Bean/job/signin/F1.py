import re


#转换图片
from functools import reduce

str = "//img30.360buyimg.com/shaidan/s128x96_jfs/t1/67138/34/17072/101481/5de21d12E2c500429/82ee74675daeba45.jpg!cc_100x100!q70.dpg.webp"


str = "https:"+ str


match_patten = re.findall(r's[0-9]{2,4}x[0-9]{2,4}_|!.*', str)

move = lambda img, o : img.replace(o, "")

def find_last(string,str):
    last_position=-1
    while True:
        position=string.find(str,last_position+1)
        if position==-1:
            return last_position
        last_position=position



for x in match_patten:
    str = move(str,x)

pos = find_last(str,".")
# reduce(move, match_patten)
# match_patten
# img = str.replace(img_size[0], "")

# str.index()
print(match_patten, str, str[pos:])