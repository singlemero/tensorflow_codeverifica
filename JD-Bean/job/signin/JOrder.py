import os
import random
import sys
# sys.path.append("..")
import time
import urllib

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.touch_actions import TouchActions
# from ..signin import logger
import ssl
import re
from job.JdPlayer import *


class Order(JdPlayer):

    job_name = "京东订单评论"
    #这是我的页面，须要手动进入订单
    # job_url = "https://home.m.jd.com/myJd/home.action"
    #这是订单，须要选已完成
    job_url = "https://wqs.jd.com/order/orderlist_merge.shtml?tab=1&ptag=7155.1.11&sceneval=2"

    order_detail_url = "https://wqs.jd.com/order/n_detail_v2.shtml?deal_id={}&isoldpin=0&sceneval=2"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info("京东订单评论")
        try:
            self.__do_process()
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()
            # time.sleep(6000)
            # self.driver.quit()


    def __do_process(self):
        """评论订单"""
        #查找未评论订单
        orderids = self.__find_uncomment_orderid()
        num = 0
        while len(orderids) <= 0 and num < 3:
            orderids = self.__find_uncomment_orderid()
            num = num + 1


        orderids = list(set(orderids))

        for id in orderids:
            product_urls = self.__find_order_detail(id)
            items = []
            for index, url in enumerate(product_urls):
                item = Item(id, index, url, self.driver)
                item.parser()
                items.append(item)
            #评论
            remark = Remark(self.broswer, id, items)
            remark.play()

    def __find_order_detail(self, orderid):
        """进入订单详情"""
        url = self.order_detail_url.format(orderid)
        self.to_page(url)
        #获取订单详情的产品列表
        products = self.driver.find_elements(By.CSS_SELECTOR, "div[data-id=product]")

        product_urls = list(map(lambda e: e.get_attribute("data-url"), products))
        return product_urls


    def __find_uncomment_orderid(self):
        """查找未评论的订单"""
        self.__toggle_finish()
        get_uncomment = lambda : self.driver.find_elements(By.CSS_SELECTOR, ".oh_btn.bg_white[data-event=onComment]")

        uncomment = get_uncomment()
        scroll_time = 0
        while len(uncomment) <= 0 or scroll_time <= 3:
            #下拉到页面底部
            self.driver.execute_script(script="window.scrollTo(0,document.body.scrollHeight)")
            #等待页面加载
            time.sleep(1)
            uncomment = get_uncomment()
            scroll_time = scroll_time + 1
        return list(map(lambda e: e.get_attribute("data-orderid"), uncomment))

    def __toggle_finish(self):
        """转到已完成订单"""
        self.to_page(self.job_url)
        if "orderlist_merge" in self.driver.current_url:
            finish_tab = self.driver.find_element_by_css_selector(".my_nav_list_item[data-tabid=waitComment]")
            finish_tab.click()
        else:
            self.logger.info("当前不是订单页面")

    def is_play(self):
        return False


class Item:

    def __init__(self, orderid, item, url, driver: WebDriver):
        self.orderid = orderid
        self.url = url
        self.item = item
        self.driver = driver
        self.comment = ""
        self.imgs = []
        self.local_imgs = []
        self.random_comment = None

    def parser(self):
        self.toggle_home_page()
        self.locate_comment()
        self.__handle_img()

    def locate_comment(self):
        comments = self.driver.find_elements(By.CSS_SELECTOR, "#evalDet_summary li")

        for ix in range(len(comments) - 1, -1, -1):
            #评论
            elem = comments[ix]
            if not self.comment:
                text_elements = elem.find_elements_by_css_selector(".cmt_cnt")
                if len(text_elements) ==0 :
                    continue
                text_ele = text_elements[0]
                comment_txt = text_ele.text
                if comment_txt and len(comment_txt) > 15:
                    self.comment = comment_txt
                    # print(self.comment)

            #图片大于等于3
            imgs = elem.find_elements(By.CSS_SELECTOR,".cmt_att span img")
            if len(imgs) >= 2:
                filter_imgs = [x for x in imgs if x.get_attribute("src")]
                load_times = 0

                while len(filter_imgs) < 2 and load_times < 3:
                    print("暂时无法加载图片")
                    imgs = elem.find_elements(By.CSS_SELECTOR, ".cmt_att span img")
                    filter_imgs = [x for x in imgs if x.get_attribute("src")]
                    load_times = load_times + 1
                    time.sleep(2)

                if len(filter_imgs) >=2:
                    # if
                    # self.comment = text_ele.text
                    img_paths = []
                    for i in filter_imgs:
                        img_paths.append(i.get_attribute("src"))
                    self.imgs = img_paths
                    break

        if len(comments) > 0:
            if not self.comment:
                self.random_comment = self.get_comment_from_all()
                if self.random_comment:
                    self.comment = self.random_comment.find_element_by_css_selector(".cmt_cnt").text
            if len(self.imgs) == 0:
                if self.random_comment:
                    imgs = self.random_comment.find_elements(By.CSS_SELECTOR, ".cmt_att span img")
                    if not len(imgs) == 0:
                        self.imgs = list(map(lambda x: x.get_attribute("src"), imgs))

                # imgs = elem.find_elements(By.CSS_SELECTOR, ".cmt_att span img")
                # if not len(imgs) == 0:
                #     self.imgs = list(map(lambda x:x.get_attribute("src"), imgs))
                # elif self.random_comment:
                #     imgs = self.random_comment.find_elements(By.CSS_SELECTOR, ".cmt_att span img")
                #     if not len(imgs) == 0:
                #         self.imgs = list(map(lambda x: x.get_attribute("src"), imgs))



    def get_comment_from_all(self):
        self.driver.execute_script(script='$(".info_label").click()')
        comments = self.driver.find_elements(By.CSS_SELECTOR, "#evalDet_summary li")
        if len(comments) <= 1 : #默认会有一条提醒没有评论的评论
            self.driver.execute_script(script='$("#evalTag2 span").filter(function(index){return this.innerHTML.indexOf("全部")!=-1}).click()')
        time.sleep(1)
        comments = self.driver.find_elements(By.CSS_SELECTOR, "#evalDet_summary li")
        if len(comments) > 1:
            random_int = random.randint(1, len(comments) -1)
            return comments[random_int]
        return None



    def toggle_home_page(self):
        if not self.driver.current_url == self.url:
            self.driver.get(self.url)
            time.sleep(1)
        self.driver.execute_script(script='$("#summaryEnterIco").click()')
        time.sleep(1)
        self.driver.execute_script(script='$(".info_label").click()')
        # summary = self.driver.find_element_by_css_selector("#summaryEnter")
        # summary.click()
        time.sleep(1)
        #仅看有图
        # spans = self.driver.find_elements(By.CSS_SELECTOR, "#evalTag2 span")
        # for span in spans:
        #     if "有图" in span.text:
        #         span.click()
        #         等待刷新
                # time.sleep(1)
                # break
        self.driver.execute_script(script='$("#evalTag2 span").filter(function(index){return this.innerHTML.indexOf("有图")!=-1}).click()')
        time.sleep(1)

        #滚动24次，查找比较靠后的评论

        for i in range(1, 24):
            self.driver.execute_script(script="window.scrollTo(0,document.body.scrollHeight)")
            time.sleep(1)
        #等待5秒
        time.sleep(5)

    def urllib_download(self, index, img_url) -> str:
        from urllib.request import urlretrieve
        img_dir = "{}/img/{}/{}/".format(os.getcwd(),self.orderid, self.item)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        img_suffix = self.find_last(img_url, ".")
        img_path = img_dir+str(index)+ img_url[img_suffix:]

        context = ssl._create_unverified_context()
        pic_data_url = urllib.request.urlopen(img_url, context=context)  # 在urllib2启用ssl字段,打开请求的数据。如果是http的话此行去除字段context=context
        pic_data = pic_data_url.read()

        f = open(img_path, "wb")
        f.write(pic_data)
        f.close()
        # urlretrieve(img_url, img_path)
        return img_path

    def find_last(self, string, str):
        last_position = -1
        while True:
            position = string.find(str, last_position + 1)
            if position == -1:
                return last_position
            last_position = position

    def __handle_img(self):
        move_unnecessary = lambda img, o: img.replace(o, "")
        try:
            for index, img in enumerate(self.imgs):
                img_url = img
                match_patten = re.findall(r's[0-9]{2,4}x[0-9]{2,4}_|!.*', img_url)
                for i in match_patten:
                    img_url = move_unnecessary(img_url, i)
                my_path = self.urllib_download(index, img_url)
                self.local_imgs.append(my_path)
        except Exception as e:
            self.logger.warning(self.imgs)
            self.logger.exception(e)


class Remark(JdPlayer):

    remark_url = "https://wqs.jd.com/wxsq_project/comment/evalProduct/index.html?orderid={}&ordertype=0&sceneval=2"

    def __init__(self, broswer: MobileBroswer, orderid, items:list):
        super().__init__(broswer)
        self.detail_url = self.remark_url.format(orderid)
        self.items = items

    def play(self):
        self.logger.info("评论订单: {}".format(self.detail_url))
        self.to_page(self.detail_url)
        time.sleep(2)
        try:
            if len(self.items) > 1:
                for index, item in enumerate(self.items):
                    self.to_page(self.detail_url)
                    action_btns = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR,u".action"))
                    )
                    reverse_index = len(action_btns) - 1 - index
                    btn = action_btns[reverse_index]
                    if btn.find_element_by_css_selector("a").text == "查看评价":
                        continue
                    btn.click()
                    self.comment(index)

            # action_btns = self.driver.find_elements(By.CSS_SELECTOR, ".action")
            # if len(action_btns) > 0:
            #     for index, btn in enumerate(action_btns):
            #         self.to_page(self.detail_url)
            #         # presence_of_all_elements_located
            #
            #         if btn.find_element_by_css_selector("a").text == "查看评价":
            #             continue
            #         btn.click()
            #         time.sleep(1)
            #         reverse_index = len(action_btns) -1 - index
            #         self.comment(reverse_index)
            else:
                self.comment(0)
        except Exception as e:
            self.logger.exception(e)
            print("订单{}评论失败！".format(self.detail_url))

    def comment(self, index):
        # 点星
        self.driver.execute_script(script='$(".stars_list li div i:last-child").click()')
        # 评论
        item = self.items[index]
        # self.driver.execute_script(script='$(".textarea_wrap textarea").val("{}")'.format(item.comment))
        text_area = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, u".textarea_wrap textarea"))
        )
        try:
            text_area.send_keys(item.comment)
        except Exception as e:
            self.logger.warning("无法发送评论内容！")
            self.logger.exception(e)

        # 上传图片local_imgs
        input_btn = self.driver.find_element_by_css_selector("input[accept='image/*']")

        for pic in item.local_imgs:
            input_btn.send_keys(pic)
        # 如果有添加图片，提交评论
        if len(item.local_imgs) > 0:
            self.driver.execute_script(script='$(".comment_btns a").click()')
            self.logger.info("订单文字评论: {}".format(item.comment))
            self.remove_file(index)
            time.sleep(1)

    def remove_file(self, index):
        item = self.items[index]
        for pic in item.local_imgs:
            if os.path.exists(pic):
                os.remove(pic)

