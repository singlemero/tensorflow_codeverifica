import sys
# sys.path.append("..")
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import re
from job.JdPlayer import *


class Bean(JdPlayer):

    job_name = "领京豆"

    job_url = "https://vip.m.jd.com/"

    sign_urls = []

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        try:
            myjobs = [
                # BeanSign(self.broswer),
                #        ViewGoods1(self.broswer),
                       ViewGoods2(self.broswer)]
            for job in myjobs:
                self.to_page(job.job_url)
                job.play_job()
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()

    def is_play(self):
        return False

class BeanSign(JdPlayer):

    job_name = "签到领京豆"

    job_url = "https://bean.m.jd.com/rank/index.action"

    job_urls = ["https://bean.m.jd.com/rank/index.action", #京豆签到
                "https://vip.m.jd.com/page/signin"
                ]

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        for url in self.job_urls:
            self.driver.get(url)
            time.sleep(1)

    def is_play(self):
        return False


class ViewGoods1(JdPlayer):

    job_name = "京东热卖"

    job_url = "https://jdde.jd.com/btyingxiao/advertMoney/html/home.html?from=kggicon "

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            i = 0
            run = True
            while run and i < 6:
                self.to_page(self.job_url)
                run = self.__view_adv(i)
                i = i + 1
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()
            # time.sleep(60000)
        self.logger.info("{} finish!".format(self.job_name))



    def __view_adv(self, index)->bool:
        try:
            elements = WebDriverWait(self.driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, u".index_list_banner"))
            )
            if len(elements) > index:

                self.__close_modal()
                self.driver.execute_script(script='$(".index_list_banner").eq({}).click()'.format(index))
                # webdriver.ActionChains(self.driver).move_to_element(elements[index]).click(elements[index]).perform()
                # elements[index].click()
                # 等待跳转
                time.sleep(5)
                section = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, u"section"))
                )
                print(section)
                divs = section.find_elements_by_css_selector("section div div")
                for div in divs:
                    if div.text and "无法重复领取" in div.text:
                        return True
                # time.sleep(15)
                present = WebDriverWait(self.driver, 15).until(
                    EC.text_to_be_present_in_element((By.CSS_SELECTOR, u"section div div i:last-child"), "领取")
                )
                if present:
                    pick = self.driver.find_element_by_css_selector("section div div i:last-child")
                    # self.driver.execute_script(script='$("section div div i:last-child").click()')
                    webdriver.ActionChains(self.driver).move_to_element(pick).click(pick).perform()
                # self.driver.find_element_by_css_selector("section div div i:last-child")
                # self.driver.execute_script('$("section div div i:last-child").click()')
                time.sleep(3)
                # for elem in self.driver.find_elements_by_css_selector("section div div i:last-child"):
                #     elem.click()
            # 转到任务页面
            self.driver.get(self.job_url)
            return True
        except Exception as e:
            self.logger.exception(e)
            return False

    def __close_modal(self):
        """关闭遮罩"""
        # try:
        #     time.sleep(1)
        #     modal = self.driver.find_elements(By.CSS_SELECTOR, ".ui-btn img")
        #     for e in modal:
        #         e.click()
        # except Exception as e:
        #     self.logger.exception(e)
        #     pass
        pass

    def is_play(self):
        return False



class ViewGoods2(JdPlayer):

    job_name = "浏览商品活动2"

    job_url = "https://jddx.jd.com/m/reward/product-list.html?from=kggicon "

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            i = 0
            run = True
            while run and i < 6:
                self.to_page(self.job_url)
                time.sleep(15)
                run = self.__view_adv()
                i = i + 1
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()
            # time.sleep(60000)
        self.logger.info("{} finish!".format(self.job_name))



    def get_view_btn(self):
        elements = self.driver.find_elements(By.CSS_SELECTOR, "section")
        btn = elements[-1].find_element(By.CSS_SELECTOR, "div div i:last-child")
        count = 0
        while "领取" != btn.text and count < 3:
            if "赚更多钱" == btn.text:
                break
            elements = self.driver.find_elements(By.CSS_SELECTOR, "section")
            btn = elements[-1].find_element(By.CSS_SELECTOR, "div div i:last-child")
            count = count + 1
        return btn

    def __view_adv(self) -> bool:
        try:
            btn = self.get_view_btn()
            if "赚更多钱" != btn.text:
                # btn.click()
                webdriver.ActionChains(self.driver).move_to_element(btn).click(btn).perform()
                time.sleep(1)
                # 因为就在此页面，不用休眠，
                # 转到任务页面
                self.to_page(self.job_url)
                return True
        except Exception as e:
            self.logger.exception(e)
            return False

    def is_play(self):
        return False



