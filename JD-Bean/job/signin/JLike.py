import sys
# sys.path.append("..")
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import re
from job.JdPlayer import *


class Like(JdPlayer):

    job_name = "取消收藏"
    job_url = "https://vip.m.jd.com/"


    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)


    def play(self):
        try:
            myjobs = [
                        Goods(self.broswer),
                        Shop(self.broswer)
                        ]
            for job in myjobs:
                self.to_page(job.job_url)
                if not job.is_play():
                    job.play_job()
            # self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()

    def is_play(self):
        return False


class Goods(JdPlayer):
    job_name = "商品收藏"

    job_url = "https://wqs.jd.com/my/fav/goods_fav.shtml?ptag=7155.1.8&sceneval=2"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:


            while self.fav_num() > 0:
                edit = WebDriverWait(self.driver, 15).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, u"#edit_btn"))
                )

                if "编辑" in edit.text:
                    edit.click()

                for i in range(0,4):
                    self.driver.execute_script(script="window.scrollTo(0,document.body.scrollHeight)")
                    time.sleep(1)

                btn = self.driver.find_element_by_css_selector("#selectAllBtn")

                if not "selected" in btn.get_attribute("class"):
                    btn.click()
                    # self.driver.execute_script(script='$("#selectAllBtn").click()')
                    time.sleep(1)

                self.driver.execute_script(script='$("#multiCancle").click()')
                time.sleep(1)
                confirm = WebDriverWait(self.driver, 15).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, u"#ui_btn_confirm"))
                )
                confirm.click()
                time.sleep(1)

            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()

            # time.sleep(60000)
        # self.logger.info("{} finish!".format(self.job_name))

    def fav_num(self) -> int:
        try:
            nothing = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u"#commlist_nothing"))
            )

            if nothing.is_displayed():
                self.logger.info("商品收藏0个")
                return 0

            fav = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u"#fav_total_num"))
            )
            if fav.text:
                self.logger.info("商品收藏{}个".format(fav.text))
                return int(fav.text)
        except Exception as e:
            self.logger.exception(e)
            self.logger.info("商品收藏0个")
            return 0

    def is_play(self):
        return self.fav_num() < 100


class Shop(JdPlayer):
    job_name = "商品收藏"

    job_url = "https://wqs.jd.com/my/fav/shop_fav.shtml?sceneval=2"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:


            while self.fav_num() > 0:
                edit = WebDriverWait(self.driver, 15).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, u"#shoplist_edit"))
                )

                if "编辑" in edit.text:
                    edit.click()

                for i in range(0,2):
                    self.driver.execute_script(script="window.scrollTo(0,document.body.scrollHeight)")
                    time.sleep(1)

                btn = self.driver.find_element_by_css_selector("#selectAllBtn")

                if not "selected" in btn.get_attribute("class"):
                    btn.click()
                    # self.driver.execute_script(script='$("#selectAllBtn").click()')
                    time.sleep(1)

                self.driver.execute_script(script='$("#multiCancle").click()')
                time.sleep(1)
                confirm = WebDriverWait(self.driver, 15).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, u"#ui_btn_confirm"))
                )
                confirm.click()
                time.sleep(1)

            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()

            # time.sleep(60000)
        # self.logger.info("{} finish!".format(self.job_name))

    def fav_num(self) -> int:
        try:
            nothing = WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u"#shoplist_nothing"))
            )

            if nothing.is_displayed():
                self.logger.info("店铺收藏0个")
                return 0

            fav = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u"#shoplist_count"))
            )
            if fav.text:
                self.logger.info("店铺收藏{}个".format(fav.text))
                return int(fav.text)
        except Exception as e:
            self.logger.exception(e)
            self.logger.info("店铺收藏0个")
            return 0

    def is_play(self):
        return self.fav_num() < 100


