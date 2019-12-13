import sys
# sys.path.append("..")
import time

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

import re
from job.JdPlayer import *


class Home(JdPlayer):

    job_name = "京东首页任务"
    job_url = "https://vip.m.jd.com/"


    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)


    def play(self):
        try:
            myjobs = [
                        Paipai(self.broswer),
                        Clock(self.broswer),
                        Life(self.broswer),
                        Cosmetics(self.broswer),
                        Pet(self.broswer),
                        Monther(self.broswer),
                        Roll(self.broswer),
                        PinGou(self.broswer)
                        ]
            for job in myjobs:
                self.to_page(job.job_url)
                if not job.is_play():
                    job.play_job()
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()

    def is_play(self):
        return False


class Paipai(JdPlayer):
    job_name = "拍拍二手"

    job_url = "https://pro.m.jd.com/mall/active/LHGZv1DrGkva1jNpUkKFuNFN6oo/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        # self.logger.info(self.job_name)
        try:
            icons = WebDriverWait(self.driver, 15).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, u"div[data-src]"))
            )
            if len(icons) == 2:
                webdriver.ActionChains(self.driver).move_to_element(icons[1]).click(icons[1]).perform()
                time.sleep(3)
                sign = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_btnTxt"))
                )
                if not "连续签到" in sign.text:
                    sign.click()
                # webdriver.ActionChains(self.driver).move_to_element(sign).click(sign).perform()

                    WebDriverWait(self.driver, 5).until(
                        EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close"))
                    )
                else:
                    self.logger.info("{}已签到".format(self.job_name))
                # time.sleep(3)
                self.job_success = True
        except Exception as e:
            self.logger.exception(e)

    def is_play(self):
        icons = WebDriverWait(self.driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, u"div[data-src]"))
        )
        return len(icons) == 1


class Block(JdPlayer):
    job_name = "各品类分馆"

    sign_btn = ".signIn_btnTxt"

    sing_close = ".signIn_Close_icon"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = self.get_sign_btn()
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, self.sing_close))
            )
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def get_sign_btn(self):
        return WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, self.sign_btn))
        )

    def is_play(self):
        sign = self.get_sign_btn()
        return "连续签到" in sign.text


class Clock(JdPlayer):
    job_name = "钟表馆"

    job_url = "https://pro.m.jd.com/mall/active/2BcJPCVVzMEtMUynXkPscCSsx68W/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_bg"))
            )
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close_icon"))
            )
            # time.sleep(2)
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False

class Life(JdPlayer):
    job_name = "生活馆"

    job_url = "https://pro.m.jd.com/mall/active/2C4Az1JUCWN8f3Y6xaxHbzTLkUzC/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_btnTxt"))
            )
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close_icon"))
            )
            # time.sleep(2)
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False


class Cosmetics(JdPlayer):
    job_name = "美妆个护"

    job_url = "https://pro.m.jd.com/mall/active/NJ1kd1PJWhwvhtim73VPsD1HwY3/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_btnTxt"))
            )
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close_icon"))
            )
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False


class Pet(JdPlayer):
    job_name = "宠物馆"

    job_url = "https://pro.m.jd.com/mall/active/3GCjZzanFWbJEU4xYEjqfPfovokM/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_btnTxt"))
            )
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close_icon"))
            )
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False


class Monther(JdPlayer):
    job_name = "母婴馆"

    job_url = "https://pro.m.jd.com/mall/active/bVs9EG4MMK4zKdqVt86UFABX2en/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".signIn_btnTxt"))
            )
            sign.click()
            close_modal = WebDriverWait(self.driver, 5).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".signIn_Close_icon"))
            )
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False


class Roll(JdPlayer):
    job_name = "摇一摇"

    job_url = "https://vip.jd.com/newPage/reward"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info(self.job_name)
        try:
            sign = WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, u".rewardBoxBot.J_ping"))
            )
            while "摇一摇" in sign.text:
                # sign.click()
                webdriver.ActionChains(self.driver).move_to_element(sign).click(sign).perform()
                time.sleep(6)
                modal = WebDriverWait(self.driver, 15).until(
                    EC.visibility_of_element_located((By.CSS_SELECTOR, u".common-popup-close"))
                )
                webdriver.ActionChains(self.driver).move_to_element(modal).click(modal).perform()
                time.sleep(1)
                sign = WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, u".rewardBoxBot.J_ping"))
                )
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False


class PinGou(JdPlayer):
    job_name = "拼购"

    job_url = "https://wqsh.jd.com/pingou/taskcenter/index.html"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        # self.logger.info(self.job_name)
        try:
            modal = WebDriverWait(self.driver, 15).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".modal_close"))
            )
            webdriver.ActionChains(self.driver).move_to_element(modal).click(modal).perform()
            time.sleep(2)
            self.job_success = True
        except Exception as e:
            self.logger.exception(e)
            # self.driver.quit()
        self.logger.info("{} finish!".format(self.job_name))

    def is_play(self):
        return False
