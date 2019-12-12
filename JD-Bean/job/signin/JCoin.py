import sys
# sys.path.append("..")
import time

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.touch_actions import TouchActions
# from ..signin import logger

import re
from job.JdPlayer import *


class Coin(JdPlayer):

    job_name = "京东金币"
    job_url = "https://wqsh.jd.com/pingou/task_center/task/index.html?tasktype=3"

    def __init__(self, broswer: MobileBroswer):
        super().__init__(broswer)

    def play(self):
        self.logger.info("Coin play")
        try:
            process = self.__find_process()
            while process[0] > 0:
                self.__view_adv()
                process = self.__find_process()
        except Exception as e:
            self.logger.exception(e)
            self.driver.quit()
            # time.sleep(60000)
        self.logger.info("{} finish!".format(self.job_name))

    def __find_process(self):
        """查找进度"""
        try:
            if not self.driver.current_url == self.job_url:
                self.driver.get(self.job_url)
            process = WebDriverWait(self.driver, 30).until(
                EC.visibility_of_element_located((By.CSS_SELECTOR, u".progress_wrap .text"))
            )
            self.__close_modal()
            if process:
                txt = process.text
                print(txt)
                num_list = re.findall('\d+', txt)
                if len(num_list) > 0:
                    return int(num_list[0]), process
            return 0, process
        except Exception as e:
            self.logger.exception(e)

    def __view_adv(self):
        elements = self.driver.find_elements(By.CSS_SELECTOR, ".task_btn.red")
        for e in elements:
            # action = TouchActions(self.driver)
            # action.tap(e)
            self.__close_modal()
            e.click()
            time.sleep(2)
            break
        # 转到任务页面
        self.driver.get(self.job_url)

    def __close_modal(self):
        """关闭遮罩"""
        try:
            time.sleep(1)
            modal = self.driver.find_elements(By.CSS_SELECTOR, ".modal_close")
            for e in modal:
                e.click()
        except Exception as e:
            self.logger.exception(e)
            pass

    def is_play(self):
        return self.__find_process()[0] == 0