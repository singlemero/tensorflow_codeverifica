import time

from job.JdBroswer import MobileBroswer
from job.log import Jlog

class JdPlayer:

    job_name = "京豆京豆"
    job_url = ""

    job_success = False

    logger = Jlog.getLogger("JdPlayer")

    def __init__(self, broswer: MobileBroswer):
        self.broswer = broswer
        self.driver = broswer.getDriver()
        pass

    def run(self):
        self.broswer.driver.get(self.job_url)
        if self.is_login():
            if not self.is_play():
                self.play_job()
                self.logger.info("Job {} {}!".format(self.job_name, self.job_success))

    def is_login(self):
        """判断是否已登录"""
        return self.broswer.login(self.job_url)

    def play_job(self):
        """执行任务防止任务异常"""
        try:
            self.logger.info('Job Start: {}'.format(self.job_name))
            self.play()
        except Exception as e:
            self.logger.error("Job {} Exception!".format(self.job_name))
            self.logger.exception(e)

    def play(self):
        """执行任务"""
        self.logger.warning("JdPlayer play")
        pass

    def to_page(self, url):
        """去到指定页面"""
        if not self.driver.current_url == url:
            self.driver.get(url)
            time.sleep(2)

    def is_play(self) -> bool:
        """任务是否已执行"""
        return True



