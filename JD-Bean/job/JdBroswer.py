import os
import pickle
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from job.log import Jlog


class MobileBroswer():

    home_page = "https://m.jd.com"
    logger = Jlog.getLogger("MobileBroswer")

    def __init_cookie(self):
        cookie_dir = os.getcwd() + "/data/"
        if not os.path.exists(cookie_dir):
            os.makedirs(cookie_dir)
        self.cookie_path = cookie_dir + "cookiesChrome"
        self.driver.get(self.home_page)
        self.load_cookie()

    def __init_option(self):
        chrome_options = webdriver.ChromeOptions()
        chrome_options.binary_location = ""
        chrome_options.accept_untrusted_certs = True
        chrome_options.assume_untrusted_cert_issuer = True
        chrome_options.add_argument("--ignore-certificate-errors")
        # chrome_options.add_argument("--headless")
        # if self.headless:
        # log.info("use headless")
        #以适应手机分辨率形式展示
        chrome_options.add_experimental_option("mobileEmulation", {"deviceName": "Pixel 2 XL" })
        chrome_options.add_experimental_option('w3c', True)
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("window-size=800*1280")
        chrome_options.add_argument('user-agent="Mozilla/5.0 (Linux; Android 8.0.0; Pixel 2 XL Build/OPD1.170816.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.97 Mobile Safari/537.36"')
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--hide-scrollbars")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-impl-side-painting")
        chrome_options.add_argument("--dtime.sleep(1)isable-setuid-sandbox")
        chrome_options.add_argument("--dtime.sleep(1)isable-breakpad")
        chrome_options.add_argument("--dtime.sleep(1)isable-client0side-phishing-detection")
        chrome_options.add_argument("--disable-cast")
        chrome_options.add_argument("--disable-cast-streaming-hw-encoding")
        chrome_options.add_argument("--disable-cloud-import")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-sesion-crashed-bubble")
        chrome_options.add_argument("--disable-ipv6")
        chrome_options.add_argument("--allow-http-screen-capture")
        chrome_options.add_argument("--start-maximized")
        # 当使用selenium进行自动化操作时，
        # 在chrome浏览器中的consloe中输入windows.navigator.webdriver会发现结果为Ture，而正常使用浏览器的时候该值为False
        # 此步骤很重要，设置为开发者模式，防止被各大网站识别出来使用了Selenium
        chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
        return chrome_options

    def __init__(self, chrome_path):
        self.driver = webdriver.Chrome(executable_path=chrome_path, options=self.__init_option())
        self.__init_cookie()
        pass

    def login(self, job_url):
        try:
            need_login = WebDriverWait(self.driver, 5).until(
                EC.title_contains(u"京东登录注册")
            )
            # need_login = False
            login_success = False
            if need_login:
                self.logger.info("未登录!")
                # return True
                roll_times = 0
                while job_url not in self.driver.current_url and roll_times < 200:
                    # print(driver.current_url)
                    # print(roll_times)
                    time.sleep(5)
                    roll_times = roll_times + 1

                if job_url not in self.driver.current_url:
                    self.logger.error("超时未登录!")
                else:
                    login_success = True
                    self.logger.info("登录成功！")
                    # cookies = self.driver.get_cookies()
                    self.save_cookie()
                return login_success
                # return True
            else:
                return True
        except Exception as e:
            self.logger.info("找不到关键字[京东登录注册]")
            print(e)
            return True
            # self.driver.quit()

    def load_cookie(self):
        if os.path.exists(self.cookie_path):
            self.logger.info("加载cookie配置！")
            with open(self.cookie_path, 'rb') as f:
                data = pickle.loads(f.read())
                for cookie in data:
                    if "expiry" in cookie:
                        cookie["expiry"] = int(cookie["expiry"])
                    self.driver.add_cookie(cookie)

    def save_cookie(self):
        self.logger.info("保存cookies")
        data = pickle.dumps(self.driver.get_cookies())
        data_file = Path(self.cookie_path)
        data_file.write_bytes(data)


    def getDriver(self):
        return self.driver

print(os.getcwd())