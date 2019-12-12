import logging

from job import *
from job.JdBroswer import MobileBroswer


def run():

    chrome_path = "/Volumes/messy/迅雷下载/chromedriver"
    broswer = MobileBroswer(chrome_path)

    for job_class in jobs:
        print(job_class)
        job = job_class(broswer)
        try:
            job.run()
        except Exception as e:
            logging.exception(e)

    print("eeeeee")


if __name__ == '__main__':
    run()