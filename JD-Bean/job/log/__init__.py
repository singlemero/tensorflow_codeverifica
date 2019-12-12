import logging
import sys

logging.basicConfig(level=logging.INFO)
__all__ = ['Jlog']


class Jlog:

    @classmethod
    def getLogger(cls, name) -> logging:
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        log_format = logging.Formatter("%(asctime)s - %(message)s")  # output format
        sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
        sh.setFormatter(log_format)
        logger.addHandler(sh)
        return logger

