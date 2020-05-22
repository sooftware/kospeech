import sys
import logging


class Logger(object):
    """ Print log message in format """
    def __init__(self):
        self.logger = logging.getLogger('root')
        FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
        self.logger.setLevel(logging.INFO)

    def info(self, message=''):
        self.logger.info(message)

    def debug(self, message=''):
        self.logger.debug(message)
