import logging
import os

def root_path():
    return os.getcwd()

default_dir_path = os.path.join(root_path(), "log")

class Logger:
    def __init__(self, logger_name, dir_path=None):
        if dir_path == None:
            dir_path = default_dir_path
        os.makedirs(dir_path, exist_ok=True)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        file_path = os.path.join(dir_path, logger_name + ".log")
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

if __name__ == '__main__':
    logger = Logger("test_log")
    logger.info("test")