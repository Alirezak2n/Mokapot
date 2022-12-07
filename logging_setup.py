import logging


def main(file_name='entrapment_logs.log'):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s: %(filename)s %(funcName)s %(name)s - %(message)s')
    formatter_console = logging.Formatter('%(asctime)s | %(message)s')

    # create logger_console handler and set level to debug
    logger_console = logging.StreamHandler()
    logger_console.setLevel(logging.INFO)

    # create file handler and set level to debug
    logger_file = logging.FileHandler(filename=file_name)
    logger_file.setLevel(logging.INFO)

    # add formatter ,logger_console and file handler
    logger_console.setFormatter(formatter_console)
    logger_file.setFormatter(formatter)
    logger.addHandler(logger_console)
    logger.addHandler(logger_file)

    return logger


def basic_logger():
    return logging.basicConfig(
        level=logging.NOTSET, format="%(asctime)s | %(levelname)s: %(filename)s %(funcName)s %(lineno)s - %(message)s"
        , filename='entapment.log')
