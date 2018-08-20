import logging


def make_logger(name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(0)
    formatter = logging.Formatter('%(asctime)s - %(name)s '
                                  '- %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
