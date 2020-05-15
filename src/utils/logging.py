import logging

_default_logger_name = 'logger'

def setup_logger(name=_default_logger_name, file_name='main'):
    """Setup logger name, handler for files and command line"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    fh = logging.FileHandler(file_name + '.log', mode='w', encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def get_logger(name=_default_logger_name):
    """Get the current logger from the global scope"""
    return logging.getLogger(name)
