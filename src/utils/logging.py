import logging
import sys

class DualHandler:

    def __init__(self, stream, file_stream):
        self.file_stream = file_stream
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file_stream.write(data)
        self.file_stream.flush()

    def flush(self):
        self.file_stream.flush()
        self.stream.flush()

class LevelFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


def setup_logger(file_name='main'):
    """Setup logger name, handler for files and command line"""

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # Hack here to have a file to record both stdout and DEBUG level loggin to a file, 
    # and record at the same time stdout and INFO level logging to console
    fh = logging.FileHandler(file_name + '.log', mode='w', encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    fh.addFilter(LevelFilter(logging.DEBUG))

    dh = logging.StreamHandler(DualHandler(sys.stdout, fh.stream))
    dh.setLevel(logging.INFO)
    dh.setFormatter(formatter)

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    rootLogger.addHandler(dh)
    rootLogger.addHandler(fh)
