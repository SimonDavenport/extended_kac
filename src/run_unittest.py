## This file discovers and runs all package unit tests

import unittest
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.utils.logging

if __name__ == '__main__':

    src.utils.logging.setup_logger(file_name='test_algebra')
    log = logging.getLogger('logger')

    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(__file__))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
